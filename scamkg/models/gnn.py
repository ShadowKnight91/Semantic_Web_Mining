"""E4 - GraphSAGE on the call knowledge graph.

Homogeneous view of the heterogeneous KG: call nodes + entity nodes, undirected call-entity edges
(optionally call-call similarity edges). Node features: sentence embeddings (SBERT, or TF-IDF+SVD
when sentence-transformers is unavailable) of the transcript / the entity name, plus a node-type
one-hot. Two SAGE layers, mean aggregation, 64 hidden units, dropout 0.3, Adam; loss on training
call nodes only, early stopping on validation PR-AUC. Used (a) end-to-end ("graphsage") and
(b) as an embedding generator for XGBoost ("graphsage_xgb").
PyTorch Geometric's SAGEConv is used when installed; otherwise an equivalent pure-PyTorch layer.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score

from scamkg.evaluation.evaluator import Evaluator
from scamkg.graph.build import split_key
from scamkg.models.classical import make_model
from scamkg.models.transformer import call_embeddings, require, sbert_encode
from scamkg.utils import get_logger, has_module, set_seed

log = get_logger(__name__)


def build_homogeneous(ctx, use_similarity_edges: bool = False) -> tuple[np.ndarray, int, int]:
    kg = ctx.kg
    n_c = kg.n_calls
    n = n_c + len(kg.entity_ids)
    coo = kg.X.tocoo()
    src = np.concatenate([coo.row, coo.col + n_c])
    dst = np.concatenate([coo.col + n_c, coo.row])
    if use_similarity_edges:
        C = kg.call_call_graph(int(ctx.cfg["graph"].get("similarity_top_k", 10)),
                               float(ctx.cfg["graph"].get("similarity_min", 0.2)))
        e = np.array([(kg.call_index[a], kg.call_index[b]) for a, b in C.edges()], dtype=int).reshape(-1, 2)
        src = np.concatenate([src, e[:, 0], e[:, 1]])
        dst = np.concatenate([dst, e[:, 1], e[:, 0]])
    return np.vstack([src, dst]).astype(np.int64), n_c, n


def node_features(ctx, method: str) -> np.ndarray:
    kg = ctx.kg
    ent_texts = []
    for key in kg.entity_ids:
        t, name = split_key(key)
        et = kg.ontology.entity_types[t]
        desc = et.values[name].descriptor if name in et.values else et.descriptor
        ent_texts.append(f"{t}: {name.replace('_', ' ')}. {desc}")
    if method == "sbert" and has_module("sentence_transformers"):
        Xc = call_embeddings(ctx, kg.call_ids)
        Xe = sbert_encode(ent_texts, ctx.cfg["models"]["sbert"]["model_name"]) if ent_texts else np.zeros((0, Xc.shape[1]))
    else:
        if method == "sbert":
            log.warning("sentence-transformers not installed - GraphSAGE node features fall back to TF-IDF + SVD")
        vec = TfidfVectorizer(min_df=2, max_features=20000, sublinear_tf=True).fit(ctx.texts(ctx.train_ids))
        dim = max(2, min(128, len(vec.vocabulary_) - 1))
        svd = TruncatedSVD(n_components=dim, random_state=ctx.seed).fit(vec.transform(ctx.texts(ctx.train_ids)))
        Xc = svd.transform(vec.transform(ctx.texts(kg.call_ids)))
        Xe = svd.transform(vec.transform(ent_texts)) if ent_texts else np.zeros((0, dim))
        for M in (Xc, Xe):
            M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
    types = ["Call", *kg.ontology.type_names]
    onehot = np.zeros((kg.n_calls + len(kg.entity_ids), len(types)), dtype=np.float32)
    onehot[:kg.n_calls, 0] = 1
    for j, t in enumerate(kg.entity_types):
        onehot[kg.n_calls + j, types.index(t)] = 1
    return np.hstack([np.vstack([Xc, Xe]).astype(np.float32), onehot])


def _model_class(use_pyg: bool):
    import torch
    import torch.nn.functional as F
    from torch import nn

    class MeanSAGEConv(nn.Module):
        """GraphSAGE layer, mean aggregator: W_self x_v + W_neigh mean(x_u, u in N(v))."""

        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.lin_self = nn.Linear(in_dim, out_dim)
            self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)

        def forward(self, x, adj):
            return self.lin_self(x) + self.lin_neigh(torch.sparse.mm(adj, x))

    if use_pyg:
        from torch_geometric.nn import SAGEConv

        def conv(i, o):
            return SAGEConv(i, o, aggr="mean")
    else:
        conv = MeanSAGEConv

    class GraphSAGE(nn.Module):
        def __init__(self, in_dim: int, hidden: int, dropout: float):
            super().__init__()
            self.conv1 = conv(in_dim, hidden)
            self.conv2 = conv(hidden, hidden)
            self.head = nn.Linear(hidden, 1)
            self.dropout = dropout

        def forward(self, x, graph):
            h = F.dropout(F.relu(self.conv1(x, graph)), p=self.dropout, training=self.training)
            emb = F.relu(self.conv2(h, graph))
            out = self.head(F.dropout(emb, p=self.dropout, training=self.training)).squeeze(-1)
            return out, emb

    return GraphSAGE


def run_graphsage(ctx) -> None:
    require("torch")
    import torch

    ctx.require_graph()
    gcfg = ctx.cfg["models"]["graphsage"]
    seed = ctx.seed
    set_seed(seed)
    use_pyg = has_module("torch_geometric")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_index, n_c, n = build_homogeneous(ctx, bool(gcfg.get("use_similarity_edges", False)))
    x = torch.tensor(node_features(ctx, gcfg.get("node_features", "sbert")), device=device)
    if use_pyg:
        graph = torch.tensor(edge_index, device=device)
    else:
        deg = np.bincount(edge_index[0], minlength=n).astype(np.float32)
        vals = 1.0 / np.maximum(deg[edge_index[0]], 1.0)
        graph = torch.sparse_coo_tensor(torch.tensor(edge_index), torch.tensor(vals), (n, n),
                                        check_invariants=True).coalesce().to(device)
    log.info("GraphSAGE (%s): %d nodes, %d directed edges, %d input features",
             "PyG SAGEConv" if use_pyg else "pure-PyTorch SAGE", n, edge_index.shape[1], x.shape[1])

    idx = ctx.kg.call_index
    tr = torch.tensor([idx[c] for c in ctx.train_ids], device=device)
    va_ids = ctx.val_ids
    va = torch.tensor([idx[c] for c in va_ids], device=device)
    y = torch.zeros(n, device=device)
    y[tr] = torch.tensor(ctx.labels(ctx.train_ids), dtype=torch.float32, device=device)
    yva = ctx.labels(va_ids)
    pos = float(y[tr].sum())
    pos_weight = torch.tensor([(len(tr) - pos) / max(pos, 1.0)], device=device)

    model = _model_class(use_pyg)(x.shape[1], int(gcfg["hidden"]), float(gcfg["dropout"])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(gcfg["lr"]), weight_decay=float(gcfg["weight_decay"]))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_ap, best_state, wait = -1.0, None, 0
    for epoch in range(int(gcfg["epochs"])):
        model.train()
        opt.zero_grad()
        logits, _ = model(x, graph)
        loss = loss_fn(logits[tr], y[tr])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            logits, _ = model(x, graph)
            ap = float(average_precision_score(yva, torch.sigmoid(logits[va]).cpu().numpy()))
        if ap > best_ap + 1e-4:
            best_ap, best_state, wait = ap, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= int(gcfg.get("patience", 30)):
                break
        if epoch % 25 == 0:
            log.info("GraphSAGE epoch %3d loss %.4f val PR-AUC %.4f", epoch, loss.item(), ap)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, emb = model(x, graph)
    prob = torch.sigmoid(logits[:n_c]).cpu().numpy()
    emb = emb[:n_c].cpu().numpy()
    ev = Evaluator(ctx)
    eval_ids = ev.eval_ids()
    ev.evaluate("E4", "graphsage", pd.Series(prob[[idx[c] for c in eval_ids]], index=eval_ids),
                extra={"val_pr_auc": best_ap, "epochs": epoch + 1})
    np.savez_compressed(embeddings_path(ctx), ids=np.array(ctx.kg.call_ids), emb=emb)
    torch.save(best_state, ctx.paths.models / "graphsage.pt")


def embeddings_path(ctx):
    return ctx.paths.models / "graphsage_call_embeddings.npz"


def run_graphsage_xgb(ctx) -> None:
    """GraphSAGE as an embedding generator for XGBoost. Kept apart from run_graphsage (and free of
    torch) because PyTorch and XGBoost cannot share a process on macOS (clashing OpenMP runtimes)."""
    path = embeddings_path(ctx)
    if not path.exists():
        log.warning("No GraphSAGE embeddings (%s) - run E4 first", path.name)
        return
    z = np.load(path, allow_pickle=False)
    emb = dict(zip(z["ids"].tolist(), z["emb"]))
    ev = Evaluator(ctx)
    eval_ids = ev.eval_ids()
    fit_ids = ctx.train_ids + ctx.val_ids
    yfit = ctx.labels(fit_ids)
    xgb = make_model("xgb", {}, ctx.seed, yfit).fit(np.stack([emb[c] for c in fit_ids]), yfit)
    ev.evaluate("E4", "graphsage_xgb", pd.Series(xgb.predict_proba(np.stack([emb[c] for c in eval_ids]))[:, 1],
                                                 index=eval_ids))
