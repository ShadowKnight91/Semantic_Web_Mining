"""E4 - GraphSAGE on the call knowledge graph, with text + topology feature fusion.

Graph: homogeneous view of the heterogeneous KG - call nodes + entity nodes, undirected call-entity
edges (optionally call-call similarity edges). Two SAGE layers reach call -> entity -> call, so a
call aggregates the calls that share its identities, tactics, assets and payment channels.

CHANGE (TA comment 2) - how the transcript text representation is combined with the structural /
topological features *before* neighbour aggregation. Before, node features were only the text
embedding + a node-type one-hot; PageRank, degree, betweenness, community size etc. were used by
E2 (XGBoost) but never reached GraphSAGE. Every node now has three feature blocks:

    s_v  text block      call: Sentence-BERT embedding of the transcript (384-d, mean over 180-word
                         chunks; TF-IDF + SVD 128-d when sentence-transformers is unavailable)
                         entity: embedding of "<type>: <value>. <descriptor>"
    g_v  topology block  call: het_degree, het_pagerank, het_betweenness, cc_degree, cc_weighted_degree,
                         cc_pagerank, community_size, entity_mean/max_popularity, n_<RELATION>, n_inferred,
                         has_entities (all label-free; log1p for heavy-tailed counts, standardised on
                         training calls)
                         entity: log(1 + number of linked calls), PageRank in the call-entity graph
    o_v  type block      one-hot node type (Call + the 6 entity types)

If the blocks were simply concatenated, the 384 text dimensions would drown out the ~19 topology
dimensions. So each block gets its own learned projection, the projections are concatenated and
layer-normalised, and only then does message passing start:

    h_v(0) = LayerNorm( [ ReLU(W_t s_v) (64) || ReLU(W_g g_v) (16) || W_o o_v (8) ] )        (88-d)
    h_v(k) = ReLU( W_self(k) h_v(k-1) + W_neigh(k) * mean_{u in N(v)} h_u(k-1) ),   k = 1, 2  (64-d)
    p_v    = sigmoid( w^T h_v(2) + b )

The projections are trained end-to-end with GraphSAGE, so the model learns how much weight each
block gets. Variants `text`, `topo` and `text+topo` (config models.graphsage.variants) give the block
ablation for RQ1.

CHANGE - inductive training (GraphSAGE is an inductive method): with `inductive: true` the loss is
computed on a graph that contains only training calls (+ entities), early stopping uses the graph with
training + validation calls, and only the final scoring uses the full graph with test and external
calls. Before, test and external calls were inside the training graph, which makes the cross-dataset
(X2) numbers optimistic. Labels are only ever read for training calls.

Training: 2 SAGE layers, mean aggregation, 64 hidden units, dropout 0.3, Adam, class-weighted BCE,
early stopping on validation PR-AUC. Used (a) end-to-end ("graphsage") and (b) as an embedding
generator for XGBoost ("graphsage_xgb", run_graphsage_xgb in a separate, torch-free step). PyTorch
Geometric's SAGEConv is used when installed, otherwise an equivalent pure-PyTorch layer.
"""
from __future__ import annotations

import copy

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from scamkg.evaluation.evaluator import Evaluator
from scamkg.graph.build import split_key
from scamkg.graph.features import STRUCTURAL
from scamkg.models.classical import make_model
from scamkg.models.transformer import call_embeddings, require, sbert_encode
from scamkg.utils import get_logger, has_module, save_json, set_seed

log = get_logger(__name__)

# heavy-tailed, non-negative columns that get log1p before standardisation
_LOG_COLS = {"het_degree", "het_pagerank", "cc_degree", "cc_weighted_degree", "cc_pagerank", "community_size"}
VARIANT_NAMES = {"text+topo": "graphsage", "text": "graphsage_text", "topo": "graphsage_topo"}


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


# ---------------------------------------------------------------------------- feature blocks
def text_block(ctx, method: str) -> np.ndarray:
    """s_v: transcript embedding for calls, description embedding for entities (L2-normalised)."""
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
            log.warning("sentence-transformers not installed - GraphSAGE text block falls back to TF-IDF + SVD")
        vec = TfidfVectorizer(min_df=2, max_features=20000, sublinear_tf=True).fit(ctx.texts(ctx.train_ids))
        dim = max(2, min(128, len(vec.vocabulary_) - 1))
        svd = TruncatedSVD(n_components=dim, random_state=ctx.seed).fit(vec.transform(ctx.texts(ctx.train_ids)))
        Xc = svd.transform(vec.transform(ctx.texts(kg.call_ids)))
        Xe = svd.transform(vec.transform(ent_texts)) if ent_texts else np.zeros((0, dim))
        for M in (Xc, Xe):
            M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
    return np.vstack([Xc, Xe]).astype(np.float32)


def topology_block(ctx) -> tuple[np.ndarray, list[str]]:
    """g_v: label-free structural features. Call columns are zero for entity rows and vice versa
    (the type block tells the model which part applies)."""
    kg = ctx.kg
    struct = ctx.fb.structural
    rel_cols = [c for c in struct.columns if c.startswith("n_")] + ["has_entities"]
    call_cols = [c for c in STRUCTURAL + ["community_size"] + rel_cols if c in struct.columns]
    C = struct.loc[kg.call_ids, call_cols].to_numpy(dtype=float)
    for j, c in enumerate(call_cols):
        if c in _LOG_COLS:
            C[:, j] = np.log1p(np.clip(C[:, j], 0, None))
    tr = [kg.call_index[c] for c in ctx.train_ids]
    C = StandardScaler().fit(C[tr]).transform(C)            # statistics from training calls only

    n_e = len(kg.entity_ids)
    E = np.zeros((n_e, 2))
    if n_e:
        pr = nx.pagerank(kg.graph, alpha=0.85, max_iter=200) if kg.graph.number_of_edges() else {}
        E[:, 0] = np.log1p(kg.entity_degree)
        E[:, 1] = [pr.get(k, 0.0) * kg.graph.number_of_nodes() for k in kg.entity_ids]
        E = StandardScaler().fit_transform(E)
    n_c = kg.n_calls
    T = np.zeros((n_c + n_e, len(call_cols) + 2), dtype=np.float32)
    T[:n_c, :len(call_cols)] = np.nan_to_num(C)
    T[n_c:, len(call_cols):] = np.nan_to_num(E)
    return T, call_cols + ["entity_log_degree", "entity_pagerank"]


def type_block(ctx) -> np.ndarray:
    kg = ctx.kg
    types = ["Call", *kg.ontology.type_names]
    onehot = np.zeros((kg.n_calls + len(kg.entity_ids), len(types)), dtype=np.float32)
    onehot[:kg.n_calls, 0] = 1
    for j, t in enumerate(kg.entity_types):
        onehot[kg.n_calls + j, types.index(t)] = 1
    return onehot


def node_features(ctx, method: str) -> np.ndarray:
    """Backward-compatible flat matrix (text block + type one-hot), e.g. for quick inspection."""
    return np.hstack([text_block(ctx, method), type_block(ctx)])


# ---------------------------------------------------------------------------- model
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

    class FusionGraphSAGE(nn.Module):
        """Per-block projections -> concat -> LayerNorm -> 2 x SAGE (mean) -> logit."""

        def __init__(self, dims: dict[str, int], out_dims: dict[str, int], hidden: int, dropout: float):
            super().__init__()
            self.blocks = list(dims)
            # keys are prefixed: "type" would clash with nn.Module.type()
            self.proj = nn.ModuleDict({f"p_{b}": nn.Linear(dims[b], out_dims[b]) for b in self.blocks})
            fused = sum(out_dims[b] for b in self.blocks)
            self.norm = nn.LayerNorm(fused)
            self.conv1 = conv(fused, hidden)
            self.conv2 = conv(hidden, hidden)
            self.head = nn.Linear(hidden, 1)
            self.dropout = dropout

        def fuse(self, feats: dict):
            parts = []
            for b in self.blocks:
                z = self.proj[f"p_{b}"](feats[b])
                parts.append(z if b == "type" else F.relu(z))   # type one-hot = learned type embedding
            return self.norm(torch.cat(parts, dim=-1))

        def forward(self, feats: dict, graph):
            h = F.dropout(self.fuse(feats), p=self.dropout, training=self.training)
            h = F.dropout(F.relu(self.conv1(h, graph)), p=self.dropout, training=self.training)
            emb = F.relu(self.conv2(h, graph))
            out = self.head(F.dropout(emb, p=self.dropout, training=self.training)).squeeze(-1)
            return out, emb

    return FusionGraphSAGE


def _make_graph(edge_index: np.ndarray, n: int, use_pyg: bool, device):
    import torch
    if use_pyg:
        return torch.tensor(edge_index, device=device)
    deg = np.bincount(edge_index[0], minlength=n).astype(np.float32)
    vals = 1.0 / np.maximum(deg[edge_index[0]], 1.0)
    return torch.sparse_coo_tensor(torch.tensor(edge_index), torch.tensor(vals), (n, n),
                                   check_invariants=True).coalesce().to(device)


def _restrict(edge_index: np.ndarray, n: int, n_c: int, allowed_calls: np.ndarray) -> np.ndarray:
    """Drop every edge that touches a call node outside `allowed_calls` (entity nodes are kept)."""
    keep_node = np.ones(n, dtype=bool)
    keep_node[:n_c] = False
    keep_node[allowed_calls] = True
    mask = keep_node[edge_index[0]] & keep_node[edge_index[1]]
    return edge_index[:, mask]


def _train_variant(ctx, variant: str, blocks_np: dict[str, np.ndarray], graphs: dict, n_c: int, use_pyg: bool,
                   device) -> dict:
    import torch
    gcfg = ctx.cfg["models"]["graphsage"]
    fcfg = {"text_dim": 64, "topo_dim": 16, "type_dim": 8, **(gcfg.get("fusion") or {})}
    use = ["text", "topo", "type"] if variant == "text+topo" else [variant, "type"]
    feats = {b: torch.tensor(blocks_np[b], device=device) for b in use}
    dims = {b: blocks_np[b].shape[1] for b in use}
    out_dims = {"text": int(fcfg["text_dim"]), "topo": int(fcfg["topo_dim"]), "type": int(fcfg["type_dim"])}
    set_seed(ctx.seed)
    model = _model_class(use_pyg)(dims, {b: out_dims[b] for b in use}, int(gcfg["hidden"]),
                                  float(gcfg["dropout"])).to(device)

    idx = ctx.kg.call_index
    tr = torch.tensor([idx[c] for c in ctx.train_ids], device=device)
    va_ids = ctx.val_ids
    va = torch.tensor([idx[c] for c in va_ids], device=device)
    y = torch.zeros(n_c + len(ctx.kg.entity_ids), device=device)
    y[tr] = torch.tensor(ctx.labels(ctx.train_ids), dtype=torch.float32, device=device)
    yva = ctx.labels(va_ids)
    pos = float(y[tr].sum())
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(tr) - pos) / max(pos, 1.0)], device=device))
    opt = torch.optim.Adam(model.parameters(), lr=float(gcfg["lr"]), weight_decay=float(gcfg["weight_decay"]))
    yva_t = torch.tensor(yva, dtype=torch.float32, device=device)
    best_ap, best_loss, best_state, wait, epoch = -1.0, float("inf"), copy.deepcopy(model.state_dict()), 0, 0
    for epoch in range(int(gcfg["epochs"])):
        model.train()
        opt.zero_grad()
        logits, _ = model(feats, graphs["train"])
        loss = loss_fn(logits[tr], y[tr])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            logits, _ = model(feats, graphs["val"])
            ap = float(average_precision_score(yva, torch.sigmoid(logits[va]).cpu().numpy()))
            vloss = float(torch.nn.functional.binary_cross_entropy_with_logits(logits[va], yva_t))
        # CHANGE: ties in validation PR-AUC are broken by validation loss. PR-AUC only ranks; when it
        # saturates early (e.g. 1.0 at epoch 0) the old rule kept that barely trained, badly calibrated
        # model, which hurts precision / F1 at the 0.5 threshold.
        better = ap > best_ap + 1e-4 or (abs(ap - best_ap) <= 1e-4 and vloss < best_loss - 1e-4)
        if better:
            best_ap, best_loss, best_state, wait = max(ap, best_ap), vloss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= int(gcfg.get("patience", 30)):
                break
        if epoch % 25 == 0:
            log.info("GraphSAGE[%s] epoch %3d loss %.4f val PR-AUC %.4f", variant, epoch, loss.item(), ap)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, emb = model(feats, graphs["full"])
    return {"prob": torch.sigmoid(logits[:n_c]).cpu().numpy(), "emb": emb[:n_c].cpu().numpy(),
            "best_ap": best_ap, "epochs": epoch + 1, "state": best_state, "dims": dims}


def run_graphsage(ctx) -> None:
    require("torch")
    import torch

    ctx.require_graph()
    gcfg = ctx.cfg["models"]["graphsage"]
    use_pyg = has_module("torch_geometric")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_index, n_c, n = build_homogeneous(ctx, bool(gcfg.get("use_similarity_edges", False)))
    idx = ctx.kg.call_index
    tr_idx = np.array([idx[c] for c in ctx.train_ids])
    va_idx = np.array([idx[c] for c in ctx.val_ids])
    if bool(gcfg.get("inductive", True)):
        e_train = _restrict(edge_index, n, n_c, tr_idx)
        e_val = _restrict(edge_index, n, n_c, np.concatenate([tr_idx, va_idx]))
    else:
        e_train = e_val = edge_index
    graphs = {"train": _make_graph(e_train, n, use_pyg, device), "val": _make_graph(e_val, n, use_pyg, device),
              "full": _make_graph(edge_index, n, use_pyg, device)}

    topo, topo_names = topology_block(ctx)
    blocks_np = {"text": text_block(ctx, gcfg.get("node_features", "sbert")), "topo": topo, "type": type_block(ctx)}
    log.info("GraphSAGE (%s, %s): %d nodes, %d directed edges (train graph %d) | blocks text %d-d, topology %d-d, "
             "type %d-d", "PyG SAGEConv" if use_pyg else "pure-PyTorch SAGE",
             "inductive" if gcfg.get("inductive", True) else "transductive", n, edge_index.shape[1],
             e_train.shape[1], blocks_np["text"].shape[1], topo.shape[1], blocks_np["type"].shape[1])

    variants = [v for v in (gcfg.get("variants") or ["text+topo"]) if v in VARIANT_NAMES]
    ev = Evaluator(ctx)
    eval_ids = ev.eval_ids()
    rows = []
    for k, variant in enumerate(variants):
        res = _train_variant(ctx, variant, blocks_np, graphs, n_c, use_pyg, device)
        name = VARIANT_NAMES[variant]
        ev.evaluate("E4", name, pd.Series(res["prob"][[idx[c] for c in eval_ids]], index=eval_ids),
                    extra={"val_pr_auc": res["best_ap"], "epochs": res["epochs"], "blocks": variant})
        rows.append({"variant": variant, "model": name, "val_pr_auc": res["best_ap"], "epochs": res["epochs"],
                     "input_dims": res["dims"]})
        if k == 0:   # main variant: its embeddings feed graphsage_xgb (run_graphsage_xgb)
            np.savez_compressed(embeddings_path(ctx), ids=np.array(ctx.kg.call_ids), emb=res["emb"])
            torch.save(res["state"], ctx.paths.models / "graphsage.pt")
    save_json(ctx.paths.results / "graphsage_fusion.json",
              {"topology_features": topo_names, "fusion": gcfg.get("fusion"), "inductive": gcfg.get("inductive", True),
               "variants": rows})


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
