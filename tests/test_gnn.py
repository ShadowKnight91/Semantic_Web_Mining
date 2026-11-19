"""E4 GraphSAGE: text + topology fusion and inductive training (TA comment 2). Skipped without torch."""
import copy

import numpy as np
import pandas as pd
import pytest

from scamkg.models import gnn
from scamkg.utils import has_module

# torch is never imported into the test process itself: on macOS, PyTorch's OpenMP runtime next to the one
# scikit-learn / XGBoost load makes later tests segfault. GraphSAGE runs in a spawned process (as in `train`).
pytestmark = pytest.mark.skipif(not has_module("torch"), reason="torch not installed")


def _ctx(demo_ctx, **overrides):
    ctx = copy.copy(demo_ctx)
    ctx.cfg = copy.deepcopy(demo_ctx.cfg)
    g = ctx.cfg["models"]["graphsage"]
    g.update({"node_features": "tfidf_svd", "epochs": 30, "patience": 10, **overrides})
    return ctx


def test_topology_block_is_label_free_and_scaled_on_train(demo_ctx):
    ctx = _ctx(demo_ctx)
    T, names = gnn.topology_block(ctx)
    n_c = ctx.kg.n_calls
    assert T.shape == (n_c + len(ctx.kg.entity_ids), len(names))
    assert "het_pagerank" in names and "community_size" in names and "entity_log_degree" in names
    assert not any(c in names for c in ("community_scam_rate", "entity_risk_mean", "knn_scam_rate"))
    tr = [ctx.kg.call_index[c] for c in ctx.train_ids]
    k = names.index("het_pagerank")
    assert abs(T[tr, k].mean()) < 1e-4                      # standardised with training statistics
    assert np.all(T[:n_c, -2:] == 0) and np.all(T[n_c:, :-2] == 0)   # call / entity columns kept apart


def test_inductive_training_graph_has_no_test_calls(demo_ctx):
    ctx = _ctx(demo_ctx)
    ei, n_c, n = gnn.build_homogeneous(ctx)
    tr = np.array([ctx.kg.call_index[c] for c in ctx.train_ids])
    sub = gnn._restrict(ei, n, n_c, tr)
    calls_in = set(sub[0][sub[0] < n_c]) | set(sub[1][sub[1] < n_c])
    assert calls_in <= set(tr) and sub.shape[1] < ei.shape[1]


def _graphsage_child(cfg: dict, run: str, use_pyg: bool) -> None:
    from scamkg.context import Context
    if not use_pyg:
        real = gnn.has_module
        gnn.has_module = lambda m: False if m == "torch_geometric" else real(m)
    gnn.run_graphsage(Context.load(cfg, run=run))


@pytest.mark.parametrize("use_pyg", [False, True])
def test_fusion_graphsage_runs_and_writes_variants(demo_ctx, use_pyg):
    if use_pyg:
        pytest.importorskip("torch_geometric")
    ctx = _ctx(demo_ctx)
    # torch runs in a fresh process, as in `train`: on macOS XGBoost (graphsage_xgb here, and every later
    # test) crashes once PyTorch has run in the same process
    import multiprocessing
    p = multiprocessing.get_context("spawn").Process(target=_graphsage_child, args=(ctx.cfg, ctx.run_name, use_pyg))
    p.start()
    p.join()
    assert p.exitcode == 0
    gnn.run_graphsage_xgb(ctx)
    m = pd.read_csv(ctx.paths.results / "metrics.csv")
    test = m[(m["experiment"] == "E4") & (m["eval_set"] == "test")].set_index("model")
    for name in ("graphsage", "graphsage_text", "graphsage_topo", "graphsage_xgb"):
        assert name in test.index and test.at[name, "pr_auc"] > 0.6
    import json
    info = json.loads((ctx.paths.results / "graphsage_fusion.json").read_text())
    assert info["variants"][0]["input_dims"]["topo"] == len(info["topology_features"])


def _loto_child(cfg: dict, run: str) -> None:
    from scamkg.context import Context
    gnn.run_graphsage_loto(Context.load(cfg, run=run), variants=("kg",))


def test_graphsage_loto_keeps_the_held_out_type_out_of_training(demo_ctx):
    ctx = _ctx(demo_ctx)
    fold = copy.copy(ctx)
    fold.calls = ctx.calls.copy()
    prim = ctx.calls[ctx.calls["dataset"] == ctx.primary]
    t = sorted(prim["subtype"].unique())[0]
    held = (fold.calls["dataset"] == ctx.primary) & (fold.calls["subtype"] == t)
    fold.calls.loc[held & fold.calls["split"].isin(["train", "val"]), "split"] = "held_out"
    assert not set(fold.calls.loc[held].index) & set(fold.train_ids + fold.val_ids)
    import multiprocessing
    p = multiprocessing.get_context("spawn").Process(target=_loto_child, args=(ctx.cfg, ctx.run_name))
    p.start()
    p.join()
    assert p.exitcode == 0
    z = np.load(gnn.loto_path(ctx), allow_pickle=False)
    assert f"{t}|kg|prob" in z.files and z[f"{t}|unsup|emb"].shape[0] == ctx.kg.n_calls


def test_every_variant_saves_its_embeddings(demo_ctx):
    ctx = _ctx(demo_ctx)
    assert gnn.embeddings_path(ctx, "text+topo") != gnn.embeddings_path(ctx, "topo") != gnn.embeddings_path(ctx)


def _score_child(cfg: dict, run: str, out: str) -> None:
    import torch

    from scamkg.context import Context
    from scamkg.utils import has_module
    ctx = Context.load(cfg, run=run)
    use_pyg = has_module("torch_geometric")
    device = torch.device("cpu")
    graphs, _, _, n_c, _ = gnn.prepare_graphs(ctx, use_pyg, device)
    text = gnn.text_block(ctx, "tfidf_svd")
    blocks = {"text": text, "kg": gnn.kg_block(text, n_c), "topo": gnn.topology_block(ctx)[0], "type": gnn.type_block(ctx)}
    res = gnn._train_variant(ctx, "kg", blocks, graphs, n_c, use_pyg, device)
    same = gnn.score_variant(ctx, "kg", res, blocks, graphs["full"], n_c, use_pyg, device)
    np.savez(out, trained=res["prob"], scored=same, kg_call_rows=blocks["kg"][:n_c])


def test_trained_weights_score_a_graph_without_retraining_and_kg_block_has_no_transcript(demo_ctx, tmp_path):
    ctx = _ctx(demo_ctx)
    import multiprocessing
    out = str(tmp_path / "scores.npz")
    p = multiprocessing.get_context("spawn").Process(target=_score_child, args=(ctx.cfg, ctx.run_name, out))
    p.start()
    p.join()
    assert p.exitcode == 0
    z = np.load(out)
    assert np.allclose(z["trained"], z["scored"], atol=1e-5)
    assert np.all(z["kg_call_rows"] == 0)
