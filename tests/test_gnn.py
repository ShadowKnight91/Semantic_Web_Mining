"""E4 GraphSAGE: text + topology fusion and inductive training (TA comment 2). Skipped without torch."""
import copy

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from scamkg.models import gnn  # noqa: E402


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


@pytest.mark.parametrize("use_pyg", [False, True])
def test_fusion_graphsage_runs_and_writes_variants(demo_ctx, monkeypatch, use_pyg):
    if use_pyg:
        pytest.importorskip("torch_geometric")
    else:
        real = gnn.has_module
        monkeypatch.setattr(gnn, "has_module", lambda m: False if m == "torch_geometric" else real(m))
    ctx = _ctx(demo_ctx)
    gnn.run_graphsage(ctx)
    m = pd.read_csv(ctx.paths.results / "metrics.csv")
    test = m[(m["experiment"] == "E4") & (m["eval_set"] == "test")].set_index("model")
    for name in ("graphsage", "graphsage_text", "graphsage_topo", "graphsage_xgb"):
        assert name in test.index and test.at[name, "pr_auc"] > 0.6
    import json
    info = json.loads((ctx.paths.results / "graphsage_fusion.json").read_text())
    assert info["variants"][0]["input_dims"]["topo"] == len(info["topology_features"])
