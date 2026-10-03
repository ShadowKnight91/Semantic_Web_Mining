import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scamkg.graph.build import CallKG
from scamkg.graph.features import GraphFeatureBuilder, label_features


def _t(rel, typ, obj):
    return {"relation": rel, "object_type": typ, "object": obj}


def _toy_kg(ontology):
    gov = _t("CLAIMS_TO_BE", "ClaimedIdentity", "government_agent")
    gift = _t("PAY_VIA", "PaymentChannel", "gift_card")
    doc = _t("CLAIMS_TO_BE", "ClaimedIdentity", "healthcare_provider")
    info = _t("REQUESTS", "RequestedAsset", "personal_info")
    ex = {"a": [gov, gift], "b": [gov, gift], "c": [gov], "d": [doc, info], "e": [doc], "f": [gov, gift], "g": []}
    return CallKG(list(ex), ex, ontology)


def test_incidence_and_jaccard(ontology):
    kg = _toy_kg(ontology)
    assert kg.X.shape == (7, 4)
    J = kg.jaccard
    i, j = kg.call_index["a"], kg.call_index["b"]
    assert J[i, j] == 1.0
    assert abs(J[i, kg.call_index["c"]] - 0.5) < 1e-6
    assert J[kg.call_index["g"]].sum() == 0


def test_entity_risk_is_leave_one_out(ontology):
    kg = _toy_kg(ontology)
    labeled = ["a", "b", "d", "e"]
    y = np.array([1, 1, 0, 0])
    idx = np.array([kg.call_index[c] for c in labeled])
    f = label_features(kg, idx, y, np.zeros(kg.n_calls), {"smoothing": 0.0, "knn_k": 3, "sim_topk": 2})
    # "a" is labeled: its gov/gift entities are seen in one *other* labeled call ("b", a scam) -> risk 1.0
    assert f.at["a", "entity_risk_mean"] == 1.0
    # unlabeled "f" sees both labeled scams -> risk 1.0 ; "c" too
    assert f.at["f", "entity_risk_mean"] == 1.0
    # "e" is labeled legit; healthcare_provider seen in "d" (legit) -> 0
    assert f.at["e", "entity_risk_max"] == 0.0
    # similarity never uses the call itself
    assert f.at["a", "sim_scam_max"] == 1.0 and f.at["d", "sim_legit_max"] == 0.5


def test_features_do_not_depend_on_unlabeled_labels(ontology):
    kg = _toy_kg(ontology)
    fb = GraphFeatureBuilder(kg, {"betweenness_samples": 0, "similarity_top_k": 3, "similarity_min": 0.1})
    labeled = ["a", "b", "d", "e"]
    f1 = fb.build(labeled, [1, 1, 0, 0])
    f2 = fb.build(labeled, [1, 1, 0, 0])
    pd.testing.assert_frame_equal(f1, f2)
    # flipping a's label changes the features of its neighbour b ...
    f3 = fb.build(labeled, [0, 1, 0, 0])
    cols = ["entity_risk_mean", "community_scam_rate", "knn_scam_rate"]
    assert not np.allclose(f1.loc["b", cols].to_numpy(dtype=float), f3.loc["b", cols].to_numpy(dtype=float))
    # ... but a's own similarity-to-scams ignores a itself (leave-one-out)
    assert f1.at["a", "sim_scam_max"] == f3.at["a", "sim_scam_max"] == 1.0


def test_training_features_do_not_encode_own_label(ontology):
    """Ten calls share one entity. Leave-one-out rates are lower for every scam than for every
    legitimate call (the label is readable from the feature); out-of-fold rates are not."""
    gov = _t("CLAIMS_TO_BE", "ClaimedIdentity", "government_agent")
    ids = [f"c{i}" for i in range(10)]
    kg = CallKG(ids, {c: [gov] for c in ids}, ontology)
    y = np.array([1, 0] * 5)
    gcfg = {"betweenness_samples": 0, "similarity_top_k": 3, "similarity_min": 0.1}
    loo = GraphFeatureBuilder(kg, {**gcfg, "label_folds": 1}).build(ids, y)
    oof = GraphFeatureBuilder(kg, {**gcfg, "label_folds": 5}).build(ids, y)
    for col in ("entity_risk_mean", "knn_scam_rate", "sim_scam_topk"):
        assert roc_auc_score(y, loo[col]) == 0.0, col
        assert roc_auc_score(y, oof[col]) == 0.5, col


def test_leak_safety_on_demo(demo_ctx):
    """Test-set labels must never influence any feature: flip them and compare."""
    ctx = demo_ctx
    labeled = ctx.train_ids
    base = ctx.fb.build(labeled, ctx.labels(labeled))
    flipped = ctx.calls.copy()
    test = ctx.test_ids
    flipped.loc[test, "label"] = 1 - flipped.loc[test, "label"]
    original = ctx.calls
    try:
        ctx.calls = flipped
        again = ctx.fb.build(labeled, ctx.labels(labeled))
    finally:
        ctx.calls = original
    pd.testing.assert_frame_equal(base, again)


def test_onehot_matrix_alignment(demo_ctx):
    fb = demo_ctx.fb
    names = fb.onehot_names()[:5] + ["ClaimedIdentity::not_a_real_value"]
    ids = demo_ctx.test_ids[:10]
    M = fb.onehot_matrix(ids, names).toarray()
    assert M.shape == (10, 6)
    assert M[:, -1].sum() == 0
    for r, cid in enumerate(ids):
        ents = set(demo_ctx.kg.entities_of(cid))
        for c, n in enumerate(names[:-1]):
            assert M[r, c] == (1.0 if n in ents else 0.0)
