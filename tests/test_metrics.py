import numpy as np

from scamkg.evaluation.evaluator import imbalanced_draws
from scamkg.evaluation.metrics import binary_metrics, cluster_purity
from scamkg.evaluation.stats import mcnemar_test


def test_binary_metrics_basic():
    m = binary_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["pr_auc"] > 0.5


def test_binary_metrics_single_class():
    m = binary_metrics([1, 1, 1], [0.9, 0.2, 0.7])
    assert np.isnan(m["pr_auc"]) and abs(m["recall"] - 2 / 3) < 1e-9


def test_mcnemar():
    y = np.ones(40, dtype=int)
    a = np.zeros(40, dtype=int)
    b = np.ones(40, dtype=int)
    res = mcnemar_test(y, a, b)
    assert res["b_only_correct"] == 40 and res["p_value"] < 1e-6
    same = mcnemar_test(y, b, b)
    assert same["p_value"] == 1.0


def test_purity():
    assert cluster_purity([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert cluster_purity([0, 1, 0, 1], [0, 0, 0, 0]) == 0.5


def test_imbalanced_draws_rate():
    ids = [f"c{i}" for i in range(200)]
    y = np.array([1] * 100 + [0] * 100)
    draws = imbalanced_draws(ids, y, 0.05, 3, seed=0)
    assert len(draws) == 3
    for d in draws:
        pos = sum(1 for c in d if int(c[1:]) < 100)
        assert len(d) == 105 and pos == 5


def test_recall_at_fixed_fpr_ignores_calibration():
    y = np.array([0] * 10 + [1] * 10)
    s = np.r_[np.linspace(0.00, 0.09, 10), np.linspace(0.10, 0.19, 10)]   # all scores far below 0.5
    m = binary_metrics(y, s)
    assert m["recall"] == 0.0                     # nothing passes the 0.5 threshold ...
    assert m["recall_at_fpr10"] == 1.0            # ... but at a 10% false-alarm rate every scam is caught
    assert np.isnan(binary_metrics([1, 1], [0.2, 0.9])["recall_at_fpr10"])
