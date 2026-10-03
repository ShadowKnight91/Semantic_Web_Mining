"""Significance testing (McNemar) and bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
from scipy.stats import binom, chi2


def mcnemar_test(y_true, pred_a, pred_b) -> dict:
    """McNemar's test on paired predictions; exact binomial when discordant pairs < 25."""
    y, a, b = (np.asarray(v).astype(int) for v in (y_true, pred_a, pred_b))
    ca, cb = a == y, b == y
    n01 = int((ca & ~cb).sum())   # A right, B wrong
    n10 = int((~ca & cb).sum())   # A wrong, B right
    n = n01 + n10
    if n == 0:
        return {"b_only_correct": n10, "a_only_correct": n01, "statistic": 0.0, "p_value": 1.0, "method": "none"}
    if n < 25:
        p = min(1.0, 2 * binom.cdf(min(n01, n10), n, 0.5))
        return {"b_only_correct": n10, "a_only_correct": n01, "statistic": float(min(n01, n10)), "p_value": float(p),
                "method": "exact"}
    stat = (abs(n01 - n10) - 1) ** 2 / n
    return {"b_only_correct": n10, "a_only_correct": n01, "statistic": float(stat), "p_value": float(chi2.sf(stat, 1)),
            "method": "chi2_cc"}


def bootstrap_ci(y_true, scores, metric_fn, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    y, s = np.asarray(y_true), np.asarray(scores)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):
            vals.append(metric_fn(y[idx], s[idx]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))
