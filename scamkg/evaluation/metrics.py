"""Classification and clustering metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, adjusted_rand_score, average_precision_score, f1_score,
                             normalized_mutual_info_score, precision_score, recall_score, roc_auc_score)

METRICS = ["precision", "recall", "f1", "pr_auc", "roc_auc", "recall_at_fpr10", "accuracy"]


def recall_at_fpr(y, s, fpr: float = 0.10) -> float:
    """Recall when the threshold flags `fpr` of the negatives: a fixed false-alarm operating point that, unlike
    recall at 0.5, does not depend on how well a model's scores are calibrated for a shifted data set."""
    neg, pos = s[y == 0], s[y == 1]
    if not len(neg) or not len(pos):
        return float("nan")
    return float((pos > np.quantile(neg, 1 - fpr)).mean())


def binary_metrics(y_true, scores, threshold: float = 0.5) -> dict:
    """Precision, recall, F1 at `threshold`; PR-AUC (primary) and ROC-AUC (NaN when one class only)."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    pred = (s >= threshold).astype(int)
    both = 0 < y.sum() < len(y)
    return {
        "n": int(len(y)), "n_pos": int(y.sum()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)) if y.sum() else float("nan"),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, s)) if both else float("nan"),
        "roc_auc": float(roc_auc_score(y, s)) if both else float("nan"),
        "recall_at_fpr10": recall_at_fpr(y, s, 0.10),
        "accuracy": float(accuracy_score(y, pred)),
        "mean_score": float(s.mean()) if len(s) else float("nan"),
    }


def cluster_purity(y_true, clusters) -> float:
    ct = pd.crosstab(np.asarray(clusters), np.asarray(y_true))
    return float(ct.max(axis=1).sum() / ct.to_numpy().sum()) if ct.size else float("nan")


def clustering_metrics(y_true, clusters) -> dict:
    y, c = np.asarray(y_true), np.asarray(clusters)
    return {"purity": cluster_purity(y, c), "nmi": float(normalized_mutual_info_score(y, c)),
            "ari": float(adjusted_rand_score(y, c)), "n_clusters": int(len(set(c) - {-1})),
            "noise_frac": float((c == -1).mean())}
