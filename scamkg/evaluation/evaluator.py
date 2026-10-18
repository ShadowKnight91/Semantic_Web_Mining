"""Shared evaluation of any model's scores on: the balanced test split, re-sampled imbalanced test
splits (~5% scams, mean +- std over draws), and every cross-dataset set (X2)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scamkg.evaluation.metrics import METRICS, binary_metrics
from scamkg.evaluation.stats import mcnemar_test
from scamkg.utils import get_logger, slugify, timestamp

log = get_logger(__name__)


class ResultStore:
    """CSV of result rows keyed by (experiment, model, eval_set); re-running replaces old rows."""

    KEYS = ["experiment", "model", "eval_set"]

    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path) if self.path.exists() else pd.DataFrame()

    def add(self, rows: list[dict]) -> None:
        if not rows:
            return
        new = pd.DataFrame(rows)
        old = self.load()
        if not old.empty:
            keys = set(map(tuple, new[self.KEYS].astype(str).to_numpy()))
            mask = old[self.KEYS].astype(str).apply(tuple, axis=1).isin(keys)
            old = old[~mask]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([old, new], ignore_index=True).to_csv(self.path, index=False)


def imbalanced_draws(ids: list[str], y: np.ndarray, rate: float, n_draws: int, seed: int) -> list[list[str]]:
    """Keep every legitimate test call and sample scams so they make up `rate` of the set."""
    ids = np.asarray(ids)
    pos, neg = ids[y == 1], ids[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return []
    n_pos = max(1, int(round(rate / (1 - rate) * len(neg))))
    n_pos = min(n_pos, len(pos))
    rng = np.random.default_rng(seed)
    return [list(neg) + list(rng.choice(pos, n_pos, replace=False)) for _ in range(n_draws)]


class Evaluator:
    def __init__(self, ctx):
        self.ctx = ctx
        cfg = ctx.cfg
        self.threshold = float(cfg["models"].get("threshold", 0.5))
        self.store = ResultStore(ctx.paths.results / "metrics.csv")
        ctx.paths.predictions.mkdir(parents=True, exist_ok=True)
        imb = cfg["data"].get("imbalanced_test", {})
        test = ctx.test_ids
        self.draws = imbalanced_draws(test, ctx.labels(test), float(imb.get("scam_rate", 0.05)),
                                      int(imb.get("n_draws", 20)), ctx.seed)

    def eval_sets(self) -> dict[str, list[str]]:
        ctx = self.ctx
        sets = {"test": ctx.test_ids}
        for ds in ctx.external_datasets:
            ids = ctx.ids(dataset=ds, labeled=True)
            if ids:
                sets[ds] = ids
        for pair in ctx.cfg["data"].get("mixed_external") or []:
            pos_ds, neg_ds = pair
            pos = [i for i in self._held_out(pos_ds) if ctx.calls.at[i, "label"] == 1]
            neg = [i for i in self._held_out(neg_ds) if ctx.calls.at[i, "label"] == 0]
            if pos and neg:
                sets[f"{self._set_name(pos_ds)}+{self._set_name(neg_ds)}"] = pos + neg
        return sets

    def _held_out(self, dataset: str) -> list[str]:
        """Calls of `dataset` no model was trained on: only the test split of the primary data set."""
        ctx = self.ctx
        return ctx.ids(split="test", dataset=dataset) if dataset == ctx.primary else ctx.ids(dataset=dataset)

    def _set_name(self, dataset: str) -> str:
        return f"{dataset}_test" if dataset == self.ctx.primary else dataset

    def eval_ids(self) -> list[str]:
        seen, out = set(), []
        for ids in self.eval_sets().values():
            for i in ids:
                if i not in seen:
                    seen.add(i)
                    out.append(i)
        return out

    def prediction_path(self, experiment: str, model: str) -> Path:
        return self.ctx.paths.predictions / f"{slugify(experiment)}__{slugify(model)}.csv"

    def evaluate(self, experiment: str, model: str, scores: pd.Series, extra: dict | None = None) -> list[dict]:
        ctx = self.ctx
        scores = pd.Series(scores, dtype=float)
        pred = pd.DataFrame({"call_id": scores.index, "score": scores.to_numpy()})
        meta = ctx.calls.loc[pred["call_id"], ["dataset", "split", "label"]].reset_index(drop=True)
        pd.concat([pred, meta], axis=1).to_csv(self.prediction_path(experiment, model), index=False)

        base = {"experiment": experiment, "model": model, "run": ctx.run_name or "", "timestamp": timestamp(),
                **(extra or {})}
        rows = []
        for name, ids in self.eval_sets().items():
            ids = [i for i in ids if i in scores.index]
            if not ids:
                continue
            rows.append({**base, "eval_set": name, **binary_metrics(ctx.labels(ids), scores.loc[ids], self.threshold)})
        if self.draws and all(i in scores.index for i in self.draws[0]):
            per = pd.DataFrame([binary_metrics(ctx.labels(d), scores.loc[d], self.threshold) for d in self.draws])
            row = {**base, "eval_set": "test_imbalanced", "n": int(per["n"].iloc[0]), "n_pos": int(per["n_pos"].iloc[0])}
            for m in METRICS:
                row[m] = float(per[m].mean())
                row[f"{m}_std"] = float(per[m].std(ddof=1)) if len(per) > 1 else 0.0
            rows.append(row)
        self.store.add(rows)
        msg = " | ".join(f"{r['eval_set']}: PR-AUC={r['pr_auc']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}"
                         for r in rows if r["eval_set"] in ("test", "test_imbalanced"))
        ext = " | ".join(f"{r['eval_set']}: R={r['recall']:.3f}" for r in rows
                         if r["eval_set"] not in ("test", "test_imbalanced"))
        log.info("%s/%s  %s%s", experiment, model, msg, f"  || {ext}" if ext else "")
        return rows


def run_significance(ctx, baseline: str = "E1", challengers: tuple[str, ...] = ("E2", "E2G")) -> pd.DataFrame:
    """McNemar test of every challenger vs. the E1 baseline, per model family and evaluation set."""
    ev = Evaluator(ctx)
    pred_dir = ctx.paths.predictions
    rows = []
    for base_file in sorted(pred_dir.glob(f"{baseline}__*.csv")):
        model = base_file.stem.split("__", 1)[1]
        a = pd.read_csv(base_file, dtype={"call_id": str}).set_index("call_id")
        for ch in challengers:
            f = pred_dir / f"{ch}__{model}.csv"
            if not f.exists():
                continue
            b = pd.read_csv(f, dtype={"call_id": str}).set_index("call_id")
            for name, ids in ev.eval_sets().items():
                ids = [i for i in ids if i in a.index and i in b.index]
                if not ids:
                    continue
                y = ctx.labels(ids)
                res = mcnemar_test(y, (a.loc[ids, "score"] >= ev.threshold).astype(int),
                                   (b.loc[ids, "score"] >= ev.threshold).astype(int))
                rows.append({"model": model, "baseline": baseline, "challenger": ch, "eval_set": name, "n": len(ids),
                             **res, "significant_0.05": res["p_value"] < 0.05})
    df = pd.DataFrame(rows)
    df.to_csv(ctx.paths.results / "significance.csv", index=False)
    log.info("McNemar tests: %d comparisons -> significance.csv", len(df))
    return df
