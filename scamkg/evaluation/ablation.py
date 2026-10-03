"""Ablations: remove each ontology entity type (rebuild the KG and all graph features) and each
graph-feature group, then re-score E2 (text + KG) and E2G (KG only) with fixed hyper-parameters."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scamkg.evaluation.evaluator import Evaluator
from scamkg.evaluation.metrics import binary_metrics
from scamkg.graph.features import ONEHOT_GROUP
from scamkg.models.classical import DEFAULT_PARAMS, assemble, fit_assembler, make_model, predict_scores
from scamkg.utils import get_logger, load_json

log = get_logger(__name__)


def _score(ctx, ev: Evaluator, experiment: str, model: str, params: dict, groups: list[str] | None) -> list[dict]:
    fit_ids = ctx.train_ids + ctx.val_ids
    y = ctx.labels(fit_ids)
    fitted, gdf = fit_assembler(ctx, experiment, fit_ids, graph_groups=groups)
    ids = ev.eval_ids()
    clf = make_model(model, params, ctx.seed, y).fit(assemble(ctx, fitted, gdf, fit_ids), y)
    s = pd.Series(predict_scores(clf, assemble(ctx, fitted, gdf, ids)), index=ids)
    rows = []
    for name, set_ids in ev.eval_sets().items():
        rows.append({"eval_set": name, **binary_metrics(ctx.labels(set_ids), s.loc[set_ids], ev.threshold)})
    if ev.draws:
        per = [binary_metrics(ctx.labels(d), s.loc[d], ev.threshold) for d in ev.draws]
        rows.append({"eval_set": "test_imbalanced", **{k: float(np.nanmean([p[k] for p in per]))
                                                        for k in ("precision", "recall", "f1", "pr_auc", "roc_auc")}})
    return rows


def run_ablation(ctx, experiments=("E2", "E2G"), models=("lr", "xgb")) -> pd.DataFrame:
    ctx.require_graph()
    ev = Evaluator(ctx)
    params_path = ctx.paths.results / "best_params.json"
    best = load_json(params_path) if params_path.exists() else {}
    all_groups = [*ctx.fb.groups, ONEHOT_GROUP]
    rows = []

    def record(kind: str, removed: str, c, groups):
        for exp in experiments:
            for m in models:
                params = best.get(f"{exp}__{m}", {}).get("params", DEFAULT_PARAMS[m])
                for r in _score(c, ev, exp, m, params, groups):
                    rows.append({"ablation": kind, "removed": removed, "experiment": exp, "model": m, **r})

    record("none", "-", ctx, None)
    for t in ctx.ontology.type_names:
        log.info("Ablation: removing entity type %s", t)
        record("entity_type", t, ctx.with_excluded_types([t]), None)
    for g in all_groups:
        log.info("Ablation: removing feature group %s", g)
        record("feature_group", g, ctx, [x for x in all_groups if x != g])

    df = pd.DataFrame(rows)
    base = df[df["ablation"] == "none"].set_index(["experiment", "model", "eval_set"])
    for metric in ("pr_auc", "recall", "f1"):
        df[f"delta_{metric}"] = [r[metric] - base.loc[(r["experiment"], r["model"], r["eval_set"]), metric]
                                 for _, r in df.iterrows()]
    df.to_csv(ctx.paths.results / "ablation.csv", index=False)
    log.info("Ablation results -> ablation.csv (%d rows)", len(df))
    return df
