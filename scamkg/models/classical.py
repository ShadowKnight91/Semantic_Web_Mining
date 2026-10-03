"""Classical DM/ML models (Phase 5).

E1  - text baseline: TF-IDF -> LR / SVM / RF / XGBoost
E2  - graph-enhanced: E1 features + Neo4j/networkx graph features + one-hot ontology entities
E2G - KG only: graph features + one-hot ontology entities, no words (tests RQ3: does KG structure transfer?)

Hyper-parameters: randomized search on the validation split (features built with train labels
only); the final model is refit on train+val (label features rebuilt with train+val labels) and
scored on the test split, the imbalanced re-samples and every external data set.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import loguniform, uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from scamkg.evaluation.evaluator import Evaluator
from scamkg.evaluation.metrics import METRICS, binary_metrics
from scamkg.graph.features import ONEHOT_GROUP
from scamkg.utils import get_logger, load_json, read_jsonl, save_json

log = get_logger(__name__)

EXPERIMENT_SPECS = {
    "E1": {"text": True, "graph": False, "onehot": False},
    "E2": {"text": True, "graph": True, "onehot": True},
    "E2G": {"text": False, "graph": True, "onehot": True},
}
DEFAULT_PARAMS = {
    "lr": {"C": 1.0},
    "svm": {"C": 0.5},
    "rf": {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
    "xgb": {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.9, "colsample_bytree": 0.8,
            "min_child_weight": 1},
}


def param_space(name: str) -> dict:
    return {
        "lr": {"C": loguniform(1e-2, 1e2)},
        "svm": {"C": loguniform(1e-3, 1e1)},
        "rf": {"n_estimators": [200, 400, 600], "max_depth": [None, 20, 40], "min_samples_leaf": [1, 2, 4],
               "max_features": ["sqrt", "log2"]},
        "xgb": {"n_estimators": [200, 400, 600], "max_depth": [3, 4, 6, 8], "learning_rate": loguniform(0.02, 0.3),
                "subsample": uniform(0.7, 0.3), "colsample_bytree": uniform(0.5, 0.5), "min_child_weight": [1, 3, 5]},
    }[name]


def make_model(name: str, params: dict, seed: int = 42, y=None):
    p = {**DEFAULT_PARAMS[name], **(params or {})}
    if name == "lr":
        return LogisticRegression(C=float(p["C"]), class_weight="balanced", max_iter=5000, solver="liblinear",
                                  random_state=seed)
    if name == "svm":
        return CalibratedClassifierCV(LinearSVC(C=float(p["C"]), class_weight="balanced", max_iter=20000,
                                                random_state=seed), method="sigmoid", cv=3)
    if name == "rf":
        return RandomForestClassifier(n_estimators=int(p["n_estimators"]), max_depth=p["max_depth"],
                                      min_samples_leaf=int(p["min_samples_leaf"]), max_features=p["max_features"],
                                      class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
    if name == "xgb":
        from xgboost import XGBClassifier
        spw = 1.0
        if y is not None:
            y = np.asarray(y)
            spw = float((y == 0).sum() / max(1, (y == 1).sum()))
        return XGBClassifier(n_estimators=int(p["n_estimators"]), max_depth=int(p["max_depth"]),
                             learning_rate=float(p["learning_rate"]), subsample=float(p["subsample"]),
                             colsample_bytree=float(p["colsample_bytree"]), min_child_weight=float(p["min_child_weight"]),
                             scale_pos_weight=spw, tree_method="hist", eval_metric="aucpr", n_jobs=-1,
                             random_state=seed, verbosity=0)
    raise ValueError(f"Unknown model {name!r}")


def predict_scores(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return 1 / (1 + np.exp(-model.decision_function(X)))


# ---------------------------------------------------------------------------- features
@dataclass
class FittedAssembler:
    experiment: str
    vectorizer: TfidfVectorizer | None
    graph_columns: list[str] = field(default_factory=list)
    scaler: StandardScaler | None = None
    onehot_names: list[str] = field(default_factory=list)
    graph_weight: float = 1.0

    def transform_parts(self, texts, graph_df: pd.DataFrame | None, onehot) -> sparse.csr_matrix:
        blocks = []
        if self.vectorizer is not None:
            blocks.append(self.vectorizer.transform(texts))
        if self.graph_columns:
            vals = self.scaler.transform(graph_df[self.graph_columns].to_numpy(dtype=float))
            blocks.append(sparse.csr_matrix(np.nan_to_num(vals) * self.graph_weight))
        if self.onehot_names:
            blocks.append(sparse.csr_matrix(onehot))
        if not blocks:
            raise ValueError(f"{self.experiment}: no feature blocks selected")
        return sparse.hstack(blocks, format="csr")

    def feature_names(self) -> list[str]:
        names = []
        if self.vectorizer is not None:
            names += [f"tfidf:{w}" for w in self.vectorizer.get_feature_names_out()]
        names += [f"graph:{c}" for c in self.graph_columns]
        names += [f"entity:{e}" for e in self.onehot_names]
        return names


def fit_assembler(ctx, experiment: str, fit_ids: list[str], labeled_ids: list[str] | None = None,
                  graph_groups: list[str] | None = None) -> tuple[FittedAssembler, pd.DataFrame | None]:
    """Fit feature transformers on `fit_ids`; label-based graph features use only `labeled_ids` labels."""
    spec = EXPERIMENT_SPECS[experiment]
    labeled_ids = labeled_ids or fit_ids
    mcfg = ctx.cfg["models"]
    vec = None
    if spec["text"]:
        t = mcfg["tfidf"]
        vec = TfidfVectorizer(ngram_range=tuple(t["ngram_range"]), min_df=t["min_df"], max_df=t["max_df"],
                              max_features=t["max_features"], sublinear_tf=t["sublinear_tf"], strip_accents="unicode")
        vec.fit(ctx.texts(fit_ids))
    graph_df, cols, scaler, onehot = None, [], None, []
    if spec["graph"]:
        ctx.require_graph()
        groups = [g for g in (graph_groups or list(ctx.fb.groups)) if g in ctx.fb.groups]
        cols = ctx.fb.columns(groups)
        if cols:
            graph_df = ctx.fb.build(labeled_ids, ctx.labels(labeled_ids))
            scaler = StandardScaler().fit(graph_df.loc[fit_ids, cols].to_numpy(dtype=float))
    if spec["onehot"] and (graph_groups is None or ONEHOT_GROUP in graph_groups):
        ctx.require_graph()
        onehot = ctx.fb.onehot_names()
    return FittedAssembler(experiment, vec, cols, scaler, onehot, float(mcfg.get("graph_feature_weight", 1.0))), graph_df


def assemble(ctx, fitted: FittedAssembler, graph_df: pd.DataFrame | None, ids: list[str]) -> sparse.csr_matrix:
    texts = ctx.texts(ids) if fitted.vectorizer is not None else None
    gdf = graph_df.loc[ids] if fitted.graph_columns else None
    oh = ctx.fb.onehot_matrix(ids, fitted.onehot_names) if fitted.onehot_names else None
    return fitted.transform_parts(texts, gdf, oh)


# ---------------------------------------------------------------------------- tuning
def random_search(name: str, Xtr, ytr, Xva, yva, n_iter: int, seed: int) -> tuple[dict, float, list[dict]]:
    cands = [DEFAULT_PARAMS[name]] + list(ParameterSampler(param_space(name), n_iter=max(0, n_iter - 1),
                                                           random_state=seed))
    history = []
    for p in cands:
        p = {k: (v.item() if hasattr(v, "item") else v) for k, v in p.items()}
        model = make_model(name, p, seed, ytr).fit(Xtr, ytr)
        score = float(average_precision_score(yva, predict_scores(model, Xva)))
        history.append({"params": p, "val_pr_auc": score})
    best = max(history, key=lambda h: h["val_pr_auc"])
    return best["params"], best["val_pr_auc"], history


def _importances(model, names: list[str], top: int = 40) -> pd.DataFrame:
    est = model
    if isinstance(model, CalibratedClassifierCV):
        inner = [getattr(c, "estimator", None) or getattr(c, "base_estimator", None)
                 for c in model.calibrated_classifiers_]
        vals = np.mean([e.coef_.ravel() for e in inner if e is not None], axis=0)
    elif hasattr(est, "coef_"):
        vals = est.coef_.ravel()
    elif hasattr(est, "feature_importances_"):
        vals = est.feature_importances_
    else:
        return pd.DataFrame()
    order = np.argsort(-np.abs(vals))[:top]
    return pd.DataFrame({"feature": [names[i] for i in order], "weight": vals[order]})


def run_classical(ctx, experiments=("E1", "E2", "E2G"), models=None, tune: bool = True) -> None:
    models = list(models or ctx.cfg["models"]["classical"])
    ev = Evaluator(ctx)
    seed = ctx.seed
    n_iter = int(ctx.cfg["models"].get("search_iter", 12))
    params_path = ctx.paths.results / "best_params.json"
    best = load_json(params_path) if params_path.exists() else {}
    tr, va = ctx.train_ids, ctx.val_ids
    fit_ids = tr + va
    ytr, yva, yfit = ctx.labels(tr), ctx.labels(va), ctx.labels(fit_ids)
    eval_ids = ev.eval_ids()
    for exp in experiments:
        if EXPERIMENT_SPECS[exp]["graph"] and ctx.kg is None:
            log.warning("Skipping %s: no knowledge graph (run extract + build-graph)", exp)
            continue
        if tune:
            f_tr, g_tr = fit_assembler(ctx, exp, tr)
            Xtr, Xva = assemble(ctx, f_tr, g_tr, tr), assemble(ctx, f_tr, g_tr, va)
        fitted, gdf = fit_assembler(ctx, exp, fit_ids)
        Xfit, Xev = assemble(ctx, fitted, gdf, fit_ids), assemble(ctx, fitted, gdf, eval_ids)
        log.info("%s: %d features", exp, Xfit.shape[1])
        for m in models:
            key = f"{exp}__{m}"
            if tune:
                params, val_score, hist = random_search(m, Xtr, ytr, Xva, yva, n_iter, seed)
                best[key] = {"params": params, "val_pr_auc": val_score, "n_candidates": len(hist)}
                log.info("%s best val PR-AUC %.4f with %s", key, val_score, params)
            else:
                params = best.get(key, {}).get("params", DEFAULT_PARAMS[m])
            model = make_model(m, params, seed, yfit).fit(Xfit, yfit)
            scores = pd.Series(predict_scores(model, Xev), index=eval_ids)
            ev.evaluate(exp, m, scores, extra={"n_features": Xfit.shape[1]})
            imp = _importances(model, fitted.feature_names())
            if not imp.empty:
                imp.to_csv(ctx.paths.results / f"feature_importance__{key}.csv", index=False)
            joblib.dump({"experiment": exp, "model_name": m, "params": params, "fitted": fitted, "model": model,
                         "labeled_ids": fit_ids, "run": ctx.run_name}, ctx.paths.models / f"{key}.joblib")
    save_json(params_path, best)


def run_cv(ctx, experiments=("E1", "E2", "E2G"), models=None, n_folds: int | None = None) -> pd.DataFrame:
    """Stratified k-fold CV on the primary data set (label features rebuilt per fold from the fold's train labels)."""
    models = list(models or ctx.cfg["models"]["classical"])
    n_folds = int(n_folds or ctx.cfg["models"].get("cv_folds", 5))
    thr = float(ctx.cfg["models"].get("threshold", 0.5))
    params_path = ctx.paths.results / "best_params.json"
    best = load_json(params_path) if params_path.exists() else {}
    ids = ctx.ids(dataset=ctx.primary, labeled=True)
    y = ctx.labels(ids)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=ctx.seed)
    rows = []
    for fold, (tri, tei) in enumerate(skf.split(ids, y)):
        tr, te = [ids[i] for i in tri], [ids[i] for i in tei]
        for exp in experiments:
            if EXPERIMENT_SPECS[exp]["graph"] and ctx.kg is None:
                continue
            fitted, gdf = fit_assembler(ctx, exp, tr)
            Xtr, Xte = assemble(ctx, fitted, gdf, tr), assemble(ctx, fitted, gdf, te)
            for m in models:
                params = best.get(f"{exp}__{m}", {}).get("params", DEFAULT_PARAMS[m])
                model = make_model(m, params, ctx.seed, y[tri]).fit(Xtr, y[tri])
                rows.append({"fold": fold, "experiment": exp, "model": m,
                             **binary_metrics(y[tei], predict_scores(model, Xte), thr)})
        log.info("CV fold %d/%d done", fold + 1, n_folds)
    folds = pd.DataFrame(rows)
    folds.to_csv(ctx.paths.results / "cv_folds.csv", index=False)
    agg = folds.groupby(["experiment", "model"])[METRICS].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(ctx.paths.results / "cv_results.csv", index=False)
    for _, r in agg.iterrows():
        log.info("CV %-4s %-4s PR-AUC %.3f +- %.3f | recall %.3f +- %.3f | F1 %.3f +- %.3f", r["experiment"], r["model"],
                 r["pr_auc_mean"], r["pr_auc_std"], r["recall_mean"], r["recall_std"], r["f1_mean"], r["f1_std"])
    return agg


def run_multiclass(ctx, experiment: str = "E2") -> dict:
    """Stretch goal: fraudster / salesperson / prankster / other_legitimate (labels from data or LLM relabeling)."""
    label3 = ctx.calls["label3"].copy()
    llm_file = ctx.paths.processed / "label3_llm.jsonl"
    if (label3 == "").all() and llm_file.exists():
        mapping = {r["call_id"]: r["label3"] for r in read_jsonl(llm_file)}
        label3 = pd.Series([mapping.get(c, "") for c in ctx.calls.index], index=ctx.calls.index)
    prim = set(ctx.ids(dataset=ctx.primary))
    has = [c for c in ctx.calls.index if c in prim and label3[c]]
    if not has:
        log.warning("No three-way labels: run `python -m scamkg relabel` (LLM-assisted relabeling) first")
        return {}
    if EXPERIMENT_SPECS[experiment]["graph"] and ctx.kg is None:
        experiment = "E1"
    split = ctx.calls["split"]
    tr = [c for c in has if split[c] in ("train", "val")]
    te = [c for c in has if split[c] == "test"]
    fitted, gdf = fit_assembler(ctx, experiment, tr)
    Xtr, Xte = assemble(ctx, fitted, gdf, tr), assemble(ctx, fitted, gdf, te)
    model = LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0).fit(Xtr, label3[tr])
    pred = model.predict(Xte)
    classes = sorted(set(label3[has]))
    out = {"experiment": experiment, "n_train": len(tr), "n_test": len(te),
           "macro_f1": float(f1_score(label3[te], pred, average="macro")),
           "report": classification_report(label3[te], pred, output_dict=True, zero_division=0),
           "labels": classes, "confusion_matrix": confusion_matrix(label3[te], pred, labels=classes).tolist()}
    save_json(ctx.paths.results / "multiclass.json", out)
    log.info("Three-way classification (%s): macro-F1 %.3f on %d test calls", experiment, out["macro_f1"], len(te))
    return out
