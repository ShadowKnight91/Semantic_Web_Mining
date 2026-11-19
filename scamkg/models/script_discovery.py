"""Hidden scam scripts (professor's feedback: GraphSAGE can uncover behavioural structure bag of words misses).

The test: cluster scam calls from every data set in several spaces with the same number of clusters, and ask
  1. does a space link real calls to synthetic training scripts, or keep each data set on its own island?
  2. do clusters share behaviour (same KG facts) even when their words differ?
  3. for each script, which scams does bag of words miss that GraphSAGE catches, at the same false-alarm rate?
  4. which scripts contain no training calls at all? Those are the hidden scripts.
  5. does a script removed from training still get caught, and does it form its own cluster?

Scripts are defined by the label-free GraphSAGE. An embedding trained to separate scams from legitimate calls groups
calls by their score, so per-script recall measured on its own clusters would be partly circular. Labels only choose
the population (scam calls) and score the results; clustering never reads them. Transcript words are not an input of
the KG-facts and label-free embeddings, so anything those find is behaviour.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA, TruncatedSVD  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics import average_precision_score, f1_score, normalized_mutual_info_score, silhouette_score  # noqa: E402

from scamkg.evaluation.metrics import recall_at_fpr  # noqa: E402
from scamkg.evaluation.plot_style import MODEL_COLORS, MUTED, SERIES, SURFACE, style_axes  # noqa: E402
from scamkg.evaluation.stats import mcnemar_test  # noqa: E402
from scamkg.models.clustering import describe_clusters  # noqa: E402
from scamkg.utils import get_logger, md_table, save_json, slugify  # noqa: E402

log = get_logger(__name__)
EMBEDDINGS = (("GraphSAGE, label-free", "unsup"), ("GraphSAGE, KG facts", "kg"), ("GraphSAGE", "text+topo"),
              ("GraphSAGE, structure only", "topo"))
SCRIPT_SPACE = "GraphSAGE, label-free"
HIDDEN_TRAIN_SHARE = 0.05     # a cluster with fewer training calls than this has no script to learn from
MIN_SCRIPT_CALLS = 10


def _unit(M: np.ndarray) -> np.ndarray:
    return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)


def dataset_mixing(clusters: np.ndarray, datasets: np.ndarray) -> float:
    """Size-weighted evenness of data sets inside clusters: 0 = every cluster holds one data set, 1 = evenly mixed.
    Counts are divided by each data set's size, so the largest data set does not count as the most mixed."""
    names, inv = np.unique(datasets, return_inverse=True)
    if len(names) < 2:
        return float("nan")
    totals = np.bincount(inv).astype(float)
    score = 0.0
    for c in np.unique(clusters):
        mask = clusters == c
        p = np.bincount(inv[mask], minlength=len(names)) / totals
        p = p / p.sum()
        h = -(p[p > 0] * np.log(p[p > 0])).sum() / np.log(len(names))
        score += h * mask.mean()
    return float(score)


def linked_share(clusters: np.ndarray, is_train: np.ndarray, min_train_share: float = HIDDEN_TRAIN_SHARE) -> float:
    """Share of non-training calls in clusters that also hold a real share of training calls."""
    linked = total = 0
    for c in np.unique(clusters):
        mask = clusters == c
        ext = int((~is_train & mask).sum())
        total += ext
        if is_train[mask].mean() >= min_train_share:
            linked += ext
    return linked / total if total else float("nan")


def _sample_pairs(idx: np.ndarray, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    a, b = rng.integers(0, len(idx), n), rng.integers(0, len(idx), n)
    keep = a != b
    return idx[a[keep]], idx[b[keep]]


def _coherence(idx: np.ndarray, tfidf, entities, rng, n_pairs: int = 2000) -> tuple[float, float]:
    """(mean word similarity, mean KG-fact similarity) of random pairs of calls inside one cluster."""
    if len(idx) < 2:
        return float("nan"), float("nan")
    a, b = _sample_pairs(idx, n_pairs, rng)
    lex = np.asarray(tfidf[a].multiply(tfidf[b]).sum(axis=1)).ravel()
    inter = np.asarray(entities[a].multiply(entities[b]).sum(axis=1)).ravel()
    sizes = np.asarray(entities.sum(axis=1)).ravel()
    union = sizes[a] + sizes[b] - inter
    jac = np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0)
    return float(lex.mean()), float(jac.mean())


def method_table(spaces: dict[str, np.ndarray], ids, calls, is_train, tfidf, entities, ks, seed) -> tuple[pd.DataFrame, dict]:
    datasets = calls.loc[ids, "dataset"].to_numpy()
    subtype = calls.loc[ids, "subtype"].to_numpy()
    prim = is_train | (calls.loc[ids, "split"].to_numpy() == "test")
    rows, assignments = [], {}
    rng = np.random.default_rng(seed)
    for name, X in spaces.items():
        for k in ks:
            km = KMeans(k, n_init=10, random_state=seed).fit(X)
            lab = km.labels_
            assignments[(name, k)] = lab
            lex, jac, sizes = [], [], []
            for c in np.unique(lab):
                member = np.flatnonzero(lab == c)
                lc, jc = _coherence(member, tfidf, entities, rng)
                if not np.isnan(lc):
                    lex.append(lc)
                    jac.append(jc)
                    sizes.append(len(member))
            rows.append({"method": name, "k": k,
                         "silhouette": float(silhouette_score(X, lab, sample_size=min(2000, len(X)), random_state=seed)),
                         "nmi_with_synthetic_types": float(normalized_mutual_info_score(subtype[prim], lab[prim])),
                         "dataset_mixing": dataset_mixing(lab, datasets),
                         "real_calls_linked_to_training": linked_share(lab, is_train),
                         "word_similarity_within_cluster": float(np.average(lex, weights=sizes)),
                         "kg_similarity_within_cluster": float(np.average(jac, weights=sizes))})
    return pd.DataFrame(rows), assignments


def _flags(ids, score: pd.Series, thr: float) -> np.ndarray:
    return score.reindex(ids).to_numpy() >= thr


def _top_words(tfidf, names: np.ndarray, member: np.ndarray, background: np.ndarray, n: int = 6) -> str:
    gap = np.asarray(tfidf[member].mean(axis=0)).ravel() - background
    return " ".join(names[np.argsort(-gap)[:n]])


def script_table(ctx, ids, labels, is_train, tfidf, vocab_names, entities, onehot, onehot_names, bow, sage, tb, ts,
                 emb) -> pd.DataFrame:
    calls = ctx.calls
    datasets = calls.loc[ids, "dataset"].to_numpy()
    y = np.ones(len(ids), dtype=int)
    described = {d["cluster"]: d for d in describe_clusters(list(ids), y, labels, onehot_names, onehot)}
    bflag, sflag = _flags(ids, bow, tb), _flags(ids, sage, ts)
    bg = np.asarray(tfidf.mean(axis=0)).ravel()
    train_idx = np.flatnonzero(is_train)
    rng = np.random.default_rng(ctx.seed)
    rows = []
    for c in np.unique(labels):
        member = np.flatnonzero(labels == c)
        ext = member[~is_train[member]]
        lex, jac = _coherence(member, tfidf, entities, rng)
        near = float((1 - emb[ext] @ emb[train_idx].T).min(axis=1).mean()) if len(ext) else float("nan")
        rows.append({"script": int(c), "calls": len(member), **{f"n_{d}": int((datasets[member] == d).sum())
                                                                for d in np.unique(datasets)},
                     "training_share": float(is_train[member].mean()),
                     "hidden": bool(is_train[member].mean() < HIDDEN_TRAIN_SHARE and len(member) >= MIN_SCRIPT_CALLS),
                     "signature": described[int(c)]["signature"], "top_words": _top_words(tfidf, vocab_names, member, bg),
                     "real_calls": len(ext), "bow_recall": float(bflag[ext].mean()) if len(ext) else float("nan"),
                     "graphsage_recall": float(sflag[ext].mean()) if len(ext) else float("nan"),
                     "missed_by_bow_caught_by_graphsage": int((~bflag[ext] & sflag[ext]).sum()),
                     "caught_by_bow_missed_by_graphsage": int((bflag[ext] & ~sflag[ext]).sum()),
                     "word_similarity": lex, "kg_similarity": jac, "distance_to_nearest_training_call": near,
                     "example_call": ids[ext[0]] if len(ext) else ""})
    df = pd.DataFrame(rows)
    return df.sort_values(["hidden", "missed_by_bow_caught_by_graphsage"], ascending=[False, False])


def rescue_table(ctx, scores: dict[str, pd.Series], neg_ids: list[str], target_fpr: float = 0.10) -> pd.DataFrame:
    """Scams from unseen data sets: bag of words against each other model, default threshold and equal false alarms."""
    calls = ctx.calls
    ext_pos = calls.index[(calls["split"] == "external") & (calls["label"] == 1)].tolist()
    bow = scores["E1__lr"]
    y = np.ones(len(ext_pos), dtype=int)
    rows = []
    for name, s in scores.items():
        if name == "E1__lr":
            continue
        for rule, tb, ts in (("threshold 0.5", 0.5, 0.5),
                             (f"{target_fpr:.0%} false alarms", float(np.quantile(bow.reindex(neg_ids), 1 - target_fpr)),
                              float(np.quantile(s.reindex(neg_ids), 1 - target_fpr)))):
            a, b = _flags(ext_pos, bow, tb), _flags(ext_pos, s, ts)
            test = mcnemar_test(y, a.astype(int), b.astype(int))
            rows.append({"challenger": name, "operating_point": rule, "scams": len(ext_pos), "bow_recall": float(a.mean()),
                         "challenger_recall": float(b.mean()), "missed_by_bow_caught": int((~a & b).sum()),
                         "caught_by_bow_missed": int((a & ~b).sum()), "p_value": test["p_value"]})
    return pd.DataFrame(rows)


def seed_table(ctx, path: Path, bow: pd.Series, target_fpr: float = 0.10) -> pd.DataFrame:
    """Recall of each GraphSAGE variant over seeds, with bag of words (it has no seed) as the reference."""
    wide = pd.read_csv(path, dtype={"call_id": str}).set_index("call_id")
    calls = ctx.calls
    neg = calls.index[(calls["split"] == "external") & (calls["label"] == 0)].tolist()
    groups = {d: calls.index[(calls["dataset"] == d) & (calls["label"] == 1)].tolist() for d in ctx.external_datasets}
    names = dict((v, n) for n, v in EMBEDDINGS)

    def measure(s: pd.Series, pos: list[str]) -> dict:
        p, n = s.reindex(pos).to_numpy(), s.reindex(neg).to_numpy()
        return {"recall_default_threshold": float((p >= 0.5).mean()),
                "recall_at_10pct_false_alarms": float((p > np.quantile(n, 1 - target_fpr)).mean()),
                "false_alarms_default_threshold": float((n >= 0.5).mean())}

    rows = []
    for ds, pos in groups.items():
        rows.append({"model": "Bag of words", "dataset": ds, "seeds": 1, **measure(bow, pos)})
        per: dict[str, list[dict]] = {}
        for col in wide.columns:
            variant, _ = col.split("|")
            per.setdefault(variant, []).append(measure(wide[col], pos))
        for variant, vals in per.items():
            f = pd.DataFrame(vals)
            rows.append({"model": names.get(variant, variant), "dataset": ds, "seeds": len(f),
                         **f.mean().to_dict(), **{f"{c}_sd": f[c].std(ddof=1) for c in f.columns}})
    return pd.DataFrame(rows)


def _best_cluster_f1(in_type: np.ndarray, X: np.ndarray, k: int, seed: int) -> float:
    lab = KMeans(k, n_init=10, random_state=seed).fit_predict(X)
    return max(float(f1_score(in_type, lab == c)) for c in np.unique(lab))


def loto_table(ctx, rctx, path: Path, variants=(("text+topo", "GraphSAGE"), ("topo", "GraphSAGE, structure only"),
                                                ("kg", "GraphSAGE, KG facts"), ("unsup", "GraphSAGE, label-free"))) -> pd.DataFrame:
    """An unseen script: is it flagged (recall), ranked (PR-AUC), and does it form its own cluster? Same calls and
    held-out rule as `loto` for E1; the bag-of-words clusters are also fitted without the held-out type."""
    z = np.load(path, allow_pickle=False)
    all_ids = z["ids"].astype(str)
    pos = {c: i for i, c in enumerate(all_ids)}
    calls = ctx.calls
    prim = calls[(calls["dataset"] == ctx.primary) & (calls["label"] >= 0)]
    rows = []
    e1 = None
    p = ctx.paths.results / "loto.csv"
    if p.exists():
        e1 = pd.read_csv(p)
        e1 = e1[(e1["experiment"] == "E1") & (e1["model"] == "lr")].set_index("held_out_type")
    for t in sorted(s for s in prim["subtype"].unique() if s):
        held = prim[prim["subtype"] == t]
        lab = int(held["label"].iloc[0])
        test = held.index.tolist() + prim[(prim["split"] == "test") & (prim["label"] != lab)].index.tolist()
        y = ctx.labels(test)
        group = prim[prim["label"] == lab]        # k = types of this class; the held-out type should come out alone
        k = max(2, group["subtype"].nunique())
        in_type = (group["subtype"] == t).to_numpy()
        for variant, name in variants:
            if f"{t}|{variant}|emb" not in z.files:
                continue
            emb = _unit(z[f"{t}|{variant}|emb"][[pos[c] for c in group.index]])
            row = {"held_out_type": t, "held_out_class": "scam" if lab else "legitimate", "model": name,
                   "own_cluster_f1": _best_cluster_f1(in_type, emb, k, ctx.seed)}
            if f"{t}|{variant}|prob" in z.files:      # the label-free embedding has no scam score of its own
                prob = z[f"{t}|{variant}|prob"][[pos[c] for c in test]]
                row.update(flag_rate=float((prob[:len(held)] >= 0.5).mean()), pr_auc=float(average_precision_score(y, prob)),
                           recall_at_10pct_false_alarms=recall_at_fpr(y, prob, 0.10) if lab else float("nan"))
            rows.append(row)
        fit = prim[(prim["subtype"] != t) & prim["split"].isin(["train", "val"])].index
        vec = TfidfVectorizer(min_df=2, sublinear_tf=True, stop_words="english", max_features=20000)
        vec.fit(ctx.texts(fit))
        svd = TruncatedSVD(64, random_state=ctx.seed).fit(vec.transform(ctx.texts(fit)))
        bow_row = {"held_out_type": t, "held_out_class": "scam" if lab else "legitimate", "model": "Bag of words",
                   "own_cluster_f1": _best_cluster_f1(in_type, _unit(svd.transform(vec.transform(ctx.texts(group.index)))), k, ctx.seed)}
        if e1 is not None and t in e1.index:
            bow_row.update(flag_rate=float(e1.loc[t, "held_out_flag_rate"]), pr_auc=float(e1.loc[t, "pr_auc"]),
                           recall_at_10pct_false_alarms=float(e1.loc[t, "recall_at_fpr10"]) if lab else float("nan"))
        rows.append(bow_row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- charts
def plot_map(spaces: dict[str, np.ndarray], datasets: np.ndarray, path: Path, seed: int) -> None:
    names = {"bothbosu": "Synthetic (training data)", "kaggle": "Kaggle", "ncsu": "FTC robocalls"}
    show = [m for m in ("Bag of words", SCRIPT_SPACE, "GraphSAGE") if m in spaces]
    fig, axes = plt.subplots(1, len(show), figsize=(4.3 * len(show), 4.6), facecolor=SURFACE)
    for ax, m in zip(np.atleast_1d(axes), show):
        P = PCA(2, random_state=seed).fit_transform(spaces[m])
        for j, d in enumerate(dict.fromkeys(["bothbosu", "kaggle", "ncsu"])):
            mask = datasets == d
            ax.scatter(P[mask, 0], P[mask, 1], s=9, color=SERIES[j], alpha=0.55, label=names[d], linewidths=0)
        style_axes(ax, m)
        ax.set_xticks([])
        ax.set_yticks([])
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, labelcolor=MUTED, markerscale=2.2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_scripts(df: pd.DataFrame, path: Path, top: int = 10) -> None:
    d = df[df["real_calls"] >= MIN_SCRIPT_CALLS].sort_values("missed_by_bow_caught_by_graphsage", ascending=False).head(top)
    d = d.iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 0.46 * len(d) + 2.3), facecolor=SURFACE)
    y = np.arange(len(d))
    ax.hlines(y, d["bow_recall"], d["graphsage_recall"], color="#c3c2b7", linewidth=2, zorder=1)
    ax.scatter(d["bow_recall"], y, s=70, color=MODEL_COLORS["E1__lr"], label="Bag of words", zorder=3, edgecolors=SURFACE, linewidths=1.5)
    ax.scatter(d["graphsage_recall"], y, s=70, color=MODEL_COLORS["E4__graphsage"], label="GraphSAGE", zorder=3, edgecolors=SURFACE, linewidths=1.5)
    labels = [f"{' '.join(str(w).split()[:4])}  (n={n}{', hidden' if h else ''})"
              for w, n, h in zip(d["top_words"], d["real_calls"], d["hidden"])]
    ax.set_yticks(y, labels, fontsize=8.5)
    ax.set_xlim(0, 1.02)
    style_axes(ax, "Scams caught per script, at the same 10% false-alarm rate", "Share of real scam calls caught")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, labelcolor=MUTED, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------- run
def run_script_discovery(ctx, rctx, k: int = 16, ks=(8, 12, 16, 24)) -> dict:
    """`ctx` has the committed results (E1 scores); `rctx` has the KG, embeddings and GraphSAGE scores of one run."""
    from scamkg.extraction.benchmark import feedback_dir
    from scamkg.models.gnn import embeddings_path, loto_path, seeds_path
    fb = feedback_dir(ctx)
    calls = rctx.calls
    ids = np.array(calls.index[calls["label"] == 1].tolist())
    is_train = calls.loc[ids, "split"].isin(["train", "val"]).to_numpy()

    vec = TfidfVectorizer(min_df=2, sublinear_tf=True, stop_words="english", max_features=20000)
    tfidf = vec.fit_transform(rctx.texts(ids))
    vocab_names = np.array(vec.get_feature_names_out())
    spaces: dict[str, np.ndarray] = {
        "Bag of words": _unit(TruncatedSVD(64, random_state=rctx.seed).fit_transform(tfidf))}
    for name, variant in EMBEDDINGS:
        p = embeddings_path(rctx, variant)
        if p.exists():
            z = np.load(p, allow_pickle=False)
            e = dict(zip(z["ids"].astype(str).tolist(), z["emb"]))
            spaces[name] = _unit(np.stack([e[c] for c in ids]))
    sb = Path(rctx.paths.cache) / "embeddings" / f"{slugify(rctx.cfg['models']['sbert']['model_name'])}.npz"
    if sb.exists():
        z = np.load(sb, allow_pickle=False)
        e = dict(zip(z["ids"].astype(str).tolist(), z["emb"]))
        if all(c in e for c in ids):
            spaces["Sentence-BERT"] = _unit(np.stack([e[c] for c in ids]))

    onehot_names = rctx.fb.onehot_names()
    onehot = rctx.fb.onehot_matrix(list(ids), onehot_names).toarray()
    rows = [rctx.kg.call_index[c] for c in ids]
    entities = rctx.kg.X[rows]
    methods, assignments = method_table(spaces, list(ids), calls, is_train, tfidf, entities, ks, rctx.seed)
    methods.to_csv(fb / "script_methods.csv", index=False)

    def load(path):
        return pd.read_csv(path, dtype={"call_id": str}).set_index("call_id")["score"]

    scores = {"E1__lr": load(ctx.paths.predictions / "E1__lr.csv")}
    for m in ("E4__graphsage", "E4__graphsage_topo", "E4__graphsage_text", "E4__graphsage_kg"):
        p = rctx.paths.predictions / f"{m}.csv"
        if p.exists():
            scores[m] = load(p)
    for m in ("E2__lr", "E2G__lr"):
        p = ctx.paths.predictions / f"{m}.csv"
        if p.exists():
            scores[m] = load(p)
    neg = calls.index[(calls["split"] == "external") & (calls["label"] == 0)].tolist()
    rescue = rescue_table(rctx, scores, neg)
    rescue.to_csv(fb / "script_rescue.csv", index=False)

    tb = float(np.quantile(scores["E1__lr"].reindex(neg), 0.9))
    main_sage = "E4__graphsage"
    ts = float(np.quantile(scores[main_sage].reindex(neg), 0.9))
    space = SCRIPT_SPACE if SCRIPT_SPACE in spaces else next(n for n in spaces if n.startswith("GraphSAGE"))
    scripts = script_table(rctx, ids, assignments[(space, k)], is_train, tfidf, vocab_names, entities, onehot,
                           onehot_names, scores["E1__lr"], scores[main_sage], tb, ts, spaces[space])
    scripts.to_csv(fb / "scripts.csv", index=False)
    pd.DataFrame({"call_id": ids, "script": assignments[(space, k)]}).to_csv(fb / "script_assignments.csv", index=False)
    figs = fb / "figures"
    plot_map(spaces, calls.loc[ids, "dataset"].to_numpy(), figs / "script_map.png", rctx.seed)
    plot_scripts(scripts, figs / "script_recall.png")
    out = {"methods": methods[methods["k"] == k], "scripts": scripts[["script", "calls", "training_share", "hidden",
                                                                      "signature", "top_words", "bow_recall", "graphsage_recall",
                                                                      "missed_by_bow_caught_by_graphsage"]],
           "rescue": rescue}
    if seeds_path(rctx).exists():
        seeds = seed_table(rctx, seeds_path(rctx), scores["E1__lr"])
        seeds.to_csv(fb / "script_seeds.csv", index=False)
        out["rescue_seeds"] = seeds
    if loto_path(rctx).exists():
        loto = loto_table(ctx, rctx, loto_path(rctx))
        loto.to_csv(fb / "script_loto.csv", index=False)
        out["loto"] = loto.groupby(["held_out_class", "model"])[["flag_rate", "pr_auc", "recall_at_10pct_false_alarms",
                                                                 "own_cluster_f1"]].mean().reset_index()
        out["loto"].to_csv(fb / "script_loto_summary.csv", index=False)
    save_json(fb / "script_summary.json", {"k": k, "run": rctx.run_name, "n_scams": len(ids),
                                           "hidden_scripts": int(scripts["hidden"].sum()),
                                           "calls_in_hidden_scripts": int(scripts.loc[scripts["hidden"], "calls"].sum())})
    log.info("Script discovery -> %s", fb)
    return out


__all__ = ["dataset_mixing", "linked_share", "md_table", "run_script_discovery"]
