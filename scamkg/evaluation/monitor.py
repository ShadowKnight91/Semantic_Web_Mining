"""Recall monitor (professor's feedback: watch recall from balanced synthetic data to noisy, imbalanced FTC calls).

`monitor-recall` works from saved model scores and runs in seconds:
  ladder         recall at the deployed threshold on each step away from the training data, with bootstrap CIs
  prevalence     what happens as scams get rare: precision and false alarms at the same threshold, and the recall
                 left once the threshold must keep precision above a target
  recalibration  how much recall a small labelled pilot from the new domain recovers
  drift          label-free warning signals (score shift, unknown words, calls without KG facts)
  stream         a simulated deployment drifting from synthetic to FTC calls: do those signals fire in time?

`monitor-recall --noise` also re-runs models on transcripts with simulated speech-to-text errors (minutes).

Recall needs labels, and a deployed system rarely has them. That is why the drift signals are checked here against
the true recall: they are only useful if they move when recall moves.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from scamkg.evaluation.plot_style import INK, MODEL_COLORS, MODEL_LABELS, MUTED, SERIES, SURFACE, style_axes  # noqa: E402
from scamkg.utils import get_logger, save_json  # noqa: E402

log = get_logger(__name__)
MODELS = ["E1__lr", "E2__lr", "E2G__lr", "E3__distilbert", "E3__sbert_mlp", "E4__graphsage", "E4__graphsage_text",
          "E4__graphsage_topo"]
CHART_MODELS = ["E1__lr", "E2__lr", "E3__distilbert", "E4__graphsage", "E4__graphsage_topo"]
DEFAULTS = {"threshold": 0.5, "recall_floor": 0.80, "max_drop": 0.10, "psi_warn": 0.10, "psi_alert": 0.25,
            "oov_warn": 0.05, "oov_alert": 0.10, "target_fpr": 0.10, "prevalences": [0.5, 0.1, 0.05, 0.01],
            "precision_targets": [0.5, 0.8], "pilot_sizes": [20, 50, 100, 200], "pilot_draws": 200,
            "bootstrap": 1000, "stream_windows": 21, "stream_window_size": 300, "stream_scam_rate": 0.10,
            "noise_levels": ["clean", "light", "medium", "heavy"]}
_TOKEN = re.compile(r"[a-z0-9']+")


def monitor_config(cfg: dict) -> dict:
    return {**DEFAULTS, **(cfg.get("monitor") or {})}


# ---------------------------------------------------------------------------- pure metrics
def bootstrap_ci(hits: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    """95% percentile interval of the mean of a 0/1 array."""
    if len(hits) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = hits[rng.integers(0, len(hits), (n_boot, len(hits)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population stability index of two score samples in [0, 1]; 0 = same distribution."""
    edges = np.linspace(0, 1, bins + 1)
    a = np.histogram(np.clip(reference, 0, 1), edges)[0].astype(float)
    b = np.histogram(np.clip(current, 0, 1), edges)[0].astype(float)
    a = (a + 0.5) / (a.sum() + 0.5 * bins)
    b = (b + 0.5) / (b.sum() + 0.5 * bins)
    return float(((b - a) * np.log(b / a)).sum())


def precision_at_prevalence(tpr: float, fpr: float, prevalence: float) -> float:
    denom = prevalence * tpr + (1 - prevalence) * fpr
    return float(prevalence * tpr / denom) if denom > 0 else float("nan")


def recall_at_precision(pos: np.ndarray, neg: np.ndarray, prevalence: float, target: float) -> tuple[float, float]:
    """(best recall, threshold) with estimated precision >= target at this prevalence, or (nan, nan) if the sample
    cannot show it. The false-positive rate is add-one smoothed: zero false alarms among 400 calls is not proof of
    a zero rate, so a finite sample can never claim perfect precision."""
    best, best_t = float("nan"), float("nan")
    for t in np.unique(pos):
        tpr = float((pos >= t).mean())
        fpr = ((neg >= t).sum() + 1) / (len(neg) + 1)
        if precision_at_prevalence(tpr, fpr, prevalence) >= target and not tpr <= best:
            best, best_t = tpr, float(t)
    return best, best_t


def _spearman(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float("nan") if a.std() == 0 or b.std() == 0 else float(spearmanr(a, b).statistic)


def status_for(recall: float, ci_high: float, drop: float, mc: dict) -> str:
    if ci_high < mc["recall_floor"]:
        return "ALERT"
    if recall < mc["recall_floor"] or drop > mc["max_drop"]:
        return "WARN"
    return "OK"


def pilot_recalibration(pos: np.ndarray, neg: np.ndarray, size: int, target_fpr: float, draws: int,
                        seed: int = 42) -> dict:
    """Set the threshold on `size` labelled calls from the new domain (half scams) and score the calls left over."""
    rng = np.random.default_rng(seed)
    rec, fpr = [], []
    half = size // 2
    for _ in range(draws):
        pi, ni = rng.permutation(len(pos)), rng.permutation(len(neg))
        t = np.quantile(neg[ni[:half]], 1 - target_fpr)
        rec.append(float((pos[pi[half:]] > t).mean()))
        fpr.append(float((neg[ni[half:]] > t).mean()))
    return {"pilot_size": size, "recall": float(np.mean(rec)), "recall_sd": float(np.std(rec)),
            "false_alarm_rate": float(np.mean(fpr)), "false_alarm_sd": float(np.std(fpr))}


# ---------------------------------------------------------------------------- inputs
def load_scores(ctx, models=MODELS) -> dict[str, pd.Series]:
    out = {}
    for m in models:
        p = ctx.paths.predictions / f"{m}.csv"
        if p.exists():
            out[m] = pd.read_csv(p, dtype={"call_id": str}).set_index("call_id")["score"]
        else:
            log.warning("No saved scores for %s (%s)", m, p.name)
    return out


def stage_sets(ctx) -> dict[str, dict[str, list[str]]]:
    """Each step away from the training data: scams (pos) and legitimate calls (neg). FTC robocalls are all scams,
    so their false alarms come from Kaggle's legitimate calls."""
    c = ctx.calls
    test = c[c["split"] == "test"]
    stages = {"synthetic_test": {"pos": test.index[test["label"] == 1].tolist(),
                                 "neg": test.index[test["label"] == 0].tolist()}}
    ext = ctx.external_datasets
    pos_only = [d for d in ext if c.loc[c["dataset"] == d, "label"].nunique() == 1]
    mixed = [d for d in ext if d not in pos_only]
    for d in mixed:
        sub = c[c["dataset"] == d]
        stages[d] = {"pos": sub.index[sub["label"] == 1].tolist(), "neg": sub.index[sub["label"] == 0].tolist()}
    negs = stages[mixed[0]]["neg"] if mixed else stages["synthetic_test"]["neg"]
    for d in pos_only:
        stages[f"{d}_robocalls"] = {"pos": c.index[c["dataset"] == d].tolist(), "neg": negs}
    return stages


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(str(text).lower())


def oov_rates(ctx, ids: list[str], vocab: set[str]) -> np.ndarray:
    out = np.zeros(len(ids))
    for i, text in enumerate(ctx.calls.loc[ids, "text_plain"]):
        toks = _tokens(text)
        out[i] = sum(w not in vocab for w in toks) / max(1, len(toks))
    return out


def training_vocabulary(ctx, min_count: int = 2) -> set[str]:
    counts: dict[str, int] = {}
    for text in ctx.calls.loc[ctx.train_ids + ctx.val_ids, "text_plain"]:
        for w in set(_tokens(text)):
            counts[w] = counts.get(w, 0) + 1
    return {w for w, n in counts.items() if n >= min_count}


# ---------------------------------------------------------------------------- tables
def ladder_table(ctx, scores: dict[str, pd.Series], mc: dict) -> pd.DataFrame:
    stages = stage_sets(ctx)
    t = float(mc["threshold"])
    rows = []
    for model, s in scores.items():
        base = None
        for stage, sets in stages.items():
            pos, neg = s.reindex(sets["pos"]).dropna().to_numpy(), s.reindex(sets["neg"]).dropna().to_numpy()
            hits = (pos >= t).astype(float)
            low, high = bootstrap_ci(hits, int(mc["bootstrap"]), ctx.seed)
            recall = float(hits.mean())
            base = recall if base is None else base
            fpr = float((neg >= t).mean())
            tf = np.quantile(neg, 1 - float(mc["target_fpr"]))
            rows.append({"model": model, "stage": stage, "n_scams": len(pos), "n_legit": len(neg), "recall": recall,
                         "recall_ci_low": low, "recall_ci_high": high, "recall_drop": base - recall,
                         "false_alarm_rate": fpr, "precision": precision_at_prevalence(recall, fpr, len(pos) / (len(pos) + len(neg))),
                         "recall_at_target_fpr": float((pos > tf).mean()),
                         "status": status_for(recall, high, base - recall, mc)})
    return pd.DataFrame(rows)


def prevalence_table(ctx, scores: dict[str, pd.Series], mc: dict) -> pd.DataFrame:
    """Always the hardest step: FTC robocalls against real-world legitimate calls."""
    sets = list(stage_sets(ctx).values())[-1]
    t = float(mc["threshold"])
    rows = []
    for model, s in scores.items():
        pos, neg = s.reindex(sets["pos"]).dropna().to_numpy(), s.reindex(sets["neg"]).dropna().to_numpy()
        tpr, fpr = float((pos >= t).mean()), float((neg >= t).mean())
        for pv in mc["prevalences"]:
            row = {"model": model, "prevalence": pv, "recall_at_threshold": tpr,
                   "precision_at_threshold": precision_at_prevalence(tpr, fpr, pv),
                   "false_alarms_per_1000_calls": 1000 * (1 - pv) * fpr, "scams_caught_per_1000_calls": 1000 * pv * tpr,
                   "scams_missed_per_1000_calls": 1000 * pv * (1 - tpr)}
            for target in mc["precision_targets"]:
                row[f"recall_at_precision_{target:g}"] = recall_at_precision(pos, neg, pv, target)[0]
            rows.append(row)
    return pd.DataFrame(rows)


def recalibration_table(ctx, scores: dict[str, pd.Series], mc: dict) -> pd.DataFrame:
    rows = []
    t = float(mc["threshold"])
    stages = stage_sets(ctx)
    for stage in list(stages)[1:]:
        sets = stages[stage]
        for model, s in scores.items():
            pos, neg = s.reindex(sets["pos"]).dropna().to_numpy(), s.reindex(sets["neg"]).dropna().to_numpy()
            base = {"model": model, "stage": stage, "frozen_recall": float((pos >= t).mean()),
                    "frozen_false_alarm_rate": float((neg >= t).mean()), "target_fpr": mc["target_fpr"]}
            for size in mc["pilot_sizes"]:
                rows.append({**base, **pilot_recalibration(pos, neg, int(size), float(mc["target_fpr"]),
                                                           int(mc["pilot_draws"]), ctx.seed)})
    return pd.DataFrame(rows)


def _kg_coverage(ctx, ids: list[str]) -> dict:
    if ctx.kg is None:
        return {}
    idx = [ctx.kg.call_index[c] for c in ids]
    n = np.asarray(ctx.kg.X[idx].sum(axis=1)).ravel()
    return {"mean_kg_facts": float(n.mean()), "no_kg_facts_rate": float((n == 0).mean())}


def drift_table(ctx, scores: dict[str, pd.Series], mc: dict, vocab: set[str]) -> pd.DataFrame:
    """No labels are read here. Reference: the in-distribution test split."""
    stages = stage_sets(ctx)
    ref_ids = stages["synthetic_test"]["pos"] + stages["synthetic_test"]["neg"]
    ref_oov = float(oov_rates(ctx, ref_ids, vocab).mean())
    rows = []
    for stage, sets in stages.items():
        ids = list(dict.fromkeys(sets["pos"] + sets["neg"]))
        oov = float(oov_rates(ctx, ids, vocab).mean())
        common = {"stage": stage, "n_calls": len(ids), "unknown_word_rate": oov, "unknown_word_shift": oov - ref_oov,
                  "median_words": float(ctx.calls.loc[ids, "n_words"].median()), **_kg_coverage(ctx, ids)}
        for model, s in scores.items():
            cur = s.reindex(ids).dropna().to_numpy()
            ref = s.reindex(ref_ids).dropna().to_numpy()
            p = psi(ref, cur)
            status = "ALERT" if p >= mc["psi_alert"] or oov - ref_oov >= mc["oov_alert"] else \
                     "WARN" if p >= mc["psi_warn"] or oov - ref_oov >= mc["oov_warn"] else "OK"
            rows.append({**common, "model": model, "alert_rate": float((cur >= mc["threshold"]).mean()),
                         "score_psi": p, "status": status})
    return pd.DataFrame(rows)


def _first_sustained(flags, run: int = 2) -> int | None:
    """First window that starts `run` consecutive True values: one noisy window is not a trend."""
    flags = list(flags)
    for i in range(len(flags) - run + 1):
        if all(flags[i:i + run]):
            return i
    return None


def stream_simulation(ctx, scores: dict[str, pd.Series], mc: dict, vocab: set[str],
                      max_share: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Windows of calls whose share of real calls grows from 0 to 100%, at a fixed scam rate. The monitor sees scores
    and words; true recall is computed afterwards to check whether its signals fired before recall fell.

    The reference is a large baseline sample at the stream's own scam rate. Comparing against the balanced test
    split would measure the change in class mix, so every window would look like drift."""
    stages = stage_sets(ctx)
    synth = stages["synthetic_test"]
    real = list(stages.values())[-1]
    n_win, size = int(mc["stream_windows"]), int(mc["stream_window_size"])
    rate = float(mc["stream_scam_rate"])
    n_pos = int(round(size * rate))
    ids = list(dict.fromkeys(synth["pos"] + synth["neg"] + real["pos"] + real["neg"]))
    oov = dict(zip(ids, oov_rates(ctx, ids, vocab)))
    rows = []
    for model, s in scores.items():
        rng = np.random.default_rng(ctx.seed)     # every model sees the same stream
        n_base = 5000
        base_pos = rng.choice(synth["pos"], int(n_base * rate))
        base_neg = rng.choice(synth["neg"], n_base - int(n_base * rate))
        base_ids = list(base_pos) + list(base_neg)
        ref = s.reindex(base_ids).to_numpy()
        ref_oov = float(np.mean([oov[c] for c in base_ids]))
        for w in range(n_win):
            share = max_share * w / (n_win - 1)

            def draw(a, b, k):
                from_b = rng.random(k) < share
                return [b[i] if f else a[j] for f, i, j in zip(from_b, rng.integers(0, len(b), k),
                                                               rng.integers(0, len(a), k))]

            pos, neg = draw(synth["pos"], real["pos"], n_pos), draw(synth["neg"], real["neg"], size - n_pos)
            sp, sn = s.reindex(pos).to_numpy(), s.reindex(neg).to_numpy()
            all_s = np.concatenate([sp, sn])
            p, o = psi(ref, all_s), float(np.mean([oov[c] for c in pos + neg])) - ref_oov
            rows.append({"model": model, "window": w, "real_call_share": share,
                         "true_recall": float((sp >= mc["threshold"]).mean()),
                         "alert_rate": float((all_s >= mc["threshold"]).mean()), "score_psi": p, "unknown_word_shift": o,
                         "warning_psi": bool(p >= mc["psi_warn"]), "warning_words": bool(o >= mc["oov_warn"]),
                         "alert": bool(p >= mc["psi_alert"] or o >= mc["oov_alert"])})
    df = pd.DataFrame(rows)
    summary = []
    for model, g in df.groupby("model", sort=False):
        base = g["true_recall"].iloc[:3].mean()
        first_drop = _first_sustained(g["true_recall"] < base - mc["max_drop"])
        first_psi = _first_sustained(g["warning_psi"])
        first_words = _first_sustained(g["warning_words"])
        first_alert = _first_sustained(g["alert"])
        known = [w for w in (first_psi, first_words) if w is not None]
        first_warn = min(known) if known else None
        summary.append({"model": model, "baseline_recall": base, "final_recall": g["true_recall"].iloc[-3:].mean(),
                        "first_recall_drop_window": first_drop, "first_warning_window": first_warn,
                        "first_psi_warning_window": first_psi, "first_word_warning_window": first_words,
                        "first_alert_window": first_alert,
                        "lead_windows": first_drop - first_warn if first_drop is not None and first_warn is not None else None,
                        "spearman_recall_vs_psi": _spearman(g["true_recall"], g["score_psi"]),
                        "spearman_recall_vs_unknown_words": _spearman(g["true_recall"], g["unknown_word_shift"]),
                        "spearman_recall_vs_alert_rate": _spearman(g["true_recall"], g["alert_rate"])})
    return df, pd.DataFrame(summary)


# ---------------------------------------------------------------------------- charts
def _models_in(df, models=CHART_MODELS) -> list[str]:
    return [m for m in models if m in set(df["model"])]


def plot_ladder(df: pd.DataFrame, path: Path) -> None:
    stages = list(dict.fromkeys(df["stage"]))
    labels = {"synthetic_test": "Synthetic test", "kaggle": "Kaggle", "ncsu_robocalls": "FTC robocalls"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True, facecolor=SURFACE)
    for ax, col, title in ((axes[0], "recall", "Default threshold (0.5)"),
                           (axes[1], "recall_at_target_fpr", "Threshold set for 10% false alarms")):
        for m in _models_in(df):
            g = df[df["model"] == m].set_index("stage").loc[stages]
            ax.plot(range(len(stages)), g[col], marker="o", markersize=7, linewidth=2, color=MODEL_COLORS[m],
                    label=MODEL_LABELS[m], markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_xticks(range(len(stages)), [labels.get(s, s) for s in stages])
        ax.set_ylim(0, 1.04)
        style_axes(ax, title)
    axes[0].set_ylabel("Share of scams caught", color=MUTED)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, frameon=False, labelcolor=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_prevalence(df: pd.DataFrame, target: float, path: Path) -> None:
    col = f"recall_at_precision_{target:g}"
    fig, ax = plt.subplots(figsize=(8, 4.3), facecolor=SURFACE)
    for m in _models_in(df):
        g = df[df["model"] == m].sort_values("prevalence")
        ax.plot(g["prevalence"], g[col], marker="o", markersize=7, linewidth=2, color=MODEL_COLORS[m],
                label=MODEL_LABELS[m], markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_xscale("log")
    ticks = sorted(df["prevalence"].unique())
    ax.set_xticks(ticks, [f"{t:.0%}" for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(0, 1.04)
    style_axes(ax, f"Recall left when at least {target:.0%} of alerts must be real scams",
               "Share of calls that are scams", "Share of FTC scams caught")
    shown = df[df["model"].isin(_models_in(df))]
    for t in ticks:
        if shown.loc[shown["prevalence"] == t, col].isna().all():
            ax.annotate(f"No model reaches\n{target:.0%} precision here", (t, 0.5), xytext=(8, 0),
                        textcoords="offset points", color=MUTED, fontsize=9, va="center")
    ax.legend(frameon=False, labelcolor=MUTED, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_recalibration(df: pd.DataFrame, size: int, path: Path) -> None:
    """One row per model: recall at the default threshold, and after a pilot sets it for the same domain."""
    d = df[(df["pilot_size"] == size) & (df["stage"].str.contains("robocalls"))].sort_values("recall")
    fig, ax = plt.subplots(figsize=(9.4, 0.5 * len(d) + 2.2), facecolor=SURFACE)
    y = np.arange(len(d))
    ax.hlines(y, d["frozen_recall"], d["recall"], color="#c3c2b7", linewidth=2, zorder=1)
    ax.scatter(d["frozen_recall"], y, s=70, color=SERIES[1], label="Default threshold (0.5)", zorder=3,
               edgecolors=SURFACE, linewidths=1.5)
    ax.scatter(d["recall"], y, s=70, color=SERIES[0], label=f"Threshold set from a {size}-call pilot", zorder=3,
               edgecolors=SURFACE, linewidths=1.5)
    # recall alone flatters a model that flags everything, so each label carries the false alarms behind its 0.5 dot
    ax.set_yticks(y, [f"{MODEL_LABELS.get(m, m)} ({fa:.0%} false alarms at 0.5)"
                      for m, fa in zip(d["model"], d["frozen_false_alarm_rate"])], fontsize=9)
    ax.set_xlim(0, 1.02)
    style_axes(ax, "FTC recall, before and after a small labelled pilot",
               "Share of FTC scams caught (the pilot targets 10% false alarms)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, labelcolor=MUTED, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_stream(df: pd.DataFrame, mc: dict, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6), sharex=True, facecolor=SURFACE)
    for m in _models_in(df):
        g = df[df["model"] == m]
        axes[0].plot(g["real_call_share"], g["true_recall"], linewidth=2, color=MODEL_COLORS[m], label=MODEL_LABELS[m])
        axes[1].plot(g["real_call_share"], g["score_psi"], linewidth=2, color=MODEL_COLORS[m], label=MODEL_LABELS[m])
    axes[0].set_ylim(0, 1.04)
    axes[1].axhline(mc["psi_alert"], color=MUTED, linewidth=1, linestyle="--")
    axes[1].annotate(f"alert level ({mc['psi_alert']})", (1, mc["psi_alert"]), xytext=(-4, -13), textcoords="offset points",
                     color=MUTED, fontsize=8.5, ha="right")
    axes[1].set_yscale("log")      # the models differ by two orders of magnitude; on a linear scale most look flat
    style_axes(axes[0], "True recall in each window (the monitor cannot see this)", ylabel="Share of scams caught")
    style_axes(axes[1], "Score shift the monitor sees without labels", "Share of real calls in the window",
               "Population stability index (log)")
    axes[1].xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[0].legend(frameon=False, labelcolor=MUTED, fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------- run
def run_monitor(ctx, out_dir: Path | None = None) -> dict:
    from scamkg.extraction.benchmark import feedback_dir
    fb = Path(out_dir) if out_dir else feedback_dir(ctx)
    (fb / "figures").mkdir(parents=True, exist_ok=True)
    mc = monitor_config(ctx.cfg)
    scores = load_scores(ctx)
    if not scores:
        raise FileNotFoundError(f"No saved model scores in {ctx.paths.predictions}: run `train` first")
    vocab = training_vocabulary(ctx)
    ladder = ladder_table(ctx, scores, mc)
    prev = prevalence_table(ctx, scores, mc)
    recal = recalibration_table(ctx, scores, mc)
    drift = drift_table(ctx, scores, mc, vocab)
    stream, stream_sum = stream_simulation(ctx, scores, mc, vocab)
    # the same monitor on a stream that never drifts: a warning here is a false alarm of the monitor itself
    quiet, _ = stream_simulation(ctx, scores, mc, vocab, max_share=0.0)
    flags = quiet.assign(flag=quiet["warning_psi"] | quiet["warning_words"]).groupby("model", sort=False)
    stream_sum = stream_sum.merge(flags["flag"].mean().rename("no_drift_warning_windows").reset_index(), on="model")
    stream_sum = stream_sum.merge(flags["flag"].apply(lambda f: _first_sustained(f) is not None)
                                  .rename("no_drift_sustained_warning").reset_index(), on="model")
    for name, df in (("recall_ladder", ladder), ("recall_prevalence", prev), ("recall_recalibration", recal),
                     ("drift_signals", drift), ("recall_stream", stream), ("recall_stream_summary", stream_sum)):
        df.to_csv(fb / f"{name}.csv", index=False)
    figs = fb / "figures"
    plot_ladder(ladder, figs / "recall_ladder.png")
    plot_prevalence(prev, 0.5, figs / "recall_prevalence.png")
    if not recal.empty:
        plot_recalibration(recal, 100 if 100 in set(recal["pilot_size"]) else int(recal["pilot_size"].max()),
                           figs / "recall_recalibration.png")
    plot_stream(stream, mc, figs / "recall_stream.png")
    save_json(fb / "monitor_config.json", mc)
    log.info("Recall monitor -> %s", fb)
    return {"ladder": ladder, "prevalence": prev, "recalibration": recal, "drift": drift, "stream": stream_sum}


# ---------------------------------------------------------------------------- noise robustness
def noisy_context(ctx, level: str):
    """A copy of the context whose external calls carry noise at `level`, with triples re-extracted by the rule
    extractor (it re-runs instantly) and the graph features rebuilt. Training calls are not touched. The label-free
    graph features are cached per level, since the classical models and the GraphSAGE worker both need them."""
    from scamkg.data.noise import noisy_calls
    from scamkg.extraction.extractor import make_extractor
    from scamkg.graph.build import CallKG
    from scamkg.graph.features import GraphFeatureBuilder
    ext = ctx.calls.index[ctx.calls["dataset"].isin(ctx.external_datasets)].tolist()
    new = copy.copy(ctx)
    new.calls = noisy_calls(ctx.calls, ext, level, ctx.seed)
    extractor = make_extractor("rule", ctx.cfg, ctx.ontology)
    extractions = dict(ctx.extractions)
    for cid in ext:
        extractions[cid] = extractor.extract(cid, new.calls.at[cid, "text"]).triples
    new.extractions = extractions
    new.kg = CallKG(new.calls.index.tolist(), extractions, ctx.ontology)
    cache = ctx.paths.graph_features / f"structural__noise_{level}__{ctx.run_name}.csv"
    structural = None
    if cache.exists():
        structural = pd.read_csv(cache, index_col=0)
        structural.index = structural.index.astype(str)
    new.fb = GraphFeatureBuilder(new.kg, ctx.cfg["graph"], structural=structural, seed=ctx.seed)
    new.fb.structural.to_csv(cache)
    return new


def _noise_rows(ctx, model_name: str, level: str, score: pd.Series, clean_threshold: dict, mc: dict) -> list[dict]:
    rows = []
    for stage, sets in list(stage_sets(ctx).items())[1:]:
        pos, neg = score.reindex(sets["pos"]).to_numpy(), score.reindex(sets["neg"]).to_numpy()
        key = (model_name, stage)
        if key not in clean_threshold:     # the first level is clean: the threshold a deployed model would keep
            clean_threshold[key] = float(np.quantile(neg, 1 - float(mc["target_fpr"])))
        t10 = clean_threshold[key]
        rows.append({"model": model_name, "noise": level, "stage": stage,
                     "recall_default": float((pos >= mc["threshold"]).mean()),
                     "false_alarm_default": float((neg >= mc["threshold"]).mean()),
                     "recall_clean_threshold": float((pos > t10).mean()),
                     "false_alarm_clean_threshold": float((neg > t10).mean()),
                     "recall_at_target_fpr": float((pos > np.quantile(neg, 1 - float(mc["target_fpr"]))).mean())})
    return rows


def noise_rows(ctx, model_name: str, level: str, score: pd.Series, clean_threshold: dict) -> list[dict]:
    return _noise_rows(ctx, model_name, level, score, clean_threshold, monitor_config(ctx.cfg))


def run_noise_classical(ctx, levels, specs=(("E1", "lr"), ("E2", "lr"), ("E2G", "lr"))) -> pd.DataFrame:
    """Train once on clean calls, as a deployment would, then score noisy external calls with those same models.
    Noise changes the KG, so label-based graph features are rebuilt, still from training labels only."""
    from scamkg.evaluation.evaluator import Evaluator
    from scamkg.models.classical import DEFAULT_PARAMS, assemble, fit_assembler, make_model, predict_scores
    from scamkg.utils import load_json
    mc = monitor_config(ctx.cfg)
    params_path = ctx.paths.results / "best_params.json"
    best = load_json(params_path) if params_path.exists() else {}
    clean = noisy_context(ctx, "clean")
    fit_ids = clean.train_ids + clean.val_ids
    y = clean.labels(fit_ids)
    trained = {}
    for exp, m in specs:
        fitted, gdf = fit_assembler(clean, exp, fit_ids)
        params = best.get(f"{exp}__{m}", {}).get("params", DEFAULT_PARAMS[m])
        trained[(exp, m)] = (fitted, make_model(m, params, clean.seed, y).fit(assemble(clean, fitted, gdf, fit_ids), y))
    rows, thresholds = [], {}
    for level in levels:
        nctx = noisy_context(ctx, level)
        eval_ids = Evaluator(nctx).eval_ids()
        gdf = nctx.fb.build(fit_ids, y)
        for (exp, m), (fitted, model) in trained.items():
            s = pd.Series(predict_scores(model, assemble(nctx, fitted, gdf, eval_ids)), index=eval_ids)
            rows += _noise_rows(nctx, f"{exp}__{m}", level, s, thresholds, mc)
        log.info("Noise level %s: classical models done", level)
    return pd.DataFrame(rows)


def plot_noise(df: pd.DataFrame, path: Path) -> None:
    from scamkg.data.noise import NOISE_LEVELS
    d = df[df["stage"].str.contains("robocalls")]
    levels = [lv for lv in NOISE_LEVELS if lv in set(d["noise"])]
    labels = [f"{NOISE_LEVELS[lv][0]:.0%}" for lv in levels]
    models = [m for m in ["E1__lr", "E2__lr", "E2G__lr", "E4__graphsage", "E4__graphsage_topo"] if m in set(d["model"])]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), facecolor=SURFACE)
    for ax, col, title, ylabel, ylim in (
            (axes[0], "recall_clean_threshold", "FTC scams caught", "Share of FTC scams caught", 1.04),
            (axes[1], "false_alarm_clean_threshold", "Legitimate calls flagged", "Share of legitimate calls flagged", None)):
        for m in models:
            g = d[d["model"] == m].groupby("noise")[col].mean().loc[levels]
            ax.plot(labels, g, marker="o", markersize=7, linewidth=2, color=MODEL_COLORS[m], label=MODEL_LABELS[m],
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        if ylim:
            ax.set_ylim(0, ylim)
        else:
            ax.set_ylim(bottom=0)
        style_axes(ax, title, "Simulated word error rate", ylabel)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, frameon=False, labelcolor=MUTED, fontsize=9)
    fig.suptitle("Threshold frozen on clean transcripts (rule-based KG)", x=0.012, ha="left", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.11, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
