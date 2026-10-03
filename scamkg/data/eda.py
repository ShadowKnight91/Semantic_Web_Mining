"""Exploratory analysis (proposal Sec. II): lengths, class balance, top entities, vocabulary overlap."""
from __future__ import annotations

import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scamkg.utils import get_logger, md_table, save_json  # noqa: E402

log = get_logger(__name__)
TOKEN_RE = re.compile(r"[a-z][a-z']+")


def _vocab(texts: pd.Series, min_count: int = 2) -> Counter:
    c = Counter()
    for t in texts:
        c.update(TOKEN_RE.findall(t.lower()))
    return Counter({w: n for w, n in c.items() if n >= min_count})


def vocabulary_overlap(calls: pd.DataFrame) -> pd.DataFrame:
    vocabs = {ds: _vocab(g["text_plain"]) for ds, g in calls.groupby("dataset")}
    rows = []
    for a, b in combinations(sorted(vocabs), 2):
        va, vb = set(vocabs[a]), set(vocabs[b])
        tokens_b = sum(vocabs[b].values())
        oov_b = sum(n for w, n in vocabs[b].items() if w not in va) / tokens_b if tokens_b else np.nan
        rows.append({"dataset_a": a, "dataset_b": b, "vocab_a": len(va), "vocab_b": len(vb),
                     "jaccard": len(va & vb) / len(va | vb) if va | vb else np.nan,
                     "oov_rate_b_given_a": oov_b})
    return pd.DataFrame(rows)


def top_entities(calls: pd.DataFrame, extractions: dict[str, list[dict]], k: int = 10) -> pd.DataFrame:
    rows = []
    for cid, triples in extractions.items():
        if cid not in calls.index:
            continue
        ds, label = calls.at[cid, "dataset"], int(calls.at[cid, "label"])
        for t in triples:
            rows.append({"dataset": ds, "label": label, "type": t["object_type"], "object": t["object"]})
    if not rows:
        return pd.DataFrame(columns=["dataset", "label", "type", "object", "calls"])
    df = pd.DataFrame(rows)
    counts = df.groupby(["dataset", "label", "type", "object"]).size().rename("calls").reset_index()
    return (counts.sort_values("calls", ascending=False).groupby(["dataset", "label", "type"]).head(k)
            .sort_values(["dataset", "type", "label", "calls"], ascending=[True, True, True, False]))


def run_eda(calls: pd.DataFrame, out_dir: Path, extractions: dict[str, list[dict]] | None = None) -> dict:
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    labeled = calls[calls["label"] >= 0]
    stats = (labeled.groupby("dataset").agg(n=("label", "size"), n_scam=("label", "sum"),
                                            scam_rate=("label", "mean"), mean_words=("n_words", "mean"),
                                            median_words=("n_words", "median"), mean_turns=("n_turns", "mean"))
             .reset_index())
    splits = calls.groupby(["dataset", "split"]).size().rename("n").reset_index()
    overlap = vocabulary_overlap(calls)

    # length distributions
    datasets = sorted(calls["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.2), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        sub = calls[calls["dataset"] == ds]
        for lab, name, color in ((0, "legitimate", "#4C78A8"), (1, "scam", "#E45756")):
            vals = sub.loc[sub["label"] == lab, "n_words"]
            if len(vals):
                ax.hist(vals, bins=30, alpha=0.6, label=name, color=color)
        ax.set_title(ds)
        ax.set_xlabel("words per transcript")
    axes[0][0].set_ylabel("calls")
    axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_length_hist.png", dpi=130)
    plt.close(fig)

    # class balance
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    x = np.arange(len(stats))
    ax.bar(x - 0.2, stats["n"] - stats["n_scam"], 0.4, label="legitimate", color="#4C78A8")
    ax.bar(x + 0.2, stats["n_scam"], 0.4, label="scam", color="#E45756")
    ax.set_xticks(x, stats["dataset"], rotation=15)
    ax.set_ylabel("calls")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "eda_class_balance.png", dpi=130)
    plt.close(fig)

    report = {"datasets": stats.to_dict(orient="records"), "splits": splits.to_dict(orient="records"),
              "vocabulary_overlap": overlap.to_dict(orient="records")}
    md = ["# Exploratory data analysis\n", "## Data sets\n", md_table(stats), "\n## Splits\n", md_table(splits),
          "\n## Vocabulary overlap (hard cross-dataset transfer = low Jaccard / high OOV)\n", md_table(overlap)]

    if extractions:
        top = top_entities(calls, extractions)
        report["top_entities"] = top.to_dict(orient="records")
        md += ["\n## Most frequent extracted entities\n", md_table(top, max_rows=120)]
        focus = top[top["type"].isin(["ClaimedIdentity", "RequestedAsset"])]
        if not focus.empty:
            agg = focus.groupby(["type", "object", "label"])["calls"].sum().unstack(fill_value=0)
            agg = agg.loc[agg.sum(axis=1).sort_values(ascending=False).index[:16]]
            fig, ax = plt.subplots(figsize=(7, 4.5))
            y = np.arange(len(agg))
            left = np.zeros(len(agg))
            for lab, name, color in ((0, "legitimate", "#4C78A8"), (1, "scam", "#E45756")):
                if lab in agg.columns:
                    ax.barh(y, agg[lab], left=left, label=name, color=color)
                    left = left + agg[lab].to_numpy()
            ax.set_yticks(y, [f"{t}: {o}" for t, o in agg.index])
            ax.invert_yaxis()
            ax.set_xlabel("calls")
            ax.set_title("Claimed identities and requested assets")
            ax.legend()
            fig.tight_layout()
            fig.savefig(fig_dir / "eda_top_entities.png", dpi=130)
            plt.close(fig)

    save_json(out_dir / "eda_summary.json", report)
    (out_dir / "eda_report.md").write_text("\n".join(md), encoding="utf-8")
    log.info("EDA written to %s", out_dir)
    return report
