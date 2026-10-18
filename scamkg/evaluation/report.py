"""Aggregate every results file into results/REPORT.md (tables for the final report)."""
from __future__ import annotations

import pandas as pd

from scamkg.utils import get_logger, load_json, md_table

log = get_logger(__name__)
SHOW = ["experiment", "model", "n", "precision", "recall", "f1", "pr_auc", "roc_auc",   # n: sampled LLM rows
        "recall_at_fpr10"]


def generalization_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Recall / PR-AUC on each external set and the drop relative to the in-distribution test split (RQ3)."""
    test = metrics[metrics["eval_set"] == "test"].set_index(["experiment", "model"])
    rows = []
    for (exp, model, es), r in metrics.set_index(["experiment", "model", "eval_set"]).iterrows():
        if es in ("test", "test_imbalanced") or (exp, model) not in test.index:
            continue
        t = test.loc[(exp, model)]
        rows.append({"experiment": exp, "model": model, "eval_set": es, "recall": r["recall"],
                     "recall_drop": t["recall"] - r["recall"], "pr_auc": r["pr_auc"],
                     "pr_auc_drop": t["pr_auc"] - r["pr_auc"] if pd.notna(r["pr_auc"]) else float("nan")})
    return pd.DataFrame(rows)


def build_report(ctx) -> str:
    res = ctx.paths.results
    md = ["# Results report\n", f"Extraction run used for the KG: `{ctx.run_name or 'n/a'}`\n"]
    mpath = res / "metrics.csv"
    if mpath.exists():
        m = pd.read_csv(mpath)
        for es, title in (("test", "Balanced test split"), ("test_imbalanced", "Imbalanced test split (~5% scams, mean over draws)")):
            sub = m[m["eval_set"] == es].sort_values(["experiment", "model"])
            if not sub.empty:
                cols = SHOW + ([f"{c}_std" for c in ("recall", "pr_auc") if f"{c}_std" in sub] if es == "test_imbalanced" else [])
                md += [f"\n## {title}\n", md_table(sub[cols])]
        ext = m[~m["eval_set"].isin(["test", "test_imbalanced"])]
        if not ext.empty:
            md += ["\n## Cross-dataset generalization (X2)\n",
                   md_table(ext.sort_values(["eval_set", "experiment", "model"])[["eval_set", *SHOW]])]
            gen = generalization_table(m)
            if not gen.empty:
                gen.to_csv(res / "generalization.csv", index=False)
                md += ["\n### Recall / PR-AUC drop vs. in-distribution test\n", md_table(gen)]
    for fname, title in (("cv_results.csv", "5-fold stratified cross-validation (mean / std)"),
                         ("loto_summary.csv", "Leave-one-type-out (flag rate = recall on unseen scam types, "
                                              "false-positive rate on unseen legitimate types)"),
                         ("significance.csv", "McNemar tests (E1 vs graph-enhanced)"),
                         ("extraction_quality.csv", "Extraction quality (X1)")):
        p = res / fname
        if p.exists():
            md += [f"\n## {title}\n", md_table(pd.read_csv(p))]
    p = res / "graphsage_fusion.json"
    if p.exists():   # CHANGE (TA comment 2): document the GraphSAGE input fusion and its block ablation
        g = load_json(p)
        fus = g.get("fusion") or {}
        md += ["\n## GraphSAGE input fusion (E4)\n",
               f"Text block -> {fus.get('text_dim')}-d, topology block ({len(g.get('topology_features', []))} "
               f"label-free features) -> {fus.get('topo_dim')}-d, node type -> {fus.get('type_dim')}-d; "
               "concatenated, LayerNorm, then 2 mean-aggregation SAGE layers. "
               f"Training graph: {'inductive (no test / external calls)' if g.get('inductive') else 'transductive'}.\n",
               "Topology features: " + ", ".join(f"`{f}`" for f in g.get("topology_features", [])) + "\n",
               md_table(pd.DataFrame([{k: v for k, v in r.items() if k != "input_dims"} for r in g.get("variants", [])]))]
    p = res / "clustering.json"
    if p.exists():
        c = load_json(p)
        md += ["\n## Clustering (E5)\n",
               md_table(pd.DataFrame([{"method": "kmeans_best", **c["kmeans_best"]}, {"method": "kmeans_k2", **c["kmeans_k2"]},
                                      {"method": "dbscan", **c.get("dbscan", {})}])),
               "\n### Discovered scripts\n",
               md_table(pd.DataFrame([{k: v for k, v in s.items() if k != "top_entities"} for s in c["scripts"]]))]
    p = res / "ablation.csv"
    if p.exists():
        a = pd.read_csv(p)
        a = a[(a["eval_set"] == "test") & (a["ablation"] != "none")]
        md += ["\n## Ablations (test split, delta vs. full model)\n",
               md_table(a[["ablation", "removed", "experiment", "model", "pr_auc", "delta_pr_auc", "delta_recall"]])]
    p = res / "multiclass.json"
    if p.exists():
        mc = load_json(p)
        md += ["\n## Three-way classification (stretch goal)\n", f"Macro-F1 {mc['macro_f1']:.3f} on {mc['n_test']} test calls\n"]
    text = "\n".join(md)
    (res / "REPORT.md").write_text(text, encoding="utf-8")
    log.info("Report -> %s", res / "REPORT.md")
    return text
