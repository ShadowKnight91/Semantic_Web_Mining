"""Results dashboard:  python -m scamkg [--config config/demo.yaml] dashboard
(or: streamlit run dashboard/app.py -- --config config/demo.yaml)"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scamkg.config import get_paths, load_config  # noqa: E402
from scamkg.utils import load_json, read_jsonl  # noqa: E402

EXAMPLE = ("caller: This is Officer Brown from the Social Security Administration. Your SSN has been suspended due "
           "to suspicious activity. To avoid arrest today, you must verify your number and pay a fine using Google "
           "Play gift cards.\nreceiver: Oh no, what do I need to do?")
CONFIGS = {"real data (config/config.yaml)": None, "real data, rule KG + Ollama (config/local.yaml)": "config/local.yaml",
           "caller types, with prank calls (config/multiclass.yaml)": "config/multiclass.yaml",
           "synthetic demo (config/demo.yaml)": "config/demo.yaml"}

st.set_page_config(page_title="ScamKG - phone scam detection", layout="wide")


def _cli_config() -> str | None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    args, _ = p.parse_known_args()
    return args.config


cli_cfg = _cli_config()
default_idx = 0
if cli_cfg:
    known = {str((ROOT / v).resolve()): k for k, v in CONFIGS.items() if v}
    resolved = str(Path(cli_cfg).resolve() if Path(cli_cfg).is_absolute() else (ROOT / cli_cfg).resolve())
    label = known.get(resolved) or f"custom ({cli_cfg})"
    CONFIGS.setdefault(label, cli_cfg)
    default_idx = list(CONFIGS).index(label)
choice = st.sidebar.selectbox("Configuration", list(CONFIGS), index=default_idx)
cfg_path = CONFIGS[choice]
cfg = load_config(cfg_path)
paths = get_paths(cfg)
res = paths.results
st.sidebar.caption(f"results: `{res.relative_to(ROOT) if res.is_relative_to(ROOT) else res}`")


@st.cache_data
def _csv(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def csv(name: str) -> pd.DataFrame | None:
    p = res / name
    return _csv(str(p), p.stat().st_mtime) if p.exists() else None


def jsonf(name: str):
    p = res / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def image(name: str, caption: str | None = None) -> None:
    p = res / "figures" / name
    if not p.exists():
        p = res / "eda" / "figures" / name
    if not p.exists():
        p = res / "feedback" / "figures" / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")


@st.cache_data
def load_calls(path: str, mtime: float) -> pd.DataFrame:
    df = pd.DataFrame(read_jsonl(path))
    return df.set_index("call_id", drop=False)


@st.cache_resource
def load_context(cfg_file: str | None):
    from scamkg.context import Context
    return Context.load(load_config(cfg_file))


@st.cache_resource
def load_detector(cfg_file: str | None, experiment: str, model: str):
    from scamkg.predict import ScamDetector
    return ScamDetector(load_config(cfg_file), experiment, model, "rule")


st.title("From call transcripts to knowledge graphs")
st.caption("LLM-powered phone scam detection - CSE 573 Semantic Web Mining, Project P10")
if "demo" in (cfg_path or ""):
    st.warning("Synthetic demo data: template-generated calls used to exercise the pipeline, not evidence for the RQs.")

tabs = st.tabs(["Overview", "Models (RQ1)", "Cross-dataset (RQ3)", "Extraction (RQ2)", "Clustering (RQ4)",
                "Ablations", "Call explorer", "Try a transcript", "Caller types", "Professor's feedback"])

with tabs[0]:
    summ_path = paths.processed / "summary.json"
    c1, c2, c3, c4 = st.columns(4)
    if summ_path.exists():
        summ = json.loads(summ_path.read_text(encoding="utf-8"))
        c1.metric("Calls", summ["n_calls"])
        c2.metric("Duplicates removed", summ["duplicates_removed"])
    gs = jsonf("graph_summary.json")
    if gs:
        c3.metric("KG entities", gs["entities"])
        c4.metric("KG edges", gs["edges"])
        st.subheader("Knowledge graph")
        a, b = st.columns(2)
        with a:
            st.json({"entities_by_type": gs["entities_by_type"], "edges_by_relation": gs["edges_by_relation"],
                     "communities": gs.get("communities"), "per_label": gs.get("per_label")}, expanded=False)
            st.dataframe(pd.DataFrame(gs["top_entities"]), hide_index=True, width="stretch")
        with b:
            image("kg_sample.png", "Sample of the call knowledge graph")
    eda = res / "eda" / "eda_summary.json"
    if eda.exists():
        st.subheader("Exploratory analysis")
        e = json.loads(eda.read_text(encoding="utf-8"))
        st.dataframe(pd.DataFrame(e["datasets"]), hide_index=True, width="stretch")
        a, b = st.columns(2)
        with a:
            image("eda_length_hist.png", "Transcript length by data set and label")
        with b:
            image("eda_top_entities.png", "Claimed identities / requested assets")
        if e.get("vocabulary_overlap"):
            st.caption("Vocabulary overlap between data sets (low Jaccard / high OOV = harder transfer)")
            st.dataframe(pd.DataFrame(e["vocabulary_overlap"]), hide_index=True)
    if not summ_path.exists():
        st.info("No processed data yet. Run `python -m scamkg --config config/demo.yaml run-all` for the demo.")

metrics = csv("metrics.csv")
with tabs[1]:
    if metrics is None:
        st.info("No model results yet (`python -m scamkg train`).")
    else:
        for es, title in (("test", "Balanced test split"), ("test_imbalanced", "Imbalanced test split (~5% scams)")):
            sub = metrics[metrics["eval_set"] == es]
            if sub.empty:
                continue
            st.subheader(title)
            cols = ["experiment", "model", "precision", "recall", "f1", "pr_auc", "roc_auc"] + \
                   [c for c in ("recall_std", "pr_auc_std") if c in sub and es == "test_imbalanced"]
            st.dataframe(sub[cols].sort_values(["experiment", "model"]).round(4), hide_index=True, width="stretch")
            chart = sub.assign(name=sub["experiment"] + "/" + sub["model"]).set_index("name")[["pr_auc", "recall"]]
            st.bar_chart(chart)
        cv = csv("cv_results.csv")
        if cv is not None:
            st.subheader("5-fold cross-validation")
            st.dataframe(cv.round(4), hide_index=True, width="stretch")
        sig = csv("significance.csv")
        if sig is not None and not sig.empty:
            st.subheader("McNemar tests vs. E1")
            st.dataframe(sig.round(4), hide_index=True, width="stretch")

with tabs[2]:
    if metrics is not None:
        ext = metrics[~metrics["eval_set"].isin(["test", "test_imbalanced"])]
        if ext.empty:
            st.info("No external data sets evaluated.")
        else:
            for es, sub in ext.groupby("eval_set"):
                st.subheader(es)
                cols = ["experiment", "model", "n", "n_pos", "precision", "recall", "f1", "pr_auc", "recall_at_fpr10"]
                st.dataframe(sub[[c for c in cols if c in sub]].round(4),
                             hide_index=True, width="stretch")
            gen = csv("generalization.csv")
            if gen is not None:
                st.subheader("Recall drop vs. in-distribution test")
                st.bar_chart(gen.assign(name=gen["experiment"] + "/" + gen["model"] + " @ " + gen["eval_set"])
                             .set_index("name")[["recall_drop"]])

with tabs[3]:
    xq = csv("extraction_quality.csv")
    if xq is None:
        st.info("Run `python -m scamkg eval-extraction`.")
    else:
        st.dataframe(xq.round(4), hide_index=True, width="stretch")
        info = jsonf("extraction_quality.json") or {}
        if info.get("inter_annotator"):
            st.metric("Inter-annotator triple F1", f"{info['inter_annotator']['triple_f1']:.3f}")
        if info.get("per_relation"):
            run = st.selectbox("Per-relation scores for run", list(info["per_relation"]))
            st.dataframe(pd.DataFrame(info["per_relation"][run]).T.round(3), width="stretch")

with tabs[4]:
    cl = jsonf("clustering.json")
    if cl is None:
        st.info("Run `python -m scamkg cluster`.")
    else:
        st.dataframe(pd.DataFrame([{"method": "K-Means (best k)", **cl["kmeans_best"]},
                                   {"method": "K-Means (k=2)", **cl["kmeans_k2"]},
                                   {"method": "DBSCAN", **cl.get("dbscan", {})}]).round(4), hide_index=True)
        image("clusters_pca.png")
        st.subheader("Discovered scripts")
        st.dataframe(pd.DataFrame([{k: v for k, v in s.items() if k != "top_entities"} for s in cl["scripts"]]),
                     hide_index=True, width="stretch")

with tabs[5]:
    ab = csv("ablation.csv")
    if ab is None:
        st.info("Run `python -m scamkg ablate`.")
    else:
        es = st.selectbox("Evaluation set", sorted(ab["eval_set"].unique()), key="ab_es")
        sub = ab[(ab["eval_set"] == es) & (ab["ablation"] != "none")]
        st.dataframe(sub[["ablation", "removed", "experiment", "model", "pr_auc", "delta_pr_auc", "delta_recall"]].round(4),
                     hide_index=True, width="stretch")
        st.bar_chart(sub.assign(name=sub["experiment"] + "/" + sub["model"] + " -" + sub["removed"])
                     .set_index("name")[["delta_pr_auc"]])

with tabs[6]:
    if not paths.calls_file.exists():
        st.info("No data yet.")
    else:
        calls = load_calls(str(paths.calls_file), paths.calls_file.stat().st_mtime)
        pred_files = sorted(paths.predictions.glob("*.csv")) if paths.predictions.exists() else []
        pf = st.selectbox("Predictions", [p.stem for p in pred_files]) if pred_files else None
        pred = pd.read_csv(paths.predictions / f"{pf}.csv", dtype={"call_id": str}) if pf else pd.DataFrame()
        only_flagged = st.checkbox("Only flagged calls (score >= threshold)", value=True)
        pool = pred
        if not pred.empty and only_flagged:
            pool = pred[pred["score"] >= float(cfg["models"].get("threshold", 0.5))]
        options = pool.sort_values("score", ascending=False)["call_id"].tolist() if not pool.empty else calls.index.tolist()[:500]
        cid = st.selectbox("Call", options)
        if cid:
            row = calls.loc[cid]
            a, b = st.columns([1, 1])
            with a:
                st.markdown(f"**{cid}** - data set `{row['dataset']}`, split `{row['split']}`, gold label "
                            f"`{'scam' if row['label'] == 1 else 'legitimate' if row['label'] == 0 else 'unknown'}`")
                if not pred.empty and cid in set(pred["call_id"]):
                    st.metric("Model score", f"{float(pred.set_index('call_id').at[cid, 'score']):.3f}")
                st.text_area("Transcript", row["text"], height=320)
            with b:
                try:
                    ctx = load_context(cfg_path)
                    from scamkg.evaluation.explain import explain_call
                    score = float(pred.set_index("call_id").at[cid, "score"]) if not pred.empty and cid in set(pred["call_id"]) else None
                    ex = explain_call(ctx, cid, score)
                    st.markdown(f"**Script path:** {ex['script_path'] or '-'}")
                    st.markdown(f"**Why:** {ex['summary'] or '-'}")
                    st.graphviz_chart(ex["dot"])
                    st.dataframe(pd.DataFrame(ex["entities"]), hide_index=True, width="stretch")
                    with st.expander("Cypher (Neo4j Browser)"):
                        st.code(ex["cypher"].replace("$call_id", f"'{cid}'"), language="cypher")
                except Exception as e:  # graph not built yet
                    st.info(f"Knowledge graph not available: {e}")

with tabs[7]:
    st.write("Paste a transcript. It is cleaned, triples are extracted with the rule-based extractor, the call is "
             "added to the knowledge graph, graph features are recomputed and a trained model scores it.")
    exp = st.selectbox("Model", ["E2/xgb", "E2/lr", "E2G/xgb", "E1/lr", "E1/xgb"])
    text = st.text_area("Transcript", EXAMPLE, height=180)
    if st.button("Score call", type="primary"):
        e_name, m_name = exp.split("/")
        try:
            with st.spinner("Extracting triples and recomputing graph features..."):
                out = load_detector(cfg_path, e_name, m_name).score(text)
            st.metric("Scam probability", f"{out['score']:.3f}", "SCAM" if out["is_scam"] else "legitimate",
                      delta_color="inverse" if out["is_scam"] else "normal")
            if "caller_type" in out:
                probs = out["caller_type_probs"]
                st.metric("Caller type", out["caller_type"].replace("_", " "),
                          f"{probs[out['caller_type']]:.0%} ({out['caller_type_model']})", delta_color="off")
                st.bar_chart(pd.Series(probs, name="probability"))
            ex = out["explanation"]
            st.markdown(f"**Script path:** {ex['script_path'] or '-'}")
            st.markdown(f"**Why:** {ex['summary'] or '-'}")
            st.graphviz_chart(ex["dot"])
            st.dataframe(pd.DataFrame(out["triples"])[["relation", "object_type", "object", "detail", "evidence"]]
                         if out["triples"] else pd.DataFrame(), hide_index=True, width="stretch")
        except FileNotFoundError as e:
            st.error(str(e))

with tabs[8]:
    mc = csv("multiclass.csv")
    p = res / "multiclass.json"
    if mc is None or not p.exists():
        st.info("No caller-type results here. Run `python -m scamkg --config config/multiclass.yaml run-multiclass` "
                "and pick that configuration in the sidebar.")
    else:
        info = load_json(p)
        missing = info.get("missing_classes") or []
        st.write(f"Fraudster / salesperson / prankster / other legitimate. Classes in this run: "
                 f"{', '.join(info['classes'])}" + (f" (no training calls for {', '.join(missing)})." if missing else ".")
                 + f" Training calls: {info['n_train']}.")
        es = st.selectbox("Evaluation set", sorted(mc["eval_set"].unique()), key="caller_type_set")
        st.dataframe(mc[mc["eval_set"] == es].drop(columns="eval_set").round(3), hide_index=True, width="stretch")
        cm = info["confusion"].get(f"{info['main'].replace('/', '__')}__{es}")
        if cm:
            st.subheader(f"Confusion matrix ({info['main']}, rows = true type)")
            st.dataframe(pd.DataFrame(cm, index=info["classes"], columns=info["classes"]), width="stretch")

with tabs[9]:
    if not (res / "feedback").exists():
        st.info("No results yet. Run `python -m scamkg monitor-recall`, `discover-scripts` and `benchmark-extractors`.")
    else:
        def table(name: str, columns: list[str] | None = None) -> None:
            df = csv(f"feedback/{name}")
            if df is not None:
                show = [c for c in (columns or df.columns) if c in df.columns]
                st.dataframe(df[show].round(3), hide_index=True, width="stretch")

        st.write("Three follow-ups: lighter extraction models, recall from synthetic to noisy FTC calls, and scam scripts "
                 "bag of words misses. Details and limits: `results/feedback/REPORT.md`.")
        light, recall, scripts = st.tabs(["Lighter extraction", "Recall monitor", "Hidden scripts"])
        with light:
            image("extractor_benchmark.png", "F1 on the 50 gold calls against seconds per call")
            table("extractor_benchmark.csv", ["model", "prompt", "constrained", "provider", "parameters", "loaded_gb",
                                              "gold_f1", "agree_ncsu_f1", "agree_kaggle_f1", "failed", "latency_mean_s",
                                              "latency_p95_s", "hours_per_100k_calls"])
            st.subheader("Detection on a KG from each extractor")
            table("downstream_summary.csv")
            st.subheader("Rules first, LLM only when they find too little")
            for p in sorted((res / "feedback").glob("cascade__*.csv")):
                table(p.name)
        with recall:
            image("recall_ladder.png", "Recall at each step away from the training data")
            image("recall_prevalence.png", "Recall when alerts must stay trustworthy as scams get rare")
            a, b = st.columns(2)
            with a:
                image("recall_recalibration.png", "What a small labelled pilot recovers")
            with b:
                image("recall_noise.png", "Simulated speech-to-text noise")
            image("recall_stream.png", "A stream drifting from synthetic to real calls")
            st.subheader("Label-free warning signals")
            table("drift_signals.csv")
        with scripts:
            image("script_map.png", "Scam calls in three spaces (first two principal components)")
            image("script_recall.png", "Recall per script, at the same false-alarm rate")
            table("scripts.csv", ["script", "calls", "training_share", "hidden", "signature", "top_words", "bow_recall",
                                  "graphsage_recall", "missed_by_bow_caught_by_graphsage"])
            st.subheader("Clustering spaces compared")
            table("script_methods.csv")
            st.subheader("Recall over seeds")
            table("script_seeds.csv")
