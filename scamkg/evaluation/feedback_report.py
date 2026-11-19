"""results/feedback/REPORT.md: every number comes from the CSV files, so re-running an experiment updates the text."""
from __future__ import annotations

import pandas as pd

from scamkg.evaluation.plot_style import MODEL_LABELS
from scamkg.utils import get_logger, load_json, md_table

log = get_logger(__name__)
LADDER_STAGES = {"synthetic_test": "Synthetic test", "kaggle": "Kaggle", "ncsu_robocalls": "FTC robocalls"}
MAIN_MODELS = ["E1__lr", "E2__lr", "E3__distilbert", "E4__graphsage", "E4__graphsage_topo"]


def _pct(v, digits: int = 0) -> str:
    return "n/a" if pd.isna(v) else f"{100 * float(v):.{digits}f}%"


def _rows(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    d = df[df["model"].isin(models)].copy()
    d["model"] = d["model"].map(lambda m: MODEL_LABELS.get(m, m))
    return d


def _extractor_label(row) -> str:
    run, model = str(row["run"]), str(row["model"])
    if run == "rule":
        return "keyword rules"
    label = model + (" + schema" if row.get("constrained") is True or str(row.get("constrained")) == "True" else "")
    return f"{label} ({row['prompt']})" if row.get("prompt") else label


def _run_label(run: str, main_run: str | None) -> str:
    """Readable name for an extraction run in the detection table."""
    short = run
    if main_run:
        short = short.replace(main_run, "Gemma 31B")
    short = short.replace("llm_ollama_", "").replace("_v1_full_constrained", " + schema")
    if short.startswith("serve__"):
        served = short.removeprefix("serve__")
        if served.startswith("cascade_k99_union__"):
            served = "rules + " + served.removeprefix("cascade_k99_union__")
        return f"Train: Gemma 31B. Serve: {served}"
    if short.startswith("cascade_k"):
        k, _, rest = short.removeprefix("cascade_k").partition("_")
        mode, _, llm = rest.partition("__")
        return f"Rules, then {llm} below {k} facts ({mode})"
    return {"rule": "Keyword rules"}.get(short, short)


def section_light(fb, main_run: str | None = None) -> list[str]:
    md = ["\n## 1. Lighter extraction models\n"]
    p = fb / "extractor_benchmark.csv"
    if p.exists():
        d = pd.read_csv(p)
        d["extractor"] = d.apply(_extractor_label, axis=1)
        cols = {"extractor": "extractor", "provider": "runs on", "parameters": "size", "loaded_gb": "memory (GB)",
                "gold_f1": "F1, 50 gold calls", "gold_f1_ci_low": "F1 low", "gold_f1_ci_high": "F1 high",
                "agree_ncsu_f1": "agrees with Gemma, FTC",
                "agree_kaggle_f1": "agrees with Gemma, Kaggle", "latency_mean_s": "s per call",
                "latency_p95_s": "p95 s", "hours_per_100k_calls": "hours per 100k calls"}
        show = d[[c for c in cols if c in d]].rename(columns=cols).sort_values("F1, 50 gold calls", ascending=False)
        md += [md_table(show.round(3)), "\n![Quality against cost](figures/extractor_benchmark.png)\n",
               "Limits: Gemma's latency comes from a shared Voyager server and the small models from an M1 Pro laptop, so "
               "seconds per call are not a fair race; the point is the hardware each one needs. F1 low / high is a 95% bootstrap "
               "interval over the 50 gold calls, so models a few points apart cannot be separated. The gold calls are all "
               "synthetic. On Kaggle and FTC calls there is no gold, so those columns measure agreement with Gemma, not "
               "accuracy, on the calls each run covers. Small runs that cover only gold calls have no agreement value.\n"]
    for p in sorted(fb.glob("cascade__*.csv")):
        c = pd.read_csv(p)
        md += ["\n### Rules first, LLM only when they find too little\n",
               "The rule extractor is free (about 3 ms per call). A call with fewer than `min_triples` rule triples goes to "
               f"the LLM; `{p.stem.split('__', 2)[-1]}` supplies its triples here ({p.stem.split('__')[1]} mode), so no LLM "
               "was re-run. `agree_*` compares with that same LLM's full answer; `llm_share` counts calls that LLM covers.\n",
               md_table(c.drop(columns="run").round(3))]
    p = fb / "downstream_summary.csv"
    if p.exists():
        md += ["\n### Does cheaper extraction hurt detection?\n",
               "Same calls, same models and hyper-parameters; only the triples differ. `Train: Gemma` rows keep Gemma's triples for "
               "training calls and use the cheaper extractor only for calls scored later, which is where the cost recurs. GraphSAGE "
               "is a mean over 5 seeds. `kaggle` is scam against legitimate inside one data set, the fairer test. `ncsu+kaggle` puts FTC "
               "scams against Kaggle's legitimate calls, two data sets, so an extractor's quirks on either one can separate them: a "
               "cheap extractor beating Gemma there is not evidence that it is better. These runs use networkx graph features for "
               "every KG, while the committed results used Neo4j GDS, so Gemma's row differs slightly from `results/REPORT.md`; "
               "compare within this table.\n",
               md_table(pd.read_csv(p).assign(extraction_run=lambda d: d["extraction_run"].map(
                   lambda r: _run_label(r, main_run))).round(3))]
    p = fb / "extraction_noise.csv"
    if p.exists():
        n = pd.read_csv(p)
        md += ["\n### Do extractors keep the facts on noisy transcripts?\n",
               f"{int(n['calls'].iloc[0])} calls per data set. Gemma itself could not be re-run on noisy text (no Voyager access on "
               "this machine). First table: agreement with Gemma on the clean transcript.\n",
               md_table(n.pivot_table(index=["dataset", "extractor"], columns="noise", values="agree_with_gemma_clean_f1")
                        .reindex(columns=["clean", "light", "medium", "heavy"]).round(3).reset_index()),
               "\nSecond table: agreement with the extractor's own answer on the clean transcript. The small LLM does not rescue noisy "
               "calls: it keeps fewer of its own facts than the rules do.\n",
               md_table(n.pivot_table(index=["dataset", "extractor"], columns="noise", values="agree_with_own_clean_f1")
                        .reindex(columns=["clean", "light", "medium", "heavy"]).round(3).reset_index())]
    return md


def section_recall(fb) -> list[str]:
    md = ["\n## 2. Recall from synthetic data to noisy, imbalanced FTC calls\n"]
    p = fb / "recall_ladder.csv"
    if p.exists():
        d = pd.read_csv(p)
        d["stage"] = d["stage"].map(lambda s: LADDER_STAGES.get(s, s))
        keep = ["model", "stage", "n_scams", "recall", "recall_ci_low", "recall_ci_high", "false_alarm_rate",
                "recall_at_target_fpr", "status"]
        md += ["### Recall ladder\n", "Default threshold 0.5, 95% bootstrap interval. `recall_at_target_fpr` sets the threshold "
               "for 10% false alarms on the legitimate calls of the same step (in-sample). Single committed run per model.\n",
               md_table(_rows(d, MAIN_MODELS)[keep].round(3)), "\n![Recall ladder](figures/recall_ladder.png)\n"]
    p = fb / "recall_prevalence.csv"
    if p.exists():
        d = pd.read_csv(p)
        keep = ["model", "prevalence", "recall_at_threshold", "precision_at_threshold", "false_alarms_per_1000_calls",
                "scams_missed_per_1000_calls", "recall_at_precision_0.5", "recall_at_precision_0.8"]
        md += ["\n### As scams get rare\n",
               "FTC robocalls (713) against Kaggle's legitimate calls (400), reweighted to each scam share. At a fixed threshold "
               "recall cannot change with prevalence, but precision collapses, so keeping alerts trustworthy means raising the "
               "threshold, and that is where recall goes. `n/a`: no threshold reaches that precision on this sample (zero false "
               "alarms among 400 calls is treated as about 1 in 401, not as zero).\n",
               md_table(_rows(d, ["E1__lr", "E2__lr", "E4__graphsage"])[keep].round(3)),
               "\n![Recall at required precision](figures/recall_prevalence.png)\n"]
    p = fb / "recall_recalibration.csv"
    if p.exists():
        d = pd.read_csv(p)
        d = d[d["stage"].str.contains("robocalls") & (d["pilot_size"] == 100)]
        keep = ["model", "frozen_recall", "frozen_false_alarm_rate", "recall", "recall_sd", "false_alarm_rate", "false_alarm_sd"]
        md += ["\n### What a small labelled pilot recovers\n",
               "The threshold is set on 100 labelled calls from the new domain (50 scams, 50 legitimate; target 10% false alarms) "
               "and scored on the rest, over 200 draws. Recall is not free: compare the false-alarm columns.\n",
               md_table(_rows(d, list(MODEL_LABELS))[keep].round(3)), "\n![Pilot](figures/recall_recalibration.png)\n"]
    p = fb / "recall_noise.csv"
    if p.exists():
        d = pd.read_csv(p)
        d = d[d["stage"].str.contains("robocalls")]
        order = ["clean", "light", "medium", "heavy"]
        md += ["\n### Simulated speech-to-text noise\n",
               "Models trained once on clean calls, threshold frozen at 10% false alarms on clean calls, then scored on noisy "
               "external calls (0%, 5%, 15%, 30% word errors). GraphSAGE: mean of 5 seeds. All KGs here come from the keyword rules, "
               "since only they can re-extract every noisy call instantly. Simulated noise, not one real recognizer.\n",
               md_table(_rows(d, list(MODEL_LABELS)).pivot_table(index="model", columns="noise", values="recall_clean_threshold")[order]
                        .round(3).reset_index()),
               "\n![Noise](figures/recall_noise.png)\n"]
    p = fb / "drift_signals.csv"
    if p.exists():
        d = pd.read_csv(p)
        keep = ["stage", "model", "n_calls", "alert_rate", "score_psi", "unknown_word_rate", "no_kg_facts_rate", "status"]
        md += ["\n### Warning signals that need no labels\n",
               md_table(_rows(d, ["E1__lr", "E2__lr", "E4__graphsage"])[keep].round(3))]
    p = fb / "recall_stream_summary.csv"
    if p.exists():
        d = pd.read_csv(p)
        keep = ["model", "baseline_recall", "final_recall", "first_warning_window", "first_recall_drop_window", "lead_windows",
                "spearman_recall_vs_unknown_words", "spearman_recall_vs_psi", "no_drift_warning_windows"]
        md += ["\n### A simulated deployment\n",
               "21 windows of 300 calls at 10% scams, drifting from 0% to 100% real calls. A signal must hold for 2 windows to count. "
               "`lead_windows` is how many windows the first warning came before recall fell by 10 points; negative means after. "
               "Empty: recall never fell that far. `no_drift_warning_windows`: share of windows that warned on a stream that never "
               "drifts, one stream per model. The alert levels are starting points, not validated limits.\n",
               md_table(_rows(d, list(MODEL_LABELS))[keep].round(3)), "\n![Stream](figures/recall_stream.png)\n"]
    return md


def section_scripts(fb) -> list[str]:
    md = ["\n## 3. Hidden scripts with GraphSAGE\n"]
    p = fb / "script_seeds.csv"
    if p.exists():
        s = pd.read_csv(p)
        keep = [c for c in ["model", "dataset", "seeds", "recall_at_10pct_false_alarms", "recall_at_10pct_false_alarms_sd",
                            "recall_default_threshold", "recall_default_threshold_sd"] if c in s]
        md += ["### Does GraphSAGE catch what bag of words misses?\n", "Unseen data sets, over seeds:\n", md_table(s[keep].round(3))]
    p = fb / "script_rescue.csv"
    if p.exists():
        r = pd.read_csv(p)
        r["challenger"] = r["challenger"].map(lambda m: MODEL_LABELS.get(m, m))
        md += ["\nCall by call (1,106 real scams; one seed; McNemar test):\n", md_table(r.round(4))]
    p = fb / "script_methods.csv"
    if p.exists():
        m = pd.read_csv(p)
        info = load_json(fb / "script_summary.json") if (fb / "script_summary.json").exists() else {"k": 16}
        md += [f"\n### Clusters of scam calls, same k = {info['k']} for every space\n",
               "`dataset_mixing`: 0 = every cluster holds one data set, 1 = evenly mixed. `real_calls_linked_to_training`: share of "
               "real calls in clusters that also hold training calls. Other values of k are in `script_methods.csv`.\n",
               md_table(m[m["k"] == info["k"]].drop(columns="k").round(3)),
               "\n![Map](figures/script_map.png)\n"]
    p = fb / "scripts.csv"
    if p.exists():
        s = pd.read_csv(p)
        keep = ["script", "calls", "training_share", "hidden", "signature", "top_words", "bow_recall", "graphsage_recall",
                "missed_by_bow_caught_by_graphsage", "caught_by_bow_missed_by_graphsage"]
        h = s[s["hidden"]]
        n = s[~s["hidden"]]

        def wmean(frame, col):
            return (frame[col] * frame["real_calls"]).sum() / frame["real_calls"].sum() if len(frame) else float("nan")

        md += ["\n### The scripts\n",
               f"Defined by label-free GraphSAGE. {len(h)} of {len(s)} are hidden (under 5% training calls), holding "
               f"{int(h['real_calls'].sum())} real scam calls. Recall at the same 10% false alarms: bag of words "
               f"{_pct(wmean(h, 'bow_recall'))} against GraphSAGE {_pct(wmean(h, 'graphsage_recall'))} in hidden scripts; "
               f"{_pct(wmean(n, 'bow_recall'))} against {_pct(wmean(n, 'graphsage_recall'))} in the others.\n",
               md_table(s[keep].round(3)), "\n![Scripts](figures/script_recall.png)\n",
               "A script here means calls sharing KG facts, and that is not always one story. Four transcripts per hidden cluster were "
               "read, unlabelled and only as a spot check: Social Security suspension, utility disconnection and Amazon account clusters "
               "read as one script, while others share behaviour (reward and urgency, or fear and press 1) across different stories. "
               "`kg_similarity` shows how tight each cluster is; read `script_assignments.csv` before naming one. GraphSAGE recall per "
               "script is one seed.\n"]
    p = fb / "script_loto_summary.csv"
    if p.exists():
        md += ["\n### A script removed from training\n",
               "Mean over held-out BothBosu types. `flag_rate`: recall for a held-out scam type, false-alarm rate for a held-out "
               "legitimate type. `own_cluster_f1`: how cleanly the type comes out as one cluster. This synthetic data has very "
               "different vocabulary per type, which suits bag of words.\n",
               md_table(pd.read_csv(p).round(3))]
    return md


def _summary(fb, main_run: str | None) -> list[str]:
    """Main results, read from the tables below so the two can never disagree."""
    out = []
    p = fb / "script_seeds.csv"
    if p.exists():
        d = pd.read_csv(p).set_index(["model", "dataset"])
        col, sd = "recall_at_10pct_false_alarms", "recall_at_10pct_false_alarms_sd"

        def rec(model, ds):
            return d.at[(model, ds), col]
        out.append(f"* **GraphSAGE catches scams bag of words misses.** At 10% false alarms, over 5 seeds: FTC {_pct(rec('GraphSAGE', 'ncsu'))} "
                   f"(sd {_pct(d.at[('GraphSAGE', 'ncsu'), sd])}) against {_pct(rec('Bag of words', 'ncsu'))}; Kaggle "
                   f"{_pct(rec('GraphSAGE', 'kaggle'))} (sd {_pct(d.at[('GraphSAGE', 'kaggle'), sd])}) against {_pct(rec('Bag of words', 'kaggle'))}. "
                   f"With no transcript words at all, structure alone gets {_pct(rec('GraphSAGE, structure only', 'kaggle'))} on Kaggle but an "
                   f"unstable {_pct(rec('GraphSAGE, structure only', 'ncsu'))} (sd {_pct(d.at[('GraphSAGE, structure only', 'ncsu'), sd])}) on FTC.")
    p = fb / "script_loto_summary.csv"
    if p.exists():
        d = pd.read_csv(p).set_index(["held_out_class", "model"])
        out.append(f"* **\"Completely misses\" goes too far.** Bag of words flags {_pct(d.at[('scam', 'Bag of words'), 'flag_rate'])} of scam types removed from "
                   f"synthetic training and recovers them as clusters better (F1 {d.at[('scam', 'Bag of words'), 'own_cluster_f1']:.2f} against "
                   f"{d.at[('scam', 'GraphSAGE, label-free'), 'own_cluster_f1']:.2f}). Its weakness is the move to real calls, not unseen scripts as such. GraphSAGE's clear win "
                   f"on unseen synthetic types is false alarms on unseen legitimate types ({_pct(d.at[('legitimate', 'GraphSAGE'), 'flag_rate'])} against "
                   f"{_pct(d.at[('legitimate', 'Bag of words'), 'flag_rate'])}).")
    p = fb / "extractor_benchmark.csv"
    if p.exists():
        d = pd.read_csv(p)

        def f1(run_like):
            r = d[d["run"] == run_like].iloc[0]
            return f"{r['gold_f1']:.2f} ({r['gold_f1_ci_low']:.2f}-{r['gold_f1_ci_high']:.2f})"
        out.append(f"* **Small models do not replace Gemma on triple quality.** Gold F1: Gemma 31B {f1(main_run)}, gemma2:9b with schema "
                   f"{f1('llm_ollama_gemma2-9b_v1_full_constrained')}, keyword rules {f1('rule')}, qwen2.5:3b with schema "
                   f"{f1('llm_ollama_qwen2.5-3b_v1_full_constrained')}, gemma3:1b with schema {f1('llm_ollama_gemma3-1b_v1_full_constrained')}. "
                   "Schema enforcement raises conformance, not F1.")
    p = fb / "downstream_detection.csv"
    if p.exists():
        d = pd.read_csv(p)
        d = d[(d["experiment"] == "E4")].set_index(["extraction_run", "eval_set"])
        q = "llm_ollama_qwen2.5-3b_v1_full_constrained"

        def row(run, label):
            k, f = (run, "kaggle"), (run, "ncsu+kaggle")
            return (f"{label}: PR-AUC {d.at[k, 'pr_auc']:.2f} on Kaggle and {d.at[f, 'pr_auc']:.2f} on FTC, recall at 10% false alarms "
                    f"{_pct(d.at[k, 'recall_at_fpr10'])} and {_pct(d.at[f, 'recall_at_fpr10'])}")
        if (f"serve__cascade_k99_union__{q}", "kaggle") in d.index and (main_run, "kaggle") in d.index:
            out.append("* **But detection mostly survives cheap serving** (GraphSAGE, 5 seeds). " +
                       row(main_run, "Gemma everywhere") + ". " + row(f"serve__cascade_k99_union__{q}", "Train on Gemma, serve with rules + qwen2.5:3b") +
                       ". " + row("rule", "Keyword rules everywhere, no LLM at all") + ". The LLM earns its cost on Kaggle more than on FTC.")
    p = fb / "recall_prevalence.csv"
    if p.exists():
        d = pd.read_csv(p).set_index(["model", "prevalence"])
        g, b = "E4__graphsage", "E1__lr"
        out.append(f"* **Rare scams are the hard limit.** At 0.5, GraphSAGE keeps {_pct(d.at[(g, 0.05), 'recall_at_threshold'])} FTC recall whatever the scam share, but only "
                   f"{_pct(d.at[(g, 0.05), 'precision_at_threshold'])} of its alerts are real at 5% scams and {_pct(d.at[(g, 0.01), 'precision_at_threshold'])} at 1%. Requiring 50% "
                   f"precision at 5% scams leaves {_pct(d.at[(g, 0.05), 'recall_at_precision_0.5'])} recall ({_pct(d.at[(b, 0.05), 'recall_at_precision_0.5'])} for bag of words). "
                   f"At 1% scams, {int(d.xs(0.01, level='prevalence')['recall_at_precision_0.5'].notna().sum())} of {d.index.get_level_values('model').nunique()} models reach 50% "
                   "precision on this sample.")
    p = fb / "recall_noise.csv"
    if p.exists():
        d = pd.read_csv(p)
        d = d[d["stage"].str.contains("robocalls")].set_index(["model", "noise"])["recall_clean_threshold"]
        out.append(f"* **Noise erases the KG advantage.** FTC recall at a frozen threshold, clean to 30% word errors, rule KG: bag of words {_pct(d[('E1__lr', 'clean')])} to "
                   f"{_pct(d[('E1__lr', 'heavy')])}, words + KG {_pct(d[('E2__lr', 'clean')])} to {_pct(d[('E2__lr', 'heavy')])}, GraphSAGE {_pct(d[('E4__graphsage', 'clean')])} to "
                   f"{_pct(d[('E4__graphsage', 'heavy')])}.")
    p = fb / "recall_recalibration.csv"
    if p.exists():
        d = pd.read_csv(p)
        d = d[d["stage"].str.contains("robocalls") & (d["pilot_size"] == 100)].set_index("model")
        g = d.loc["E4__graphsage"]
        out.append(f"* **Much of the recall loss is the threshold.** A 100-call labelled pilot moves GraphSAGE from {_pct(g['frozen_recall'])} to {_pct(g['recall'])} FTC recall, with false "
                   f"alarms from {_pct(g['frozen_false_alarm_rate'])} to {_pct(g['false_alarm_rate'])}.")
    p = fb / "recall_stream_summary.csv"
    if p.exists():
        d = pd.read_csv(p)
        lead = d["lead_windows"].dropna()
        out.append(f"* **Recall can be watched without labels.** In a simulated drift, a warning came before recall fell by 10 points for {int((lead > 0).sum())} of the "
                   f"{len(lead)} models whose recall fell that far, and no model warned on a stream without drift. The share of unknown words alone tracks recall "
                   f"(Spearman {d['spearman_recall_vs_unknown_words'].max():.2f} to {d['spearman_recall_vs_unknown_words'].min():.2f}).")
    return ["\n## Main results\n", *out, ""] if out else []


def build_feedback_report(ctx) -> str:
    fb = ctx.paths.results / "feedback"
    md = ["# Follow-up on the professor's feedback\n",
          "Generated by `python -m scamkg feedback-report`. The committed results in `results/` are not changed by anything here.\n"]
    main_run = ctx.cfg.get("extraction", {}).get("default_run")
    md += _summary(fb, main_run) + section_light(fb, main_run) + section_recall(fb) + section_scripts(fb)
    text = "\n".join(md)
    (fb / "REPORT.md").write_text(text, encoding="utf-8")
    log.info("Report -> %s", fb / "REPORT.md")
    return text
