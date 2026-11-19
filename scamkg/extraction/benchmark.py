"""Lighter extraction models (professor's feedback: the 31B / 30B models are a computational bottleneck).

Three questions, each with its own command:
  benchmark-extractors  What do small local models cost and how good are their triples? F1 on the 50 gold calls,
                        agreement with the Gemma run on Kaggle and FTC calls (no gold there), latency, memory.
  cascade               How many LLM calls can a free first pass save? The rule extractor runs on every call and
                        only calls where it finds too little go to an LLM.
  compare-runs          Does cheaper extraction hurt detection? Builds a KG from each run and scores the same
                        models on the same unseen data sets. Every run gets its own results and artifact folder.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

from scamkg.config import deep_merge  # noqa: E402
from scamkg.evaluation.plot_style import INK, MUTED, SERIES, SURFACE, style_axes  # noqa: E402
from scamkg.extraction.evaluate import compare_to_gold, load_annotations  # noqa: E402
from scamkg.extraction.extractor import (latest_records, load_extraction_run, make_extractor, resolve_run,  # noqa: E402
                                         run_extraction, summarize_records)
from scamkg.ontology.standardize import EntityStandardizer  # noqa: E402
from scamkg.utils import (get_logger, load_json, md_table, read_jsonl, save_json, slugify, timestamp,  # noqa: E402
                          write_jsonl)

log = get_logger(__name__)
HEAVY_PARAMETERS = {"gemma4-31b-it": "31B", "qwen3-30b-a3b": "30B"}   # not on this machine: size from the model name


def feedback_dir(ctx) -> Path:
    path = ctx.paths.results / "feedback"
    (path / "figures").mkdir(parents=True, exist_ok=True)
    return path


def load_gold(ctx) -> dict[str, list[dict]]:
    """The X1 reference: adjudicated gold.jsonl if it exists, otherwise annotator A."""
    ann = ctx.paths.annotations
    std = EntityStandardizer(ctx.ontology)
    if (ann / "gold.jsonl").exists():
        return load_annotations(ann / "gold.jsonl", ctx.ontology, std, all_annotated=True)
    return load_annotations(ann / "annotator_A.jsonl", ctx.ontology, std)


def benchmark_ids(ctx, n_each: int = 50) -> list[str]:
    """The gold calls, then `n_each` FTC robocalls and `n_each` Kaggle calls (half scams): seeded and label-balanced."""
    calls = ctx.calls
    rng = np.random.default_rng(ctx.seed)
    ids = [c for c in load_gold(ctx) if c in calls.index]

    def pick(pool, n):
        return list(rng.choice(pool, min(n, len(pool)), replace=False)) if n and len(pool) else []

    for ds in ctx.external_datasets:
        sub = calls[calls["dataset"] == ds]
        if sub["label"].nunique() > 1:
            ids += pick(sub.index[sub["label"] == 1].to_numpy(), n_each // 2)
            ids += pick(sub.index[sub["label"] == 0].to_numpy(), n_each - n_each // 2)
        else:
            ids += pick(sub.index.to_numpy(), n_each)
    return [str(i) for i in ids]


def ollama_model_info(cfg: dict, model: str) -> dict:
    """Disk size and memory when loaded with the extraction context size, from the local Ollama server."""
    pcfg = cfg["llm"]["providers"]["ollama"]
    base = (pcfg.get("base_url") or "http://localhost:11434").rstrip("/")
    info: dict = {"model": model}
    tags = requests.get(f"{base}/api/tags", timeout=10).json().get("models", [])
    tag = next((m for m in tags if m["name"] == model), None)
    if tag:
        d = tag.get("details", {})
        info.update(disk_gb=round(tag["size"] / 1e9, 2), parameters=d.get("parameter_size"),
                    quantization=d.get("quantization_level"))
    requests.post(f"{base}/api/generate", json={"model": model, "keep_alive": "2m",
                                                "options": {"num_ctx": int(pcfg.get("num_ctx", 8192))}}, timeout=300)
    loaded = next((m for m in requests.get(f"{base}/api/ps", timeout=10).json().get("models", [])
                   if m["name"] == model), None)
    if loaded:
        info["loaded_gb"] = round(loaded["size"] / 1e9, 2)
    return info


def run_benchmark(ctx, models: list[str], prompts: list[str], constrained_modes=(True, False), n_each: int = 50,
                  gold_only: bool = False, workers: int = 1) -> list[Path]:
    """Extract the benchmark calls with every model / prompt / decoding combination (resumable, cached)."""
    cfg = deep_merge(ctx.cfg, {"llm": {"provider": "ollama", "fallback_model": None}})
    ids = benchmark_ids(ctx, 0 if gold_only else n_each)
    calls = ctx.calls.loc[ids]
    fb = feedback_dir(ctx)
    info_path = fb / "ollama_models.json"
    info = load_json(info_path) if info_path.exists() else {}
    paths = []
    for model in models:
        info[model] = ollama_model_info(cfg, model)
        save_json(info_path, info)
        for prompt in prompts:
            for constrained in constrained_modes:
                ex = make_extractor("llm", cfg, ctx.ontology, cache_dir=ctx.paths.cache, model=model,
                                    provider="ollama", prompt_version=prompt, constrained=constrained)
                paths.append(run_extraction(calls, ex, ctx.paths.extractions, max_workers=workers))
    return paths


def gold_f1_interval(pred: dict, gold: dict, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    """95% bootstrap interval of triple F1, resampling calls: 50 calls cannot separate models a few points apart."""
    ids = [c for c in gold if c in pred]
    counts = []
    for c in ids:
        g = {(t["relation"], str(t["object"]).lower()) for t in gold[c]}
        p = {(t["relation"], str(t["object"]).lower()) for t in pred[c]}
        counts.append((len(g & p), len(p - g), len(g - p)))
    arr = np.array(counts, dtype=float)
    rng = np.random.default_rng(seed)
    sums = arr[rng.integers(0, len(arr), (n_boot, len(arr)))].sum(axis=1)
    f1 = 2 * sums[:, 0] / np.maximum(2 * sums[:, 0] + sums[:, 1] + sums[:, 2], 1e-9)
    return float(np.percentile(f1, 2.5)), float(np.percentile(f1, 97.5))


def _latency_stats(records: list[dict]) -> dict:
    fresh = [r["latency_s"] for r in records if not r.get("cached") and r.get("latency_s")]
    if not fresh:
        return {}
    return {"latency_mean_s": float(np.mean(fresh)), "latency_p50_s": float(np.median(fresh)),
            "latency_p95_s": float(np.percentile(fresh, 95))}


def run_row(ctx, run_path: Path, gold: dict, reference: dict | None, info: dict) -> dict:
    """One benchmark row. `reference` (the Gemma run) stands in for gold on data sets without annotations:
    this measures agreement with the heavy model, not accuracy."""
    records = latest_records(read_jsonl(run_path))
    meta = load_json(run_path.with_suffix(".meta.json")) if run_path.with_suffix(".meta.json").exists() else {}
    pred = {r["call_id"]: r.get("triples", []) for r in records}
    summ = summarize_records(records)
    model = meta.get("model", meta.get("extractor", run_path.stem))
    m = info.get(model, {})
    row = {"run": run_path.stem, "model": model, "prompt": meta.get("prompt_version", ""),
           "constrained": bool(meta.get("constrained", False)), "provider": meta.get("provider", "local"),
           "parameters": m.get("parameters", HEAVY_PARAMETERS.get(model, "")), "disk_gb": m.get("disk_gb"),
           "loaded_gb": m.get("loaded_gb"), "n_calls": summ.get("n_calls", 0),
           "json_valid": summ.get("json_validity_rate"), "schema_conformant": summ.get("schema_conformance_rate"),
           "failed": summ.get("failed_rate"), "triples_per_call": summ.get("mean_triples_per_call"),
           "no_triples": summ.get("calls_without_triples"), **_latency_stats(records)}
    covered = {k: v for k, v in gold.items() if k in pred}
    if covered:
        cmp = compare_to_gold(pred, covered)
        low, high = gold_f1_interval(pred, covered)
        row.update(gold_calls=cmp["n_calls"], gold_precision=cmp["triple_precision"], gold_recall=cmp["triple_recall"],
                   gold_f1=cmp["triple_f1"], gold_f1_ci_low=low, gold_f1_ci_high=high)
    if reference:
        for ds in ctx.external_datasets:
            ids = [c for c in ctx.calls.index[ctx.calls["dataset"] == ds] if c in pred and c in reference]
            if ids:
                cmp = compare_to_gold({c: pred[c] for c in ids}, {c: reference[c] for c in ids})
                row[f"agree_{ds}_f1"] = cmp["triple_f1"]
                row[f"agree_{ds}_calls"] = len(ids)
    if row.get("latency_mean_s"):
        row["hours_per_100k_calls"] = row["latency_mean_s"] * 100_000 / 3600
    return row


def benchmark_table(ctx, runs: list[str] | None = None, reference_run: str | None = None) -> pd.DataFrame:
    """Table of every Ollama run + the rule extractor + the heavy Voyager runs; written to results/feedback."""
    ex_dir = ctx.paths.extractions
    fb = feedback_dir(ctx)
    reference_run = reference_run or ctx.cfg.get("extraction", {}).get("default_run")
    if runs:
        paths = [resolve_run(ex_dir, r) for r in runs]
    else:
        heavy = [r for r in (reference_run, "llm_voyager_qwen3-30b-a3b_v1_full") if r]
        paths = [p for p in sorted(ex_dir.glob("llm_ollama_*.jsonl")) if "_v0_open" not in p.stem and
                 "_v3_no_example" not in p.stem]
        paths += [ex_dir / f"{r}.jsonl" for r in heavy + ["rule"] if (ex_dir / f"{r}.jsonl").exists()]
    gold = load_gold(ctx)
    ref = None
    if reference_run and (ex_dir / f"{reference_run}.jsonl").exists():
        ref = load_extraction_run(ex_dir / f"{reference_run}.jsonl")
    info_path = fb / "ollama_models.json"
    info = load_json(info_path) if info_path.exists() else {}
    rows = []
    for p in paths:
        rows.append(run_row(ctx, p, gold, None if p.stem == reference_run else ref, info))
    df = pd.DataFrame(rows).sort_values("gold_f1", ascending=False)
    df.to_csv(fb / "extractor_benchmark.csv", index=False)
    plot_benchmark(df, fb / "figures" / "extractor_benchmark.png")
    log.info("Extractor benchmark: %d runs -> %s", len(df), fb / "extractor_benchmark.csv")
    return df


def _short_label(row) -> str:
    if row["model"] in ("rule", "rule_based") or row["run"] == "rule":
        return "keyword rules"
    label = str(row["model"])
    if row.get("constrained"):
        label += " + schema"
    return label


def plot_benchmark(df: pd.DataFrame, path: Path) -> None:
    """Quality against cost for the best prompt of each model. One axis pair; x is logarithmic."""
    d = df.dropna(subset=["gold_f1", "latency_mean_s"]).copy()
    if d.empty:
        return
    d["label"] = d.apply(_short_label, axis=1)
    # the best configuration of each model, so the chart compares models rather than prompts
    d = d.sort_values("gold_f1", ascending=False).drop_duplicates("model")
    fig, ax = plt.subplots(figsize=(8.4, 4.8), facecolor=SURFACE)
    heavy = d["provider"].eq("voyager")
    local = d["provider"].eq("ollama")
    rule = ~heavy & ~local
    for mask, color, name in ((local, SERIES[0], "Local Ollama (this laptop)"),
                              (heavy, SERIES[1], "ASU Voyager (shared server)"), (rule, SERIES[2], "Keyword rules")):
        if mask.any():
            ax.scatter(d.loc[mask, "latency_mean_s"], d.loc[mask, "gold_f1"], s=70, color=color, label=name, zorder=3,
                       edgecolors=SURFACE, linewidths=2)
    for _, r in d.iterrows():
        # a neighbour to the right at nearly the same height: label this point on its left instead
        crowded = ((d["latency_mean_s"] > r["latency_mean_s"]) & (d["latency_mean_s"] < 4 * r["latency_mean_s"]) &
                   ((d["gold_f1"] - r["gold_f1"]).abs() < 0.04)).any()
        ax.annotate(r["label"], (r["latency_mean_s"], r["gold_f1"]), xytext=(-8, 6) if crowded else (8, 6),
                    textcoords="offset points", fontsize=8.5, color=INK, ha="right" if crowded else "left")
    ax.set_xscale("log")
    style_axes(ax, "Extraction quality against cost", "Seconds per call (log scale; the machines differ)",
               "Triple F1 on the 50 gold calls")
    ax.set_xlim(right=d["latency_mean_s"].max() * 6)     # room for the labels of the slowest models
    ax.legend(frameon=False, loc="upper left", labelcolor=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def extraction_noise_table(ctx, model: str | None, prompt: str = "v4_compact", constrained: bool = True,
                           n_each: int = 50, levels=("clean", "light", "medium", "heavy")) -> pd.DataFrame:
    """Do extractors keep the facts when a transcript gets noisy? Each one is scored on the same noisy calls against
    the Gemma run on the clean transcript, and against its own clean answer. No gold exists for these calls, so this
    measures how much agreement survives, not accuracy. The Gemma model itself is not re-run on noisy text."""
    from scamkg.data.noise import add_asr_noise
    from scamkg.data.preprocess import clean_transcript
    calls = ctx.calls
    gold = set(load_gold(ctx))
    ids = [c for c in benchmark_ids(ctx, n_each) if c not in gold]
    ref_run = ctx.cfg.get("extraction", {}).get("default_run")
    ref = load_extraction_run(resolve_run(ctx.paths.extractions, ref_run))
    cfg = deep_merge(ctx.cfg, {"llm": {"provider": "ollama", "fallback_model": None}})
    extractors = {"keyword rules": make_extractor("rule", cfg, ctx.ontology)}
    if model:
        extractors[model + (" + schema" if constrained else "")] = make_extractor(
            "llm", cfg, ctx.ontology, cache_dir=ctx.paths.cache, model=model, provider="ollama", prompt_version=prompt,
            constrained=constrained)
    rows = []
    for name, ex in extractors.items():
        own: dict[str, list[dict]] = {}
        for level in levels:
            pred = {}
            for c in ids:
                text = calls.at[c, "text"]
                if level != "clean":
                    text = clean_transcript(add_asr_noise(text, level, c, ctx.seed))[0]
                pred[c] = ex.extract(c, text).triples
            if level == "clean":
                own = pred
            for ds in ctx.external_datasets:
                sub = [c for c in ids if calls.at[c, "dataset"] == ds]
                vs_ref = compare_to_gold({c: pred[c] for c in sub}, {c: ref[c] for c in sub})
                vs_own = compare_to_gold({c: pred[c] for c in sub}, {c: own[c] for c in sub})
                rows.append({"extractor": name, "noise": level, "dataset": ds, "calls": len(sub),
                             "agree_with_gemma_clean_f1": vs_ref["triple_f1"],
                             "agree_with_gemma_clean_recall": vs_ref["triple_recall"],
                             "agree_with_own_clean_f1": vs_own["triple_f1"],
                             "triples_per_call": sum(len(pred[c]) for c in sub) / len(sub)})
            log.info("Extraction under noise: %s, %s done", name, level)
    df = pd.DataFrame(rows)
    df.to_csv(feedback_dir(ctx) / "extraction_noise.csv", index=False)
    return df


# ---------------------------------------------------------------------------- cascade
def cascade_records(rule: dict[str, dict], llm: dict[str, dict], min_triples: int, mode: str = "escalate") -> list[dict]:
    """Rule extractor first. A call with fewer than `min_triples` rule triples goes to the LLM.
    mode "escalate": the LLM's triples replace the rule triples. mode "union": they are added to them."""
    out = []
    for cid, r in rule.items():
        rt = r.get("triples", [])
        esc = len(rt) < min_triples and cid in llm
        triples = rt
        if esc and mode == "escalate":
            triples = llm[cid].get("triples", [])
        elif esc:
            seen = {(t["relation"], str(t["object"]).lower()) for t in rt}
            triples = rt + [t for t in llm[cid].get("triples", [])
                            if (t["relation"], str(t["object"]).lower()) not in seen]
        out.append({"call_id": cid, "extractor": f"cascade_k{min_triples}", "triples": triples, "escalated": bool(esc),
                    "json_valid": True, "schema_conformant": True, "status": "ok",
                    "latency_s": float(r.get("latency_s", 0)) + (float(llm[cid].get("latency_s", 0)) if esc else 0.0)})
    return out


def build_cascade_run(ctx, rule_run: str, llm_run: str, min_triples: int, mode: str = "escalate",
                      name: str | None = None) -> Path:
    ex_dir = ctx.paths.extractions
    rule = {r["call_id"]: r for r in latest_records(read_jsonl(resolve_run(ex_dir, rule_run)))}
    llm = {r["call_id"]: r for r in latest_records(read_jsonl(resolve_run(ex_dir, llm_run)))}
    records = cascade_records(rule, llm, min_triples, mode)
    frame = pd.DataFrame({"call_id": [r["call_id"] for r in records], "escalated": [r["escalated"] for r in records]})
    frame["dataset"] = ctx.calls.loc[frame["call_id"], "dataset"].to_numpy()
    name = slugify(name or f"cascade_k{min_triples}_{mode}__{llm_run}")
    path = ex_dir / f"{name}.jsonl"
    write_jsonl(path, records)
    meta = {"extractor": "cascade", "model": f"rules, then {load_json(resolve_run(ex_dir, llm_run).with_suffix('.meta.json')).get('model', llm_run)}",
            "provider": "cascade", "rule_run": rule_run, "llm_run": llm_run, "min_triples": min_triples, "mode": mode,
            "escalation_rate": float(frame["escalated"].mean()),
            "escalation_by_dataset": frame.groupby("dataset")["escalated"].mean().to_dict(),
            "run_name": name, "updated": timestamp(), **summarize_records(records)}
    save_json(path.with_suffix(".meta.json"), meta)
    log.info("Cascade %s: %.1f%% of calls go to the LLM", name, 100 * meta["escalation_rate"])
    return path


def cascade_table(ctx, rule_run: str, llm_run: str, ks=(1, 2, 3, 4, 5, 6), mode: str = "escalate") -> pd.DataFrame:
    """Escalation share against gold F1 and agreement with the LLM run, for every threshold (no LLM calls)."""
    gold = load_gold(ctx)
    ex_dir = ctx.paths.extractions
    llm_path = resolve_run(ex_dir, llm_run)
    ref = load_extraction_run(llm_path)
    rows = []
    for k in ks:
        p = build_cascade_run(ctx, rule_run, llm_run, k, mode)
        meta = load_json(p.with_suffix(".meta.json"))
        pred = load_extraction_run(p)
        covered = {c: t for c, t in gold.items() if c in pred}
        row = {"run": p.stem, "min_triples": k, "llm_share": meta["escalation_rate"],
               **{f"llm_share_{d}": v for d, v in meta["escalation_by_dataset"].items()},
               "gold_f1": compare_to_gold(pred, covered)["triple_f1"] if covered else float("nan")}
        for ds in ctx.external_datasets:
            ids = [c for c in ctx.calls.index[ctx.calls["dataset"] == ds] if c in pred and c in ref]
            row[f"agree_{ds}_f1"] = compare_to_gold({c: pred[c] for c in ids}, {c: ref[c] for c in ids})["triple_f1"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(feedback_dir(ctx) / f"cascade__{mode}__{slugify(llm_run)}.csv", index=False)
    return df


# ---------------------------------------------------------------------------- downstream detection
def run_config(cfg: dict, run: str) -> dict:
    """The same configuration, with results and artifacts of this KG kept apart from the committed ones."""
    base = Path(cfg["paths"]["results"]) / "feedback" / "runs" / slugify(run)
    art = Path(cfg["paths"]["artifacts"]) / "feedback" / slugify(run)
    return deep_merge(cfg, {"paths": {"results": str(base), "artifacts": str(art)},
                            "extraction": {"default_run": run}})


def build_serving_run(ctx, train_run: str | None, serve_run: str, name: str | None = None) -> Path:
    """Train once, serve light: training, validation and test calls keep the heavy run's triples (extracted once,
    offline) and only the calls arriving later use the light one. That is where the cost per call actually recurs."""
    ex_dir = ctx.paths.extractions
    train_run = train_run or ctx.cfg.get("extraction", {}).get("default_run")
    heavy = {r["call_id"]: r for r in latest_records(read_jsonl(resolve_run(ex_dir, train_run)))}
    light = {r["call_id"]: r for r in latest_records(read_jsonl(resolve_run(ex_dir, serve_run)))}
    external = set(ctx.calls.index[ctx.calls["split"] == "external"])
    missing = [c for c in external if c not in light]
    if missing:
        raise ValueError(f"Run {serve_run} has no triples for {len(missing)} external calls (e.g. {missing[:3]})")
    records = [light[c] if c in external else heavy[c] for c in ctx.calls.index if c in heavy or c in external]
    name = slugify(name or f"serve__{serve_run}")
    path = ex_dir / f"{name}.jsonl"
    write_jsonl(path, records)
    save_json(path.with_suffix(".meta.json"), {
        "extractor": "serving", "model": f"trained on {train_run}, served by {serve_run}", "provider": "serving",
        "train_run": train_run, "serve_run": serve_run, "run_name": name, "updated": timestamp(),
        **summarize_records(records)})
    return path


def _seed_rows(rctx, run: str) -> list[dict]:
    """GraphSAGE on this KG, mean over seeds: a single seed moves recall by several points, more than most gaps here."""
    from scamkg.evaluation.evaluator import Evaluator
    from scamkg.evaluation.metrics import binary_metrics
    from scamkg.models.gnn import seeds_path
    wide = pd.read_csv(seeds_path(rctx), dtype={"call_id": str}).set_index("call_id")
    wide = wide[[c for c in wide.columns if c.startswith("text+topo|")]]
    sets = Evaluator(rctx).eval_sets()
    rows = []
    for name, ids in sets.items():
        if name not in ("kaggle", "ncsu", "ncsu+kaggle"):
            continue
        y = rctx.labels(ids)
        per = pd.DataFrame([binary_metrics(y, wide.loc[ids, c].to_numpy()) for c in wide.columns])
        rows.append({"extraction_run": run, "experiment": "E4", "model": "graphsage (5 seeds)", "eval_set": name,
                     "n": len(ids), "n_pos": int(y.sum()), **per[["precision", "recall", "f1", "pr_auc", "roc_auc",
                                                                  "recall_at_fpr10"]].mean().to_dict(),
                     "recall_sd": float(per["recall"].std(ddof=1)), "pr_auc_sd": float(per["pr_auc"].std(ddof=1)),
                     "recall_at_fpr10_sd": float(per["recall_at_fpr10"].std(ddof=1))})
    return rows


def compare_runs(ctx, runs: list[str], experiments=("E1", "E2", "E2G"), models=("lr", "xgb"), deep: bool = False) -> pd.DataFrame:
    """Detection on KGs from different extractors: same calls, models and hyper-parameters; only the triples differ.
    Results accumulate in downstream_detection.csv, so a run added later keeps the earlier ones."""
    from scamkg.cli import run_deep_isolated
    from scamkg.context import Context
    from scamkg.models.classical import run_classical

    frames = []
    keep = ["extraction_run", "experiment", "model", "eval_set", "n", "n_pos", "precision", "recall", "f1", "pr_auc",
            "roc_auc", "recall_at_fpr10"]
    for run in runs:
        rcfg = run_config(ctx.cfg, run)
        rctx = Context.load(rcfg, run=run)
        params = ctx.paths.results / "best_params.json"
        if params.exists():   # hyper-parameters fixed across runs, so the extractor is the only thing that changes
            shutil.copyfile(params, rctx.paths.results / "best_params.json")
        run_classical(rctx, list(experiments), list(models), tune=False)
        m = pd.read_csv(rctx.paths.results / "metrics.csv")
        m = m[m["experiment"].isin(experiments) & m["model"].isin(models)].assign(extraction_run=run)
        frames.append(m[m["eval_set"].isin(["kaggle", "ncsu", "ncsu+kaggle"])][keep])
        if deep:
            run_deep_isolated("E4_SEEDS", rcfg, run)
            frames.append(pd.DataFrame(_seed_rows(rctx, run)))
    df = pd.concat(frames, ignore_index=True)
    path = feedback_dir(ctx) / "downstream_detection.csv"
    if path.exists():
        old = pd.read_csv(path)
        done = set(map(tuple, df[["extraction_run", "experiment", "model", "eval_set"]].to_numpy()))
        mask = old[["extraction_run", "experiment", "model", "eval_set"]].apply(tuple, axis=1).isin(done)
        df = pd.concat([old[~mask], df], ignore_index=True)
    df.to_csv(path, index=False)
    downstream_summary(df).to_csv(feedback_dir(ctx) / "downstream_summary.csv", index=False)
    return df


def downstream_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per run and model: PR-AUC and recall on Kaggle and on FTC robocalls vs. legitimate calls."""
    wide = df[df["eval_set"].isin(["kaggle", "ncsu+kaggle"])].pivot_table(
        index=["extraction_run", "experiment", "model"], columns="eval_set",
        values=["pr_auc", "recall", "recall_at_fpr10"])
    wide.columns = [f"{m}_{s}" for m, s in wide.columns]
    return wide.reset_index()


__all__ = ["benchmark_table", "build_cascade_run", "build_serving_run", "cascade_table", "compare_runs", "downstream_summary",
           "feedback_dir", "md_table", "run_benchmark", "run_config"]
