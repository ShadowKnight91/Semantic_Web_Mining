"""Command-line interface:  python -m scamkg [--config FILE] <command> [options]

Pipeline order (or just `run-all`):
  download | demo-data -> preprocess -> extract -> build-graph -> eda -> annotate-sample -> eval-extraction
  -> train (E1 E2 E2G E3 E4 E5 LLM) -> cv -> loto -> significance -> ablate -> explain -> report -> dashboard

Professor's feedback (results in results/feedback/, the committed results are not touched):
  benchmark-extractors -> cascade -> compare-runs | monitor-recall [--noise] | discover-scripts -> feedback-report
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

from scamkg.config import ROOT, get_paths, load_config
from scamkg.utils import get_logger, has_module, save_json, set_seed

log = get_logger("scamkg")
ALL_EXPERIMENTS = ["E1", "E2", "E2G", "E3", "E4", "E5", "LLM"]


# ---------------------------------------------------------------------------- data
def cmd_download(args, cfg):
    from scamkg.data.download import download_all
    download_all(cfg, get_paths(cfg).ensure())


def cmd_demo_data(args, cfg):
    from scamkg.data.synthetic import generate_demo_data
    paths = get_paths(cfg).ensure()
    generate_demo_data(paths.raw, args.n_primary, args.n_shifted, args.n_robocall, int(cfg["project"]["seed"]))


def cmd_generate_pranks(args, cfg):
    from scamkg.data.pranks import generate
    paths = get_paths(cfg).ensure()
    src = (cfg["data"].get("sources") or {}).get("pranks") or {}
    out = Path(args.out) if args.out else ROOT / src.get("local_dir", "data/pranks") / "pranks.jsonl"
    generate(cfg, out, paths.cache, n_pranks=args.n, n_controls=args.controls, model=args.model,
             provider=args.provider, seed=int(cfg["project"]["seed"]))


def cmd_preprocess(args, cfg):
    from scamkg.data.preprocess import preprocess
    preprocess(cfg, get_paths(cfg).ensure())


def cmd_eda(args, cfg):
    from scamkg.context import Context
    from scamkg.data.eda import run_eda
    from scamkg.extraction.extractor import load_extraction_run, resolve_run
    ctx = Context.load(cfg, with_graph=False)
    extractions = None
    try:
        extractions = load_extraction_run(resolve_run(ctx.paths.extractions, args.run))
    except FileNotFoundError:
        log.info("No extraction run yet - EDA without entity statistics")
    run_eda(ctx.calls, ctx.paths.results / "eda", extractions)


# ---------------------------------------------------------------------------- extraction
def cmd_extract(args, cfg):
    from scamkg.context import Context
    from scamkg.extraction.extractor import make_extractor, run_extraction
    ctx = Context.load(cfg, with_graph=False)
    calls = ctx.calls
    if args.datasets:
        calls = calls[calls["dataset"].isin(args.datasets)]
    if args.splits:
        calls = calls[calls["split"].isin(args.splits)]
    if args.ids_file:
        ids = _read_call_ids(Path(args.ids_file))
        missing = [c for c in ids if c not in calls.index]
        if missing:
            log.warning("%d call ids from %s are not in the processed calls (e.g. %s)", len(missing), args.ids_file,
                        missing[:3])
        calls = calls[calls.index.isin(ids)]
    models = args.model or [None]
    versions = args.prompt_version or [None]
    if getattr(args, "no_fallback", False):
        cfg = {**cfg, "llm": {**cfg["llm"], "fallback_model": None}}
    for model in (models if args.extractor == "llm" else [None]):
        for version in (versions if args.extractor == "llm" else [None]):
            ex = make_extractor(args.extractor, cfg, ctx.ontology, cache_dir=ctx.paths.cache, model=model,
                                provider=args.provider, prompt_version=version,
                                constrained=getattr(args, "constrained", False))
            run_name = args.run_name if (args.run_name and len(models) * len(versions) == 1) else None
            # rule extraction takes seconds -> always recompute; LLM runs resume (responses are also cached)
            resume = not args.no_resume and args.extractor == "llm"
            run_extraction(calls, ex, ctx.paths.extractions, run_name=run_name, resume=resume,
                           max_workers=args.workers or int(cfg["llm"].get("max_workers", 1)), limit=args.limit,
                           seed=ctx.seed)


def _read_call_ids(path: Path) -> list[str]:
    """Call ids from a JSONL file with a `call_id` field (e.g. data/annotations/sample.jsonl) or one id per line."""
    from scamkg.utils import read_jsonl
    if path.suffix == ".jsonl":
        return [str(r["call_id"]) for r in read_jsonl(path)]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cmd_annotate_sample(args, cfg):
    from scamkg.context import Context
    from scamkg.extraction.evaluate import create_annotation_sample
    ctx = Context.load(cfg, with_graph=False)
    create_annotation_sample(ctx.calls, ctx.paths.annotations, n=args.n, seed=ctx.seed, primary=ctx.primary,
                             from_synthetic=args.from_synthetic)


def cmd_eval_extraction(args, cfg):
    from scamkg.extraction.evaluate import evaluate_extraction_runs
    from scamkg.ontology.schema import Ontology
    paths = get_paths(cfg).ensure()
    df = evaluate_extraction_runs(paths.extractions, paths.annotations, Ontology.load(cfg["ontology"]["path"]),
                                  paths.results, gold_path=args.gold)
    if not df.empty:
        print(df.to_string(index=False))


def cmd_benchmark_extractors(args, cfg):
    from scamkg.context import Context
    from scamkg.extraction.benchmark import benchmark_table, run_benchmark
    ctx = Context.load(cfg, with_graph=False)
    if args.models:
        modes = {"both": (True, False), "on": (True,), "off": (False,)}[args.constrained]
        run_benchmark(ctx, args.models, args.prompts, modes, n_each=args.n_each, gold_only=args.gold_only,
                      workers=args.workers)
    if args.noise_model:
        from scamkg.extraction.benchmark import extraction_noise_table
        table = extraction_noise_table(ctx, args.noise_model, args.prompts[0], args.constrained != "off", args.n_each)
        print(table.round(3).to_string(index=False))
    df = benchmark_table(ctx)
    show = ["model", "prompt", "constrained", "parameters", "loaded_gb", "n_calls", "gold_f1", "gold_precision",
            "gold_recall", "agree_ncsu_f1", "agree_kaggle_f1", "schema_conformant", "failed", "latency_mean_s",
            "latency_p95_s", "hours_per_100k_calls"]
    print(df[[c for c in show if c in df]].round(3).to_string(index=False))


def cmd_cascade(args, cfg):
    from scamkg.context import Context
    from scamkg.extraction.benchmark import cascade_table
    ctx = Context.load(cfg, with_graph=False)
    llm_run = args.llm_run or cfg.get("extraction", {}).get("default_run")
    print(cascade_table(ctx, args.rule_run, llm_run, args.k, args.mode).round(3).to_string(index=False))


def cmd_compare_runs(args, cfg):
    from scamkg.context import Context
    from scamkg.extraction.benchmark import compare_runs, downstream_summary
    ctx = Context.load(cfg, with_graph=False)
    df = compare_runs(ctx, args.runs, args.experiments, args.models, deep=args.deep)
    print(downstream_summary(df).round(3).to_string(index=False))


def cmd_serving_run(args, cfg):
    from scamkg.context import Context
    from scamkg.extraction.benchmark import build_serving_run
    print(build_serving_run(Context.load(cfg, with_graph=False), args.train_run, args.serve_run, args.name).stem)


# ---------------------------------------------------------------------------- graph
def cmd_build_graph(args, cfg):
    from scamkg.context import Context
    from scamkg.extraction.extractor import load_extraction_run, resolve_run
    from scamkg.graph.build import CallKG
    from scamkg.graph.rdf_export import export_rdf, validate_shacl
    from scamkg.graph.visualize import graph_summary, plot_entity_degrees, plot_kg_sample
    ctx = Context.load(cfg, with_graph=False)
    run_path = resolve_run(ctx.paths.extractions, args.run or cfg.get("extraction", {}).get("default_run"))
    backend = args.backend or cfg["graph"].get("backend", "networkx")
    structural = None
    if backend == "neo4j":
        from scamkg.graph.neo4j_store import Neo4jStore
        kg = CallKG(ctx.calls.index.tolist(), load_extraction_run(run_path), ctx.ontology)
        with Neo4jStore.from_config(cfg) as store:
            store.clear()
            store.load(kg, ctx.calls)
            structural = store.compute_features(kg, cfg["graph"], ctx.seed)
            store.export_csv(ctx.paths.kg / "neo4j_export")
    ctx.attach_graph(str(run_path), structural=structural, use_cache=False)
    kg, run = ctx.kg, ctx.run_name
    kg.write_graphml(ctx.paths.kg / f"kg__{run}.graphml", ctx.calls)
    nodes, edges = kg.to_frames(ctx.calls)
    nodes.to_csv(ctx.paths.kg / f"nodes__{run}.csv", index=False)
    edges.to_csv(ctx.paths.kg / f"edges__{run}.csv", index=False)
    if not args.no_rdf:
        g = export_rdf(kg, ctx.calls, ctx.ontology, ctx.paths.kg / "rdf" / f"kg__{run}.ttl",
                       owl_path=ROOT / cfg["ontology"]["owl"])
        conforms, text = validate_shacl(g, ROOT / cfg["ontology"]["shapes"])
        (ctx.paths.kg / "rdf" / f"shacl_report__{run}.txt").write_text(text, encoding="utf-8")
        log.info("SHACL: %s", {True: "graph conforms", False: "violations - see shacl report", None: text}[conforms])
    summary = graph_summary(kg, ctx.calls, ctx.fb.structural)
    summary["backend"] = backend
    summary["run"] = run
    save_json(ctx.paths.results / "graph_summary.json", summary)
    plot_kg_sample(kg, ctx.calls, ctx.paths.figures / "kg_sample.png", seed=ctx.seed)
    plot_entity_degrees(kg, ctx.paths.figures / "kg_entity_degrees.png")
    log.info("Graph built (%s backend): %d entities, %d communities", backend, summary["entities"],
             summary.get("communities", 0))


def cmd_sparql(args, cfg):
    from rdflib import Graph
    from scamkg.graph.rdf_export import run_sparql_file
    paths = get_paths(cfg)
    ttl = Path(args.ttl) if args.ttl else max((paths.kg / "rdf").glob("kg__*.ttl"), key=lambda p: p.stat().st_mtime)
    g = Graph().parse(str(ttl), format="turtle")
    for title, df in run_sparql_file(g, Path(args.query_file)):
        print(f"\n== {title} ==\n{df.to_string(index=False) if not df.empty else '(no rows)'}")


# ---------------------------------------------------------------------------- models
def _deep_ok(*mods: str) -> bool:
    missing = [m for m in mods if not has_module(m)]
    if missing:
        log.warning("Skipping (missing %s) - pip install -r requirements-deep.txt", ", ".join(missing))
    return not missing


def _deep_worker(experiment: str, cfg: dict, run: str | None) -> None:
    from scamkg.context import Context
    set_seed(int(cfg["project"]["seed"]))
    ctx = Context.load(cfg, with_graph=experiment.startswith("E4"), run=run)
    ctx.run_name = ctx.run_name or run  # recorded in metrics.csv
    if experiment == "E3":
        from scamkg.models.transformer import run_distilbert, run_sbert_mlp
        if _deep_ok("torch", "sentence_transformers"):
            run_sbert_mlp(ctx)
        if _deep_ok("torch", "transformers"):
            run_distilbert(ctx)
    elif experiment == "E4" and _deep_ok("torch"):
        from scamkg.models.gnn import run_graphsage
        run_graphsage(ctx)
    elif experiment == "E4_SCRIPTS" and _deep_ok("torch"):
        from scamkg.models.gnn import run_graphsage_loto, run_graphsage_scripts, run_graphsage_seed_scores
        run_graphsage_scripts(ctx)
        run_graphsage_seed_scores(ctx)
        run_graphsage_loto(ctx)
    elif experiment == "E4_SEEDS" and _deep_ok("torch"):
        from scamkg.models.gnn import run_graphsage_seed_scores
        run_graphsage_seed_scores(ctx)   # every variant: discover-scripts reads the same file


def run_deep_isolated(experiment: str, cfg: dict, run: str | None) -> None:
    """Run the torch part of E3/E4 in a fresh process. PyTorch and XGBoost bundle different OpenMP
    runtimes on macOS; once both have run in one process XGBoost segfaults or deadlocks, which would
    break every XGBoost model, CV fold and ablation that follows E3/E4."""
    import multiprocessing
    p = multiprocessing.get_context("spawn").Process(target=_deep_worker, args=(experiment, cfg, run))
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"{experiment} failed in its worker process (exit code {p.exitcode})")


def cmd_train(args, cfg):
    from scamkg.context import Context
    exps = args.experiments or ["E1", "E2", "E2G", "E5"] + (["E3", "E4"] if has_module("torch") else [])
    ctx = Context.load(cfg, with_graph=any(e in exps for e in ("E2", "E2G", "E4", "E5")), run=args.run)
    classical = [e for e in exps if e in ("E1", "E2", "E2G")]
    if classical:
        from scamkg.models.classical import run_classical
        run_classical(ctx, classical, args.models, tune=not args.no_tune)
    if "E3" in exps:
        run_deep_isolated("E3", cfg, ctx.run_name)
    if "E4" in exps and _deep_ok("torch"):
        from scamkg.models.gnn import run_graphsage_xgb
        run_deep_isolated("E4", cfg, ctx.run_name)
        run_graphsage_xgb(ctx)
    if "E5" in exps:
        from scamkg.models.clustering import run_clustering
        run_clustering(ctx)
    if "LLM" in exps:
        from scamkg.models.llm_zeroshot import run_zeroshot
        run_zeroshot(ctx, limit=args.limit)


def cmd_cv(args, cfg):
    from scamkg.context import Context
    from scamkg.models.classical import run_cv
    exps = args.experiments or ["E1", "E2", "E2G"]
    ctx = Context.load(cfg, with_graph=any(e != "E1" for e in exps), run=args.run)
    run_cv(ctx, exps, args.models, args.folds)


def cmd_loto(args, cfg):
    from scamkg.context import Context
    from scamkg.models.classical import run_loto
    exps = args.experiments or ["E1", "E2", "E2G"]
    ctx = Context.load(cfg, with_graph=any(e != "E1" for e in exps), run=args.run)
    run_loto(ctx, exps, args.models)


def cmd_significance(args, cfg):
    from scamkg.context import Context
    from scamkg.evaluation.evaluator import run_significance
    df = run_significance(Context.load(cfg, with_graph=False))
    if not df.empty:
        print(df.to_string(index=False))


def cmd_monitor_recall(args, cfg):
    from scamkg.context import Context
    from scamkg.evaluation.monitor import run_monitor
    ctx = Context.load(cfg, run=args.run)
    out = run_monitor(ctx)
    keep = ["model", "stage", "n_scams", "recall", "recall_ci_low", "recall_ci_high", "recall_drop", "false_alarm_rate",
            "recall_at_target_fpr", "status"]
    print(out["ladder"][keep].round(3).to_string(index=False))
    print(out["stream"].round(3).to_string(index=False))
    if args.noise:
        from scamkg.evaluation.noise_run import run_noise_robustness
        print(run_noise_robustness(cfg, ctx, args.noise_run).round(3).to_string(index=False))


def cmd_discover_scripts(args, cfg):
    from scamkg.context import Context
    from scamkg.extraction.benchmark import run_config
    from scamkg.config import deep_merge
    from scamkg.models.gnn import embeddings_path, loto_path
    from scamkg.models.script_discovery import run_script_discovery
    main = Context.load(cfg, with_graph=False)
    run = args.run or cfg.get("extraction", {}).get("default_run")
    rcfg = deep_merge(run_config(cfg, run), {"models": {"graphsage": {"variants": ["text+topo", "text", "topo", "kg"]}}})
    rctx = Context.load(rcfg, run=run)
    if args.retrain or not embeddings_path(rctx, "kg").exists():
        run_deep_isolated("E4", rcfg, rctx.run_name)
    if args.retrain or not embeddings_path(rctx, "unsup").exists() or not loto_path(rctx).exists():
        run_deep_isolated("E4_SCRIPTS", rcfg, rctx.run_name)
    out = run_script_discovery(main, rctx, k=args.k)
    for name in ("methods", "rescue", "rescue_seeds", "scripts", "loto"):
        if name in out:
            print(f"\n== {name} ==\n{out[name].round(3).to_string(index=False)}")


def cmd_cluster(args, cfg):
    from scamkg.context import Context
    from scamkg.models.clustering import run_clustering
    run_clustering(Context.load(cfg, run=args.run))


def cmd_ablate(args, cfg):
    from scamkg.context import Context
    from scamkg.evaluation.ablation import run_ablation
    run_ablation(Context.load(cfg, run=args.run), args.experiments or ("E2", "E2G"), args.models or ("lr", "xgb"))


def cmd_explain(args, cfg):
    from scamkg.context import Context
    from scamkg.evaluation.explain import explain_call, run_explanations
    ctx = Context.load(cfg, run=args.run)
    if args.call_id:
        e = explain_call(ctx, args.call_id)
        print(json.dumps({k: v for k, v in e.items() if k != "dot"}, indent=2, default=str))
    else:
        run_explanations(ctx, args.experiment, args.model, args.max_calls)


def cmd_zeroshot(args, cfg):
    from scamkg.context import Context
    from scamkg.models.llm_zeroshot import run_zeroshot
    run_zeroshot(Context.load(cfg, with_graph=False), args.model, args.provider, args.limit)


def cmd_relabel(args, cfg):
    from scamkg.context import Context
    from scamkg.models.llm_zeroshot import run_relabel
    run_relabel(Context.load(cfg, with_graph=False), args.model, args.provider, args.limit)


def cmd_multiclass(args, cfg):
    from scamkg.context import Context
    from scamkg.models.classical import run_multiclass
    ctx = Context.load(cfg, with_graph=False)
    try:
        ctx.attach_graph(args.run)
    except FileNotFoundError:
        pass
    run_multiclass(ctx, getattr(args, "experiments", None), getattr(args, "models", None))


def cmd_run_multiclass(args, cfg):
    """Caller-type pipeline (use with config/multiclass.yaml): preprocess -> rule-based triples (unless --run
    names an existing run) -> KG -> binary models (for predict / the dashboard) -> caller-type models -> report."""
    ns = argparse.Namespace
    cmd_preprocess(args, cfg)
    run = args.run
    if run is None:
        cmd_extract(ns(extractor="rule", datasets=None, splits=None, ids_file=None, model=None, prompt_version=None,
                       provider=None, run_name=None, no_resume=False, workers=None, limit=None, no_fallback=False), cfg)
        run = "rule"
    cmd_build_graph(ns(run=run, backend=None, no_rdf=True), cfg)
    cmd_train(ns(experiments=["E1", "E2", "E2G"], models=["lr", "xgb"], no_tune=False, run=run, limit=None), cfg)
    cmd_multiclass(ns(run=run, experiments=None, models=None), cfg)
    cmd_report(args, cfg)


def cmd_report(args, cfg):
    from scamkg.context import Context
    from scamkg.evaluation.report import build_report
    ctx = Context.load(cfg, with_graph=False)
    try:
        from scamkg.extraction.extractor import resolve_run
        ctx.run_name = resolve_run(ctx.paths.extractions, cfg.get("extraction", {}).get("default_run")).stem
    except FileNotFoundError:
        pass
    print(build_report(ctx))


def cmd_feedback_report(args, cfg):
    from scamkg.context import Context
    from scamkg.evaluation.feedback_report import build_feedback_report
    print(build_feedback_report(Context.load(cfg, with_graph=False)))


def cmd_predict(args, cfg):
    from scamkg.predict import ScamDetector
    text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
    if not text:
        raise SystemExit("Give --text or --file")
    det = ScamDetector(cfg, args.experiment, args.model, args.extractor)
    res = det.score(text)
    ex = res["explanation"]
    caller = {k: res[k] for k in ("caller_type", "caller_type_probs") if k in res}
    print(json.dumps({"score": round(res["score"], 4), "is_scam": res["is_scam"], "model": res["model"], **caller,
                      "script_path": ex["script_path"], "summary": ex["summary"],
                      "triples": [f"{t['relation']} {t['object']}" for t in res["triples"]],
                      "similar_calls": ex["similar_calls"]}, indent=2))


def cmd_banksim(args, cfg):
    from scamkg.models.banksim import run_banksim
    print(run_banksim(cfg, get_paths(cfg).ensure(), args.path).to_string(index=False))


def cmd_dashboard(args, cfg):
    cmd = [sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py"), "--"]
    if cfg.get("_config_file"):
        cmd += ["--config", cfg["_config_file"]]
    subprocess.run(cmd, check=False)


def cmd_run_all(args, cfg):
    from scamkg.context import Context
    paths = get_paths(cfg).ensure()
    demo = "demo" in cfg["data"]["load"]
    ns = argparse.Namespace
    if demo and (args.regenerate or not any(paths.raw.glob("*.jsonl"))):
        cmd_demo_data(ns(n_primary=600, n_shifted=240, n_robocall=120), cfg)
    elif args.download:
        cmd_download(args, cfg)
    cmd_preprocess(args, cfg)
    cmd_extract(ns(extractor=args.extractor, datasets=None, splits=None, ids_file=None, model=args.model,
                   prompt_version=None, provider=None, run_name=None, no_resume=False, workers=None, limit=None,
                   no_fallback=False), cfg)
    # every later step uses the run just extracted (the most recent one), not extraction.default_run
    cfg = {**cfg, "extraction": {**(cfg.get("extraction") or {}), "default_run": None}}
    cmd_build_graph(ns(run=None, backend="neo4j" if args.with_neo4j else None, no_rdf=False), cfg)
    cmd_eda(ns(run=None), cfg)
    cmd_annotate_sample(ns(n=50, from_synthetic=demo), cfg)
    cmd_eval_extraction(ns(gold=None), cfg)
    exps = ["E1", "E2", "E2G", "E5"]
    if not args.skip_deep and has_module("torch"):
        exps += ["E3", "E4"]
    cmd_train(ns(experiments=exps, models=None, no_tune=False, run=None, limit=None), cfg)
    cmd_cv(ns(experiments=None, models=None, folds=None, run=None), cfg)
    cmd_loto(ns(experiments=None, models=None, run=None), cfg)
    cmd_significance(args, cfg)
    cmd_ablate(ns(experiments=None, models=None, run=None), cfg)
    ctx = Context.load(cfg)
    if (ctx.calls["label3"] != "").any():
        from scamkg.models.classical import run_multiclass
        run_multiclass(ctx)
    from scamkg.evaluation.explain import run_explanations
    run_explanations(ctx, "E2", "xgb")
    cmd_report(args, cfg)
    log.info("Pipeline finished. Results in %s - open the dashboard with: python -m scamkg%s dashboard",
             paths.results, f" --config {cfg['_config_file']}" if cfg.get("_config_file") else "")


# ---------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scamkg", description="From call transcripts to knowledge graphs: "
                                                            "LLM-powered phone scam detection (CSE 573 P10)")
    p.add_argument("--config", default=None, help="YAML file merged over config/config.yaml (e.g. config/demo.yaml)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="download BothBosu (HF) + NCSU robocalls (+ Kaggle via CLI)").set_defaults(fn=cmd_download)
    s = sub.add_parser("demo-data", help="generate synthetic demo calls (use with --config config/demo.yaml)")
    s.add_argument("--n-primary", type=int, default=600)
    s.add_argument("--n-shifted", type=int, default=240)
    s.add_argument("--n-robocall", type=int, default=120)
    s.set_defaults(fn=cmd_demo_data)
    s = sub.add_parser("generate-pranks", help="LLM-generated prank calls (+ style controls) for the caller-type task")
    s.add_argument("--n", type=int, default=120, help="prank calls")
    s.add_argument("--controls", type=int, default=36, help="fraud / sales calls by the same model (test only)")
    s.add_argument("--model", default="gemma2:9b")
    s.add_argument("--provider", default=None)
    s.add_argument("--out", default=None, help="default: data/pranks/pranks.jsonl")
    s.set_defaults(fn=cmd_generate_pranks)
    sub.add_parser("preprocess", help="clean, dedupe, unify labels, create splits").set_defaults(fn=cmd_preprocess)
    s = sub.add_parser("eda", help="exploratory data analysis")
    s.add_argument("--run", default=None, help="extraction run for entity statistics")
    s.set_defaults(fn=cmd_eda)

    s = sub.add_parser("extract", help="ontology-guided triple extraction (rule-based or LLM)")
    s.add_argument("--extractor", choices=["rule", "llm"], default="rule")
    s.add_argument("--provider", default=None, help="ollama | voyager (default: config llm.provider)")
    s.add_argument("--model", nargs="+", default=None, help="one or more LLM names, e.g. qwen2.5:7b gemma2:9b")
    s.add_argument("--prompt-version", nargs="+", default=None,
                   help="v1_full | v2_no_descriptors | v3_no_example | v0_open | v4_compact (several = prompt ablation)")
    s.add_argument("--datasets", nargs="+", default=None)
    s.add_argument("--splits", nargs="+", default=None)
    s.add_argument("--limit", type=int, default=None, help="a seeded random sample of N pending calls (pilot run)")
    s.add_argument("--ids-file", default=None,
                   help="only these calls: JSONL with call_id (e.g. data/annotations/sample.jsonl) or one id per line")
    s.add_argument("--workers", type=int, default=None)
    s.add_argument("--run-name", default=None)
    s.add_argument("--no-resume", action="store_true")
    s.add_argument("--no-fallback", action="store_true",
                   help="disable llm.fallback_model (use for the model / prompt comparisons of RQ2)")
    s.add_argument("--constrained", action="store_true",
                   help="constrained decoding: Ollama enforces the ontology as a JSON schema while generating")
    s.set_defaults(fn=cmd_extract)
    s = sub.add_parser("benchmark-extractors", help="lighter models: quality, agreement, latency, memory")
    s.add_argument("--models", nargs="+", default=None, help="Ollama models to run; omit to only rebuild the table")
    s.add_argument("--prompts", nargs="+", default=["v1_full", "v4_compact"])
    s.add_argument("--constrained", choices=["both", "on", "off"], default="both")
    s.add_argument("--n-each", type=int, default=50, help="calls per external data set, besides the gold calls")
    s.add_argument("--gold-only", action="store_true", help="only the gold calls (a quick screen)")
    s.add_argument("--workers", type=int, default=1, help="keep 1 when latency matters")
    s.add_argument("--noise-model", default=None, help="also compare this model with the rules on noisy transcripts")
    s.set_defaults(fn=cmd_benchmark_extractors)
    s = sub.add_parser("cascade", help="rule extractor first; only low-coverage calls go to an LLM")
    s.add_argument("--rule-run", default="rule")
    s.add_argument("--llm-run", default=None, help="default: extraction.default_run")
    s.add_argument("--k", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6], help="escalate below this many rule triples")
    s.add_argument("--mode", choices=["escalate", "union"], default="escalate")
    s.set_defaults(fn=cmd_cascade)
    s = sub.add_parser("compare-runs", help="does cheaper extraction hurt detection? one KG per extraction run")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--experiments", nargs="+", choices=["E1", "E2", "E2G"], default=["E2", "E2G"])
    s.add_argument("--models", nargs="+", choices=["lr", "svm", "rf", "xgb"], default=["lr", "xgb"])
    s.add_argument("--deep", action="store_true", help="also train GraphSAGE on each KG (5 seeds)")
    s.set_defaults(fn=cmd_compare_runs)
    s = sub.add_parser("serving-run", help="train once with a heavy extractor, serve new calls with a light one")
    s.add_argument("--train-run", default=None, help="triples of training calls (default: extraction.default_run)")
    s.add_argument("--serve-run", required=True, help="triples of the calls scored afterwards")
    s.add_argument("--name", default=None)
    s.set_defaults(fn=cmd_serving_run)

    s = sub.add_parser("annotate-sample", help="sample calls + templates for the 2-annotator gold standard")
    s.add_argument("--n", type=int, default=50)
    s.add_argument("--from-synthetic", action="store_true", help="demo only: gold triples from the generator")
    s.set_defaults(fn=cmd_annotate_sample)
    s = sub.add_parser("eval-extraction", help="X1: extraction quality of every run")
    s.add_argument("--gold", default=None)
    s.set_defaults(fn=cmd_eval_extraction)

    s = sub.add_parser("build-graph", help="build the KG, graph features, GraphML/CSV/RDF exports")
    s.add_argument("--run", default=None)
    s.add_argument("--backend", choices=["networkx", "neo4j"], default=None)
    s.add_argument("--no-rdf", action="store_true")
    s.set_defaults(fn=cmd_build_graph)
    s = sub.add_parser("sparql", help="run SPARQL queries over the exported RDF KG")
    s.add_argument("--query-file", default=str(ROOT / "ontology" / "queries.sparql"))
    s.add_argument("--ttl", default=None)
    s.set_defaults(fn=cmd_sparql)

    s = sub.add_parser("train", help="train/evaluate experiments " + " ".join(ALL_EXPERIMENTS))
    s.add_argument("--experiments", nargs="+", choices=ALL_EXPERIMENTS, default=None)
    s.add_argument("--models", nargs="+", choices=["lr", "svm", "rf", "xgb"], default=None)
    s.add_argument("--no-tune", action="store_true", help="reuse best_params.json / defaults")
    s.add_argument("--run", default=None)
    s.add_argument("--limit", type=int, default=None, help="LLM zero-shot: sampled calls per evaluation set")
    s.set_defaults(fn=cmd_train)
    s = sub.add_parser("cv", help="5-fold stratified cross-validation (mean +- std)")
    s.add_argument("--experiments", nargs="+", choices=["E1", "E2", "E2G"], default=None)
    s.add_argument("--models", nargs="+", choices=["lr", "svm", "rf", "xgb"], default=None)
    s.add_argument("--folds", type=int, default=None)
    s.add_argument("--run", default=None)
    s.set_defaults(fn=cmd_cv)
    s = sub.add_parser("loto", help="leave-one-type-out: generalization to unseen scam / legitimate call types")
    s.add_argument("--experiments", nargs="+", choices=["E1", "E2", "E2G"], default=None)
    s.add_argument("--models", nargs="+", choices=["lr", "svm", "rf", "xgb"], default=None)
    s.add_argument("--run", default=None)
    s.set_defaults(fn=cmd_loto)
    sub.add_parser("significance", help="McNemar tests E1 vs E2/E2G").set_defaults(fn=cmd_significance)
    s = sub.add_parser("monitor-recall", help="recall from synthetic to FTC calls: ladder, prevalence, drift, stream")
    s.add_argument("--run", default=None, help="KG run for the coverage signals (default: extraction.default_run)")
    s.add_argument("--noise", action="store_true", help="also score transcripts with simulated speech-to-text errors")
    s.add_argument("--noise-run", default="rule", help="KG of the noise test; its triples must re-run on noisy text")
    s.set_defaults(fn=cmd_monitor_recall)
    s = sub.add_parser("discover-scripts", help="GraphSAGE: scam scripts across data sets that bag of words misses")
    s.add_argument("--run", default=None, help="KG run (default: extraction.default_run)")
    s.add_argument("--k", type=int, default=16, help="number of script clusters; the same for every method")
    s.add_argument("--retrain", action="store_true", help="retrain GraphSAGE even if embeddings exist")
    s.set_defaults(fn=cmd_discover_scripts)
    s = sub.add_parser("cluster", help="E5: K-Means / DBSCAN scam-script discovery")
    s.add_argument("--run", default=None)
    s.set_defaults(fn=cmd_cluster)
    s = sub.add_parser("ablate", help="entity-type and feature-group ablations")
    s.add_argument("--experiments", nargs="+", choices=["E2", "E2G"], default=None)
    s.add_argument("--models", nargs="+", choices=["lr", "svm", "rf", "xgb"], default=None)
    s.add_argument("--run", default=None)
    s.set_defaults(fn=cmd_ablate)
    s = sub.add_parser("explain", help="KG evidence for flagged calls (or one --call-id)")
    s.add_argument("--call-id", default=None)
    s.add_argument("--experiment", default="E2")
    s.add_argument("--model", default="xgb")
    s.add_argument("--max-calls", type=int, default=300)
    s.add_argument("--run", default=None)
    s.set_defaults(fn=cmd_explain)
    for name, fn, hlp in (("zeroshot", cmd_zeroshot, "LLM zero-shot classification baseline"),
                          ("relabel", cmd_relabel, "LLM-assisted fraudster/salesperson/prankster relabeling")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("--model", default=None)
        s.add_argument("--provider", default=None)
        s.add_argument("--limit", type=int, default=None,
                       help="a seeded, label-stratified sample of N calls (per evaluation set for zeroshot)")
        s.set_defaults(fn=fn)
    s = sub.add_parser("multiclass", help="caller type: fraudster / salesperson / prankster / other legitimate")
    s.add_argument("--run", default=None)
    s.add_argument("--experiments", nargs="+", default=None, choices=["E1", "E2", "E2G"])
    s.add_argument("--models", nargs="+", default=None, choices=["lr", "rf"])
    s.set_defaults(fn=cmd_multiclass)
    s = sub.add_parser("run-multiclass", help="caller-type pipeline end to end (--config config/multiclass.yaml)")
    s.add_argument("--run", default=None, help="existing extraction run for the KG (default: rule-based)")
    s.set_defaults(fn=cmd_run_multiclass)
    sub.add_parser("report", help="write results/REPORT.md").set_defaults(fn=cmd_report)
    sub.add_parser("feedback-report", help="write results/feedback/REPORT.md").set_defaults(fn=cmd_feedback_report)
    s = sub.add_parser("predict", help="score a new transcript with a trained model + KG explanation")
    s.add_argument("--text", default=None)
    s.add_argument("--file", default=None)
    s.add_argument("--experiment", default="E2", choices=["E1", "E2", "E2G"])
    s.add_argument("--model", default="xgb", choices=["lr", "svm", "rf", "xgb"])
    s.add_argument("--extractor", default=None, choices=["rule", "llm"],
                   help="default: the extractor that built the knowledge graph")
    s.set_defaults(fn=cmd_predict)
    s = sub.add_parser("banksim", help="optional BankSim graph baseline from prior projects")
    s.add_argument("--path", default=None)
    s.set_defaults(fn=cmd_banksim)
    sub.add_parser("dashboard", help="Streamlit results dashboard").set_defaults(fn=cmd_dashboard)
    s = sub.add_parser("run-all", help="the whole pipeline end-to-end")
    s.add_argument("--extractor", choices=["rule", "llm"], default="rule")
    s.add_argument("--model", nargs="+", default=None, help="LLM model(s) when --extractor llm")
    s.add_argument("--with-neo4j", action="store_true", help="compute graph features with Neo4j GDS")
    s.add_argument("--skip-deep", action="store_true", help="skip E3/E4 even if torch is installed")
    s.add_argument("--download", action="store_true", help="download the real data sets first")
    s.add_argument("--regenerate", action="store_true", help="demo: regenerate synthetic data")
    s.set_defaults(fn=cmd_run_all)
    return p


def main(argv: list[str] | None = None) -> int:
    # scikit-learn/joblib version-interplay noise from RandomForest's parallel predict (harmless)
    warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    set_seed(int(cfg["project"]["seed"]))
    args.fn(args, cfg)
    return 0
