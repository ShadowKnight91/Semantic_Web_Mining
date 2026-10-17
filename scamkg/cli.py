"""Command-line interface:  python -m scamkg [--config FILE] <command> [options]

Pipeline order (or just `run-all`):
  download | demo-data -> preprocess -> extract -> build-graph -> eda -> annotate-sample -> eval-extraction
  -> train (E1 E2 E2G E3 E4 E5 LLM) -> cv -> significance -> ablate -> explain -> report -> dashboard
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
    models = args.model or [None]
    versions = args.prompt_version or [None]
    if getattr(args, "no_fallback", False):
        cfg = {**cfg, "llm": {**cfg["llm"], "fallback_model": None}}
    for model in (models if args.extractor == "llm" else [None]):
        for version in (versions if args.extractor == "llm" else [None]):
            ex = make_extractor(args.extractor, cfg, ctx.ontology, cache_dir=ctx.paths.cache, model=model,
                                provider=args.provider, prompt_version=version)
            run_name = args.run_name if (args.run_name and len(models) * len(versions) == 1) else None
            # rule extraction takes seconds -> always recompute; LLM runs resume (responses are also cached)
            resume = not args.no_resume and args.extractor == "llm"
            run_extraction(calls, ex, ctx.paths.extractions, run_name=run_name, resume=resume,
                           max_workers=args.workers or int(cfg["llm"].get("max_workers", 1)), limit=args.limit)


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
    ctx.attach_graph(str(run_path), structural=structural)
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


def cmd_train(args, cfg):
    from scamkg.context import Context
    exps = args.experiments or ["E1", "E2", "E2G", "E5"] + (["E3", "E4"] if has_module("torch") else [])
    ctx = Context.load(cfg, with_graph=any(e in exps for e in ("E2", "E2G", "E4", "E5")), run=args.run)
    classical = [e for e in exps if e in ("E1", "E2", "E2G")]
    if classical:
        from scamkg.models.classical import run_classical
        run_classical(ctx, classical, args.models, tune=not args.no_tune)
    if "E3" in exps:
        from scamkg.models.transformer import run_distilbert, run_sbert_mlp
        if _deep_ok("torch", "sentence_transformers"):
            run_sbert_mlp(ctx)
        if _deep_ok("torch", "transformers"):
            run_distilbert(ctx)
    if "E4" in exps and _deep_ok("torch"):
        from scamkg.models.gnn import run_graphsage
        run_graphsage(ctx)
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


def cmd_significance(args, cfg):
    from scamkg.context import Context
    from scamkg.evaluation.evaluator import run_significance
    df = run_significance(Context.load(cfg, with_graph=False))
    if not df.empty:
        print(df.to_string(index=False))


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
    run_multiclass(ctx)


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


def cmd_predict(args, cfg):
    from scamkg.predict import ScamDetector
    text = Path(args.file).read_text(encoding="utf-8") if args.file else args.text
    if not text:
        raise SystemExit("Give --text or --file")
    det = ScamDetector(cfg, args.experiment, args.model, args.extractor)
    res = det.score(text)
    ex = res["explanation"]
    print(json.dumps({"score": round(res["score"], 4), "is_scam": res["is_scam"], "model": res["model"],
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
    cmd_extract(ns(extractor=args.extractor, datasets=None, splits=None, model=args.model, prompt_version=None,
                   provider=None, run_name=None, no_resume=False, workers=None, limit=None, no_fallback=False), cfg)
    cmd_build_graph(ns(run=None, backend="neo4j" if args.with_neo4j else None, no_rdf=False), cfg)
    cmd_eda(ns(run=None), cfg)
    cmd_annotate_sample(ns(n=50, from_synthetic=demo), cfg)
    cmd_eval_extraction(ns(gold=None), cfg)
    exps = ["E1", "E2", "E2G", "E5"]
    if not args.skip_deep and has_module("torch"):
        exps += ["E3", "E4"]
    cmd_train(ns(experiments=exps, models=None, no_tune=False, run=None, limit=None), cfg)
    cmd_cv(ns(experiments=None, models=None, folds=None, run=None), cfg)
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
    sub.add_parser("preprocess", help="clean, dedupe, unify labels, create splits").set_defaults(fn=cmd_preprocess)
    s = sub.add_parser("eda", help="exploratory data analysis")
    s.add_argument("--run", default=None, help="extraction run for entity statistics")
    s.set_defaults(fn=cmd_eda)

    s = sub.add_parser("extract", help="ontology-guided triple extraction (rule-based or LLM)")
    s.add_argument("--extractor", choices=["rule", "llm"], default="rule")
    s.add_argument("--provider", default=None, help="ollama | voyager (default: config llm.provider)")
    s.add_argument("--model", nargs="+", default=None, help="one or more LLM names, e.g. qwen2.5:7b gemma2:9b")
    s.add_argument("--prompt-version", nargs="+", default=None,
                   help="v1_full | v2_no_descriptors | v3_no_example | v0_open (several = prompt ablation)")
    s.add_argument("--datasets", nargs="+", default=None)
    s.add_argument("--splits", nargs="+", default=None)
    s.add_argument("--limit", type=int, default=None, help="only the first N pending calls (pilot run)")
    s.add_argument("--workers", type=int, default=None)
    s.add_argument("--run-name", default=None)
    s.add_argument("--no-resume", action="store_true")
    s.add_argument("--no-fallback", action="store_true",
                   help="disable llm.fallback_model (use for the model / prompt comparisons of RQ2)")
    s.set_defaults(fn=cmd_extract)

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
    s.add_argument("--limit", type=int, default=None, help="LLM zero-shot: calls per evaluation set")
    s.set_defaults(fn=cmd_train)
    s = sub.add_parser("cv", help="5-fold stratified cross-validation (mean +- std)")
    s.add_argument("--experiments", nargs="+", choices=["E1", "E2", "E2G"], default=None)
    s.add_argument("--models", nargs="+", choices=["lr", "svm", "rf", "xgb"], default=None)
    s.add_argument("--folds", type=int, default=None)
    s.add_argument("--run", default=None)
    s.set_defaults(fn=cmd_cv)
    sub.add_parser("significance", help="McNemar tests E1 vs E2/E2G").set_defaults(fn=cmd_significance)
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
        s.add_argument("--limit", type=int, default=None)
        s.set_defaults(fn=fn)
    s = sub.add_parser("multiclass", help="three-way classification (stretch goal)")
    s.add_argument("--run", default=None)
    s.set_defaults(fn=cmd_multiclass)
    sub.add_parser("report", help="write results/REPORT.md").set_defaults(fn=cmd_report)
    s = sub.add_parser("predict", help="score a new transcript with a trained model + KG explanation")
    s.add_argument("--text", default=None)
    s.add_argument("--file", default=None)
    s.add_argument("--experiment", default="E2", choices=["E1", "E2", "E2G"])
    s.add_argument("--model", default="xgb", choices=["lr", "svm", "rf", "xgb"])
    s.add_argument("--extractor", default="rule", choices=["rule", "llm"])
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
