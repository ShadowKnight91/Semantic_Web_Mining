"""End-to-end smoke test on a tiny synthetic data set (no LLM, no Neo4j, no torch needed)."""
import pandas as pd

from scamkg.evaluation.explain import explain_call
from scamkg.extraction.evaluate import create_annotation_sample, evaluate_extraction_runs
from scamkg.graph.rdf_export import build_rdf_graph, run_sparql_file
from scamkg.config import ROOT
from scamkg.evaluation.evaluator import Evaluator
from scamkg.models.classical import run_classical, run_cv, run_loto
from scamkg.models.clustering import run_clustering


def test_preprocess_splits(demo_ctx):
    calls = demo_ctx.calls
    prim = calls[calls["dataset"] == "demo_primary"]
    assert set(prim["split"]) == {"train", "val", "test"}
    assert (calls[calls["dataset"] != "demo_primary"]["split"] == "external").all()
    assert calls["text"].str.contains("caller:").any()
    assert calls["call_id"].is_unique


def test_classical_e1_e2_and_cv(demo_ctx):
    run_classical(demo_ctx, ["E1", "E2", "E2G"], ["lr"], tune=True)
    m = pd.read_csv(demo_ctx.paths.results / "metrics.csv")
    for exp in ("E1", "E2", "E2G"):
        row = m[(m["experiment"] == exp) & (m["model"] == "lr") & (m["eval_set"] == "test")]
        assert len(row) == 1 and row["pr_auc"].iloc[0] > 0.6
    assert {"test_imbalanced", "demo_shifted", "demo_robocall", "demo_robocall+demo_shifted"} <= set(m["eval_set"])
    assert (demo_ctx.paths.models / "E2__lr.joblib").exists()
    agg = run_cv(demo_ctx, ["E1", "E2"], ["lr"], n_folds=3)
    assert len(agg) == 2 and (agg["pr_auc_mean"] > 0.6).all()


def test_clustering_and_explain(demo_ctx):
    res = run_clustering(demo_ctx)
    assert res["kmeans_best"]["k"] >= 2 and 0 <= res["kmeans_best"]["nmi"] <= 1
    assert res["scripts"]
    scam = next(c for c in demo_ctx.test_ids if demo_ctx.calls.at[c, "label"] == 1 and demo_ctx.kg.triples[c])
    ex = explain_call(demo_ctx, scam, 0.9)
    assert ex["entities"] and ex["dot"].startswith("digraph")


def test_extraction_quality_with_synthetic_gold(demo_ctx):
    create_annotation_sample(demo_ctx.calls, demo_ctx.paths.annotations, n=20, primary=demo_ctx.primary,
                             from_synthetic=True)
    df = evaluate_extraction_runs(demo_ctx.paths.extractions, demo_ctx.paths.annotations, demo_ctx.ontology,
                                  demo_ctx.paths.results)
    row = df.iloc[0]
    assert row["json_validity_rate"] == 1.0
    assert 0.3 < row["triple_precision"] <= 1.0 and 0.3 < row["triple_recall"] <= 1.0


def test_rdf_export_and_sparql(demo_ctx):
    g = build_rdf_graph(demo_ctx.kg, demo_ctx.calls, demo_ctx.ontology)
    assert len(g) > 1000
    results = run_sparql_file(g, ROOT / "ontology" / "queries.sparql")
    assert len(results) == 4
    title, df = results[1]
    assert not df.empty


def test_leave_one_type_out(demo_ctx):
    df = run_loto(demo_ctx, ["E1", "E2G"], ["lr"])
    assert set(df["held_out_label"]) == {0, 1}
    assert df["held_out_type"].nunique() == demo_ctx.calls.query("dataset == 'demo_primary'")["subtype"].nunique()
    assert df["held_out_flag_rate"].between(0, 1).all()
    assert (demo_ctx.paths.results / "loto_summary.csv").exists()


def test_mixed_external_uses_only_held_out_primary_calls(demo_ctx):
    import copy
    ctx = copy.copy(demo_ctx)
    ctx.cfg = copy.deepcopy(demo_ctx.cfg)
    ctx.cfg["data"]["mixed_external"] = [["demo_robocall", "demo_primary"]]
    ids = Evaluator(ctx).eval_sets()["demo_robocall+demo_primary_test"]
    neg = [i for i in ids if ctx.calls.at[i, "dataset"] == "demo_primary"]
    assert neg and set(ctx.calls.loc[neg, "split"]) == {"test"} and (ctx.calls.loc[neg, "label"] == 0).all()


def test_pilot_limit_samples_across_types(demo_ctx, tmp_path):
    """`--limit` must not just take the first N rows: data files are often sorted by scam type."""
    from scamkg.extraction.extractor import make_extractor, run_extraction
    from scamkg.utils import read_jsonl
    ex = make_extractor("rule", demo_ctx.cfg, demo_ctx.ontology)
    calls = demo_ctx.calls.sort_values("subtype")
    first = [r["call_id"] for r in read_jsonl(run_extraction(calls, ex, tmp_path / "a", limit=20, seed=1))]
    again = [r["call_id"] for r in read_jsonl(run_extraction(calls, ex, tmp_path / "b", limit=20, seed=1))]
    assert len(first) == 20 and first == again
    assert calls.loc[first, "subtype"].nunique() > 3


def test_zeroshot_sample_is_stratified(demo_ctx):
    from scamkg.models.llm_zeroshot import sample_ids
    ids = demo_ctx.calls.sort_values(["label", "subtype"]).index.tolist()   # sorted file: one class first
    pick = sample_ids(demo_ctx, ids, 20, seed=0)
    assert len(pick) == 20 and set(demo_ctx.labels(pick)) == {0, 1}
    assert pick == sample_ids(demo_ctx, ids, 20, seed=0) and sample_ids(demo_ctx, ids, None, 0) == ids


def test_run_all_end_to_end(tmp_path):
    """The run-all entry point wires every command together (catches argument-plumbing errors)."""
    from scamkg.cli import main
    from scamkg.config import get_paths, load_config
    override = tmp_path / "cfg.yaml"
    override.write_text(
        "paths: {" + ", ".join(f"{k}: '{tmp_path / k}'" for k in ("data_raw", "data_processed", "kg", "cache",
                                                                    "annotations", "results", "artifacts")) + "}\n"
        "data: {load: [demo], primary: demo_primary, external: [demo_shifted, demo_robocall],"
        " mixed_external: [[demo_robocall, demo_shifted]], imbalanced_test: {n_draws: 3}}\n"
        "models: {search_iter: 1, classical: [lr], cv_folds: 2}\n"
        "graph: {betweenness_samples: 30}\n", encoding="utf-8")
    from scamkg.data.synthetic import generate_demo_data
    cfg = load_config(override)
    generate_demo_data(get_paths(cfg).ensure().raw, n_primary=120, n_shifted=40, n_robocall=20, seed=3)
    assert main(["--config", str(override), "run-all", "--skip-deep"]) == 0
    report = (tmp_path / "results" / "REPORT.md").read_text(encoding="utf-8")
    assert "Leave-one-type-out" in report and "Cross-dataset generalization" in report
