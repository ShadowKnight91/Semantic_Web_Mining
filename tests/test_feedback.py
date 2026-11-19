"""Professor's feedback: constrained decoding, the cascade, the noise simulator, the recall monitor, script discovery."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from scamkg.data.noise import NOISE_LEVELS, add_asr_noise, noisy_calls
from scamkg.evaluation import monitor as mon
from scamkg.extraction.benchmark import build_cascade_run, cascade_records, run_config
from scamkg.extraction.extractor import LLMExtractor, parse_json_response, validate_triples
from scamkg.extraction.llm_client import BaseLLMClient, CachedLLMClient
from scamkg.models import script_discovery as sd
from scamkg.ontology.standardize import EntityStandardizer
from scamkg.utils import load_json

TEXT = ("caller: This is Officer Brown from the Social Security Administration. Press 1 to verify your number.\n"
        "receiver: Oh no, what do I need to do?\ncaller: Pay a fine with gift cards today. Call 1-888-555-0142 now.")


# ---------------------------------------------------------------------------- constrained decoding
def test_json_schema_ties_each_relation_to_its_type_and_values(ontology):
    schema = ontology.json_schema()
    json.dumps(schema)
    branches = schema["properties"]["triples"]["items"]["anyOf"]
    assert len(branches) == len(ontology.relations)
    for b in branches:
        rel = b["properties"]["relation"]["const"]
        target = ontology.target_of(rel)
        assert b["properties"]["object_type"]["const"] == target
        if ontology.is_closed(target):
            assert b["properties"]["object"]["enum"] == ontology.allowed_values(target)
        else:
            assert "enum" not in b["properties"]["object"]
    assert schema["properties"]["triples"]["maxItems"] >= 11   # the most facts Gemma found in one call


def test_compact_schema_drops_what_the_relation_already_implies(ontology):
    branch = ontology.json_schema(compact=True)["properties"]["triples"]["items"]["anyOf"][0]
    assert set(branch["properties"]) == {"relation", "object", "evidence", "inferred"}
    answer = ('{"triples": [{"relation": "CLAIMS_TO_BE", "object": "government_agent", "evidence": "Officer Brown",'
              ' "inferred": false}]}')
    triples, stats = validate_triples(parse_json_response(answer), ontology, EntityStandardizer(ontology))
    assert stats["conformant"] and triples[0]["object_type"] == "ClaimedIdentity" and triples[0]["raw_type"] == "ClaimedIdentity"


def test_compact_prompt_is_much_shorter_and_existing_prompts_are_unchanged(ontology):
    from scamkg.extraction.prompts import build_extraction_messages, prompt_fingerprint
    full = build_extraction_messages("x", ontology, "v1_full")[0]["content"]
    assert len(build_extraction_messages("x", ontology, "v4_compact")[0]["content"]) < 0.6 * len(full)
    # committed runs resume by this hash: a new prompt version must not change the old ones
    assert prompt_fingerprint(ontology, "v1_full") == "f7572cc124cb"


class _Client(BaseLLMClient):
    provider = "ollama"

    def __init__(self, format_schema=None):
        super().__init__("tiny:1b")
        self.format_schema = format_schema

    def chat(self, messages):
        return '{"triples": []}'


def test_cache_key_stable_without_schema_and_separate_with_one(tmp_path):
    msgs = [{"role": "user", "content": "x"}]
    plain = CachedLLMClient(_Client(), tmp_path)._key(msgs)
    # keys of unconstrained runs must not change: the committed runs replay from this cache
    from scamkg.utils import sha256_text
    expected = sha256_text(json.dumps({"p": "ollama", "m": "tiny:1b", "t": 0.1, "s": 42, "j": True, "n": 2048,
                                       "msg": msgs}, sort_keys=True))
    assert plain == expected
    assert CachedLLMClient(_Client({"type": "object"}), tmp_path)._key(msgs) != plain


def test_constrained_run_has_its_own_name(ontology):
    assert not LLMExtractor(_Client(), ontology).name.endswith("_constrained")
    assert LLMExtractor(_Client({"type": "object"}), ontology).name.endswith("_constrained")


def test_schema_shaped_answer_passes_validation_as_conformant(ontology):
    answer = ('{"triples": [{"relation": "CLAIMS_TO_BE", "object_type": "ClaimedIdentity", "object": "government_agent",'
              ' "detail": "", "evidence": "This is Officer Brown", "inferred": false}]}')
    triples, stats = validate_triples(parse_json_response(answer), ontology, EntityStandardizer(ontology))
    assert stats["conformant"] and triples[0]["object"] == "government_agent"


# ---------------------------------------------------------------------------- cascade
def _rec(triples, latency=0.0):
    return {"triples": [{"relation": r, "object": o} for r, o in triples], "latency_s": latency}


def test_cascade_escalates_only_low_coverage_calls():
    rule = {"a": _rec([("USES_TACTIC", "fear")], 0.01), "b": _rec([("REQUESTS", "ssn"), ("USES_TACTIC", "fear")], 0.01)}
    llm = {"a": _rec([("REQUESTS", "money")], 5.0), "b": _rec([("REQUESTS", "otp")], 5.0)}
    out = {r["call_id"]: r for r in cascade_records(rule, llm, min_triples=2)}
    assert out["a"]["escalated"] and not out["b"]["escalated"]
    assert out["a"]["triples"] == llm["a"]["triples"] and out["b"]["triples"] == rule["b"]["triples"]
    assert out["a"]["latency_s"] == pytest.approx(5.01) and out["b"]["latency_s"] == pytest.approx(0.01)


def test_cascade_union_keeps_rule_triples_without_duplicates():
    rule = {"a": _rec([("USES_TACTIC", "fear")])}
    llm = {"a": _rec([("USES_TACTIC", "fear"), ("REQUESTS", "money")])}
    triples = cascade_records(rule, llm, 2, mode="union")[0]["triples"]
    assert [t["object"] for t in triples] == ["fear", "money"]


def test_cascade_keeps_rules_when_the_llm_has_no_answer():
    out = cascade_records({"a": _rec([])}, {}, 3)[0]
    assert not out["escalated"] and out["triples"] == []


def test_cascade_run_files(demo_ctx):
    path = build_cascade_run(demo_ctx, "rule", "rule", 3)
    meta = load_json(path.with_suffix(".meta.json"))
    assert 0 <= meta["escalation_rate"] <= 1 and meta["min_triples"] == 3
    assert set(meta["escalation_by_dataset"]) == set(demo_ctx.calls["dataset"])
    path.unlink()
    path.with_suffix(".meta.json").unlink()


def test_run_config_keeps_committed_results_apart():
    cfg = {"paths": {"results": "results", "artifacts": "artifacts"}, "extraction": {"default_run": "x"}}
    out = run_config(cfg, "My Run")
    assert out["paths"]["results"] != "results" and out["paths"]["artifacts"] != "artifacts"
    assert out["extraction"]["default_run"] == "My Run" and cfg["paths"]["results"] == "results"


# ---------------------------------------------------------------------------- noise
def test_noise_is_deterministic_and_clean_is_untouched():
    assert add_asr_noise(TEXT, "clean", "a") == TEXT
    assert add_asr_noise(TEXT, "heavy", "a") == add_asr_noise(TEXT, "heavy", "a")
    assert add_asr_noise(TEXT, "heavy", "a") != add_asr_noise(TEXT, "heavy", "b")


def test_noise_looks_like_speech_recognition():
    out = add_asr_noise(TEXT, "light", "a")
    assert out == out.lower() and "." not in out.replace("1-888", "") and "," not in out
    assert all(line.startswith(("caller: ", "receiver: ")) for line in out.split("\n"))   # who said it survives
    assert "press 1" not in out


def test_noise_error_rate_grows_with_level():
    text = " ".join(["the caller asked about the account number today"] * 200)

    def errors(level):
        a, b = text.split(), add_asr_noise(text, level, "x").split()
        n = min(len(a), len(b))
        return sum(x != y for x, y in zip(a[:n], b[:n])) / n + abs(len(a) - len(b)) / len(a)
    rates = [errors(lv) for lv in ("light", "medium", "heavy")]
    assert rates[0] < rates[1] < rates[2]
    assert NOISE_LEVELS["heavy"][0] > NOISE_LEVELS["light"][0]


def test_noisy_calls_changes_only_the_requested_calls(demo_ctx):
    calls = demo_ctx.calls
    ext = calls.index[calls["split"] == "external"].tolist()[:5]
    out = noisy_calls(calls, ext, "medium", 42)
    other = calls.index.difference(ext)
    assert (out.loc[other, "text_plain"] == calls.loc[other, "text_plain"]).all()
    assert (out.loc[ext, "text_plain"] != calls.loc[ext, "text_plain"]).any()
    assert (calls.loc[ext, "text_plain"] != out.loc[ext, "text_plain"]).sum() > 0 and len(out) == len(calls)


# ---------------------------------------------------------------------------- monitor metrics
def test_psi_zero_for_same_scores_and_large_for_shifted():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 0.3, 2000)
    assert mon.psi(a, a) == pytest.approx(0, abs=1e-9)
    assert mon.psi(a, rng.uniform(0.7, 1, 2000)) > 0.25


def test_precision_falls_with_prevalence():
    assert mon.precision_at_prevalence(0.8, 0.1, 0.5) == pytest.approx(0.4 / 0.45)
    assert mon.precision_at_prevalence(0.8, 0.1, 0.01) < 0.1
    assert np.isnan(mon.precision_at_prevalence(0.0, 0.0, 0.5))


def test_recall_at_precision_never_claims_precision_a_sample_cannot_show():
    pos, neg = np.linspace(0.6, 1.0, 100), np.linspace(0.0, 0.5, 400)   # perfectly separated
    assert mon.recall_at_precision(pos, neg, 0.5, 0.8)[0] == pytest.approx(1.0)
    # no false alarms among 400 calls cannot prove 80% precision when 1 call in 1,000 is a scam
    assert np.isnan(mon.recall_at_precision(pos, neg, 0.001, 0.8)[0])


def test_status_needs_the_interval_below_the_floor_to_alert():
    mc = mon.DEFAULTS
    assert mon.status_for(0.95, 0.97, 0.03, mc) == "OK"
    assert mon.status_for(0.78, 0.85, 0.03, mc) == "WARN"
    assert mon.status_for(0.85, 0.90, 0.15, mc) == "WARN"
    assert mon.status_for(0.60, 0.70, 0.40, mc) == "ALERT"


def test_bootstrap_interval_contains_the_mean():
    hits = (np.random.default_rng(0).random(500) < 0.7).astype(float)
    low, high = mon.bootstrap_ci(hits)
    assert low < hits.mean() < high and high - low < 0.15


def test_pilot_recalibration_recovers_recall_lost_to_a_bad_threshold():
    rng = np.random.default_rng(0)
    pos, neg = np.clip(rng.normal(0.30, 0.05, 700), 0, 1), np.clip(rng.normal(0.10, 0.05, 400), 0, 1)
    assert (pos >= 0.5).mean() < 0.01                       # separable, but every score sits below the 0.5 threshold
    out = mon.pilot_recalibration(pos, neg, 100, 0.10, 100)
    assert out["recall"] > 0.9
    assert 0.03 < out["false_alarm_rate"] < 0.2


def test_first_sustained_ignores_a_single_noisy_window():
    assert mon._first_sustained([False, True, False, True, True]) == 3
    assert mon._first_sustained([False, True, False]) is None


# ---------------------------------------------------------------------------- monitor end to end
@pytest.fixture()
def scored_ctx(demo_ctx, tmp_path):
    ctx = copy.copy(demo_ctx)
    ctx.paths = copy.copy(demo_ctx.paths)
    ctx.paths.results = tmp_path / "results"
    ctx.paths.predictions.mkdir(parents=True)
    calls = ctx.calls
    rng = np.random.default_rng(1)
    for name in ("E1__lr", "E4__graphsage", "E4__graphsage_topo"):
        s = np.clip(calls["label"].to_numpy() * 0.5 + rng.uniform(0, 0.5, len(calls)), 0, 1)
        pd.DataFrame({"call_id": calls.index, "score": s}).to_csv(ctx.paths.predictions / f"{name}.csv", index=False)
    return ctx


def test_run_monitor_writes_every_table_and_reads_no_labels_in_drift(scored_ctx):
    ctx = scored_ctx
    (ctx.paths.results / "feedback").mkdir()
    out = mon.run_monitor(ctx)
    fb = ctx.paths.results / "feedback"
    for f in ("recall_ladder.csv", "recall_prevalence.csv", "recall_recalibration.csv", "drift_signals.csv",
              "recall_stream.csv", "recall_stream_summary.csv", "figures/recall_ladder.png",
              "figures/recall_stream.png"):
        assert (fb / f).exists(), f
    assert set(out["ladder"]["status"]) <= {"OK", "WARN", "ALERT"}
    first = out["ladder"][out["ladder"]["stage"] == "synthetic_test"]
    assert (first["recall_drop"] == 0).all()
    # the labels are flipped: every signal of the drift table must stay exactly the same
    flipped = copy.copy(ctx)
    flipped.calls = ctx.calls.copy()
    stages = mon.stage_sets(ctx)
    a = mon.drift_table(ctx, mon.load_scores(ctx), mon.monitor_config(ctx.cfg), mon.training_vocabulary(ctx))
    orig = mon.stage_sets
    try:
        mon.stage_sets = lambda _c: stages        # the same call sets; only the label column changes
        flipped.calls["label"] = 1 - flipped.calls["label"].clip(lower=0)
        b = mon.drift_table(flipped, mon.load_scores(ctx), mon.monitor_config(ctx.cfg), mon.training_vocabulary(ctx))
    finally:
        mon.stage_sets = orig
    pd.testing.assert_frame_equal(a, b)


def test_stream_baseline_matches_the_scam_rate_so_no_drift_means_no_warning(scored_ctx):
    ctx = scored_ctx
    mc = mon.monitor_config(ctx.cfg)
    scores = {"E1__lr": mon.load_scores(ctx)["E1__lr"]}
    df, summary = mon.stream_simulation(ctx, scores, mc, mon.training_vocabulary(ctx))
    assert df["window"].nunique() == mc["stream_windows"] and df["real_call_share"].iloc[-1] == 1.0
    # window 0 has no real calls: the same distribution as the baseline, whatever the class mix of the test split
    assert df["score_psi"].iloc[0] < mc["psi_alert"]


def test_noisy_context_only_changes_external_calls_and_keeps_labels(demo_ctx):
    n = mon.noisy_context(demo_ctx, "medium")
    tr = demo_ctx.train_ids
    assert (n.calls.loc[tr, "text"] == demo_ctx.calls.loc[tr, "text"]).all()
    assert (n.calls["label"] == demo_ctx.calls["label"]).all()
    ext = demo_ctx.calls.index[demo_ctx.calls["split"] == "external"]
    assert (n.calls.loc[ext, "text"] != demo_ctx.calls.loc[ext, "text"]).any()
    assert n.kg.n_calls == demo_ctx.kg.n_calls and demo_ctx.kg is not n.kg


# ---------------------------------------------------------------------------- script discovery
def test_dataset_mixing_runs_from_islands_to_even():
    ds = np.array(["a"] * 50 + ["b"] * 50)
    assert sd.dataset_mixing(np.array([0] * 50 + [1] * 50), ds) == pytest.approx(0)
    assert sd.dataset_mixing(np.array([0, 1] * 50), ds) == pytest.approx(1)


def test_dataset_mixing_corrects_for_data_set_size():
    ds = np.array(["a"] * 90 + ["b"] * 10)
    assert sd.dataset_mixing(np.zeros(100, dtype=int), ds) == pytest.approx(1)


def test_linked_share_counts_real_calls_next_to_training_calls():
    clusters = np.array([0, 0, 0, 1, 1, 1])
    is_train = np.array([True, False, False, False, False, False])
    assert sd.linked_share(clusters, is_train) == pytest.approx(2 / 5)


def test_script_discovery_end_to_end(scored_ctx, demo_ctx, tmp_path):
    from scamkg.models.gnn import embeddings_path
    ctx = scored_ctx
    (ctx.paths.results / "feedback").mkdir()
    rctx = copy.copy(demo_ctx)
    rctx.paths = copy.copy(ctx.paths)
    rctx.paths.artifacts = tmp_path / "art"
    rctx.paths.models.mkdir(parents=True)
    ids = np.array(rctx.kg.call_ids)
    rng = np.random.default_rng(0)
    # the first entity of a call decides its embedding: "scripts" here are shared facts, not words
    emb = rctx.kg.X.toarray()[:, :8].astype(np.float32) + rng.normal(0, 0.01, (len(ids), 8)).astype(np.float32)
    for v in ("text+topo", "topo", "kg", "unsup"):
        np.savez_compressed(embeddings_path(rctx, v), ids=ids, emb=emb)
    out = sd.run_script_discovery(ctx, rctx, k=4, ks=(4,))
    fb = ctx.paths.results / "feedback"
    for f in ("script_methods.csv", "scripts.csv", "script_rescue.csv", "script_summary.json", "figures/script_map.png"):
        assert (fb / f).exists(), f
    methods = out["methods"].set_index("method")
    assert {"GraphSAGE", "GraphSAGE, label-free", "GraphSAGE, KG facts", "Bag of words"} <= set(methods.index)
    assert out["scripts"]["calls"].sum() == (ctx.calls["label"] == 1).sum()
    # a space built from KG facts must group calls with shared facts better than a random baseline would
    assert methods.at["GraphSAGE", "kg_similarity_within_cluster"] > 0
    scripts = pd.read_csv(fb / "scripts.csv")
    assert ((scripts["bow_recall"].dropna() >= 0) & (scripts["bow_recall"].dropna() <= 1)).all()


# ---------------------------------------------------------------------------- train heavy, serve light
def test_serving_run_only_swaps_calls_scored_later(demo_ctx):
    from scamkg.extraction.benchmark import build_serving_run
    from scamkg.utils import read_jsonl, write_jsonl
    ex = demo_ctx.paths.extractions
    records = read_jsonl(ex / "rule.jsonl")
    light = [{**r, "triples": [], "extractor": "light"} for r in records]
    write_jsonl(ex / "light_test.jsonl", light)
    path = build_serving_run(demo_ctx, "rule", "light_test", name="serve_test")
    out = {r["call_id"]: r for r in read_jsonl(path)}
    external = set(demo_ctx.calls.index[demo_ctx.calls["split"] == "external"])
    assert all(out[c]["extractor"] == "light" for c in external)
    assert all(out[c]["extractor"] != "light" for c in demo_ctx.train_ids)
    meta = load_json(path.with_suffix(".meta.json"))
    assert meta["train_run"] == "rule" and meta["serve_run"] == "light_test"
    for p in (path, path.with_suffix(".meta.json"), ex / "light_test.jsonl"):
        p.unlink()


def test_serving_run_refuses_a_partial_light_run(demo_ctx):
    from scamkg.extraction.benchmark import build_serving_run
    from scamkg.utils import read_jsonl, write_jsonl
    ex = demo_ctx.paths.extractions
    records = [r for r in read_jsonl(ex / "rule.jsonl") if r["call_id"] in set(demo_ctx.train_ids)]
    write_jsonl(ex / "partial_test.jsonl", records)
    with pytest.raises(ValueError, match="no triples"):
        build_serving_run(demo_ctx, "rule", "partial_test")
    (ex / "partial_test.jsonl").unlink()


def test_extraction_noise_table_with_rules(demo_ctx, monkeypatch):
    from scamkg.extraction import benchmark as bm
    ctx = copy.copy(demo_ctx)
    ctx.cfg = {**demo_ctx.cfg, "extraction": {"default_run": "rule"}}
    monkeypatch.setattr(bm, "load_gold", lambda _c: {})
    df = bm.extraction_noise_table(ctx, None, n_each=6, levels=("clean", "heavy"))
    clean = df[df["noise"] == "clean"]
    assert set(df["extractor"]) == {"keyword rules"}
    assert (clean["agree_with_own_clean_f1"] == 1).all()     # rules on clean text are the reference itself
    assert (clean["agree_with_gemma_clean_f1"] == 1).all()
    heavy = df[df["noise"] == "heavy"]
    assert heavy["agree_with_own_clean_f1"].mean() < 1


def test_gold_f1_interval_contains_the_point_estimate():
    from scamkg.extraction.benchmark import gold_f1_interval
    from scamkg.extraction.evaluate import compare_to_gold
    rng = np.random.default_rng(0)
    gold, pred = {}, {}
    for i in range(50):
        g = [{"relation": "R", "object": f"o{j}"} for j in range(4)]
        gold[str(i)] = [{**t, "object_type": "T"} for t in g]
        pred[str(i)] = [{**t, "object_type": "T"} for t in g if rng.random() < 0.7]
    low, high = gold_f1_interval(pred, gold)
    point = compare_to_gold(pred, gold)["triple_f1"]
    assert low < point < high and high - low < 0.3


def test_quiet_stream_has_no_real_calls(scored_ctx):
    ctx = scored_ctx
    mc = mon.monitor_config(ctx.cfg)
    df, _ = mon.stream_simulation(ctx, {"E1__lr": mon.load_scores(ctx)["E1__lr"]}, mc, mon.training_vocabulary(ctx),
                                  max_share=0.0)
    assert (df["real_call_share"] == 0).all()


def test_seed_table_reports_every_variant_saved(scored_ctx, tmp_path):
    ctx = scored_ctx
    ids = ctx.calls.index.tolist()
    rng = np.random.default_rng(0)
    wide = {"call_id": ids}
    for v in ("text+topo", "topo", "kg"):
        for seed in (42, 1):
            wide[f"{v}|{seed}"] = rng.random(len(ids))
    path = tmp_path / "seeds.csv"
    pd.DataFrame(wide).to_csv(path, index=False)
    out = sd.seed_table(ctx, path, mon.load_scores(ctx)["E1__lr"])
    assert {"Bag of words", "GraphSAGE", "GraphSAGE, structure only", "GraphSAGE, KG facts"} <= set(out["model"])
    assert (out.loc[out["model"] == "GraphSAGE", "seeds"] == 2).all()


def test_derived_runs_can_be_named_but_are_never_the_default(tmp_path):
    from scamkg.extraction.extractor import list_runs, resolve_run
    from scamkg.utils import write_jsonl
    rows = [{"call_id": str(i), "triples": []} for i in range(3)]
    write_jsonl(tmp_path / "real_run.jsonl", rows)
    write_jsonl(tmp_path / "serve__real_run.jsonl", rows)      # newer and just as large
    write_jsonl(tmp_path / "cascade_k3_escalate__real_run.jsonl", rows)
    assert [p.stem for p in list_runs(tmp_path)] == ["real_run"]
    assert resolve_run(tmp_path).stem == "real_run"
    assert resolve_run(tmp_path, "serve__real_run").stem == "serve__real_run"
