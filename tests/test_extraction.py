import json

from scamkg.extraction.extractor import LLMExtractor, parse_json_response, validate_triples
from scamkg.extraction.prompts import EXAMPLE_OUTPUT, build_extraction_messages
from scamkg.extraction.rule_based import RuleBasedExtractor
from scamkg.ontology.standardize import EntityStandardizer

PAPER_EXAMPLE = ("This is Officer Brown from the Social Security Administration. Your SSN has been suspended due to "
                 "suspicious activity. To avoid arrest today, you must verify your number and pay a fine using "
                 "Google Play gift cards.")


def _pairs(triples):
    return {(t["relation"], t["object"]) for t in triples}


def test_rule_extractor_on_proposal_example(ontology):
    rec = RuleBasedExtractor(ontology).extract("call_0412", PAPER_EXAMPLE)
    got = _pairs(rec.triples)
    expected = {("CLAIMS_TO_BE", "government_agent"), ("MENTIONS_ORG", "Social Security Administration"),
                ("USES_TACTIC", "threat"), ("USES_TACTIC", "urgency"), ("REQUESTS", "ssn"),
                ("PAY_VIA", "gift_card")}
    assert expected <= got, expected - got


def test_rule_extractor_ignores_receiver_and_negations(ontology):
    text = ("caller: Hi, this is Sarah from Chase. We will never ask for your PIN or password over the phone.\n"
            "receiver: I bought gift cards yesterday, is that a problem?")
    got = _pairs(RuleBasedExtractor(ontology).extract("c1", text).triples)
    assert ("CLAIMS_TO_BE", "bank") in got
    assert ("REQUESTS", "password") not in got
    assert ("PAY_VIA", "gift_card") not in got


def test_parse_json_response_variants():
    assert parse_json_response('```json\n{"triples": []}\n```') == {"triples": []}
    assert parse_json_response('Sure! {"triples": [{"relation": "REQUESTS"}]} Hope this helps.')["triples"]
    assert parse_json_response("not json") is None


def test_validate_repairs_and_drops(ontology):
    s = EntityStandardizer(ontology)
    raw = {"triples": [
        {"relation": "claims to be", "object_type": "ClaimedIdentity", "object": "IRS agent"},
        {"relation": "REQUESTS", "object_type": "RequestedAsset", "object": "gift card"},   # wrong type -> repaired
        {"relation": "LIKES", "object": "pizza"},                                             # unknown -> dropped
        {"relation": "PAY_VIA", "object_type": "PaymentChannel", "object": "gift_card"},     # duplicate after repair
        {"relation": "USES_TACTIC", "object_type": "PressureTactic", "object": "urgency", "inferred": "true"},
    ]}
    triples, stats = validate_triples(raw, ontology, s)
    got = _pairs(triples)
    assert ("CLAIMS_TO_BE", "government_agent") in got
    assert ("PAY_VIA", "gift_card") in got
    assert len([t for t in triples if t["relation"] == "PAY_VIA"]) == 1
    assert stats["n_dropped"] == 1 and stats["n_repaired"] >= 1 and not stats["conformant"]
    assert next(t for t in triples if t["object"] == "urgency")["inferred"] is True
    # the model's own (wrong) type is kept for X1 entity-type accuracy
    assert next(t for t in triples if t["mention"] == "gift card")["raw_type"] == "RequestedAsset"


def test_annotations_done_flag_and_entity_type_accuracy(ontology, tmp_path):
    from scamkg.extraction.evaluate import compare_to_gold, load_annotations
    path = tmp_path / "gold.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [
        {"call_id": "scam", "triples": [{"relation": "PAY_VIA", "object": "gift_card"}]},
        {"call_id": "wrong_number", "done": True, "triples": []},     # annotated: no caller facts
        {"call_id": "todo", "done": False, "triples": []},            # not annotated yet
    ]), encoding="utf-8")
    gold = load_annotations(path, ontology, EntityStandardizer(ontology))
    assert gold.keys() == {"scam", "wrong_number"}
    pred = {"scam": [{"relation": "PAY_VIA", "object_type": "PaymentChannel", "object": "gift_card",
                      "raw_type": "RequestedAsset"}],
            "wrong_number": [{"relation": "USES_TACTIC", "object_type": "PressureTactic", "object": "urgency"}]}
    res = compare_to_gold(pred, gold)
    assert res["triple_precision"] == 0.5 and res["triple_recall"] == 1.0   # the invented triple is a FP
    assert res["entity_type_accuracy"] == 0.0                              # judged on the raw type


def test_prompt_example_is_schema_conformant(ontology):
    triples, stats = validate_triples(EXAMPLE_OUTPUT, ontology, EntityStandardizer(ontology))
    assert stats["conformant"] and stats["n_dropped"] == 0
    assert len(triples) == len(EXAMPLE_OUTPUT["triples"])
    msgs = build_extraction_messages("caller: hello", ontology, "v1_full")
    assert "CLAIMS_TO_BE" in msgs[0]["content"] and "caller: hello" in msgs[1]["content"]


class FakeClient:
    provider, model, temperature, seed, json_mode = "fake", "fake-llm", 0.1, 42, True

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        return self.responses.pop(0)

    def describe(self):
        return {"provider": self.provider, "model": self.model}


def test_llm_extractor_retries_on_invalid_json(ontology):
    good = json.dumps({"triples": [{"relation": "PAY_VIA", "object_type": "PaymentChannel", "object": "Zelle"}]})
    client = FakeClient(["I think the answer is...", good])
    rec = LLMExtractor(client, ontology, max_retries=2).extract("c1", "caller: pay me with Zelle")
    assert rec.json_valid and rec.attempts == 2 and client.calls == 2
    assert _pairs(rec.triples) == {("PAY_VIA", "zelle")}


def test_grounding_drops_copied_example(ontology):
    """A small model that parrots the few-shot example must not inject its triples."""
    client = FakeClient([json.dumps(EXAMPLE_OUTPUT)])
    text = "caller: Hi, this is Dr. Lee's office, confirming your appointment on Monday. receiver: Thanks!"
    rec = LLMExtractor(client, ontology).extract("c1", text)
    assert rec.json_valid and rec.triples == [] and rec.n_ungrounded == len(EXAMPLE_OUTPUT["triples"])
    # the same output on the example transcript itself is fully grounded
    from scamkg.extraction.prompts import EXAMPLE_TRANSCRIPT
    rec2 = LLMExtractor(FakeClient([json.dumps(EXAMPLE_OUTPUT)]), ontology).extract("c2", EXAMPLE_TRANSCRIPT)
    assert rec2.n_ungrounded == 0 and len(rec2.triples) == len(EXAMPLE_OUTPUT["triples"])


def test_llm_extractor_gives_up(ontology):
    client = FakeClient(["nope", "still nope", "no"])
    rec = LLMExtractor(client, ontology, max_retries=2).extract("c1", "caller: hi")
    assert not rec.json_valid and rec.triples == [] and rec.attempts == 3


# ---------------------------------------------------------------- fallback / retry logic (TA comment 1)
def test_salvages_truncated_json(ontology):
    """Answer cut off at the token cap: the complete triples are kept, the unfinished one is dropped."""
    cut = ('{"triples": [{"relation": "PAY_VIA", "object_type": "PaymentChannel", "object": "gift_card", '
           '"evidence": "pay with gift cards"}, {"relation": "REQUESTS", "object_type": "RequestedAs')
    rec = LLMExtractor(FakeClient([cut]), ontology, max_retries=0).extract("c1", "caller: pay with gift cards now")
    assert rec.status == "salvaged" and rec.salvaged and not rec.json_valid
    assert _pairs(rec.triples) == {("PAY_VIA", "gift_card")}


def test_retries_when_every_triple_is_off_schema(ontology):
    """Valid JSON whose triples all miss the ontology -> schema-specific retry, not silent loss."""
    bad = json.dumps({"triples": [{"predicate": "LIKES", "value": "pizza"}, {"relation": "EATS", "object": "cake"}]})
    good = json.dumps({"triples": [{"relation": "PAY_VIA", "object_type": "PaymentChannel", "object": "Zelle"}]})
    client = FakeClient([bad, good])
    seen = []
    orig = client.chat
    client.chat = lambda m: (seen.append(m[-1]["content"]), orig(m))[1]
    rec = LLMExtractor(client, ontology, max_retries=2).extract("c1", "caller: send it by Zelle")
    assert client.calls == 2 and rec.status == "retried" and "off_schema" in rec.retry_reasons
    assert "PAY_VIA -> PaymentChannel" in seen[-1] and "unknown relation" in seen[-1]
    assert _pairs(rec.triples) == {("PAY_VIA", "zelle")}


def test_honest_empty_answer_is_not_retried(ontology):
    client = FakeClient([json.dumps({"triples": []})])
    rec = LLMExtractor(client, ontology, max_retries=2).extract("c1", "caller: see you at dinner")
    assert client.calls == 1 and rec.status == "empty" and rec.json_valid


def test_fallback_model_after_primary_fails(ontology):
    good = json.dumps({"triples": [{"relation": "REQUESTS", "object_type": "RequestedAsset", "object": "otp",
                                    "evidence": "read me the code"}]})
    primary = FakeClient(["nope", "still nope", "no"])
    fallback = FakeClient([good])
    fallback.model = "fallback-llm"
    rec = LLMExtractor(primary, ontology, max_retries=2, fallback_client=fallback).extract(
        "c1", "caller: read me the code we sent you")
    assert rec.status == "fallback" and rec.fallback_used and rec.model_used == "fallback-llm"
    assert rec.attempts == 4 and _pairs(rec.triples) == {("REQUESTS", "otp")}


def test_failed_calls_are_retried_on_resume(ontology, tmp_path):
    import pandas as pd
    from scamkg.extraction.extractor import read_jsonl, run_extraction
    from scamkg.extraction.llm_client import LLMError

    class Flaky(FakeClient):
        def chat(self, messages):
            self.calls += 1
            if self.calls == 1:
                raise LLMError("timed out")
            return json.dumps({"triples": []})

    calls = pd.DataFrame({"text": ["caller: hello there friend", "caller: good morning to you"]}, index=["a", "b"])
    ex = LLMExtractor(Flaky([]), ontology, max_retries=0)
    out = run_extraction(calls, ex, tmp_path, run_name="r", resume=True)
    first = {r["call_id"]: r["status"] for r in read_jsonl(out)}
    assert first == {"a": "failed", "b": "empty"}
    run_extraction(calls, ex, tmp_path, run_name="r", resume=True)       # only "a" is redone
    from scamkg.extraction.extractor import latest_records, summarize_records
    recs = latest_records(read_jsonl(out))
    assert {r["call_id"]: r["status"] for r in recs} == {"a": "empty", "b": "empty"}
    assert summarize_records(read_jsonl(out))["n_calls"] == 2


def test_annotated_empty_calls_count_in_gold(ontology, tmp_path):
    """A call annotated as fact-free must stay in the gold set, so extra predicted triples are false positives."""
    from scamkg.extraction.evaluate import compare_to_gold, load_annotations
    from scamkg.utils import write_jsonl
    path = tmp_path / "annotator_A.jsonl"
    write_jsonl(path, [
        {"call_id": "a", "annotated": True, "triples": []},
        {"call_id": "b", "triples": [{"relation": "PAY_VIA", "object": "gift_card"}]},
        {"call_id": "c", "annotated": False, "triples": []},              # not done yet -> skipped
    ])
    gold = load_annotations(path, ontology, EntityStandardizer(ontology))
    assert set(gold) == {"a", "b"} and gold["a"] == []
    pred = {"a": [{"relation": "REQUESTS", "object": "otp", "object_type": "RequestedAsset"}],
            "b": [{"relation": "PAY_VIA", "object": "gift_card", "object_type": "PaymentChannel"}]}
    assert compare_to_gold(pred, gold)["triple_precision"] == 0.5


def test_bare_ivr_digit_is_recovered_from_evidence(ontology):
    raw = {"triples": [
        {"relation": "PROVIDES_NUMBER", "object_type": "CallbackNumber", "object": "1", "evidence": "press 1 to start your claim"},
        {"relation": "PROVIDES_NUMBER", "object_type": "CallbackNumber", "object": "42", "evidence": "dial extension 42"},
        {"relation": "PROVIDES_NUMBER", "object_type": "CallbackNumber", "object": "7", "evidence": "it takes 7 days"},
    ]}
    triples, stats = validate_triples(raw, ontology, EntityStandardizer(ontology))
    assert _pairs(triples) == {("PROVIDES_NUMBER", "ivr:press-1"), ("PROVIDES_NUMBER", "ext-42")}
    assert stats["n_dropped"] == 1   # a bare number with no IVR / extension context is still dropped
