"""X1 - extraction quality (RQ2).

Gold standard: two annotators hand-annotate triples for 50 transcripts
(data/annotations/annotator_A.jsonl, annotator_B.jsonl; adjudicated file gold.jsonl if present).
For every extraction run we report triple precision / recall / F1 (micro + macro), entity-type
accuracy, JSON validity rate and schema-conformance rate, plus inter-annotator agreement.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from scamkg.extraction.extractor import list_runs, summarize_records, validate_triples
from scamkg.ontology.schema import Ontology
from scamkg.ontology.standardize import EntityStandardizer, match_key
from scamkg.utils import get_logger, load_json, read_jsonl, save_json, write_jsonl

log = get_logger(__name__)

ANNOTATION_GUIDE = """# Annotation guide (X1: extraction quality)

Each line of `annotator_A.jsonl` / `annotator_B.jsonl` is one call. Read the transcript in
`sample.jsonl` and fill `triples` with every fact the CALLER states, using only the ontology:

| relation | object type | object |
|---|---|---|
| CLAIMS_TO_BE | ClaimedIdentity | one allowed value, e.g. `government_agent`, `bank`, `tech_support` |
| MENTIONS_ORG | Organization | the organization name, e.g. `Social Security Administration` |
| USES_TACTIC | PressureTactic | `urgency`, `threat`, `reward`, `fear`, `secrecy`, `other` |
| REQUESTS | RequestedAsset | `otp`, `ssn`, `card_number`, `bank_account`, `password`, `remote_access`, `money`, `personal_info`, `other` |
| PAY_VIA | PaymentChannel | `gift_card`, `wire_transfer`, `crypto`, `zelle`, `p2p_app`, `bank_transfer`, `cash`, `check_or_money_order`, `credit_card_payment`, `other` |
| PROVIDES_NUMBER | CallbackNumber | the number as given, e.g. `1-888-555-0142`, `press 1` |

Example line:
```json
{"call_id": "bothbosu_00012", "annotator": "A", "done": true, "triples": [
  {"relation": "CLAIMS_TO_BE", "object": "government_agent"},
  {"relation": "USES_TACTIC", "object": "threat", "inferred": false}]}
```
Set `"done": true` on every call you have annotated - also when the caller states no ontology facts
at all (e.g. a wrong number): such a call stays a gold example with no triples, so any triple an
extractor invents for it counts as a false positive. Calls with `"done": false` and no triples are
treated as not yet annotated. Full allowed values and descriptors: `ontology/ontology.yaml`.

Decision rules (used for annotator A; follow them so agreement measures the extractors, not the rules):
* CLAIMS_TO_BE - who the caller presents as. "Officer X from the SSA" = `government_agent`; a caller from an
  insurance company = `insurance`; other sellers = `telemarketer_sales`; a prize / gift-card giveaway =
  `prize_or_lottery` (if the caller also claims another type, e.g. "we're a marketing firm", annotate both).
  A private person (wrong number, a customer calling a company) gets no identity; `relative_or_friend` only
  for the receiver's own family or friends.
* MENTIONS_ORG - every company or agency the caller names, placeholders ("XYZ Company") and brands (Dell)
  included; not generic references ("your bank"), not organizations only the receiver names, and "social
  security number" is not a mention of the SSA.
* USES_TACTIC - `urgency`: explicit time pressure or a deadline (today, immediately, within 48 hours, limited
  time); `threat`: arrest, legal action, suspension, fines, account closure; `reward`: refunds, prizes, free
  gifts, discounts, special deals (sales calls too); `fear`: suspicious activity, unauthorized charges, hacked /
  compromised accounts, viruses; `secrecy`: an explicit instruction to stay on the line or not to contact / call
  back the real organization. A delivery arriving "today" is not urgency.
* REQUESTS - only what the caller asks the receiver for: `ssn` (also the last four digits), `personal_info`
  (name, address, date of birth, email, phone number, policy details, "whose number is this?"),
  `card_number`, `bank_account`, `password`, `otp`, `remote_access` (also downloading a "support tool" or being
  talked through system settings - mark it inferred), `money` (fees, fines, payments).
* PAY_VIA - only how money is meant to move; a gift card that is the prize is not a payment channel, and a
  refund "deposited into your account" is `bank_transfer`.
* PROVIDES_NUMBER - phone numbers, extensions and "press N" options the caller gives; not badge, order,
  tracking or case numbers, and not websites.
* Mark `"inferred": true` when a fact is implied rather than stated.

Annotate independently; afterwards
adjudicate disagreements into `gold.jsonl` (same format). `python -m scamkg eval-extraction`
reports agreement and scores every extraction run.
"""


def create_annotation_sample(calls: pd.DataFrame, out_dir: Path, n: int = 50, seed: int = 42,
                             primary: str | None = None, from_synthetic: bool = False) -> Path:
    """Sample calls for annotation (stratified by label) and write empty annotator templates."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = calls[calls["split"].isin(["test", "val"])] if primary is None else calls[calls["dataset"] == primary]
    if len(pool) < n:
        pool = calls
    parts = []
    for _, grp in pool.groupby("label"):
        k = max(1, round(n * len(grp) / len(pool)))
        parts.append(grp.sample(min(k, len(grp)), random_state=seed))
    sample = pd.concat(parts).head(n).sort_index()
    write_jsonl(out_dir / "sample.jsonl", ({"call_id": cid, "dataset": r["dataset"], "label": int(r["label"]),
                                             "text": r["text"]} for cid, r in sample.iterrows()))
    for annot in ("A", "B"):
        path = out_dir / f"annotator_{annot}.jsonl"
        if not path.exists():
            write_jsonl(path, ({"call_id": cid, "annotator": annot, "done": False, "triples": []}
                               for cid in sample.index))
    (out_dir / "README.md").write_text(ANNOTATION_GUIDE, encoding="utf-8")
    if from_synthetic and "gold_triples" in sample.columns:
        write_jsonl(out_dir / "gold.jsonl", ({"call_id": cid, "annotator": "synthetic", "done": True,
                                               "triples": r["gold_triples"] or []} for cid, r in sample.iterrows()))
        log.info("Wrote synthetic gold triples for %d calls", len(sample))
    log.info("Annotation sample of %d calls written to %s", len(sample), out_dir)
    return out_dir / "sample.jsonl"


def load_annotations(path: Path, ontology: Ontology, standardizer: EntityStandardizer) -> dict[str, list[dict]]:
    """Annotated calls only: a record counts once it has triples or is marked `"done": true` (a call whose
    caller states no ontology facts, e.g. a wrong number, is a valid gold example with no triples)."""
    out = {}
    for rec in read_jsonl(path):
        if not rec.get("triples") and not rec.get("done"):
            continue
        raw = []
        for t in rec.get("triples", []):
            t = dict(t)
            t.setdefault("object_type", ontology.target_of(str(t.get("relation", "")).upper()))
            raw.append(t)
        triples, _ = validate_triples({"triples": raw}, ontology, standardizer)
        out[rec["call_id"]] = triples
    return out


def _key(t: dict) -> tuple[str, str]:
    return t["relation"], str(t["object"]).lower()


def compare_to_gold(pred: dict[str, list[dict]], gold: dict[str, list[dict]]) -> dict:
    """Triple-level P/R/F1 (micro and macro over calls), per relation, and entity-type accuracy."""
    tp = fp = fn = 0
    per_rel = defaultdict(lambda: [0, 0, 0])
    macro_p, macro_r = [], []
    type_total = type_correct = 0
    for cid, gtriples in gold.items():
        g = {_key(t) for t in gtriples}
        p = {_key(t) for t in pred.get(cid, [])}
        inter = g & p
        tp += len(inter)
        fp += len(p - g)
        fn += len(g - p)
        for rel, _ in inter:
            per_rel[rel][0] += 1
        for rel, _ in p - g:
            per_rel[rel][1] += 1
        for rel, _ in g - p:
            per_rel[rel][2] += 1
        macro_p.append(len(inter) / len(p) if p else (1.0 if not g else 0.0))
        macro_r.append(len(inter) / len(g) if g else 1.0)
        # entity-type accuracy judges the type the extractor itself gave (before schema repair) for objects
        # that also occur in the gold triples; "other" exists in several types, so it identifies nothing
        gold_types = {match_key(t["object"]): t["object_type"] for t in gtriples if t["object"] != "other"}
        for t in pred.get(cid, []):
            k = match_key(t["object"])
            if k in gold_types:
                type_total += 1
                type_correct += int(gold_types[k] == t.get("raw_type", t["object_type"]))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    per_relation = {}
    for rel, (a, b, c) in per_rel.items():
        p_ = a / (a + b) if a + b else 0.0
        r_ = a / (a + c) if a + c else 0.0
        per_relation[rel] = {"precision": p_, "recall": r_, "f1": 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0.0,
                             "support": a + c}
    return {"n_calls": len(gold), "n_gold": tp + fn, "n_pred": tp + fp, "triple_precision": prec,
            "triple_recall": rec, "triple_f1": f1, "macro_precision": float(np.mean(macro_p)) if macro_p else 0.0,
            "macro_recall": float(np.mean(macro_r)) if macro_r else 0.0,
            "entity_type_accuracy": type_correct / type_total if type_total else float("nan"),
            "per_relation": per_relation}


def evaluate_extraction_runs(extractions_dir: Path, annotations_dir: Path, ontology: Ontology,
                             results_dir: Path, gold_path: Path | None = None) -> pd.DataFrame:
    standardizer = EntityStandardizer(ontology)
    annotations_dir = Path(annotations_dir)
    gold_path = Path(gold_path) if gold_path else annotations_dir / "gold.jsonl"
    a_path, b_path = annotations_dir / "annotator_A.jsonl", annotations_dir / "annotator_B.jsonl"
    report: dict = {}
    gold: dict[str, list[dict]] = {}
    if gold_path.exists():
        gold = load_annotations(gold_path, ontology, standardizer)
    elif a_path.exists():
        gold = load_annotations(a_path, ontology, standardizer)
        log.info("No gold.jsonl - using annotator A as reference")
    if a_path.exists() and b_path.exists():
        a = load_annotations(a_path, ontology, standardizer)
        b = load_annotations(b_path, ontology, standardizer)
        common = {k: v for k, v in a.items() if k in b}
        if common:
            agree = compare_to_gold({k: b[k] for k in common}, common)
            report["inter_annotator"] = {"calls": len(common), "triple_f1": agree["triple_f1"],
                                         "precision_B_vs_A": agree["triple_precision"],
                                         "recall_B_vs_A": agree["triple_recall"]}
            log.info("Inter-annotator agreement (triple F1): %.3f on %d calls", agree["triple_f1"], len(common))
    if not gold:
        log.warning("No annotated gold triples found in %s - reporting run statistics only", annotations_dir)

    rows = []
    for run_path in list_runs(extractions_dir):
        records = read_jsonl(run_path)
        meta_path = run_path.with_suffix(".meta.json")
        meta = load_json(meta_path) if meta_path.exists() else {}
        summ = summarize_records(records)
        row = {"run": run_path.stem, "model": meta.get("model", meta.get("extractor", "")),
               "prompt_version": meta.get("prompt_version", ""), "n_calls": summ.get("n_calls", 0),
               "json_validity_rate": summ.get("json_validity_rate"),
               "schema_conformance_rate": summ.get("schema_conformance_rate"),
               "mean_triples_per_call": summ.get("mean_triples_per_call"),
               "repaired_rate": summ.get("repaired_rate"), "inferred_rate": summ.get("inferred_rate"),
               "mean_latency_s": summ.get("mean_latency_s")}
        if gold:
            pred = {r["call_id"]: r.get("triples", []) for r in records}
            covered = {k: v for k, v in gold.items() if k in pred}
            if covered:
                cmp = compare_to_gold(pred, covered)
                row.update({k: cmp[k] for k in ("n_calls", "triple_precision", "triple_recall", "triple_f1",
                                                 "macro_precision", "macro_recall", "entity_type_accuracy")})
                row["n_calls"] = summ.get("n_calls", 0)
                row["gold_calls"] = cmp["n_calls"]
                report.setdefault("per_relation", {})[run_path.stem] = cmp["per_relation"]
        rows.append(row)
    df = pd.DataFrame(rows)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "extraction_quality.csv", index=False)
    save_json(results_dir / "extraction_quality.json", {"runs": rows, **report})
    return df
