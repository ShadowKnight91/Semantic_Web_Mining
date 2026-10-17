"""LLM zero-shot classification (reference point) and LLM-assisted three-way relabeling (stretch goal)."""
from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from scamkg.evaluation.evaluator import Evaluator
from scamkg.extraction.extractor import parse_json_response, truncate_transcript
from scamkg.extraction.llm_client import LLMError, make_client
from scamkg.extraction.prompts import CLASSIFY_PROMPT_VERSION, build_classification_messages
from scamkg.utils import dumps, get_logger, read_jsonl, slugify

log = get_logger(__name__)
CATEGORIES = {"fraudster", "salesperson", "prankster", "other_legitimate"}
# CHANGE: an unusable answer (request failed / no JSON) used to get scam_probability 0.5, and the 0.5
# threshold (score >= 0.5) then counted it as a scam detection, inflating zero-shot recall. It now gets
# 0.0 = "no alert", the honest outcome for a detector that did not answer; valid_rate is reported.
NO_ANSWER_SCORE = 0.0


def classify(client, text: str, max_chars: int = 6000) -> dict:
    raw = None
    try:
        raw = client.chat(build_classification_messages(truncate_transcript(text, max_chars)))
    except LLMError as e:
        return {"valid": False, "error": str(e), "scam_probability": NO_ANSWER_SCORE, "category": "", "label": ""}
    obj = parse_json_response(raw)
    if not isinstance(obj, dict):
        return {"valid": False, "raw": (raw or "")[:500], "scam_probability": NO_ANSWER_SCORE, "category": "",
                "label": ""}
    label = str(obj.get("label", "")).strip().lower()
    default = 1.0 if label in ("scam", "fraud") else 0.0 if label == "legitimate" else NO_ANSWER_SCORE
    try:
        p = float(obj.get("scam_probability", obj.get("confidence", default)))
    except (TypeError, ValueError):
        p = default
    p = min(1.0, max(0.0, p))
    if "scam_probability" not in obj and "confidence" in obj:
        # a confidence in the stated label, not a scam probability. CHANGE: before, only "scam" answers
        # were converted, so "legitimate" with confidence 0.9 was scored as a 0.9 scam probability.
        if label in ("scam", "fraud"):
            p = max(p, 1 - p)
        elif label == "legitimate":
            p = 1 - max(p, 1 - p)
    cat = str(obj.get("category", "")).strip().lower()
    return {"valid": label in ("scam", "legitimate"), "label": label, "scam_probability": p,
            "category": cat if cat in CATEGORIES else "", "reason": str(obj.get("reason", ""))[:300]}


def _run(ctx, ids: list[str], out_path, model, provider) -> dict[str, dict]:
    llm = ctx.cfg["llm"]
    client = make_client(ctx.cfg, cache_dir=ctx.paths.cache, model=model, provider=provider)
    # CHANGE: answers that failed with a request error (timeout, server down) are asked again on resume
    done = {r["call_id"]: r for r in read_jsonl(out_path) if not r.get("error")}
    todo = [c for c in ids if c not in done]
    with open(out_path, "a", encoding="utf-8") as f:
        for cid in tqdm(todo, desc=f"llm[{client.model}]", disable=len(todo) < 10):
            rec = {"call_id": cid, "model": client.model, "prompt_version": CLASSIFY_PROMPT_VERSION,
                   **classify(client, ctx.calls.at[cid, "text"], int(llm.get("max_chars", 6000)))}
            f.write(dumps(rec) + "\n")
            f.flush()
            done[cid] = rec
    return done


def run_zeroshot(ctx, model: str | None = None, provider: str | None = None, limit: int | None = None) -> None:
    ev = Evaluator(ctx)
    ids = []
    for name, set_ids in ev.eval_sets().items():
        ids.extend(set_ids[:limit] if limit else set_ids)
    ids = list(dict.fromkeys(ids))
    model = model or ctx.cfg["llm"]["model"]
    out = ctx.paths.results / f"zeroshot__{slugify(model)}.jsonl"
    recs = _run(ctx, ids, out, model, provider)
    scores = pd.Series({c: recs[c]["scam_probability"] for c in ids})
    validity = sum(recs[c]["valid"] for c in ids) / max(1, len(ids))
    log.info("Zero-shot %s: valid JSON answers %.3f", model, validity)
    ev.evaluate("LLM", f"zeroshot_{slugify(model)}", scores, extra={"valid_rate": validity})


def run_relabel(ctx, model: str | None = None, provider: str | None = None, limit: int | None = None) -> pd.DataFrame:
    """Three-way labels: scams stay 'fraudster'; the LLM splits non-scams into salesperson / prankster / other."""
    ids = ctx.ids(labeled=True)
    legit = [c for c in ids if ctx.calls.at[c, "label"] == 0]
    if limit:
        legit = legit[:limit]
    model = model or ctx.cfg["llm"]["model"]
    recs = _run(ctx, legit, ctx.paths.results / f"relabel__{slugify(model)}.jsonl", model, provider)
    rows = []
    for c in ids:
        if ctx.calls.at[c, "label"] == 1:
            rows.append({"call_id": c, "label3": "fraudster", "llm_category": "", "disagreement": False})
        elif c in recs:
            cat = recs[c].get("category") or "other_legitimate"
            rows.append({"call_id": c, "label3": cat if cat != "fraudster" else "other_legitimate",
                         "llm_category": cat, "disagreement": cat == "fraudster"})
    df = pd.DataFrame(rows)
    path = ctx.paths.processed / "label3_llm.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in df.to_dict(orient="records"):
            f.write(dumps(r) + "\n")
    log.info("Three-way labels -> %s: %s", path, df["label3"].value_counts().to_dict())
    return df
