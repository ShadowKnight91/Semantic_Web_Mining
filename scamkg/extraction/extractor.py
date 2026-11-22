"""Extraction runs: LLM extractor with JSON parsing, schema validation, retry and standardization.

Output of a run: data/kg/extractions/<run>.jsonl (one record per call) + <run>.meta.json
(model name, provider, prompt version + hash, seed, temperature, summary statistics).
Each stored triple: {relation, object_type, object (canonical), mention, detail, evidence, inferred, repaired}.
"""
from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from scamkg.extraction.llm_client import LLMError
from scamkg.extraction.prompts import (PROMPT_VERSIONS, RETRY_MESSAGE, RETRY_WRAPPER_MESSAGE,
                                       build_extraction_messages, prompt_fingerprint, schema_retry_message)
from scamkg.ontology.schema import Ontology
from scamkg.ontology.standardize import EntityStandardizer, match_key
from scamkg.utils import dumps, get_logger, load_json, read_jsonl, save_json, slugify, timestamp

log = get_logger(__name__)


@dataclass
class ExtractionRecord:
    call_id: str
    extractor: str
    triples: list[dict] = field(default_factory=list)
    json_valid: bool = True
    schema_conformant: bool = True
    attempts: int = 1
    n_raw: int = 0
    n_dropped: int = 0
    n_repaired: int = 0
    n_ungrounded: int = 0
    latency_s: float = 0.0
    cached: bool = False
    error: str | None = None
    raw_response: str | None = None
    # CHANGE (TA comment 1): record what the fallback logic did for every call, so the report can show
    # JSON-repair, retry, fallback and failure rates instead of a single validity number.
    # status: ok | repaired | salvaged | fallback | empty | failed
    status: str = "ok"
    retry_reasons: list = field(default_factory=list)
    salvaged: bool = False
    fallback_used: bool = False
    model_used: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class BaseExtractor:
    name = "base"

    def extract(self, call_id: str, text: str) -> ExtractionRecord:
        raise NotImplementedError

    def metadata(self) -> dict:
        return {"extractor": self.name}

    @property
    def run_name(self) -> str:
        return slugify(self.name)


# ---------------------------------------------------------------------------- parsing / validation
def parse_json_response(text: str | None) -> Any | None:
    """Parse an LLM answer into JSON, tolerating code fences and surrounding prose."""
    if not text:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = s.find(open_c), s.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(s[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


def salvage_triples(text: str | None) -> dict | None:
    """Recover the complete triple objects from a cut-off answer.

    CHANGE (TA comment 1): when the model hits the output-token cap (max_tokens / num_predict) the
    JSON ends mid-object, e.g. '{"triples": [{...}, {...}, {"relation": "PAY_'. json.loads fails
    on the whole answer, so every triple was lost before. Here each complete {...} inside the
    "triples" array is parsed on its own and the unfinished last one is dropped.
    """
    if not text:
        return None
    start = text.find('"triples"')
    start = text.find("[", start) if start != -1 else text.find("[")
    if start == -1:
        return None
    objs, depth, obj_start, in_str, esc = [], 0, None, False, False
    for i in range(start + 1, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and obj_start is not None:
                try:
                    objs.append(json.loads(text[obj_start:i + 1]))
                except json.JSONDecodeError:
                    pass
                obj_start = None
        elif ch == "]" and depth == 0:
            break
    return {"triples": objs} if objs else None


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"true", "yes", "1", "y"}
    return bool(v) if isinstance(v, (int, float)) else False


def _expand_short_number(mention: str, evidence) -> str:
    """LLMs often give an IVR option ("press 1") or an extension as the bare digit; recover it from the
    evidence so it standardizes like the rule-based extractor's output (ivr:press-1, ext-42)."""
    if mention.isdigit() and len(mention) <= 6:
        m = re.search(rf"\b(press|extension|ext\.?)\s*{re.escape(mention)}\b", str(evidence or ""), re.IGNORECASE)
        if m:
            return f"{m.group(1)} {mention}"
    return mention


def _norm_relation(rel) -> str:
    return re.sub(r"[\s\-]+", "_", str(rel or "").strip()).upper()


def validate_triples(obj: Any, ontology: Ontology, standardizer: EntityStandardizer) -> tuple[list[dict], dict]:
    """Validate raw LLM output against the ontology, repair what is safely repairable, standardize entities.

    Returns (triples, stats) where stats has n_raw, n_dropped, n_repaired and `conformant`
    (True when every raw triple already used a valid relation, the right object type and, for
    closed types, an allowed value).
    """
    # CHANGE (TA comment 1): `problems` lists why each triple was dropped; it feeds the schema-specific retry.
    stats = {"n_raw": 0, "n_dropped": 0, "n_repaired": 0, "conformant": True, "problems": []}
    if isinstance(obj, dict):
        raw = obj.get("triples")
        if raw is None:
            raw = next((v for v in obj.values() if isinstance(v, list)), None)
            stats["conformant"] = False
    elif isinstance(obj, list):
        raw = obj
        stats["conformant"] = False
    else:
        raw = None
    if not isinstance(raw, list):
        return [], {**stats, "conformant": False}

    out: dict[tuple[str, str], dict] = {}
    for t in raw:
        stats["n_raw"] += 1
        if not isinstance(t, dict):
            stats["n_dropped"] += 1
            stats["conformant"] = False
            stats["problems"].append("a triple is not a JSON object")
            continue
        rel = _norm_relation(t.get("relation") or t.get("predicate"))
        otype = str(t.get("object_type") or "").strip() or None
        mention = t.get("object")
        if isinstance(mention, (list, dict)) or mention is None or not str(mention).strip():
            stats["n_dropped"] += 1
            stats["conformant"] = False
            stats["problems"].append("missing or non-text 'object' field")
            continue
        mention = str(mention).strip()
        # the type the model itself assigned (stated, else implied by its relation) - for X1 type accuracy
        raw_type = otype if otype in ontology.entity_types else ontology.target_of(rel)
        repaired = False
        conform = True
        if rel not in ontology.relations:
            conform = False
            fixed = ontology.relation_for_target(otype) if otype in ontology.entity_types else None
            if fixed is None:
                stats["n_dropped"] += 1
                stats["conformant"] = False
                stats["problems"].append(f"unknown relation {rel or '(missing)'!r}" +
                                         ("" if otype in ontology.entity_types else f" and object_type {otype!r}"))
                continue
            rel, repaired = fixed, True
        target = ontology.target_of(rel)
        if otype is not None and otype != target:
            conform = False
        if not standardizer.is_exact_value(target, mention):
            conform = False
        canon = None
        if ontology.is_closed(target) and match_key(mention) not in standardizer.lookup.get(target, {}):
            # an exact vocabulary term of *another* type is strong evidence of a mistyped triple
            alt = standardizer.infer_type(mention, exclude=target)
            if alt is not None:
                target, canon = alt
                rel = ontology.relation_for_target(target)
                repaired = True
        if canon is None:
            if target == "CallbackNumber":
                mention = _expand_short_number(mention, t.get("evidence"))
            canon = standardizer.standardize(target, mention)
        if canon is None:
            alt = standardizer.infer_type(mention, exclude=target)
            if alt is not None:
                target, canon = alt
                rel = ontology.relation_for_target(target)
                repaired = True
        if canon is None:
            stats["n_dropped"] += 1
            stats["conformant"] = False
            stats["problems"].append(f"value {mention[:40]!r} is not allowed for {target}")
            continue
        if not conform:
            stats["conformant"] = False
        stats["n_repaired"] += int(repaired)
        key = (rel, canon)
        triple = {"relation": rel, "object_type": target, "object": canon, "mention": mention[:200],
                  "detail": str(t.get("detail") or "")[:120], "evidence": str(t.get("evidence") or "")[:300],
                  "inferred": _as_bool(t.get("inferred", False)), "repaired": repaired,
                  "raw_type": raw_type or target}
        if key in out:
            # keep the stated (non-inferred) version when the same fact appears twice
            if out[key]["inferred"] and not triple["inferred"]:
                out[key] = triple
        else:
            out[key] = triple
    return list(out.values()), stats


_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> list[str]:
    return _TOKEN.findall(str(s).lower())


def ground_triples(triples: list[dict], transcript: str, min_overlap: float = 0.8) -> tuple[list[dict], int]:
    """Drop hallucinated triples (e.g. copied from the few-shot example): keep a triple only when its
    evidence quote (>= min_overlap of its tokens) or its mention occurs in the transcript. Call-back
    numbers must appear digit-for-digit."""
    text_tokens = set(_tokens(transcript))
    text_flat = " ".join(_tokens(transcript))
    text_digits = re.sub(r"\D", "", transcript)
    kept, dropped = [], 0
    for t in triples:
        if t["object_type"] == "CallbackNumber":
            digits = re.sub(r"\D", "", t["object"])
            ok = t["object"].startswith(("ivr:", "ext-")) or (digits and digits[-7:] in text_digits)
        else:
            ev, mention = _tokens(t.get("evidence", "")), _tokens(t.get("mention", ""))
            ev_ok = bool(ev) and (" ".join(ev) in text_flat or
                                  sum(w in text_tokens for w in ev) / len(ev) >= min_overlap)
            mention_ok = bool(mention) and all(w in text_tokens for w in mention)
            ok = ev_ok or mention_ok
        if ok:
            kept.append(t)
        else:
            dropped += 1
    return kept, dropped


def truncate_transcript(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.7)
    tail = max_chars - head
    return text[:head] + "\n[...]\n" + text[-tail:]


# ---------------------------------------------------------------------------- LLM extractor
class LLMExtractor(BaseExtractor):
    """Ontology-guided LLM extraction with a layered fallback (TA comment 1).

    For every transcript:
      1. Ask for JSON (Ollama `format: json` / OpenAI `response_format`), temperature 0.1, fixed seed.
      2. Parse: strip code fences / prose; if the answer was cut off at the token cap, salvage the
         complete triple objects (`salvage_triples`).
      3. Validate every triple against the ontology (`validate_triples`): missing fields -> dropped;
         unknown relation -> repaired from object_type when possible, else dropped; wrong object type or
         a value of another closed vocabulary -> re-typed; free text -> standardized to the vocabulary.
      4. Retry (up to `max_retries`) with feedback for the specific failure:
           invalid JSON            -> RETRY_MESSAGE
           JSON without "triples"  -> RETRY_WRAPPER_MESSAGE
           all triples off-schema  -> schema_retry_message (allowed relations + what was wrong)
         An honest empty answer {"triples": []} is accepted (legitimate calls may state no facts).
      5. If the primary model still gives nothing usable (or the request fails), run the same loop once
         with `fallback_model` (e.g. gemma2:9b when the primary is qwen2.5:7b).
      6. Grounding check drops triples whose evidence/mention is not in the transcript.
      7. A call that still fails keeps status "failed" and no triples: it stays in the data set as a
         Call node with zero graph features (has_entities = 0), so no call is silently dropped, and
         `run_extraction` retries it on the next resume.
    """

    def __init__(self, client, ontology: Ontology, standardizer: EntityStandardizer | None = None,
                 prompt_version: str = "v1_full", max_retries: int = 2, max_chars: int = 6000,
                 grounding: bool = True, fallback_client=None, schema_retry: bool = True):
        if prompt_version not in PROMPT_VERSIONS:
            raise ValueError(f"Unknown prompt version {prompt_version}")
        self.grounding = grounding
        self.client = client
        self.fallback_client = fallback_client
        self.schema_retry = schema_retry
        self.ontology = ontology
        self.standardizer = standardizer or EntityStandardizer(ontology)
        self.prompt_version = prompt_version
        self.max_retries = max_retries
        self.max_chars = max_chars
        self.name = f"llm_{client.provider}_{client.model}_{prompt_version}"
        if getattr(client, "format_schema", None) is not None:
            self.name += "_constrained"
        if fallback_client is not None:   # own run name, so it never mixes with a single-model comparison run
            self.name += f"_fb-{fallback_client.model}"

    # -------------------------------------------------------------- one model, with retries
    def _run_model(self, client, messages: list[dict]) -> dict:
        res = {"obj": None, "triples": [], "stats": None, "raw": None, "attempts": 0, "error": None,
               "cached": True, "salvaged": False, "reasons": [], "usable": False}
        best = None   # a parsed-but-imperfect answer kept in case the retries do not improve on it
        while res["attempts"] <= self.max_retries:
            res["attempts"] += 1
            try:
                raw = client.chat(messages)
                res["cached"] = res["cached"] and bool(getattr(client, "last_cached", False))
            except LLMError as e:
                res["error"], res["cached"] = str(e), False
                break
            res["raw"] = raw
            parsed, salvaged = parse_json_response(raw), False
            if parsed is None:
                parsed = salvage_triples(raw)
                salvaged = parsed is not None
            if isinstance(parsed, dict) and isinstance(parsed.get("triples"), list):
                triples, stats = validate_triples(parsed, self.ontology, self.standardizer)
                cand = {"obj": parsed, "triples": triples, "stats": stats, "salvaged": salvaged}
                if triples or stats["n_raw"] == 0 or not self.schema_retry:
                    res.update(cand, usable=True, error=None)
                    if salvaged:
                        res["reasons"].append("truncated_json_salvaged")
                    return res
                # the model answered, but nothing fits the ontology -> schema-specific retry
                res["reasons"].append("off_schema")
                best = best or cand
                feedback = schema_retry_message(self.ontology, stats["problems"])
            elif parsed is not None:
                res["reasons"].append("missing_triples_key")
                if isinstance(parsed, list):   # a bare list of triples: usable if retries do not fix it
                    triples, stats = validate_triples(parsed, self.ontology, self.standardizer)
                    best = best or {"obj": parsed, "triples": triples, "stats": stats, "salvaged": False}
                feedback = RETRY_WRAPPER_MESSAGE
            else:
                res["reasons"].append("invalid_json")
                feedback = RETRY_MESSAGE
            messages = messages + [{"role": "assistant", "content": (raw or "")[:1500]},
                                   {"role": "user", "content": feedback}]
        if best is not None:
            res.update(best, usable=bool(best["triples"]), error=None)
        return res

    def extract(self, call_id: str, text: str) -> ExtractionRecord:
        messages = build_extraction_messages(truncate_transcript(text, self.max_chars), self.ontology,
                                             self.prompt_version)
        t0 = time.time()
        res = self._run_model(self.client, messages)
        model_used, fallback_used = self.client.model, False
        attempts, reasons = res["attempts"], list(res["reasons"])
        if not res["usable"] and self.fallback_client is not None:
            reasons.append("fallback_model")
            fb = self._run_model(self.fallback_client, messages)
            attempts += fb["attempts"]
            reasons += fb["reasons"]
            if fb["usable"] or (res["obj"] is None and fb["obj"] is not None):
                res, model_used, fallback_used = fb, self.fallback_client.model, True
        obj, stats = res["obj"], res["stats"] or {"n_raw": 0, "n_dropped": 0, "n_repaired": 0, "conformant": False}
        triples = res["triples"]
        n_ungrounded = 0
        if self.grounding:
            triples, n_ungrounded = ground_triples(triples, text)
        if obj is None:
            status = "failed"
        elif fallback_used:
            status = "fallback"
        elif res["salvaged"]:
            status = "salvaged"
        elif res["attempts"] > 1:
            status = "retried"
        elif stats["n_repaired"]:
            status = "repaired"
        elif not triples:
            status = "empty"
        else:
            status = "ok"
        json_valid = obj is not None and not res["salvaged"]
        return ExtractionRecord(
            call_id=call_id, extractor=self.name, triples=triples, json_valid=json_valid,
            schema_conformant=json_valid and stats["conformant"], attempts=attempts, n_raw=stats["n_raw"],
            n_dropped=stats["n_dropped"], n_repaired=stats["n_repaired"], n_ungrounded=n_ungrounded,
            latency_s=round(time.time() - t0, 3), cached=res["cached"], error=res["error"] if obj is None else None,
            raw_response=(res["raw"] or "")[:4000], status=status, retry_reasons=reasons,
            salvaged=res["salvaged"], fallback_used=fallback_used, model_used=model_used)

    def metadata(self) -> dict:
        return {"extractor": "llm", **self.client.describe(), "prompt_version": self.prompt_version,
                "prompt_hash": prompt_fingerprint(self.ontology, self.prompt_version),
                "ontology_version": self.ontology.version, "ontology_hash": self.ontology.fingerprint,
                "max_retries": self.max_retries, "max_chars": self.max_chars, "grounding": self.grounding,
                "schema_retry": self.schema_retry,
                "fallback_model": self.fallback_client.model if self.fallback_client is not None else None}


def make_extractor(kind: str, cfg: dict, ontology: Ontology, cache_dir: Path | None = None,
                   model: str | None = None, provider: str | None = None,
                   prompt_version: str | None = None, constrained: bool = False) -> BaseExtractor:
    standardizer = EntityStandardizer(ontology)
    if kind == "rule":
        from scamkg.extraction.rule_based import RuleBasedExtractor
        return RuleBasedExtractor(ontology, standardizer)
    if kind == "llm":
        from scamkg.extraction.llm_client import make_client
        llm = cfg["llm"]
        version = prompt_version or llm.get("prompt_version", "v1_full")
        compact = PROMPT_VERSIONS.get(version, {}).get("compact", False)
        client = make_client(cfg, cache_dir=cache_dir, model=model, provider=provider,
                             format_schema=ontology.json_schema(compact=compact) if constrained else None)
        fallback = None
        fb_model = llm.get("fallback_model")
        if fb_model and fb_model != client.model:
            fallback = make_client(cfg, cache_dir=cache_dir, model=fb_model,
                                   provider=llm.get("fallback_provider") or provider)
        return LLMExtractor(client, ontology, standardizer, version, int(llm.get("max_retries", 2)), int(llm.get("max_chars", 6000)),
                            grounding=bool(llm.get("grounding", True)), fallback_client=fallback,
                            schema_retry=bool(llm.get("schema_retry", True)))
    raise ValueError(f"Unknown extractor {kind!r} (use 'rule' or 'llm')")


# ---------------------------------------------------------------------------- running + IO
def latest_records(records: list[dict]) -> list[dict]:
    """One record per call: the last one written (a resumed run appends retries of failed calls)."""
    return list({r["call_id"]: r for r in records}.values())


def _is_failed(r: dict) -> bool:
    return r.get("status") == "failed" or (bool(r.get("error")) and not r.get("triples"))


def summarize_records(records: list[dict]) -> dict:
    records = latest_records(records)
    n = len(records)
    if n == 0:
        return {"n_calls": 0}
    triples = [t for r in records for t in r.get("triples", [])]
    per_rel: dict[str, int] = {}
    for t in triples:
        per_rel[t["relation"]] = per_rel.get(t["relation"], 0) + 1
    n_raw = sum(r.get("n_raw", 0) for r in records)
    fresh = [r for r in records if not r.get("cached")]   # replays from the response cache take ~0 s
    return {
        "n_calls": n,
        "json_validity_rate": sum(bool(r.get("json_valid")) for r in records) / n,
        "schema_conformance_rate": sum(bool(r.get("schema_conformant")) for r in records) / n,
        "error_rate": sum(bool(r.get("error")) for r in records) / n,
        "mean_triples_per_call": len(triples) / n,
        "calls_without_triples": sum(1 for r in records if not r.get("triples")) / n,
        "mean_attempts": sum(r.get("attempts", 1) for r in records) / n,
        "raw_triples": n_raw,
        "dropped_rate": (sum(r.get("n_dropped", 0) for r in records) / n_raw) if n_raw else 0.0,
        "repaired_rate": (sum(r.get("n_repaired", 0) for r in records) / n_raw) if n_raw else 0.0,
        "ungrounded_rate": (sum(r.get("n_ungrounded", 0) for r in records) / n_raw) if n_raw else 0.0,
        "inferred_rate": (sum(bool(t.get("inferred")) for t in triples) / len(triples)) if triples else 0.0,
        "mean_latency_s": sum(r.get("latency_s", 0.0) for r in fresh) / len(fresh) if fresh else None,
        "cached_rate": 1 - len(fresh) / n,
        # how often each fallback layer was needed (TA comment 1)
        "retry_rate": sum(r.get("attempts", 1) > 1 for r in records) / n,
        "salvage_rate": sum(bool(r.get("salvaged")) for r in records) / n,
        "fallback_rate": sum(bool(r.get("fallback_used")) for r in records) / n,
        "failed_rate": sum(_is_failed(r) for r in records) / n,
        "status_counts": pd.Series([r.get("status", "ok") for r in records]).value_counts().to_dict(),
        "triples_per_relation": per_rel,
    }


def run_extraction(calls: pd.DataFrame, extractor: BaseExtractor, out_dir: Path, run_name: str | None = None,
                   resume: bool = True, max_workers: int = 1, limit: int | None = None, seed: int = 42) -> Path:
    """Extract triples for every call (column `text`) and append them to <out_dir>/<run>.jsonl.
    `limit` takes a seeded random sample of the pending calls (a pilot run): the data files are often
    sorted by type, so the first N calls would all come from one scam script."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = slugify(run_name or extractor.run_name)
    out_path = out_dir / f"{run_name}.jsonl"
    meta_path = out_path.with_suffix(".meta.json")
    if resume and out_path.exists() and meta_path.exists():
        old = load_json(meta_path)
        new = extractor.metadata()
        changed = [k for k in ("model", "prompt_hash", "ontology_hash", "rules_version", "grounding", "max_tokens")
                   if k in new and old.get(k) != new.get(k)]
        if changed:
            log.warning("Run %s was produced with different settings (%s) - re-extracting from scratch",
                        run_name, ", ".join(changed))
            resume = False
    if out_path.exists() and not resume:
        out_path.unlink()
    # CHANGE: calls whose last record failed (timeout, server down, nothing usable) are retried on resume.
    # Before, any written record counted as done, so one Ollama timeout lost that call for good.
    done = {r["call_id"] for r in latest_records(read_jsonl(out_path)) if not _is_failed(r)} if resume else set()
    todo = [(cid, text) for cid, text in calls["text"].items() if cid not in done]
    if limit is not None and limit < len(todo):
        pick = np.sort(np.random.default_rng(seed).choice(len(todo), size=limit, replace=False))
        todo = [todo[i] for i in pick]
    log.info("Extraction run %s: %d calls to process (%d already done)", run_name, len(todo), len(done))
    lock = threading.Lock()
    with out_path.open("a", encoding="utf-8") as f:
        def work(item):
            cid, text = item
            try:
                rec = extractor.extract(cid, text)
            except Exception as e:  # keep the run going; the failure is recorded
                rec = ExtractionRecord(call_id=cid, extractor=extractor.name, json_valid=False,
                                       schema_conformant=False, error=f"{type(e).__name__}: {e}")
            with lock:
                f.write(dumps(rec.to_dict()) + "\n")
                f.flush()
            return rec

        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(work, it) for it in todo]
                for _ in tqdm(as_completed(futures), total=len(futures), desc=f"extract[{run_name}]"):
                    pass
        else:
            for it in tqdm(todo, desc=f"extract[{run_name}]", disable=len(todo) < 20):
                work(it)
    records = latest_records(read_jsonl(out_path))
    meta = {**extractor.metadata(), "run_name": run_name, "updated": timestamp(), **summarize_records(records)}
    save_json(out_path.with_suffix(".meta.json"), meta)
    log.info("Run %s: %d calls, JSON validity %.3f, schema conformance %.3f, %.2f triples/call | retried %.3f, "
             "salvaged %.3f, fallback %.3f, failed %.3f", run_name, meta["n_calls"], meta.get("json_validity_rate", 0),
             meta.get("schema_conformance_rate", 0), meta.get("mean_triples_per_call", 0), meta.get("retry_rate", 0),
             meta.get("salvage_rate", 0), meta.get("fallback_rate", 0), meta.get("failed_rate", 0))
    return out_path


def is_derived_run(path: Path) -> bool:
    """Cascade and serving runs only recombine other runs' triples (scamkg/extraction/benchmark.py)."""
    return Path(path).stem.startswith(("cascade_", "serve__"))


def list_runs(extractions_dir: Path) -> list[Path]:
    """Extraction runs, oldest first. Derived runs can still be named (--run), but are never picked as "the latest
    run" and are not scored as extractors, so creating one cannot change what another command uses."""
    runs = [p for p in Path(extractions_dir).glob("*.jsonl") if not is_derived_run(p)]
    return sorted(runs, key=lambda p: p.stat().st_mtime)


def _n_records(path: Path) -> int:
    return len(latest_records(read_jsonl(path)))


def resolve_run(extractions_dir: Path, run: str | None = None) -> Path:
    """Path of an extraction run by name/path. Default: the most recently updated run among those
    covering (nearly) as many calls as the largest run, so a small pilot run is never picked silently."""
    extractions_dir = Path(extractions_dir)
    if run:
        p = Path(run)
        if p.suffix == ".jsonl" and p.exists():
            return p
        cand = extractions_dir / f"{slugify(run)}.jsonl"
        if cand.exists():
            return cand
        raise FileNotFoundError(f"Extraction run {run!r} not found in {extractions_dir}")
    runs = list_runs(extractions_dir)
    sizes = {p: _n_records(p) for p in runs}
    runs = [p for p in runs if sizes[p] > 0]
    if not runs:
        raise FileNotFoundError(f"No extraction runs in {extractions_dir}; run `python -m scamkg extract` first")
    biggest = max(sizes[p] for p in runs)
    full = [p for p in runs if sizes[p] >= 0.9 * biggest]
    return full[-1]


def load_extraction_run(path: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in read_jsonl(path):
        out[r["call_id"]] = r.get("triples", [])  # later records (re-runs) override earlier ones
    return out
