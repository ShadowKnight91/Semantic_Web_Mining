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
from scamkg.extraction.prompts import (PROMPT_VERSIONS, RETRY_MESSAGE, build_extraction_messages,
                                       prompt_fingerprint)
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
    stats = {"n_raw": 0, "n_dropped": 0, "n_repaired": 0, "conformant": True}
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
            continue
        rel = _norm_relation(t.get("relation") or t.get("predicate"))
        otype = str(t.get("object_type") or "").strip() or None
        mention = t.get("object")
        if isinstance(mention, (list, dict)) or mention is None:
            stats["n_dropped"] += 1
            stats["conformant"] = False
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
    def __init__(self, client, ontology: Ontology, standardizer: EntityStandardizer | None = None,
                 prompt_version: str = "v1_full", max_retries: int = 2, max_chars: int = 6000,
                 grounding: bool = True):
        if prompt_version not in PROMPT_VERSIONS:
            raise ValueError(f"Unknown prompt version {prompt_version}")
        self.grounding = grounding
        self.client = client
        self.ontology = ontology
        self.standardizer = standardizer or EntityStandardizer(ontology)
        self.prompt_version = prompt_version
        self.max_retries = max_retries
        self.max_chars = max_chars
        self.name = f"llm_{client.provider}_{client.model}_{prompt_version}"

    def extract(self, call_id: str, text: str) -> ExtractionRecord:
        messages = build_extraction_messages(truncate_transcript(text, self.max_chars), self.ontology,
                                             self.prompt_version)
        t0 = time.time()
        raw, obj, error, attempts, cached = None, None, None, 0, True
        while attempts <= self.max_retries:
            attempts += 1
            try:
                raw = self.client.chat(messages)
                cached = cached and bool(getattr(self.client, "last_cached", False))
            except LLMError as e:
                error = str(e)
                cached = False
                break
            parsed = parse_json_response(raw)
            if isinstance(parsed, dict) and isinstance(parsed.get("triples"), list):
                obj = parsed
                break
            if isinstance(parsed, list) and attempts > self.max_retries:
                obj = parsed
                break
            messages = messages + [{"role": "assistant", "content": (raw or "")[:1500]},
                                   {"role": "user", "content": RETRY_MESSAGE}]
        json_valid = obj is not None
        triples, stats = validate_triples(obj, self.ontology, self.standardizer) if json_valid else ([], {
            "n_raw": 0, "n_dropped": 0, "n_repaired": 0, "conformant": False})
        n_ungrounded = 0
        if self.grounding:
            triples, n_ungrounded = ground_triples(triples, text)
        return ExtractionRecord(
            call_id=call_id, extractor=self.name, triples=triples, json_valid=json_valid,
            schema_conformant=json_valid and stats["conformant"], attempts=attempts, n_raw=stats["n_raw"],
            n_dropped=stats["n_dropped"], n_repaired=stats["n_repaired"], n_ungrounded=n_ungrounded,
            latency_s=round(time.time() - t0, 3), cached=cached, error=error, raw_response=(raw or "")[:4000])

    def metadata(self) -> dict:
        return {"extractor": "llm", **self.client.describe(), "prompt_version": self.prompt_version,
                "prompt_hash": prompt_fingerprint(self.ontology, self.prompt_version),
                "ontology_version": self.ontology.version, "ontology_hash": self.ontology.fingerprint,
                "max_retries": self.max_retries, "max_chars": self.max_chars, "grounding": self.grounding}


def make_extractor(kind: str, cfg: dict, ontology: Ontology, cache_dir: Path | None = None,
                   model: str | None = None, provider: str | None = None,
                   prompt_version: str | None = None) -> BaseExtractor:
    standardizer = EntityStandardizer(ontology)
    if kind == "rule":
        from scamkg.extraction.rule_based import RuleBasedExtractor
        return RuleBasedExtractor(ontology, standardizer)
    if kind == "llm":
        from scamkg.extraction.llm_client import make_client
        client = make_client(cfg, cache_dir=cache_dir, model=model, provider=provider)
        llm = cfg["llm"]
        return LLMExtractor(client, ontology, standardizer, prompt_version or llm.get("prompt_version", "v1_full"),
                            int(llm.get("max_retries", 2)), int(llm.get("max_chars", 6000)),
                            grounding=bool(llm.get("grounding", True)))
    raise ValueError(f"Unknown extractor {kind!r} (use 'rule' or 'llm')")


# ---------------------------------------------------------------------------- running + IO
def summarize_records(records: list[dict]) -> dict:
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
    done = {r["call_id"] for r in read_jsonl(out_path)} if resume else set()
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
    records = read_jsonl(out_path)
    meta = {**extractor.metadata(), "run_name": run_name, "updated": timestamp(), **summarize_records(records)}
    save_json(out_path.with_suffix(".meta.json"), meta)
    log.info("Run %s: %d calls, JSON validity %.3f, schema conformance %.3f, %.2f triples/call", run_name,
             meta["n_calls"], meta.get("json_validity_rate", 0), meta.get("schema_conformance_rate", 0),
             meta.get("mean_triples_per_call", 0))
    return out_path


def list_runs(extractions_dir: Path) -> list[Path]:
    runs = [p for p in Path(extractions_dir).glob("*.jsonl")]
    return sorted(runs, key=lambda p: p.stat().st_mtime)


def _n_records(path: Path) -> int:
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


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
