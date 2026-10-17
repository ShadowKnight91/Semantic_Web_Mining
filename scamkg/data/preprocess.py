"""Phase 1 - cleaning, speaker-tag normalization, de-duplication, label unification, splits.

Output: data/processed/calls.jsonl with one row per call:
call_id, dataset, source_id, text (normalized "caller:/receiver:" lines), text_plain (no tags,
used by text models), label, label3, subtype, split (train/val/test for the primary data set,
"external" for cross-dataset sets), n_words, n_turns, gold_triples.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from scamkg.config import Paths
from scamkg.data.loaders import load_source
from scamkg.utils import get_logger, read_jsonl, save_json, write_jsonl

log = get_logger(__name__)

ROLE_MAP = {
    "caller": "caller", "scammer": "caller", "fraudster": "caller", "suspect": "caller", "agent": "caller",
    "representative": "caller", "rep": "caller", "speaker 1": "caller", "person 1": "caller",
    "speaker a": "caller", "person a": "caller", "spk 1": "caller",
    "receiver": "receiver", "victim": "receiver", "innocent": "receiver", "customer": "receiver",
    "recipient": "receiver", "callee": "receiver", "client": "receiver", "speaker 2": "receiver",
    "person 2": "receiver", "speaker b": "receiver", "person b": "receiver", "spk 2": "receiver",
}
TAG_RE = re.compile(
    r"(?i)(?<![A-Za-z])(caller|receiver|scammer|fraudster|suspect|victim|innocent|agent|customer|representative|rep|"
    r"recipient|callee|client|speaker\s*[12ab]|person\s*[12ab]|spk\s*[12])\s*:\s*")


def _norm_tag(tag: str) -> str:
    t = re.sub(r"\s+", " ", tag.lower()).strip()
    return re.sub(r"^(speaker|person|spk)\s*", r"\1 ", t)


def split_turns(text: str) -> list[tuple[str, str]]:
    parts = TAG_RE.split(text)
    if len(parts) < 3:
        return [("unknown", text.strip())] if text.strip() else []
    turns = []
    if parts[0].strip():
        turns.append(("unknown", parts[0].strip()))
    for tag, utt in zip(parts[1::2], parts[2::2]):
        utt = utt.strip()
        if utt:
            turns.append((ROLE_MAP.get(_norm_tag(tag), "unknown"), utt))
    return turns


def clean_transcript(raw: str) -> tuple[str, str, int]:
    """Return (text with normalized speaker lines, plain text without tags, number of turns)."""
    s = unicodedata.normalize("NFKC", str(raw or ""))
    s = s.replace("\r", "\n")
    s = re.sub(r"\[[^\]]{0,40}\]", " ", s)                      # [noise], [inaudible], [00:01:23]
    s = re.sub(r"<[^>]{0,20}>", " ", s)                         # <unk>, <laugh>
    s = re.sub(r"\((?:inaudible|crosstalk|laughs?|laughter|silence|music|noise|pause|beep)[^)]*\)", " ", s, flags=re.I)
    s = re.sub(r"([!?.,])\1{2,}", r"\1", s)                     # repeated punctuation
    s = re.sub(r"[ \t ]+", " ", s)
    turns = split_turns(s)
    roles = {r for r, _ in turns}
    if roles & {"caller", "receiver"}:
        text = "\n".join(f"{r if r != 'unknown' else 'caller'}: {u}" for r, u in turns)
        plain = " ".join(u for _, u in turns)
    else:
        text = re.sub(r"\s*\n\s*", "\n", s).strip()
        plain = re.sub(r"\s+", " ", text)
    return text, plain.strip(), max(1, len(turns))


def assign_splits(df: pd.DataFrame, primary: str, val_size: float, test_size: float, seed: int) -> pd.DataFrame:
    df = df.copy()
    df["split"] = "external"
    prim = df[(df["dataset"] == primary) & (df["label"] >= 0)]
    if prim.empty:
        raise RuntimeError(f"Primary data set {primary!r} has no labeled calls")
    df.loc[(df["dataset"] == primary) & (df["label"] < 0), "split"] = "unlabeled"
    ids, y = prim.index.to_numpy(), prim["label"].to_numpy()
    tr, tmp = train_test_split(ids, test_size=val_size + test_size, stratify=y, random_state=seed)
    y_tmp = df.loc[tmp, "label"].to_numpy()
    va, te = train_test_split(tmp, test_size=test_size / (val_size + test_size), stratify=y_tmp, random_state=seed)
    df.loc[tr, "split"], df.loc[va, "split"], df.loc[te, "split"] = "train", "val", "test"
    return df


def preprocess(cfg: dict, paths: Paths) -> pd.DataFrame:
    dcfg = cfg["data"]
    frames = []
    for name in dcfg["load"]:
        f = load_source(name, cfg, paths)
        if f is None or f.empty:
            continue
        log.info("Loaded %-10s %6d rows", name, len(f))
        frames.append(f)
    if not frames:
        raise RuntimeError("No data sets found. Run `python -m scamkg download` (real data) or "
                           "`python -m scamkg --config config/demo.yaml demo-data` (synthetic demo).")
    df = pd.concat(frames, ignore_index=True)
    cleaned = [clean_transcript(t) for t in df["raw_text"]]
    df["text"] = [c[0] for c in cleaned]
    df["text_plain"] = [c[1] for c in cleaned]
    df["n_turns"] = [c[2] for c in cleaned]
    df["n_words"] = df["text_plain"].str.split().str.len().fillna(0).astype(int)
    before = len(df)
    df = df[df["n_words"] >= int(dcfg.get("min_words", 5))]
    log.info("Dropped %d calls shorter than %s words", before - len(df), dcfg.get("min_words", 5))

    key = df["text_plain"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    labeled = df["label"] >= 0
    conflicts = df[labeled].assign(_k=key[labeled]).groupby("_k")["label"].nunique()
    bad_keys = set(conflicts[conflicts > 1].index)
    n_conflict = len(bad_keys)
    # CHANGE: a transcript that appears with both labels is dropped entirely. Before, the first copy was
    # kept with whichever label it happened to have, i.e. a coin-flip label in train or test.
    conflict_rows = key.isin(bad_keys)
    df, key = df[~conflict_rows.to_numpy()], key[~conflict_rows]
    dup = key.duplicated(keep="first")
    log.info("Removed %d duplicate transcripts and %d transcripts with conflicting labels (%d rows)",
             int(dup.sum()), n_conflict, int(conflict_rows.sum()))
    df = df[~dup.to_numpy()].reset_index(drop=True)

    df["call_id"] = [f"{ds}_{i:05d}" for ds, i in zip(df["dataset"], df.groupby("dataset").cumcount())]
    df = df.set_index("call_id", drop=False)
    df = assign_splits(df, dcfg["primary"], float(dcfg["val_size"]), float(dcfg["test_size"]),
                       int(cfg["project"]["seed"]))
    cols = ["call_id", "dataset", "source_id", "split", "label", "label3", "subtype", "n_words", "n_turns",
            "text", "text_plain", "gold_triples"]
    paths.processed.mkdir(parents=True, exist_ok=True)
    write_jsonl(paths.calls_file, df[cols].to_dict(orient="records"))
    summary = (df.groupby(["dataset", "split", "label"]).size().rename("n").reset_index())
    save_json(paths.processed / "summary.json", {"n_calls": len(df), "duplicates_removed": int(dup.sum()),
                                                 "label_conflicts": n_conflict,
                                                 "counts": summary.to_dict(orient="records")})
    log.info("Wrote %d calls to %s", len(df), paths.calls_file)
    for (ds, split), grp in df.groupby(["dataset", "split"]):
        log.info("  %-14s %-9s n=%5d scam_rate=%.3f", ds, split, len(grp), (grp["label"] == 1).mean())
    return df


def load_calls(paths: Paths) -> pd.DataFrame:
    if not paths.calls_file.exists():
        raise FileNotFoundError(f"{paths.calls_file} not found - run `python -m scamkg preprocess` first")
    df = pd.DataFrame(read_jsonl(paths.calls_file))
    df["call_id"] = df["call_id"].astype(str)
    df = df.set_index("call_id", drop=False)
    df["label"] = df["label"].astype(int)
    for col in ("label3", "subtype"):
        df[col] = df[col].fillna("").astype(str) if col in df else ""
    if "gold_triples" not in df:
        df["gold_triples"] = [[] for _ in range(len(df))]
    df["gold_triples"] = [g if isinstance(g, list) else [] for g in df["gold_triples"]]
    return df
