"""Loaders that turn each raw data set into a common frame.

Common columns: source_id, dataset, raw_text, label (1 scam / 0 not scam / -1 unknown),
subtype, label3 (fraudster / salesperson / prankster / other_legitimate, optional), gold_triples.
Column names are auto-detected; override them in config (data.sources.<name>.*_col).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from scamkg.config import Paths
from scamkg.utils import get_logger, read_jsonl

log = get_logger(__name__)

TEXT_CANDIDATES = ["dialogue", "dialog", "conversation", "transcript", "text", "content", "call_transcript",
                   "utterance", "message", "sentence"]
LABEL_CANDIDATES = ["label", "labels", "is_scam", "scam", "class", "target", "fraud", "is_fraud", "y"]
ID_CANDIDATES = ["conversation_id", "conv_id", "dialogue_id", "dialog_id", "call_id", "conversation_no", "id"]
ORDER_CANDIDATES = ["turn", "turn_id", "step", "conversation_step", "order", "utterance_id", "line"]
SPEAKER_CANDIDATES = ["speaker", "role", "author", "from", "sender"]
SUBTYPE_CANDIDATES = ["type", "scam_type", "category", "subtype", "scenario"]

SCAM_STRINGS = {"1", "scam", "fraud", "fraudulent", "yes", "true", "spam", "malicious", "phishing", "vishing",
                "scammer", "suspicious"}
LEGIT_STRINGS = {"0", "non scam", "nonscam", "not scam", "no scam", "legit", "legitimate", "normal", "no", "false",
                 "ham", "genuine", "benign", "non fraud", "not fraud"}
# BothBosu's dialogue types give gold three-way labels (stretch goal): every scam type is a fraudster,
# insurance / telemarketing offers are salespeople, deliveries and wrong numbers are other legitimate calls.
BOTHBOSU_LABEL3 = {"insurance": "salesperson", "telemarketing": "salesperson", "delivery": "other_legitimate",
                   "wrong": "other_legitimate"}


def unify_label(v) -> int:
    if v is None:
        return -1
    if isinstance(v, (bool, np.bool_)):
        return int(v)
    if isinstance(v, (int, np.integer)):
        return 1 if v >= 1 else 0
    if isinstance(v, (float, np.floating)):
        return -1 if np.isnan(v) else (1 if v >= 0.5 else 0)
    s = str(v).strip().lower().replace("_", " ").replace("-", " ")
    if s in SCAM_STRINGS:
        return 1
    if s in LEGIT_STRINGS:
        return 0
    if any(w in s for w in ("non scam", "not scam", "legit", "normal", "genuine")):
        return 0
    if any(w in s for w in ("scam", "fraud", "spam", "malicious")):
        return 1
    return -1


def _find_col(df: pd.DataFrame, candidates: list[str], override: str | None = None) -> str | None:
    if override:
        if override not in df.columns:
            raise KeyError(f"Configured column {override!r} not in {list(df.columns)}")
        return override
    lower = {str(c).lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _read_tables(directory: Path) -> pd.DataFrame | None:
    files = sorted([*directory.glob("*.csv"), *directory.glob("*.tsv"), *directory.glob("*.jsonl"),
                    *directory.glob("*.json"), *directory.glob("*.parquet")]) if directory.exists() else []
    frames = []
    for f in files:
        try:
            if f.suffix == ".csv":
                frames.append(pd.read_csv(f))
            elif f.suffix == ".tsv":
                frames.append(pd.read_csv(f, sep="\t"))
            elif f.suffix == ".jsonl":
                frames.append(pd.read_json(f, lines=True))
            elif f.suffix == ".json":
                frames.append(pd.read_json(f))
            else:
                frames.append(pd.read_parquet(f))
            frames[-1]["_file"] = f.name
        except Exception as e:  # a stray non-data file should not stop the pipeline
            log.warning("Could not read %s: %s", f, e)
    return pd.concat(frames, ignore_index=True) if frames else None


def _read_labeled_txt(directory: Path) -> pd.DataFrame | None:
    """Kaggle "Scam and Non-Scam Call Conversation" layout (teeconnie): English_Scam.txt / English_NonScam.txt,
    one call per paragraph (scam calls numbered "1.<tab>"); the label comes from the file name."""
    files = sorted(directory.glob("*.txt")) if directory.exists() else []
    if any("english" in f.stem.lower() for f in files):     # other-language files of the same release
        files = [f for f in files if "english" in f.stem.lower()]
    rows = []
    for f in files:
        name = re.sub(r"[^a-z]", "", f.stem.lower())
        label = 0 if ("nonscam" in name or "notscam" in name) else 1 if "scam" in name else None
        if label is None:
            continue
        text = f.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
        blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
        rows += [(f"{f.stem}_{k:04d}", re.sub(r"^\d+\.\s+", "", b), label) for k, b in enumerate(blocks)]
    return pd.DataFrame(rows, columns=["id", "text", "label"]) if rows else None


def _frame(dataset: str, raw_text, label, source_id=None, subtype=None) -> pd.DataFrame:
    n = len(raw_text)
    return pd.DataFrame({
        "source_id": list(source_id) if source_id is not None else [str(i) for i in range(n)],
        "dataset": dataset,
        "raw_text": list(raw_text),
        "label": [unify_label(v) for v in label] if label is not None else [-1] * n,
        "subtype": [("" if pd.isna(s) else str(s)) for s in subtype] if subtype is not None else [""] * n,
        "label3": [""] * n,
        "gold_triples": [[] for _ in range(n)],
    })


def _src(cfg: dict, name: str) -> dict:
    return (cfg["data"].get("sources") or {}).get(name, {}) or {}


def _dir(paths: Paths, src: dict, default: str) -> Path:
    d = src.get("local_dir")
    if not d:
        return paths.raw / default
    p = Path(d)
    return p if p.is_absolute() else paths.root / p


def load_bothbosu(cfg: dict, paths: Paths) -> pd.DataFrame | None:
    src = _src(cfg, "bothbosu")
    df = _read_tables(_dir(paths, src, "bothbosu"))
    if df is None:
        log.warning("BothBosu data not found - run `python -m scamkg download`")
        return None
    text_col = _find_col(df, TEXT_CANDIDATES, src.get("text_col"))
    label_col = _find_col(df, LABEL_CANDIDATES, src.get("label_col"))
    sub_col = _find_col(df, SUBTYPE_CANDIDATES)
    if text_col is None or label_col is None:
        raise KeyError(f"BothBosu: cannot find text/label columns in {list(df.columns)}")
    out = _frame("bothbosu", df[text_col].astype(str), df[label_col], df.index.astype(str),
                 df[sub_col] if sub_col else None)
    out["label3"] = ["fraudster" if lab == 1 else BOTHBOSU_LABEL3.get(st.lower(), "") if lab == 0 else ""
                     for lab, st in zip(out["label"], out["subtype"])]
    return out


def load_kaggle(cfg: dict, paths: Paths) -> pd.DataFrame | None:
    """Kaggle scam / non-scam call conversations; handles one-row-per-call and one-row-per-turn layouts."""
    src = _src(cfg, "kaggle")
    directory = _dir(paths, src, "kaggle")
    df = _read_tables(directory)
    if df is None:
        txt = _read_labeled_txt(directory)
        if txt is None:
            log.warning("Kaggle data not found in data/raw/kaggle - see README (Data sets)")
            return None
        log.info("Kaggle: %d calls from labeled text files", len(txt))
        return _frame("kaggle", txt["text"], txt["label"], txt["id"])
    text_col = _find_col(df, TEXT_CANDIDATES, src.get("text_col"))
    label_col = _find_col(df, LABEL_CANDIDATES, src.get("label_col"))
    id_col = _find_col(df, ID_CANDIDATES, src.get("id_col"))
    if text_col is None or label_col is None:
        raise KeyError(f"Kaggle: cannot find text/label columns in {list(df.columns)}; set data.sources.kaggle.*_col")
    if id_col is not None and df[id_col].duplicated().any():
        spk_col = _find_col(df, SPEAKER_CANDIDATES, src.get("speaker_col"))
        order_col = _find_col(df, ORDER_CANDIDATES)
        rows = []
        for cid, grp in df.groupby(id_col, sort=False):
            if order_col:
                grp = grp.sort_values(order_col)
            if spk_col:
                text = "\n".join(f"{s}: {t}" for s, t in zip(grp[spk_col].astype(str), grp[text_col].astype(str)))
            else:
                text = "\n".join(grp[text_col].astype(str))
            labels = [unify_label(v) for v in grp[label_col]]
            rows.append((str(cid), text, max(labels)))
        log.info("Kaggle: grouped %d turns into %d conversations", len(df), len(rows))
        return _frame("kaggle", [r[1] for r in rows], [r[2] for r in rows], [r[0] for r in rows])
    ids = df[id_col].astype(str) if id_col else df.index.astype(str)
    return _frame("kaggle", df[text_col].astype(str), df[label_col], ids)


def load_ncsu(cfg: dict, paths: Paths) -> pd.DataFrame | None:
    """NCSU robocall audio data set metadata (transcripts); every call is an illegal robocall."""
    src = _src(cfg, "ncsu")
    df = _read_tables(_dir(paths, src, "ncsu"))
    if df is None:
        log.warning("NCSU robocall metadata not found - run `python -m scamkg download`")
        return None
    text_col = next((c for c in df.columns if "transcript" in str(c).lower()), None) or _find_col(df, TEXT_CANDIDATES)
    if text_col is None:
        raise KeyError(f"NCSU: no transcript column in {list(df.columns)}")
    lang = src.get("language")
    lang_col = _find_col(df, ["language", "lang"])
    if lang and lang_col:
        df = df[df[lang_col].astype(str).str.lower().str.startswith(str(lang).lower()[:2])]
    df = df[df[text_col].notna() & (df[text_col].astype(str).str.strip() != "")]
    id_col = _find_col(df, ["file_name", "filename", "file", "id"])
    ids = df[id_col].astype(str) if id_col else df.index.astype(str)
    label = [src.get("label", 1)] * len(df)
    return _frame("ncsu", df[text_col].astype(str), label, ids, ["robocall"] * len(df))


def load_demo(cfg: dict, paths: Paths) -> pd.DataFrame | None:
    files = sorted(paths.raw.glob("*.jsonl"))
    if not files:
        log.warning("No demo data in %s - run `python -m scamkg --config config/demo.yaml demo-data`", paths.raw)
        return None
    recs = [r for f in files for r in read_jsonl(f)]
    df = pd.DataFrame(recs)
    df["label"] = df["label"].map(unify_label)
    for col, default in (("subtype", ""), ("label3", "")):
        if col not in df:
            df[col] = default
    if "gold_triples" not in df:
        df["gold_triples"] = [[] for _ in range(len(df))]
    return df[["source_id", "dataset", "raw_text", "label", "subtype", "label3", "gold_triples"]]


LOADERS = {"bothbosu": load_bothbosu, "kaggle": load_kaggle, "ncsu": load_ncsu, "demo": load_demo}


def load_source(name: str, cfg: dict, paths: Paths) -> pd.DataFrame | None:
    if name not in LOADERS:
        raise KeyError(f"Unknown data set {name!r}; available: {sorted(LOADERS)}")
    return LOADERS[name](cfg, paths)
