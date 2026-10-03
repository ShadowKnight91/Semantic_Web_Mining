"""Small shared helpers: logging, seeding, JSON/JSONL IO, hashing, optional imports."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import logging
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_LOGGING_CONFIGURED = False


def get_logger(name: str = "scamkg") -> logging.Logger:
    global _LOGGING_CONFIGURED
    if not _LOGGING_CONFIGURED:
        logging.basicConfig(level=os.environ.get("SCAMKG_LOGLEVEL", "INFO"),
                            format="%(asctime)s %(levelname)-7s %(name)s | %(message)s", datefmt="%H:%M:%S")
        for noisy in ("httpx", "urllib3", "neo4j", "matplotlib", "sentence_transformers", "datasets"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        _LOGGING_CONFIGURED = True
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def optional_import(name: str):
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


class _Encoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (Path, datetime)):
            return str(o)
        if isinstance(o, set):
            return sorted(o)
        return super().default(o)


def dumps(obj: Any, **kw) -> str:
    return json.dumps(obj, cls=_Encoder, ensure_ascii=False, **kw)


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(obj, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                get_logger(__name__).warning("Skipping malformed line %d in %s", n, path)
    return out


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(dumps(r) + "\n")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(s)).strip("-")


def timestamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def md_table(df, floatfmt: str = "{:.3f}", max_rows: int | None = None) -> str:
    """Render a DataFrame as a GitHub markdown table (no tabulate dependency)."""
    if df is None or len(df) == 0:
        return "_(no rows)_\n"
    if max_rows:
        df = df.head(max_rows)
    cols = [str(c) for c in df.columns]

    def fmt(v: Any) -> str:
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v).replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row.values) + " |")
    return "\n".join(lines) + "\n"
