"""Configuration loading (config/config.yaml + optional override file) and project paths."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "config.yaml"


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (KEY=VALUE lines); existing environment variables win."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.strip().strip('"').strip("'")
        if val:
            os.environ.setdefault(key.strip(), val)


def deep_merge(base: dict, override: dict | None) -> dict:
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if (Path.cwd() / p).exists():
        return Path.cwd() / p
    return ROOT / p


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Load config/config.yaml, deep-merge an optional override file and a dict of overrides."""
    load_dotenv()
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if path:
        p = _resolve(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        if p.resolve() != DEFAULT_CONFIG.resolve():
            cfg = deep_merge(cfg, yaml.safe_load(p.read_text(encoding="utf-8")) or {})
        cfg["_config_file"] = str(p)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def config_hash(obj: Any, n: int = 8) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


@dataclass
class Paths:
    root: Path
    raw: Path
    processed: Path
    kg: Path
    cache: Path
    annotations: Path
    results: Path
    artifacts: Path

    @property
    def calls_file(self) -> Path:
        return self.processed / "calls.jsonl"

    @property
    def extractions(self) -> Path:
        return self.kg / "extractions"

    @property
    def graph_features(self) -> Path:
        return self.kg / "features"

    @property
    def figures(self) -> Path:
        return self.results / "figures"

    @property
    def predictions(self) -> Path:
        return self.results / "predictions"

    @property
    def models(self) -> Path:
        return self.artifacts / "models"

    def ensure(self) -> "Paths":
        for p in (self.raw, self.processed, self.kg, self.extractions, self.graph_features, self.cache,
                  self.annotations, self.results, self.figures, self.predictions, self.models):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(cfg: dict) -> Paths:
    p = cfg["paths"]

    def r(key: str) -> Path:
        path = Path(p[key])
        return path if path.is_absolute() else ROOT / path

    return Paths(root=ROOT, raw=r("data_raw"), processed=r("data_processed"), kg=r("kg"), cache=r("cache"),
                 annotations=r("annotations"), results=r("results"), artifacts=r("artifacts"))
