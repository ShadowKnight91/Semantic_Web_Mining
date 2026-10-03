"""Download the public data sets (proposal Table I).

* BothBosu scam-dialogue (Hugging Face)      -> data/raw/bothbosu/bothbosu.csv
* NCSU robocall audio data set (GitHub)      -> data/raw/ncsu/metadata.csv
* Kaggle scam / non-scam call conversations  -> data/raw/kaggle/ (Kaggle CLI + slug, or manual download)
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd
import requests

from scamkg.config import Paths
from scamkg.utils import get_logger, optional_import

log = get_logger(__name__)
HF_ROWS_API = "https://datasets-server.huggingface.co"


def _dir(paths: Paths, cfg: dict, name: str) -> Path:
    d = ((cfg["data"].get("sources") or {}).get(name) or {}).get("local_dir")
    p = Path(d) if d else paths.raw / name
    p = p if p.is_absolute() else paths.root / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def _hf_rows(name: str) -> pd.DataFrame:
    """Fetch a Hugging Face data set through the public datasets-server rows API (no extra deps)."""
    splits = requests.get(f"{HF_ROWS_API}/splits", params={"dataset": name}, timeout=60).json()["splits"]
    frames = []
    for sp in splits:
        rows, offset = [], 0
        while True:
            r = requests.get(f"{HF_ROWS_API}/rows", params={"dataset": name, "config": sp["config"],
                                                            "split": sp["split"], "offset": offset, "length": 100},
                             timeout=60)
            r.raise_for_status()
            js = r.json()
            batch = [row["row"] for row in js.get("rows", [])]
            if not batch:
                break
            rows.extend(batch)
            offset += len(batch)
            if offset >= js.get("num_rows_total", offset):
                break
        frames.append(pd.DataFrame(rows).assign(hf_split=sp["split"]))
    return pd.concat(frames, ignore_index=True)


def download_bothbosu(cfg: dict, paths: Paths) -> Path:
    name = cfg["data"]["sources"]["bothbosu"]["hf_name"]
    out = _dir(paths, cfg, "bothbosu") / "bothbosu.csv"
    datasets = optional_import("datasets")
    if datasets is not None:
        ds = datasets.load_dataset(name)
        df = pd.concat([ds[s].to_pandas().assign(hf_split=s) for s in ds], ignore_index=True)
    else:
        log.info("`datasets` not installed - using the datasets-server API")
        df = _hf_rows(name)
    df.to_csv(out, index=False)
    log.info("BothBosu: %d rows -> %s (columns: %s)", len(df), out, list(df.columns))
    return out


def download_ncsu(cfg: dict, paths: Paths) -> Path | None:
    src = cfg["data"]["sources"]["ncsu"]
    out = _dir(paths, cfg, "ncsu") / "metadata.csv"
    for url in src.get("urls", []):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code == 200 and len(r.content) > 100:
                out.write_bytes(r.content)
                log.info("NCSU robocall metadata -> %s", out)
                return out
        except requests.RequestException as e:
            log.warning("NCSU download from %s failed: %s", url, e)
    log.warning("Could not download NCSU metadata; place metadata.csv from "
                "https://github.com/wspr-ncsu/robocall-audio-dataset in %s", out.parent)
    return None


def download_kaggle(cfg: dict, paths: Paths) -> Path | None:
    src = cfg["data"]["sources"]["kaggle"]
    out_dir = _dir(paths, cfg, "kaggle")
    slug = src.get("slug")
    if slug and shutil.which("kaggle"):
        subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(out_dir), "--unzip"], check=True)
        log.info("Kaggle data set %s -> %s", slug, out_dir)
        return out_dir
    log.warning("Kaggle data set not downloaded automatically. Either set data.sources.kaggle.slug and install/"
                "configure the Kaggle CLI, or download the 'Scam / Non-Scam call conversations' CSV manually into %s",
                out_dir)
    return None


def download_all(cfg: dict, paths: Paths) -> None:
    for fn in (download_bothbosu, download_ncsu, download_kaggle):
        try:
            fn(cfg, paths)
        except Exception as e:
            log.error("%s failed: %s", fn.__name__, e)
