"""Experiment context: calls + splits + ontology + knowledge graph + graph-feature builder."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from scamkg.config import Paths, config_hash, get_paths
from scamkg.data.preprocess import load_calls
from scamkg.extraction.extractor import load_extraction_run, resolve_run
from scamkg.graph.build import CallKG
from scamkg.graph.features import GraphFeatureBuilder, structural_features
from scamkg.ontology.schema import Ontology
from scamkg.utils import get_logger, sha256_text

log = get_logger(__name__)

GRAPH_CFG_KEYS = ("similarity_top_k", "similarity_min", "louvain_resolution", "betweenness_samples")


def structural_cache_path(paths: Paths, run_path: Path, cfg: dict, call_ids: list[str]):
    """Keyed by the graph settings and the content of the extraction run and call set, so a run that
    is re-extracted under the same name (e.g. after a prompt change) never reuses stale features."""
    key = config_hash({k: cfg["graph"].get(k) for k in GRAPH_CFG_KEYS} | {
        "seed": cfg["project"]["seed"], "run": sha256_text(run_path.read_text(encoding="utf-8")),
        "calls": sha256_text("\n".join(call_ids))})
    return paths.graph_features / f"structural__{run_path.stem}__{key}.csv"


class Context:
    def __init__(self, cfg: dict, paths: Paths, calls: pd.DataFrame, ontology: Ontology):
        self.cfg = cfg
        self.paths = paths
        self.calls = calls
        self.ontology = ontology
        self.seed = int(cfg["project"]["seed"])
        self.run_name: str | None = None
        self.extractions: dict[str, list[dict]] = {}
        self.kg: CallKG | None = None
        self.fb: GraphFeatureBuilder | None = None

    @classmethod
    def load(cls, cfg: dict, with_graph: bool = True, run: str | None = None) -> "Context":
        paths = get_paths(cfg).ensure()
        ctx = cls(cfg, paths, load_calls(paths), Ontology.load(cfg["ontology"]["path"]))
        if with_graph:
            ctx.attach_graph(run)
        return ctx

    # ------------------------------------------------------------------ graph
    def attach_graph(self, run: str | None = None, structural: pd.DataFrame | None = None,
                     use_cache: bool = True) -> "Context":
        """`structural` (e.g. Neo4j GDS features) or the cached features of this run are reused; build-graph
        passes use_cache=False so the backend it runs decides the features every later command uses."""
        run_path = resolve_run(self.paths.extractions, run or self.cfg.get("extraction", {}).get("default_run"))
        self.run_name = run_path.stem
        self.extractions = load_extraction_run(run_path)
        missing = [c for c in self.calls.index if c not in self.extractions]
        if missing:
            log.warning("%d calls have no extraction in run %s (treated as having no entities)", len(missing),
                        self.run_name)
        self.kg = CallKG(self.calls.index.tolist(), self.extractions, self.ontology)
        cache = structural_cache_path(self.paths, run_path, self.cfg, self.kg.call_ids)
        if structural is None and use_cache and cache.exists():
            structural = pd.read_csv(cache, index_col=0)
            structural.index = structural.index.astype(str)
            log.info("Using cached structural graph features %s", cache.name)
        self.fb = GraphFeatureBuilder(self.kg, self.cfg["graph"], structural=structural, seed=self.seed)
        self.fb.structural.to_csv(cache)
        log.info("KG from run %s: %d calls, %d entities, %d edges", self.run_name, self.kg.n_calls,
                 len(self.kg.entity_ids), int(self.kg.X.sum()))
        return self

    def with_excluded_types(self, types) -> "Context":
        """Copy of the context whose KG drops the given entity types (ablation); features recomputed."""
        new = copy.copy(self)
        new.kg = CallKG(self.calls.index.tolist(), self.extractions, self.ontology, exclude_types=types)
        new.fb = GraphFeatureBuilder(new.kg, self.cfg["graph"], structural=structural_features(
            new.kg, self.cfg["graph"], self.seed), seed=self.seed)
        return new

    def require_graph(self) -> None:
        if self.kg is None or self.fb is None:
            raise RuntimeError("This step needs the knowledge graph: run `extract` and `build-graph` first")

    # ------------------------------------------------------------------ splits
    @property
    def primary(self) -> str:
        return self.cfg["data"]["primary"]

    def ids(self, split: str | None = None, dataset: str | None = None, labeled: bool = False) -> list[str]:
        df = self.calls
        if split is not None:
            df = df[df["split"] == split]
        if dataset is not None:
            df = df[df["dataset"] == dataset]
        if labeled:
            df = df[df["label"] >= 0]
        return df.index.tolist()

    @property
    def train_ids(self) -> list[str]:
        return self.ids("train")

    @property
    def val_ids(self) -> list[str]:
        return self.ids("val")

    @property
    def test_ids(self) -> list[str]:
        return self.ids("test")

    @property
    def external_datasets(self) -> list[str]:
        present = set(self.calls["dataset"])
        return [d for d in self.cfg["data"].get("external", []) if d in present]

    def labels(self, ids) -> np.ndarray:
        return self.calls.loc[list(ids), "label"].to_numpy().astype(int)

    def texts(self, ids) -> list[str]:
        return self.calls.loc[list(ids), "text_plain"].tolist()
