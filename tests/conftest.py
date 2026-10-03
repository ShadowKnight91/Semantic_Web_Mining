"""Shared fixtures: a tiny synthetic data set processed and extracted into a temporary directory."""
from __future__ import annotations

import pytest

from scamkg.config import load_config
from scamkg.context import Context
from scamkg.data.preprocess import preprocess
from scamkg.data.synthetic import generate_demo_data
from scamkg.extraction.extractor import make_extractor, run_extraction
from scamkg.ontology.schema import Ontology


@pytest.fixture(scope="session")
def ontology() -> Ontology:
    return Ontology.load()


@pytest.fixture(scope="session")
def demo_cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("scamkg")
    overrides = {
        "paths": {k: str(root / k) for k in ("data_raw", "data_processed", "kg", "cache", "annotations", "results",
                                             "artifacts")},
        "data": {"load": ["demo"], "primary": "demo_primary", "external": ["demo_shifted", "demo_robocall"],
                 "mixed_external": [["demo_robocall", "demo_shifted"]], "imbalanced_test": {"n_draws": 5}},
        "models": {"search_iter": 2, "classical": ["lr", "xgb"], "cv_folds": 3},
        "graph": {"betweenness_samples": 50},
    }
    cfg = load_config(overrides=overrides)
    from scamkg.config import get_paths
    paths = get_paths(cfg).ensure()
    generate_demo_data(paths.raw, n_primary=160, n_shifted=60, n_robocall=30, seed=7)
    preprocess(cfg, paths)
    ctx = Context.load(cfg, with_graph=False)
    run_extraction(ctx.calls, make_extractor("rule", cfg, ctx.ontology), paths.extractions)
    return cfg


@pytest.fixture(scope="session")
def demo_ctx(demo_cfg) -> Context:
    return Context.load(demo_cfg)
