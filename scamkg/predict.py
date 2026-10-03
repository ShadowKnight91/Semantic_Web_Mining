"""Score a new, unseen transcript end-to-end: clean -> extract triples -> add the call to the KG ->
recompute graph features -> trained E1/E2/E2G model -> score + KG explanation."""
from __future__ import annotations

import joblib
import pandas as pd

from scamkg.context import Context
from scamkg.data.preprocess import clean_transcript
from scamkg.evaluation.explain import explain_call
from scamkg.extraction.extractor import make_extractor
from scamkg.graph.build import CallKG
from scamkg.graph.features import GraphFeatureBuilder
from scamkg.models.classical import EXPERIMENT_SPECS
from scamkg.utils import get_logger

log = get_logger(__name__)
NEW_ID = "__new_call__"


class ScamDetector:
    def __init__(self, cfg: dict, experiment: str = "E2", model: str = "xgb", extractor: str = "rule",
                 run: str | None = None, llm_model: str | None = None):
        self.cfg = cfg
        if experiment not in EXPERIMENT_SPECS:
            raise ValueError(f"ScamDetector supports {sorted(EXPERIMENT_SPECS)}")
        self.ctx = Context.load(cfg, with_graph=True, run=run)  # the KG is needed for the explanation
        path = self.ctx.paths.models / f"{experiment}__{model}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run `python -m scamkg train` first")
        self.bundle = joblib.load(path)
        self.extractor = make_extractor(extractor, cfg, self.ctx.ontology, cache_dir=self.ctx.paths.cache,
                                        model=llm_model)
        self.labeled_ids = [c for c in self.bundle["labeled_ids"] if c in self.ctx.calls.index]

    def score(self, transcript: str) -> dict:
        ctx = self.ctx
        text, plain, n_turns = clean_transcript(transcript)
        rec = self.extractor.extract(NEW_ID, text)
        fitted = self.bundle["fitted"]
        row = pd.DataFrame([{"call_id": NEW_ID, "dataset": "new", "source_id": "", "split": "new", "label": -1,
                             "label3": "", "subtype": "", "n_words": len(plain.split()), "n_turns": n_turns,
                             "text": text, "text_plain": plain, "gold_triples": []}]).set_index("call_id", drop=False)
        calls = pd.concat([ctx.calls, row])
        extractions = {**ctx.extractions, NEW_ID: rec.triples}
        kg = CallKG(calls.index.tolist(), extractions, ctx.ontology)
        tmp = Context(ctx.cfg, ctx.paths, calls, ctx.ontology)
        tmp.run_name, tmp.extractions, tmp.kg = ctx.run_name, extractions, kg
        graph_df = None
        if fitted.graph_columns or fitted.onehot_names:
            tmp.fb = GraphFeatureBuilder(kg, ctx.cfg["graph"], seed=ctx.seed)
            if fitted.graph_columns:
                graph_df = tmp.fb.build(self.labeled_ids, tmp.labels(self.labeled_ids))
        oh = tmp.fb.onehot_matrix([NEW_ID], fitted.onehot_names) if fitted.onehot_names else None
        X = fitted.transform_parts([plain], graph_df.loc[[NEW_ID]] if graph_df is not None else None, oh)
        score = float(self.bundle["model"].predict_proba(X)[0, 1])
        thr = float(self.cfg["models"].get("threshold", 0.5))
        explanation = explain_call(tmp, NEW_ID, score, labeled_ids=self.labeled_ids)
        return {"score": score, "is_scam": score >= thr, "triples": rec.triples, "explanation": explanation,
                "model": f"{self.bundle['experiment']}/{self.bundle['model_name']}", "extractor": self.extractor.name}
