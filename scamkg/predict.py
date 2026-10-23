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
from scamkg.utils import get_logger, load_json

log = get_logger(__name__)
NEW_ID = "__new_call__"


class ScamDetector:
    def __init__(self, cfg: dict, experiment: str = "E2", model: str = "xgb", extractor: str | None = None,
                 run: str | None = None, llm_model: str | None = None):
        """`extractor=None` uses the extractor (and LLM settings) that built the KG, so the new call's
        triples and graph features are comparable with the ones the model was trained on."""
        self.cfg = cfg
        if experiment not in EXPERIMENT_SPECS:
            raise ValueError(f"ScamDetector supports {sorted(EXPERIMENT_SPECS)}")
        self.ctx = Context.load(cfg, with_graph=True, run=run)  # the KG is needed for the explanation
        path = self.ctx.paths.models / f"{experiment}__{model}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found - run `python -m scamkg train` first")
        self.bundle = joblib.load(path)
        meta_path = self.ctx.paths.extractions / f"{self.ctx.run_name}.meta.json"
        meta = load_json(meta_path) if meta_path.exists() else {}
        kg_extractor = meta.get("extractor", "rule")
        extractor = extractor or kg_extractor
        if extractor != kg_extractor:
            log.warning("The KG (run %s) was built with the %s extractor but new calls use %s: their triples and "
                        "graph features are not comparable", self.ctx.run_name, kg_extractor, extractor)
        same_llm = extractor == "llm" and kg_extractor == "llm"
        self.extractor = make_extractor(extractor, cfg, self.ctx.ontology, cache_dir=self.ctx.paths.cache,
                                        model=llm_model or (meta.get("model") if same_llm else None),
                                        provider=meta.get("provider") if same_llm else None,
                                        prompt_version=meta.get("prompt_version") if same_llm else None)
        self.labeled_ids = [c for c in self.bundle["labeled_ids"] if c in self.ctx.calls.index]
        # caller type (fraudster / salesperson / prankster / other legitimate), when `multiclass` saved a model
        main = (cfg["models"].get("multiclass") or {}).get("main") or ["E2", "lr"]
        mc_path = self.ctx.paths.models / f"multiclass__{main[0]}__{main[1]}.joblib"
        self.caller_model = joblib.load(mc_path) if mc_path.exists() else None
        if self.caller_model is not None and self.caller_model.get("run") != self.ctx.run_name:
            log.info("The caller-type model was trained on the KG of run %s, not %s - not used",
                        self.caller_model.get("run"), self.ctx.run_name)
            self.caller_model = None
        self.caller_labeled = ([c for c in self.caller_model["labeled_ids"] if c in self.ctx.calls.index]
                               if self.caller_model is not None else [])

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
        built = {}

        def features(fitted, labeled: list[str]):
            """Feature row of the new call for one trained model (graph features built once per label set)."""
            graph_row = oh = None
            if (fitted.graph_columns or fitted.onehot_names) and tmp.fb is None:
                tmp.fb = GraphFeatureBuilder(kg, ctx.cfg["graph"], seed=ctx.seed)
            if fitted.graph_columns:
                key = tuple(labeled)
                if key not in built:
                    built[key] = tmp.fb.build(labeled, tmp.labels(labeled))
                graph_row = built[key].loc[[NEW_ID]]
            if fitted.onehot_names:
                oh = tmp.fb.onehot_matrix([NEW_ID], fitted.onehot_names)
            return fitted.transform_parts([plain], graph_row, oh)

        score = float(self.bundle["model"].predict_proba(features(fitted, self.labeled_ids))[0, 1])
        thr = float(self.cfg["models"].get("threshold", 0.5))
        explanation = explain_call(tmp, NEW_ID, score, labeled_ids=self.labeled_ids)
        out = {"score": score, "is_scam": score >= thr, "triples": rec.triples, "explanation": explanation,
               "model": f"{self.bundle['experiment']}/{self.bundle['model_name']}", "extractor": self.extractor.name}
        if self.caller_model is not None:
            mc = self.caller_model
            p = mc["model"].predict_proba(features(mc["fitted"], self.caller_labeled))[0]
            probs = {str(c): round(float(v), 4) for c, v in zip(mc["model"].classes_, p)}
            out.update(caller_type=max(probs, key=probs.get), caller_type_probs=probs,
                       caller_type_model=f"{mc['experiment']}/{mc['model_name']}")
        return out
