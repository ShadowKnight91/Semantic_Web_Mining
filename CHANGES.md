# Code review changes (after the proposal feedback)

The code already implemented the proposal (E1-E5, X1, X2, ablations, Neo4j, RDF). The changes below
answer the two TA comments and fix a few correctness problems found while checking it. Every changed
spot in the code is marked with a `CHANGE` comment that gives the reason.

Tests: 30 -> 42 (`python -m pytest`); all pass. The offline demo (`run-all`) runs end to end.

## TA comment 1 - fallback / retry for malformed JSON, out-of-schema entities, missing fields

| File | Change | Reason |
|---|---|---|
| `scamkg/extraction/extractor.py` | `salvage_triples()` recovers the complete triples from an answer cut off at the token cap | before, one truncated answer lost every triple of that call |
| same | retry when the JSON parses but **no** triple fits the ontology (`schema_retry`) | before, off-schema answers were silently dropped with no retry |
| `scamkg/extraction/prompts.py` | error-specific retry messages (invalid JSON / missing `triples` key / off-schema list with the allowed relations and what was wrong) | one generic message for every failure fixed fewer errors |
| `extractor.py`, `config.yaml` | `llm.fallback_model` (`qwen3-30b-a3b` on Voyager; `gemma2:9b` in `config/local.yaml` for Ollama) runs when the primary model gives nothing usable or the request fails; `--no-fallback` for the model / prompt comparisons | the TA asked for a fallback; comparisons must still measure one model each |
| `extractor.py` | per-call `status`, `attempts`, `retry_reasons`, `salvaged`, `fallback_used`, `model_used`; run-level `retry_rate`, `salvage_rate`, `fallback_rate`, `failed_rate` (also in `extraction_quality.csv`) | makes the fallback logic measurable for the report |
| `extractor.py` | `run_extraction` retries calls whose last record failed when the run is resumed, and keeps one record per call | before, one Ollama timeout lost that call for good, and resumed runs double-counted calls |
| `extractor.py` | a fallback run gets its own name (`..._fb-<fallback model>`) | so it never mixes with a single-model comparison run |
| `cli.py` | `run-all` passes `no_fallback` to `extract` | the new flag would otherwise crash `run-all` |

## TA comment 2 - combining text representations with topological features before GraphSAGE

| File | Change | Reason |
|---|---|---|
| `scamkg/models/gnn.py` | three node-feature blocks: text (SBERT / TF-IDF-SVD), topology (19 label-free features: degree, PageRank, betweenness, call-call degree / PageRank, community size, relation counts; entity degree / PageRank), node type | before, GraphSAGE got only text + node type; the graph features never reached it |
| same | each block gets its own linear projection (64 / 16 / 8), then concat + LayerNorm, then 2 mean-aggregation SAGE layers | plain concatenation lets the 384 text dimensions drown out ~19 topology dimensions |
| same, `config.yaml` | variants `text+topo` (main, `graphsage`), `text`, `topo` | block ablation for RQ1 |
| same | inductive training: the loss graph has no val / test / external calls; early stopping uses train + val; scoring uses the full graph | test and external calls were inside the training graph, which makes the X2 numbers optimistic |
| same | early stopping breaks PR-AUC ties with validation loss | PR-AUC reached 1.0 at epoch 0 on easy data and the barely trained model was kept (F1 0.93 instead of 1.00 on the demo) |
| `evaluation/report.py` | REPORT.md lists the fusion inputs and the variant results; `results/graphsage_fusion.json` | documents the design for the presentation |

## Other fixes

| File | Change | Reason |
|---|---|---|
| `scamkg/graph/features.py` | label-based features (community scam rate, entity risk, ...) for **training** calls use 5-fold out-of-fold values instead of leave-one-out (`graph.label_folds: 5`) | with leave-one-out, scams get a slightly lower value than legitimate calls in the same community; tree models learn that and it reverses on test data. Simulation (1,200 calls, 60 communities, 20 seeds): XGBoost test PR-AUC 0.587 (LOO) vs 0.659 (OOF); logistic regression 0.699 both |
| `scamkg/extraction/evaluate.py` | X1 keeps calls annotated as having **no** facts (`"annotated": true`) | they were dropped, so false-positive triples on fact-free calls were never counted and triple precision was inflated |
| `scamkg/models/llm_zeroshot.py` | a failed answer scores 0.0 ("no alert") instead of 0.5; "legitimate, confidence 0.9" becomes scam probability 0.1; failed requests are retried on resume | 0.5 was counted as a scam detection by the >= 0.5 threshold (inflated recall); the confidence of a "legitimate" answer was read as a scam probability |
| `scamkg/data/preprocess.py` | a transcript that appears with both labels is dropped | the first copy used to be kept with an arbitrary label |
| `config/config.yaml` | Kaggle slug `teeconnie/scam-and-non-scam-call-conversation-dataset` | the data set the proposal now cites as [18] |
| `README.md` | documents all of the above, the new commands and both TA answers | |

## Not verified here

* LLM extraction with real Ollama models (tested with a fake Ollama server that returns prose,
  cut-off JSON, off-schema triples and bare lists; every case ended in the expected status).
* Neo4j / GDS backend (no Neo4j server available).
* E3 (DistilBERT, SBERT) and the SBERT text block: the model weights could not be downloaded in the
  test environment; GraphSAGE was tested with the TF-IDF + SVD text block.
* Real BothBosu / Kaggle data (download blocked in the test environment). The NCSU loader was checked on
  the real `metadata.csv` (1,378 English robocalls).

## Merged with the `review-fixes` work (Neo4j / LLM verification, Kaggle, X1 gold)

Both branches fixed some of the same things independently; where they overlapped, one version was kept:

| Topic | Kept | Note |
|---|---|---|
| Call ids | stable per-data-set ids (`preprocess.py`) | adding a data set no longer shifts another one's ids. The committed Gemma run was re-keyed to these ids by transcript text (562 NCSU ids moved by one; every call matched exactly once, 99.3% of its evidence quotes are found in the matching transcript) |
| Conflicting labels | dropped (this branch) | applied after the ids are assigned, so it cannot shift them either |
| Out-of-fold label features | one implementation, `graph.label_folds: 5` (replaces `label_encoding` / `label_encoding_folds`) | plain `KFold`, so a call's own label never decides its fold |
| Kaggle `.txt` loader | the `review-fixes` reader (English files only) + the `legit` / `fraud` file-name rules from this branch | |
| Annotation flag | `"done": true` (used by the committed `annotator_A.jsonl`); `"annotated": true` is accepted too | fact-free annotated calls count in X1, as in this branch |
| GraphSAGE | the fusion model of this branch; `graphsage_xgb` runs as a separate, torch-free step | PyTorch and XGBoost crash in one process on macOS; `train` runs E3 / E4 in a spawned process, and so does `tests/test_gnn.py` |
| `run-all` | builds the graph from the run it just extracted | `extraction.default_run` (the Gemma run) is for the individual commands |

Tests: 56, all pass.

## Annotator B (X1 inter-annotator agreement)

`annotator_B.jsonl` was still the empty template. It now holds a second, independent AI-assisted pass over the
50 calls (blind to annotator A and to every extractor), which the course instructor approved. Agreement with A:
triple F1 0.993 on 50 calls, 3 borderline disagreements. `report` now prints this agreement under the X1 table;
`data/annotations/PROVENANCE.md` documents both passes and why the number is an upper bound.

## Caller types: fraudster vs. prankster vs. salesperson (proposal)

The proposal asks to tell fraudsters, pranksters and salespeople apart. Only fraudster / salesperson /
other-legitimate labels existed (BothBosu types), and no data set has a single prank call (Gemma's zero-shot
categories on ~1,000 real calls contain none either), so the old three-way run was saturated and had no
prankster class.

| What | Where |
|---|---|
| 120 generated prank calls (28 scenarios, several imitating a scam or sales opening) + 36 fraud / sales calls by the same model and prompt template as a test-only style check; `gemma2:9b` via Ollama, seeded and cached | `scamkg/data/pranks.py`, `generate-pranks`, `data/pranks/pranks.jsonl` |
| `pranks` loader; `data.split_datasets` gives the pranks their own stratified split (BothBosu's split is identical) | `scamkg/data/loaders.py`, `scamkg/data/preprocess.py` |
| Separate setup, so the binary results and data stay untouched | `config/multiclass.yaml`, `run-multiclass` |
| Caller-type models: E1 / E2 / E2G x LR / RF, per-class F1, confusion matrices, style check; main model saved | `run_multiclass` in `scamkg/models/classical.py` |
| `predict` and the dashboard ("Try a transcript", new "Caller types" tab) report the caller type | `scamkg/predict.py`, `dashboard/app.py` |

Results (`results/multiclass/REPORT.md`, rule-based KG): on the test split (240 BothBosu + 18 prank calls) text
and text + KG models are perfect, so the split cannot separate them. The style check can: text-only models label
22% (LR) / 53% (RF) of the same-generator fraud / sales calls `prankster`, the KG models 0% (E2 / LR gets all 36
right; KG-only E2G 97%). The KG alone is weak at telling pranks from other legitimate calls (prankster F1 about
0.5), because pranks state few ontology facts. Limits: synthetic pranks, 18 test pranks, 36 control calls.
