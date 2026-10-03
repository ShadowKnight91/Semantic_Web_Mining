"""Explainability: for a flagged call, the KG subgraph (identity -> tactic -> asset -> payment path)
that supports the prediction, the training scam rate of each entity, and the most similar known calls."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scamkg.graph.build import entity_key, split_key
from scamkg.graph.neo4j_store import Neo4jStore
from scamkg.utils import dumps, get_logger, slugify

log = get_logger(__name__)
REL_ORDER = ["CLAIMS_TO_BE", "MENTIONS_ORG", "USES_TACTIC", "REQUESTS", "PAY_VIA", "PROVIDES_NUMBER"]
TYPE_COLORS = {"ClaimedIdentity": "#4C78A8", "Organization": "#72B7B2", "PressureTactic": "#F58518",
               "RequestedAsset": "#E45756", "PaymentChannel": "#54A24B", "CallbackNumber": "#B279A2"}


def _esc(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def to_dot(call_id: str, entities: list[dict], similar: list[dict], score: float | None) -> str:
    title = f"{call_id}\\nscore={score:.2f}" if score is not None else call_id
    lines = ["digraph G {", "rankdir=LR;", 'node [fontname="Helvetica", fontsize=10, style=filled];',
             f'"call" [label="{_esc(title)}", shape=box, fillcolor="#fde0dd"];']
    for k, e in enumerate(entities):
        rate = e.get("train_scam_rate")
        lab = f"{e['object']}" + (f"\\n{rate:.0%} scam of {e['train_calls']}" if rate is not None else "\\n(unseen)")
        lines.append(f'"e{k}" [label="{_esc(lab)}", shape=ellipse, fillcolor="{TYPE_COLORS.get(e["type"], "#ddd")}", '
                     f'fontcolor="white"];')
        style = ', style=dashed' if e.get("inferred") else ""
        lines.append(f'"call" -> "e{k}" [label="{e["relation"]}"{style}];')
    for k, s in enumerate(similar):
        lab = f"{s['call_id']}\\n{'scam' if s['label'] == 1 else 'legit'} J={s['jaccard']:.2f}"
        color = "#f4a6a6" if s["label"] == 1 else "#a6c8f4"
        lines.append(f'"s{k}" [label="{_esc(lab)}", shape=box, fillcolor="{color}"];')
        lines.append(f'"call" -> "s{k}" [label="SIMILAR_TO", style=dotted, arrowhead=none];')
    lines.append("}")
    return "\n".join(lines)


def explain_call(ctx, call_id: str, score: float | None = None, labeled_ids: list[str] | None = None,
                 top_similar: int = 3) -> dict:
    kg = ctx.kg
    labeled_ids = labeled_ids if labeled_ids is not None else ctx.train_ids + ctx.val_ids
    labeled_ids = [c for c in labeled_ids if c in kg.call_index]
    lab_idx = np.array([kg.call_index[c] for c in labeled_ids], dtype=int)
    y = ctx.labels(labeled_ids).astype(float)
    i = kg.call_index[call_id]
    XL = kg.X[lab_idx]
    e_sum = np.asarray(XL.T @ y).ravel()
    e_cnt = np.asarray(XL.sum(axis=0)).ravel()
    self_lab = call_id in set(labeled_ids)
    own_y = float(ctx.calls.at[call_id, "label"]) if self_lab else 0.0
    entities = []
    triples = sorted(kg.triples.get(call_id, []), key=lambda t: REL_ORDER.index(t["relation"])
                     if t["relation"] in REL_ORDER else 99)
    for t in triples:
        j = kg.entity_index[entity_key(t["object_type"], str(t["object"]))]
        n = e_cnt[j] - (1 if self_lab else 0)
        s = e_sum[j] - (own_y if self_lab else 0)
        entities.append({"relation": t["relation"], "type": t["object_type"], "object": t["object"],
                         "detail": t.get("detail", ""), "evidence": t.get("evidence", ""),
                         "inferred": bool(t.get("inferred")), "train_calls": int(n),
                         "train_scam_rate": float(s / n) if n > 0 else None})
    sims = kg.jaccard[i, lab_idx].astype(float).copy()
    if self_lab:
        sims[labeled_ids.index(call_id)] = -1
    similar = []
    mine = set(kg.entities_of(call_id))
    for k in np.argsort(-sims)[:top_similar]:
        if sims[k] <= 0:
            break
        other = labeled_ids[k]
        shared = sorted(split_key(e)[1] for e in mine & set(kg.entities_of(other)))
        similar.append({"call_id": other, "label": int(y[k]), "jaccard": float(sims[k]), "shared": shared})
    path = " -> ".join(f"{e['relation']}:{e['object']}" for e in entities
                       if e["relation"] in ("CLAIMS_TO_BE", "USES_TACTIC", "REQUESTS", "PAY_VIA"))
    risky = [e for e in entities if (e["train_scam_rate"] or 0) >= 0.7 and e["train_calls"] >= 3]
    parts = []
    if risky:
        parts.append("high-risk entities: " + ", ".join(
            f"{e['object']} ({e['train_scam_rate']:.0%} of {e['train_calls']} known calls were scams)" for e in risky[:4]))
    if similar:
        s0 = similar[0]
        parts.append(f"most similar known call {s0['call_id']} is {'a scam' if s0['label'] else 'legitimate'} "
                     f"(Jaccard {s0['jaccard']:.2f}, shares {', '.join(s0['shared'][:4])})")
    if not entities:
        parts.append("no ontology entities were extracted - the prediction relies on text features only")
    row = ctx.calls.loc[call_id]
    return {"call_id": call_id, "dataset": row["dataset"], "split": row["split"], "gold_label": int(row["label"]),
            "score": score, "script_path": path, "entities": entities, "similar_calls": similar,
            "summary": "; ".join(parts), "cypher": Neo4jStore.explain_query(call_id),
            "dot": to_dot(call_id, entities, similar, score)}


def run_explanations(ctx, experiment: str = "E2", model: str = "xgb", max_calls: int = 300) -> int:
    ctx.require_graph()
    pred_file = ctx.paths.predictions / f"{slugify(experiment)}__{slugify(model)}.csv"
    if not pred_file.exists():
        log.warning("No predictions for %s/%s - train the model first", experiment, model)
        return 0
    thr = float(ctx.cfg["models"].get("threshold", 0.5))
    pred = pd.read_csv(pred_file, dtype={"call_id": str})
    flagged = pred[pred["score"] >= thr].sort_values("score", ascending=False).head(max_calls)
    out = ctx.paths.results / "explanations.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in flagged.itertuples():
            f.write(dumps(explain_call(ctx, r.call_id, float(r.score))) + "\n")
    log.info("Explanations for %d flagged calls (%s/%s) -> %s", len(flagged), experiment, model, out)
    return len(flagged)
