"""Graph statistics and static visualizations of the call KG."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scamkg.graph.build import CallKG, call_node, split_key  # noqa: E402

TYPE_COLORS = {"Call": "#9e9e9e", "ClaimedIdentity": "#4C78A8", "Organization": "#72B7B2", "PressureTactic": "#F58518",
               "RequestedAsset": "#E45756", "PaymentChannel": "#54A24B", "CallbackNumber": "#B279A2"}


def graph_summary(kg: CallKG, calls: pd.DataFrame, structural: pd.DataFrame | None = None) -> dict:
    G = kg.graph
    type_counts = pd.Series(kg.entity_types).value_counts().to_dict() if len(kg.entity_types) else {}
    rel_counts = kg.relation_counts().drop(columns=["n_inferred", "has_entities"]).sum().to_dict()
    labels = calls.reindex(kg.call_ids)["label"]
    rc = kg.relation_counts()
    per_label = {}
    for lab in (0, 1):
        ids = labels.index[labels == lab]
        if len(ids):
            per_label["scam" if lab else "legitimate"] = {
                "mean_entities": float(rc.loc[ids].drop(columns=["n_inferred", "has_entities"]).sum(axis=1).mean()),
                "share_without_entities": float(1 - rc.loc[ids, "has_entities"].mean())}
    top = []
    Xc = kg.X.tocsc()
    lab_arr = labels.to_numpy().astype(float)
    lab_arr[lab_arr < 0] = np.nan
    for j in np.argsort(-kg.entity_degree)[:25]:
        t, name = split_key(kg.entity_ids[j])
        rows = Xc.indices[Xc.indptr[j]:Xc.indptr[j + 1]]
        vals = lab_arr[rows]
        scam_rate = float(np.nanmean(vals)) if np.isfinite(vals).any() else None
        top.append({"type": t, "entity": name, "calls": int(kg.entity_degree[j]), "scam_rate_all_labels": scam_rate})
    out = {"calls": kg.n_calls, "entities": len(kg.entity_ids), "entities_by_type": type_counts,
           "edges": G.number_of_edges(), "edges_by_relation": {k.replace("n_", ""): int(v) for k, v in rel_counts.items()},
           "connected_components": nx.number_connected_components(G), "per_label": per_label, "top_entities": top}
    if structural is not None:
        out["communities"] = int(structural["community_id"].nunique())
        out["largest_community"] = int(structural["community_size"].max())
        out["mean_cc_degree"] = float(structural["cc_degree"].mean())
    return out


def plot_kg_sample(kg: CallKG, calls: pd.DataFrame, path: Path, n_calls: int = 40, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    labeled = [c for c in kg.call_ids if c in calls.index and calls.at[c, "label"] >= 0 and kg.triples[c]]
    if not labeled:
        return
    chosen = list(rng.choice(labeled, size=min(n_calls, len(labeled)), replace=False))
    nodes = {call_node(c) for c in chosen}
    for c in chosen:
        nodes.update(kg.entities_of(c))
    H = kg.graph.subgraph(nodes)
    pos = nx.spring_layout(H, seed=seed, k=0.6)
    fig, ax = plt.subplots(figsize=(11, 8))
    colors, sizes = [], []
    for v in H.nodes:
        kind = H.nodes[v].get("kind", "Call")
        if kind == "Call":
            lab = calls.at[H.nodes[v]["call_id"], "label"]
            colors.append("#d62728" if lab == 1 else "#1f77b4")
            sizes.append(60)
        else:
            colors.append(TYPE_COLORS.get(kind, "#bbbbbb"))
            sizes.append(80 + 25 * H.degree(v))
    nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.25, width=0.8)
    nx.draw_networkx_nodes(H, pos, ax=ax, node_color=colors, node_size=sizes, linewidths=0)
    ent_labels = {v: H.nodes[v]["name"] for v in H.nodes if H.nodes[v].get("kind") != "Call" and H.degree(v) >= 2}
    nx.draw_networkx_labels(H, pos, labels=ent_labels, font_size=7, ax=ax)
    handles = [plt.Line2D([], [], marker="o", ls="", color="#d62728", label="scam call"),
               plt.Line2D([], [], marker="o", ls="", color="#1f77b4", label="legitimate call")]
    handles += [plt.Line2D([], [], marker="o", ls="", color=c, label=t) for t, c in TYPE_COLORS.items() if t != "Call"]
    ax.legend(handles=handles, loc="lower left", fontsize=8, frameon=False)
    ax.set_title(f"Knowledge-graph sample: {len(chosen)} calls and their ontology entities")
    ax.axis("off")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_entity_degrees(kg: CallKG, path: Path, top: int = 25) -> None:
    order = np.argsort(-kg.entity_degree)[:top]
    if not len(order):
        return
    names = [f"{split_key(kg.entity_ids[i])[0][:10]}: {split_key(kg.entity_ids[i])[1]}" for i in order]
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(order) + 1))
    ax.barh(range(len(order)), kg.entity_degree[order],
            color=[TYPE_COLORS.get(kg.entity_types[i], "#999") for i in order])
    ax.set_yticks(range(len(order)), names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("linked calls (degree)")
    ax.set_title("Most connected ontology entities")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
