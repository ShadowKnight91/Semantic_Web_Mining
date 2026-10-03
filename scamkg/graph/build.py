"""Heterogeneous call knowledge graph G = (V, E): Call nodes linked to ontology entity nodes.

Two calls become connected through shared claimed identities, requested assets, tactics, payment
channels, organizations and call-back numbers. A call-call similarity graph (Jaccard over shared
entities) is derived from the call x entity incidence matrix.
"""
from __future__ import annotations

from functools import cached_property

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from scamkg.ontology.schema import Ontology


def entity_key(type_name: str, obj: str) -> str:
    return f"{type_name}::{obj}"


def call_node(call_id: str) -> str:
    return f"Call::{call_id}"


def split_key(key: str) -> tuple[str, str]:
    t, _, name = key.partition("::")
    return t, name


class CallKG:
    def __init__(self, call_ids: list[str], extractions: dict[str, list[dict]], ontology: Ontology,
                 exclude_types: tuple | list | set = ()):
        self.ontology = ontology
        self.exclude_types = set(exclude_types)
        self.call_ids = list(call_ids)
        self.call_index = {c: i for i, c in enumerate(self.call_ids)}
        self.triples: dict[str, list[dict]] = {}
        entity_index: dict[str, int] = {}
        rows, cols = [], []
        for cid in self.call_ids:
            kept, seen = [], set()
            for t in extractions.get(cid, []) or []:
                if t.get("object_type") in self.exclude_types or t.get("relation") not in ontology.relations:
                    continue
                key = entity_key(t["object_type"], str(t["object"]))
                if key in seen:
                    continue
                seen.add(key)
                kept.append(t)
                if key not in entity_index:
                    entity_index[key] = len(entity_index)
                rows.append(self.call_index[cid])
                cols.append(entity_index[key])
            self.triples[cid] = kept
        self.entity_index = entity_index
        self.entity_ids = list(entity_index)
        self.entity_types = np.array([split_key(k)[0] for k in self.entity_ids])
        self.X = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                                   shape=(len(self.call_ids), len(self.entity_ids)))
        self.X.data[:] = 1.0

    # ------------------------------------------------------------------ basic views
    @property
    def n_calls(self) -> int:
        return len(self.call_ids)

    @cached_property
    def entity_degree(self) -> np.ndarray:
        """Number of calls linked to each entity."""
        return np.asarray(self.X.sum(axis=0)).ravel()

    def entities_of(self, call_id: str) -> list[str]:
        i = self.call_index[call_id]
        return [self.entity_ids[j] for j in self.X.indices[self.X.indptr[i]:self.X.indptr[i + 1]]]

    def relation_counts(self) -> pd.DataFrame:
        rels = self.ontology.relation_names
        data = np.zeros((self.n_calls, len(rels)), dtype=np.int32)
        inferred = np.zeros(self.n_calls, dtype=np.int32)
        ridx = {r: k for k, r in enumerate(rels)}
        for cid, ts in self.triples.items():
            i = self.call_index[cid]
            for t in ts:
                data[i, ridx[t["relation"]]] += 1
                inferred[i] += int(bool(t.get("inferred")))
        df = pd.DataFrame(data, index=self.call_ids, columns=[f"n_{r}" for r in rels])
        df["n_inferred"] = inferred
        df["has_entities"] = (data.sum(axis=1) > 0).astype(int)
        return df

    # ------------------------------------------------------------------ graphs
    @cached_property
    def graph(self) -> nx.Graph:
        """Undirected heterogeneous graph (Call + entity nodes) for centrality algorithms."""
        G = nx.Graph()
        for cid in self.call_ids:
            G.add_node(call_node(cid), kind="Call", call_id=cid)
        for key in self.entity_ids:
            t, name = split_key(key)
            G.add_node(key, kind=t, name=name)
        for cid, ts in self.triples.items():
            for t in ts:
                G.add_edge(call_node(cid), entity_key(t["object_type"], str(t["object"])), relation=t["relation"],
                           inferred=bool(t.get("inferred")))
        return G

    @cached_property
    def jaccard(self) -> np.ndarray:
        """Dense call x call Jaccard similarity over entity sets (float32)."""
        X = self.X
        inter = (X @ X.T).toarray().astype(np.float32)
        size = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
        union = size[:, None] + size[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            J = np.where(union > 0, inter / union, 0.0).astype(np.float32)
        return J

    def call_call_graph(self, top_k: int = 10, min_sim: float = 0.2) -> nx.Graph:
        """Top-k Jaccard neighbours per call (>= min_sim), symmetrized, weighted."""
        J = self.jaccard.copy()
        np.fill_diagonal(J, 0.0)
        C = nx.Graph()
        C.add_nodes_from(self.call_ids)
        n = J.shape[0]
        if n < 2:
            return C
        k = min(top_k, n - 1)
        idx = np.argpartition(-J, k - 1, axis=1)[:, :k]
        for i in range(n):
            for j in idx[i]:
                w = float(J[i, j])
                if w >= min_sim and w > 0:
                    a, b = self.call_ids[i], self.call_ids[j]
                    if not C.has_edge(a, b):
                        C.add_edge(a, b, weight=w)
        return C

    # ------------------------------------------------------------------ exports
    def to_frames(self, calls: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        nodes = [{"id": call_node(c), "kind": "Call", "name": c,
                  "dataset": calls.at[c, "dataset"] if calls is not None else "",
                  "split": calls.at[c, "split"] if calls is not None else "",
                  "label": int(calls.at[c, "label"]) if calls is not None else -1} for c in self.call_ids]
        for key, deg in zip(self.entity_ids, self.entity_degree):
            t, name = split_key(key)
            nodes.append({"id": key, "kind": t, "name": name, "dataset": "", "split": "", "label": -1,
                          "degree": int(deg)})
        edges = [{"source": call_node(c), "target": entity_key(t["object_type"], str(t["object"])),
                  "relation": t["relation"], "inferred": bool(t.get("inferred")), "detail": t.get("detail", ""),
                  "evidence": t.get("evidence", "")} for c, ts in self.triples.items() for t in ts]
        return pd.DataFrame(nodes), pd.DataFrame(edges)

    def write_graphml(self, path, calls: pd.DataFrame | None = None) -> None:
        G = nx.Graph()
        nodes, edges = self.to_frames(calls)
        for r in nodes.to_dict(orient="records"):
            # GraphML needs one type per attribute key -> store node attributes as strings
            G.add_node(r.pop("id"), **{k: ("" if v is None or (isinstance(v, float) and np.isnan(v)) else
                                           str(int(v)) if isinstance(v, float) else str(v))
                                        for k, v in r.items()})
        for r in edges.to_dict(orient="records"):
            G.add_edge(r["source"], r["target"], relation=r["relation"], inferred=r["inferred"],
                       detail=str(r["detail"]))
        nx.write_graphml(G, path)
