"""Graph features for every call (Phase 4).

Label-free ("structural") features are computed once on the full graph - transductive but
unsupervised, so no labels leak:
  het_degree, het_pagerank, het_betweenness            (heterogeneous Call-Entity graph)
  cc_degree, cc_weighted_degree, cc_pagerank          (call-call Jaccard similarity graph)
  community_id, community_size                        (Louvain on the call-call graph)
  entity_mean/max_popularity, n_<RELATION>, n_inferred, has_entities

Label-based features are recomputed for every training set and use ONLY the labels of the
"labeled" calls (train, or train+val for the final model). Training calls get out-of-fold values
(K-fold cross-fitting, `graph.label_folds`), so their own label never enters their own features:
  community_scam_rate, sim_scam_max, sim_legit_max, sim_scam_topk, sim_legit_topk, sim_diff,
  knn_scam_rate, entity_risk_mean/max/min, n_known_entities
Leave-one-out alone is not enough: within a community (or for an entity) the LOO rate of a scam
call is always lower than that of a legitimate one, so a tree can read the label back from it.
K-fold values break that deterministic link (the same cross-fitting as scikit-learn's TargetEncoder).
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import KFold

from scamkg.graph.build import CallKG, call_node
from scamkg.ontology.schema import Ontology
from scamkg.utils import get_logger

log = get_logger(__name__)

STRUCTURAL = ["het_degree", "het_pagerank", "het_betweenness", "cc_degree", "cc_weighted_degree", "cc_pagerank",
              "entity_mean_popularity", "entity_max_popularity"]
COMMUNITY = ["community_size", "community_scam_rate"]
SIMILARITY = ["sim_scam_max", "sim_legit_max", "sim_scam_topk", "sim_legit_topk", "sim_diff", "knn_scam_rate"]
ENTITY_RISK = ["entity_risk_mean", "entity_risk_max", "entity_risk_min", "n_known_entities"]
ONEHOT_GROUP = "ontology_onehot"
LABEL_BASED = ["community_scam_rate", *SIMILARITY, *ENTITY_RISK]


def feature_groups(ontology: Ontology) -> dict[str, list[str]]:
    return {"structural": STRUCTURAL, "community": COMMUNITY, "similarity": SIMILARITY, "entity_risk": ENTITY_RISK,
            "relation_counts": [f"n_{r}" for r in ontology.relation_names] + ["n_inferred", "has_entities"]}


def local_features(kg: CallKG) -> pd.DataFrame:
    """Features that only need the incidence matrix (shared by the networkx and Neo4j backends)."""
    pop = kg.entity_degree / max(1, kg.n_calls)
    mean_pop = np.zeros(kg.n_calls)
    max_pop = np.zeros(kg.n_calls)
    X = kg.X
    for i in range(kg.n_calls):
        ents = X.indices[X.indptr[i]:X.indptr[i + 1]]
        if len(ents):
            mean_pop[i], max_pop[i] = pop[ents].mean(), pop[ents].max()
    df = kg.relation_counts()
    df["entity_mean_popularity"] = mean_pop
    df["entity_max_popularity"] = max_pop
    return df


def structural_features(kg: CallKG, gcfg: dict, seed: int = 42) -> pd.DataFrame:
    """Label-free graph features with networkx (mirrors the Neo4j GDS algorithms)."""
    G = kg.graph
    nodes = [call_node(c) for c in kg.call_ids]
    n_nodes = G.number_of_nodes()
    log.info("Structural features on %d nodes / %d edges", n_nodes, G.number_of_edges())
    pr = nx.pagerank(G, alpha=0.85, max_iter=200) if G.number_of_edges() else {v: 1.0 / max(1, n_nodes) for v in G}
    k = int(gcfg.get("betweenness_samples", 300))
    bc = nx.betweenness_centrality(G, k=k if 0 < k < n_nodes else None, seed=seed, normalized=True)
    C = kg.call_call_graph(int(gcfg.get("similarity_top_k", 10)), float(gcfg.get("similarity_min", 0.2)))
    if C.number_of_edges():
        cc_pr = nx.pagerank(C, weight="weight", max_iter=200)
        comms = nx.community.louvain_communities(C, weight="weight",
                                                 resolution=float(gcfg.get("louvain_resolution", 1.0)), seed=seed)
    else:
        cc_pr = {c: 1.0 / max(1, kg.n_calls) for c in kg.call_ids}
        comms = [{c} for c in kg.call_ids]
    comm_of, comm_size = {}, {}
    for cid, members in enumerate(sorted(comms, key=len, reverse=True)):
        for m in members:
            comm_of[m] = cid
        comm_size[cid] = len(members)
    df = pd.DataFrame(index=kg.call_ids)
    df["het_degree"] = [G.degree(v) for v in nodes]
    df["het_pagerank"] = [pr[v] * n_nodes for v in nodes]  # scaled so the mean is ~1
    df["het_betweenness"] = [bc[v] for v in nodes]
    df["cc_degree"] = [C.degree(c) for c in kg.call_ids]
    df["cc_weighted_degree"] = [C.degree(c, weight="weight") for c in kg.call_ids]
    df["cc_pagerank"] = [cc_pr[c] * kg.n_calls for c in kg.call_ids]
    df["community_id"] = [comm_of[c] for c in kg.call_ids]
    df["community_size"] = [comm_size[comm_of[c]] for c in kg.call_ids]
    df = df.join(local_features(kg))
    log.info("Louvain: %d communities (largest %d calls)", len(comm_size), max(comm_size.values()) if comm_size else 0)
    return df


def label_features(kg: CallKG, labeled_idx: np.ndarray, y_lab: np.ndarray, community: np.ndarray,
                   gcfg: dict) -> pd.DataFrame:
    """Leak-safe label-derived features using only (labeled_idx, y_lab); leave-one-out for labeled calls."""
    n = kg.n_calls
    labeled_idx = np.asarray(labeled_idx, dtype=int)
    y_lab = np.asarray(y_lab, dtype=float)
    m = float(gcfg.get("smoothing", 2.0))
    k = int(gcfg.get("knn_k", 10))
    topk = int(gcfg.get("sim_topk", 5))
    prior = float(y_lab.mean()) if len(y_lab) else 0.5
    is_lab = np.zeros(n, dtype=bool)
    is_lab[labeled_idx] = True
    y_full = np.zeros(n)
    y_full[labeled_idx] = y_lab
    own = y_full * is_lab

    # community scam rate (leave-one-out, smoothed)
    _, inv = np.unique(np.asarray(community), return_inverse=True)
    c_sum = np.bincount(inv, weights=own)
    c_cnt = np.bincount(inv, weights=is_lab.astype(float))
    comm_rate = (c_sum[inv] - own + prior * m) / (c_cnt[inv] - is_lab + m)

    # similarity to labeled calls (self excluded)
    S = kg.jaccard[:, labeled_idx].astype(np.float32, copy=True)
    if len(labeled_idx):
        S[labeled_idx, np.arange(len(labeled_idx))] = -1.0
    pos, neg = y_lab == 1, y_lab == 0

    def _max(M: np.ndarray) -> np.ndarray:
        return M.max(axis=1).clip(min=0) if M.shape[1] else np.zeros(n)

    def _topk_mean(M: np.ndarray, kk: int) -> np.ndarray:
        if M.shape[1] == 0:
            return np.zeros(n)
        kk = min(kk, M.shape[1])
        part = -np.partition(-M, kk - 1, axis=1)[:, :kk]
        return part.clip(min=0).mean(axis=1)

    sim_scam_max, sim_legit_max = _max(S[:, pos]), _max(S[:, neg])
    sim_scam_topk, sim_legit_topk = _topk_mean(S[:, pos], topk), _topk_mean(S[:, neg], topk)
    if S.shape[1]:
        kk = min(k, S.shape[1])
        idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        w = np.take_along_axis(S, idx, axis=1).clip(min=0)
        eps = 1e-6
        knn = ((w * y_lab[idx]).sum(axis=1) + prior * eps) / (w.sum(axis=1) + eps)
    else:
        knn = np.full(n, prior)

    # entity risk: smoothed leave-one-out scam rate of each linked entity among labeled calls
    X = kg.X
    XL = X[labeled_idx]
    e_sum = np.asarray(XL.T @ y_lab).ravel() if len(labeled_idx) else np.zeros(X.shape[1])
    e_cnt = np.asarray(XL.sum(axis=0)).ravel() if len(labeled_idx) else np.zeros(X.shape[1])
    r_mean, r_max, r_min = np.full(n, prior), np.full(n, prior), np.full(n, prior)
    n_known = np.zeros(n)
    for i in range(n):
        ents = X.indices[X.indptr[i]:X.indptr[i + 1]]
        if not len(ents):
            continue
        s = e_sum[ents] - own[i]
        c = e_cnt[ents] - float(is_lab[i])
        denom = c + m
        r = np.where(denom > 0, (s + prior * m) / np.where(denom > 0, denom, 1.0), prior)
        r_mean[i], r_max[i], r_min[i] = r.mean(), r.max(), r.min()
        n_known[i] = float((c > 0).sum())

    return pd.DataFrame({
        "community_scam_rate": comm_rate, "sim_scam_max": sim_scam_max, "sim_legit_max": sim_legit_max,
        "sim_scam_topk": sim_scam_topk, "sim_legit_topk": sim_legit_topk, "sim_diff": sim_scam_max - sim_legit_max,
        "knn_scam_rate": knn, "entity_risk_mean": r_mean, "entity_risk_max": r_max, "entity_risk_min": r_min,
        "n_known_entities": n_known}, index=kg.call_ids)


class GraphFeatureBuilder:
    """Caches structural features of a KG and builds leak-safe feature tables for a labeled set."""

    def __init__(self, kg: CallKG, gcfg: dict, structural: pd.DataFrame | None = None, seed: int = 42):
        self.kg = kg
        self.gcfg = gcfg
        self.seed = seed
        if structural is not None:
            structural = structural.reindex(kg.call_ids)
            if structural.isna().any().any():
                log.warning("Cached structural features do not cover every call - recomputing")
                structural = None
        self.structural = structural if structural is not None else structural_features(kg, gcfg, seed)
        self.groups = feature_groups(kg.ontology)

    def build(self, labeled_ids: list[str], labels) -> pd.DataFrame:
        """Structural features + label-based features from `labels` of `labeled_ids` only.

        Unlabeled calls (val / test / external) see every label. Labeled (training) calls get K-fold
        out-of-fold values (`graph.label_folds`; leave-one-out when there are too few labeled calls). With
        leave-one-out alone, a training call's community_scam_rate / entity_risk is (sum - own_label) / (n - 1),
        so inside one community scams always get a slightly *lower* value than legitimate calls. Tree models
        learn that artefact and it reverses on test calls, which get the plain rate (simulation with 1,200
        calls, 60 communities, 20 seeds: XGBoost test PR-AUC 0.587 with LOO vs 0.659 with 5-fold OOF).
        """
        idx = np.array([self.kg.call_index[c] for c in labeled_ids], dtype=int)
        y = np.asarray(labels, dtype=float)
        community = self.structural["community_id"].to_numpy()
        lab = label_features(self.kg, idx, y, community, self.gcfg)
        k = int(self.gcfg.get("label_folds", 5))
        if k > 1 and len(idx) >= 2 * k:
            # plain (not stratified) folds: the fold assignment itself must not depend on any label, so a
            # call's own label can never move it to another fold and change its features
            splitter = KFold(n_splits=k, shuffle=True, random_state=self.seed)
            for fit, held in splitter.split(idx):
                part = label_features(self.kg, idx[fit], y[fit], community, self.gcfg)
                lab.iloc[idx[held]] = part.iloc[idx[held]].to_numpy()
        return self.structural.drop(columns=["community_id"]).join(lab)

    def columns(self, groups: list[str] | None = None) -> list[str]:
        groups = groups or list(self.groups)
        return [c for g in groups if g in self.groups for c in self.groups[g]]

    def onehot_names(self, min_count: int | None = None) -> list[str]:
        min_count = int(self.gcfg.get("onehot_min_count", 2)) if min_count is None else min_count
        return [e for e, d in zip(self.kg.entity_ids, self.kg.entity_degree) if d >= min_count]

    def onehot_matrix(self, call_ids: list[str], names: list[str]) -> sparse.csr_matrix:
        """Entity multi-hot rows for `call_ids` with columns in the order of `names`."""
        cols = [self.kg.entity_index.get(n, -1) for n in names]
        rows = [self.kg.call_index[c] for c in call_ids]
        present = np.array([k for k, c in enumerate(cols) if c >= 0], dtype=int)
        if len(present) == 0:
            return sparse.csr_matrix((len(rows), len(names)), dtype=np.float32)
        sub = self.kg.X[rows][:, [cols[k] for k in present]].tocoo()
        return sparse.csr_matrix((sub.data, (sub.row, present[sub.col])), shape=(len(rows), len(names)),
                                 dtype=np.float32)
