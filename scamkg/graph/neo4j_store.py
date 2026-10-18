"""Neo4j storage of the call KG and Graph Data Science (GDS) features.

Graph model:  (:Call {call_id, dataset, split, gold_label, n_words})
              -[:CLAIMS_TO_BE|MENTIONS_ORG|USES_TACTIC|REQUESTS|PAY_VIA|PROVIDES_NUMBER {inferred, detail, evidence}]->
              (:ClaimedIdentity|:Organization|:PressureTactic|:RequestedAsset|:PaymentChannel|:CallbackNumber {name})
Derived:      (:Call)-[:SIMILAR_TO {score}]->(:Call)   (gds.nodeSimilarity, Jaccard)
              Call.community                            (gds.louvain on SIMILAR_TO)

`compute_features` returns the same label-free columns as graph.features.structural_features, so the
rest of the pipeline is backend-agnostic. Labels are stored as `gold_label` for inspection only;
GDS never reads them, and label-based features are computed in Python from training labels.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

from scamkg.graph.build import CallKG
from scamkg.graph.features import local_features
from scamkg.utils import get_logger, optional_import

log = get_logger(__name__)
_IDENT = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
HET_GRAPH, DIR_GRAPH, CC_GRAPH = "scamkg_het", "scamkg_dir", "scamkg_cc"


def _ident(s: str) -> str:
    if not _IDENT.match(s):
        raise ValueError(f"Unsafe Cypher identifier {s!r}")
    return s


class Neo4jStore:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j", batch_size: int = 500):
        neo4j = optional_import("neo4j")
        if neo4j is None:
            raise RuntimeError("The neo4j driver is not installed: pip install neo4j")
        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        self.driver.verify_connectivity()
        self.database = database
        self.batch_size = batch_size

    @classmethod
    def from_config(cls, cfg: dict) -> "Neo4jStore":
        n = cfg["neo4j"]
        uri = os.environ.get(n.get("uri_env", "NEO4J_URI")) or n.get("uri", "bolt://localhost:7687")
        user = os.environ.get(n.get("user_env", "NEO4J_USER")) or n.get("user", "neo4j")
        password = os.environ.get(n.get("password_env", "NEO4J_PASSWORD"))
        if not password:
            raise RuntimeError("Set NEO4J_PASSWORD in .env (see .env.example)")
        return cls(uri, user, password, n.get("database", "neo4j"), int(n.get("batch_size", 500)))

    def close(self) -> None:
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def run(self, query: str, **params) -> list[dict]:
        with self.driver.session(database=self.database) as s:
            return [r.data() for r in s.run(query, **params)]

    # ------------------------------------------------------------------ loading
    def clear(self) -> None:
        self.run("MATCH (n) CALL { WITH n DETACH DELETE n } IN TRANSACTIONS OF 5000 ROWS")

    def create_constraints(self, kg: CallKG) -> None:
        self.run("CREATE CONSTRAINT call_id IF NOT EXISTS FOR (c:Call) REQUIRE c.call_id IS UNIQUE")
        for t in kg.ontology.type_names:
            self.run(f"CREATE CONSTRAINT {_ident(t).lower()}_name IF NOT EXISTS FOR (n:{_ident(t)}) "
                     f"REQUIRE n.name IS UNIQUE")

    def _batched(self, query: str, rows: list[dict]) -> None:
        for i in range(0, len(rows), self.batch_size):
            self.run(query, rows=rows[i:i + self.batch_size])

    def load(self, kg: CallKG, calls: pd.DataFrame) -> None:
        self.create_constraints(kg)
        rows = [{"call_id": c, "dataset": str(calls.at[c, "dataset"]), "split": str(calls.at[c, "split"]),
                 "gold_label": int(calls.at[c, "label"]), "n_words": int(calls.at[c, "n_words"])} for c in kg.call_ids]
        self._batched("UNWIND $rows AS r MERGE (c:Call {call_id: r.call_id}) "
                      "SET c.dataset = r.dataset, c.split = r.split, c.gold_label = r.gold_label, c.n_words = r.n_words",
                      rows)
        by_rel: dict[str, list[dict]] = {}
        for cid, ts in kg.triples.items():
            for t in ts:
                by_rel.setdefault(t["relation"], []).append(
                    {"call_id": cid, "name": str(t["object"]), "inferred": bool(t.get("inferred")),
                     "detail": str(t.get("detail", ""))[:200], "evidence": str(t.get("evidence", ""))[:300]})
        for rel, rrows in by_rel.items():
            target = _ident(kg.ontology.target_of(rel))
            self._batched(f"UNWIND $rows AS r MATCH (c:Call {{call_id: r.call_id}}) "
                          f"MERGE (e:{target} {{name: r.name}}) "
                          f"MERGE (c)-[x:{_ident(rel)}]->(e) "
                          f"SET x.inferred = r.inferred, x.detail = r.detail, x.evidence = r.evidence", rrows)
        log.info("Neo4j: loaded %d calls and %d relationships", len(rows), sum(len(v) for v in by_rel.values()))

    # ------------------------------------------------------------------ GDS features
    def _drop(self, name: str) -> None:
        self.run("CALL gds.graph.drop($name, false) YIELD graphName RETURN graphName", name=name)

    def _stream_calls(self, proc: str, graph: str, config: dict) -> dict[str, float]:
        q = (f"CALL {proc}($graph, $config) YIELD nodeId, score "
             "WITH gds.util.asNode(nodeId) AS n, score WHERE n:Call RETURN n.call_id AS call_id, score")
        return {r["call_id"]: r["score"] for r in self.run(q, graph=graph, config=config)}

    def compute_features(self, kg: CallKG, gcfg: dict, seed: int = 42) -> pd.DataFrame:
        labels = ["Call", *kg.ontology.type_names]
        rels = kg.ontology.relation_names
        for g in (HET_GRAPH, DIR_GRAPH, CC_GRAPH):
            self._drop(g)
        self.run("CALL gds.graph.project($name, $labels, $rels)", name=HET_GRAPH, labels=labels,
                 rels={r: {"orientation": "UNDIRECTED"} for r in rels})
        deg = self._stream_calls("gds.degree.stream", HET_GRAPH, {})
        pr = self._stream_calls("gds.pageRank.stream", HET_GRAPH, {"dampingFactor": 0.85, "maxIterations": 50})
        bc = self._stream_calls("gds.betweenness.stream", HET_GRAPH,
                                {"samplingSize": int(gcfg.get("betweenness_samples", 300)), "samplingSeed": seed})
        n_nodes = self.run("CALL gds.graph.list($name) YIELD nodeCount RETURN nodeCount", name=HET_GRAPH)[0]["nodeCount"]

        # call-call similarity graph via node similarity (Jaccard over shared entities)
        self.run("CALL gds.graph.project($name, $labels, $rels)", name=DIR_GRAPH, labels=labels,
                 rels={r: {"orientation": "NATURAL"} for r in rels})
        self.run("MATCH (:Call)-[s:SIMILAR_TO]->(:Call) CALL { WITH s DELETE s } IN TRANSACTIONS OF 10000 ROWS")
        self.run("CALL gds.nodeSimilarity.write($name, {writeRelationshipType: 'SIMILAR_TO', writeProperty: 'score', "
                 "similarityCutoff: $cut, topK: $k}) YIELD relationshipsWritten RETURN relationshipsWritten",
                 name=DIR_GRAPH, cut=float(gcfg.get("similarity_min", 0.2)), k=int(gcfg.get("similarity_top_k", 10)))
        # keep one direction per pair
        self.run("MATCH (a:Call)-[s1:SIMILAR_TO]->(b:Call)-[s2:SIMILAR_TO]->(a) WHERE elementId(a) < elementId(b) DELETE s2")
        self.run("CALL gds.graph.project($name, 'Call', {SIMILAR_TO: {orientation: 'UNDIRECTED', properties: 'score'}})",
                 name=CC_GRAPH)
        cc_deg = self._stream_calls("gds.degree.stream", CC_GRAPH, {})
        cc_wdeg = self._stream_calls("gds.degree.stream", CC_GRAPH, {"relationshipWeightProperty": "score"})
        cc_pr = self._stream_calls("gds.pageRank.stream", CC_GRAPH, {"relationshipWeightProperty": "score"})
        self.run("CALL gds.louvain.write($name, {writeProperty: 'community', relationshipWeightProperty: 'score', "
                 "tolerance: 0.0001}) YIELD communityCount RETURN communityCount", name=CC_GRAPH)
        comm = {r["call_id"]: r["community"] for r in
                self.run("MATCH (c:Call) RETURN c.call_id AS call_id, c.community AS community")}
        for g in (HET_GRAPH, DIR_GRAPH, CC_GRAPH):
            self._drop(g)

        df = pd.DataFrame(index=kg.call_ids)
        df["het_degree"] = [deg.get(c, 0.0) for c in kg.call_ids]
        # GDS PageRank is un-normalized (mean ~1 per node), matching networkx's pagerank * n_nodes
        df["het_pagerank"] = [pr.get(c, 0.0) for c in kg.call_ids]
        # networkx reports normalized betweenness, extrapolated by n/k when it samples k sources;
        # GDS reports raw pair counts from the sampled sources only
        norm = max(1.0, (n_nodes - 1) * (n_nodes - 2) / 2)
        samples = int(gcfg.get("betweenness_samples", 300))
        if 0 < samples < n_nodes:
            norm *= samples / n_nodes
        df["het_betweenness"] = [bc.get(c, 0.0) / norm for c in kg.call_ids]
        df["cc_degree"] = [cc_deg.get(c, 0.0) for c in kg.call_ids]
        df["cc_weighted_degree"] = [cc_wdeg.get(c, 0.0) for c in kg.call_ids]
        df["cc_pagerank"] = [cc_pr.get(c, 0.0) for c in kg.call_ids]
        raw_comm = pd.Series([comm.get(c, -1) for c in kg.call_ids], index=kg.call_ids)
        sizes = raw_comm.map(raw_comm.value_counts())
        order = {cid: k for k, cid in enumerate(raw_comm.value_counts().index)}
        df["community_id"] = raw_comm.map(order).astype(int)
        df["community_size"] = sizes.astype(int)
        df = df.join(local_features(kg))
        log.info("Neo4j GDS features computed (%d communities)", raw_comm.nunique())
        return df

    # ------------------------------------------------------------------ exports / explain
    def export_csv(self, out_dir: Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        nodes = self.run("MATCH (n) RETURN elementId(n) AS id, labels(n)[0] AS label, "
                         "coalesce(n.call_id, n.name) AS name, n.split AS split, n.community AS community")
        rels = self.run("MATCH (a)-[r]->(b) RETURN elementId(a) AS source, type(r) AS type, elementId(b) AS target, "
                        "r.inferred AS inferred, r.score AS score")
        pd.DataFrame(nodes).to_csv(out_dir / "neo4j_nodes.csv", index=False)
        pd.DataFrame(rels).to_csv(out_dir / "neo4j_relationships.csv", index=False)

    @staticmethod
    def explain_query(call_id: str) -> str:
        return ("MATCH (c:Call {call_id: $call_id})-[r]->(e) WHERE NOT e:Call "
                "OPTIONAL MATCH (e)<-[r2]-(o:Call) WHERE o.split = 'train' AND o <> c "
                "RETURN c, r, e, r2, o LIMIT 60")
