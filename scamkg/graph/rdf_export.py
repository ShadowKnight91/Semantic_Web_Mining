"""Semantic-web export of the call KG: OWL T-box + A-box in RDF/Turtle, SHACL validation, SPARQL."""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS, XSD

from scamkg.config import ROOT
from scamkg.graph.build import CallKG
from scamkg.ontology.schema import Ontology
from scamkg.utils import get_logger, optional_import

log = get_logger(__name__)


def _uri_part(s: str) -> str:
    return quote(re.sub(r"\s+", "_", str(s).strip()), safe="")


def build_rdf_graph(kg: CallKG, calls: pd.DataFrame, ontology: Ontology, owl_path: Path | None = None,
                    scores: dict[str, float] | None = None, include_transcripts: bool = False) -> Graph:
    NS = Namespace(ontology.namespace)
    DATA = Namespace(ontology.namespace.rstrip("#/") + "/data/")
    g = Graph()
    g.bind("", NS)
    g.bind("kg", DATA)
    g.bind("skos", SKOS)
    g.bind("owl", OWL)
    owl_path = Path(owl_path) if owl_path else ROOT / "ontology" / "fraud_call_ontology.ttl"
    if owl_path.exists():
        g.parse(owl_path, format="turtle")

    # controlled vocabularies as SKOS concepts / typed individuals
    for t in ontology.extractable_types:
        for v in t.values.values():
            node = DATA[f"{t.name}/{_uri_part(v.name)}"]
            g.add((node, RDF.type, NS[t.name]))
            g.add((node, RDF.type, SKOS.Concept))
            g.add((node, NS["canonicalName"], Literal(v.name)))
            g.add((node, SKOS.prefLabel, Literal(v.name.replace("_", " "), lang="en")))
            if v.descriptor:
                g.add((node, RDFS.comment, Literal(v.descriptor, lang="en")))
            for syn in v.synonyms:
                g.add((node, SKOS.altLabel, Literal(syn, lang="en")))

    for cid in kg.call_ids:
        c = DATA[f"call/{_uri_part(cid)}"]
        g.add((c, RDF.type, NS["Call"]))
        g.add((c, NS["callId"], Literal(cid)))
        if cid in calls.index:
            row = calls.loc[cid]
            g.add((c, NS["dataset"], Literal(str(row["dataset"]))))
            g.add((c, NS["split"], Literal(str(row["split"]))))
            if int(row["label"]) >= 0:
                g.add((c, NS["isScam"], Literal(bool(int(row["label"])), datatype=XSD.boolean)))
            if include_transcripts:
                g.add((c, NS["transcript"], Literal(str(row["text"]))))
        if scores and cid in scores:
            g.add((c, NS["scamScore"], Literal(round(float(scores[cid]), 6), datatype=XSD.decimal)))
        for t in kg.triples[cid]:
            rel = ontology.relations[t["relation"]]
            e = DATA[f"{t['object_type']}/{_uri_part(t['object'])}"]
            g.add((e, RDF.type, NS[t["object_type"]]))
            g.add((e, NS["canonicalName"], Literal(str(t["object"]))))
            g.add((c, NS[rel.rdf_property], e))
            if t.get("inferred") or t.get("evidence"):
                st = BNode()
                g.add((st, RDF.type, RDF.Statement))
                g.add((st, RDF.subject, c))
                g.add((st, RDF.predicate, NS[rel.rdf_property]))
                g.add((st, RDF.object, e))
                g.add((st, NS["inferred"], Literal(bool(t.get("inferred")), datatype=XSD.boolean)))
                if t.get("evidence"):
                    g.add((st, NS["evidence"], Literal(str(t["evidence"])[:300])))
                if t.get("detail"):
                    g.add((st, NS["detail"], Literal(str(t["detail"])[:120])))
    return g


def export_rdf(kg: CallKG, calls: pd.DataFrame, ontology: Ontology, out_path: Path, **kw) -> Graph:
    g = build_rdf_graph(kg, calls, ontology, **kw)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(out_path), format="turtle")
    log.info("RDF knowledge graph: %d triples -> %s", len(g), out_path)
    return g


def validate_shacl(data_graph: Graph, shapes_path: Path) -> tuple[bool | None, str]:
    pyshacl = optional_import("pyshacl")
    if pyshacl is None:
        return None, "pyshacl not installed (pip install pyshacl) - SHACL validation skipped"
    shapes = Graph().parse(str(shapes_path), format="turtle")
    conforms, _, text = pyshacl.validate(data_graph, shacl_graph=shapes, inference="rdfs", abort_on_first=False)
    return bool(conforms), text


def run_sparql_file(graph: Graph, query_file: Path) -> list[tuple[str, pd.DataFrame]]:
    text = Path(query_file).read_text(encoding="utf-8")
    blocks = [b.strip() for b in re.split(r"^#---", text, flags=re.M)[1:] if b.strip()]
    out = []
    for block in blocks:
        title = block.splitlines()[0].strip()
        query = "\n".join(block.splitlines()[1:])
        res = graph.query(query)
        cols = [str(v) for v in res.vars]
        out.append((title, pd.DataFrame([[None if v is None else v.toPython() for v in row] for row in res],
                                        columns=cols)))
    return out
