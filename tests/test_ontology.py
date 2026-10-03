from rdflib import Graph
from rdflib.namespace import OWL, RDF

from scamkg.config import ROOT
from scamkg.ontology.schema import Ontology


def test_ontology_loads(ontology: Ontology):
    assert set(ontology.type_names) == {"ClaimedIdentity", "Organization", "PressureTactic", "RequestedAsset",
                                        "PaymentChannel", "CallbackNumber"}
    assert ontology.target_of("PAY_VIA") == "PaymentChannel"
    assert ontology.relation_for_target("RequestedAsset") == "REQUESTS"
    assert {"urgency", "threat", "reward", "fear", "secrecy"} <= set(ontology.allowed_values("PressureTactic"))
    assert "gift_card" in ontology.allowed_values("PaymentChannel")


def test_owl_tbox_matches_yaml(ontology: Ontology):
    g = Graph().parse(ROOT / "ontology" / "fraud_call_ontology.ttl", format="turtle")
    ns = ontology.namespace
    classes = {str(s).replace(ns, "") for s in g.subjects(RDF.type, OWL.Class)}
    props = {str(s).replace(ns, "") for s in g.subjects(RDF.type, OWL.ObjectProperty)}
    assert set(ontology.entity_types) <= classes
    assert {r.rdf_property for r in ontology.relations.values()} <= props


def test_prompt_block_mentions_every_relation(ontology: Ontology):
    block = ontology.prompt_block()
    for rel in ontology.relation_names:
        assert rel in block
    assert "government_agent" in block
