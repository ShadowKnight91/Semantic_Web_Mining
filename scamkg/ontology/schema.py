"""The minimal fraud-call ontology as Python objects (loaded from ontology/ontology.yaml)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from scamkg.config import ROOT
from scamkg.utils import sha256_text

DEFAULT_ONTOLOGY = ROOT / "ontology" / "ontology.yaml"
_IDENT = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass
class ValueSpec:
    name: str
    descriptor: str = ""
    synonyms: list[str] = field(default_factory=list)


@dataclass
class EntityType:
    name: str
    descriptor: str
    closed: bool = False
    extractable: bool = True
    values: dict[str, ValueSpec] = field(default_factory=dict)
    aliases: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class Relation:
    name: str
    source: str
    target: str
    descriptor: str
    rdf_property: str


def _camel(name: str) -> str:
    parts = name.lower().split("_")
    return parts[0] + "".join(p.title() for p in parts[1:])


class Ontology:
    def __init__(self, name: str, version: str, namespace: str, entity_types: dict[str, EntityType],
                 relations: dict[str, Relation], source_text: str = "", path: Path | None = None):
        self.name = name
        self.version = version
        self.namespace = namespace
        self.entity_types = entity_types
        self.relations = relations
        self.path = path
        self._fingerprint = sha256_text(source_text)[:12] if source_text else "none"
        for ident in list(entity_types) + list(relations):
            if not _IDENT.match(ident):
                raise ValueError(f"Invalid ontology identifier: {ident!r}")
        for r in relations.values():
            if r.source not in entity_types or r.target not in entity_types:
                raise ValueError(f"Relation {r.name} references an unknown type")
        self._by_target = {r.target: r.name for r in relations.values()}

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Ontology":
        path = Path(path) if path else DEFAULT_ONTOLOGY
        if not path.is_absolute() and not path.exists():
            path = ROOT / path
        text = path.read_text(encoding="utf-8")
        raw = yaml.safe_load(text)
        types: dict[str, EntityType] = {}
        for tname, spec in raw["entity_types"].items():
            spec = spec or {}
            values = {}
            for vname, vspec in (spec.get("values") or {}).items():
                vspec = vspec or {}
                values[str(vname)] = ValueSpec(str(vname), vspec.get("descriptor", ""),
                                               [str(s) for s in (vspec.get("synonyms") or [])])
            aliases = {str(canon): [str(a) for a in (al or [])] for canon, al in (spec.get("aliases") or {}).items()}
            types[tname] = EntityType(tname, spec.get("descriptor", ""), bool(spec.get("closed", False)),
                                      bool(spec.get("extractable", True)), values, aliases)
        rels = {}
        for rname, r in raw["relations"].items():
            rels[rname] = Relation(rname, r["source"], r["target"], r.get("descriptor", ""),
                                   r.get("rdf_property") or _camel(rname))
        return cls(raw.get("name", "Ontology"), str(raw.get("version", "0")), raw["namespace"], types, rels,
                   source_text=text, path=path)

    # ------------------------------------------------------------------ lookups
    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def extractable_types(self) -> list[EntityType]:
        return [t for t in self.entity_types.values() if t.extractable]

    @property
    def type_names(self) -> list[str]:
        return [t.name for t in self.extractable_types]

    @property
    def relation_names(self) -> list[str]:
        return list(self.relations)

    def relation_for_target(self, type_name: str) -> str | None:
        return self._by_target.get(type_name)

    def target_of(self, relation: str) -> str | None:
        r = self.relations.get(relation)
        return r.target if r else None

    def is_closed(self, type_name: str) -> bool:
        t = self.entity_types.get(type_name)
        return bool(t and t.closed)

    def allowed_values(self, type_name: str) -> list[str]:
        t = self.entity_types.get(type_name)
        return list(t.values) if t else []

    # ------------------------------------------------------------------ constrained decoding
    def json_schema(self, max_triples: int = 15, max_text: int = 200, compact: bool = False) -> dict:
        """JSON schema of the extraction answer, for decoders that enforce a grammar (Ollama `format`).

        Each triple is one branch per relation, so the relation fixes its object type and, for closed types,
        the allowed values: a model cannot produce an off-ontology triple at all, whatever its size. The
        length caps stop small models from looping until the token limit. `compact` leaves out `object_type`
        (the relation implies it) and `detail` (nothing scores it): fewer tokens to generate for every triple."""
        text = {"type": "string", "maxLength": 100 if compact else max_text}
        branches = []
        for r in self.relations.values():
            obj = {"type": "string", "enum": self.allowed_values(r.target)} if self.is_closed(r.target) else text
            if compact:
                props = {"relation": {"const": r.name}, "object": obj, "evidence": text,
                         "inferred": {"type": "boolean"}}
            else:
                props = {"relation": {"const": r.name}, "object_type": {"const": r.target}, "object": obj,
                         "detail": text, "evidence": text, "inferred": {"type": "boolean"}}
            branches.append({"type": "object", "properties": props, "required": list(props)})
        return {"type": "object", "required": ["triples"],
                "properties": {"triples": {"type": "array", "maxItems": max_triples, "items": {"anyOf": branches}}}}

    # ------------------------------------------------------------------ prompts
    def prompt_block(self, descriptors: bool = True) -> str:
        """Human-readable ontology description injected into the extraction prompt."""
        lines = ["Entity types:"]
        for t in self.extractable_types:
            lines.append(f"- {t.name}: {t.descriptor}" if descriptors else f"- {t.name}")
            if t.closed:
                if descriptors:
                    vals = "; ".join(f"{v.name} ({v.descriptor})" for v in t.values.values())
                else:
                    vals = ", ".join(t.values)
                lines.append(f"    allowed values: {vals}")
            elif t.name == "CallbackNumber":
                lines.append("    value: the phone number, extension or IVR option exactly as given (e.g. 1-888-555-0142, ext 42, press 1)")
            else:
                lines.append("    value: the name as said in the call (e.g. Social Security Administration, Amazon)")
        lines.append("Relations (the subject is always the Call being analysed):")
        for r in self.relations.values():
            suffix = f": {r.descriptor}" if descriptors else ""
            lines.append(f"- {r.name} (Call -> {r.target}){suffix}")
        return "\n".join(lines)
