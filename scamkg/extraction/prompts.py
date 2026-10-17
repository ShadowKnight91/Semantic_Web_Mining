"""Versioned prompts for triple extraction (with ablation variants) and zero-shot classification."""
from __future__ import annotations

import json

from scamkg.ontology.schema import Ontology
from scamkg.utils import sha256_text

# Prompt ablation variants (RQ2). v0_open is *open* extraction: no ontology, the model invents predicates.
PROMPT_VERSIONS: dict[str, dict] = {
    "v1_full": {"ontology": True, "descriptors": True, "example": True},
    "v2_no_descriptors": {"ontology": True, "descriptors": False, "example": True},
    "v3_no_example": {"ontology": True, "descriptors": True, "example": False},
    "v0_open": {"ontology": False, "descriptors": False, "example": False},
}

EXAMPLE_TRANSCRIPT = (
    "caller: This is Officer Brown from the Social Security Administration. Your SSN has been suspended due to "
    "suspicious activity. To avoid arrest today, you must verify your number and pay a fine using Google Play "
    "gift cards. Call us back at 1-888-555-0142.\n"
    "receiver: Oh no, what do I need to do?"
)

EXAMPLE_OUTPUT = {"triples": [
    {"relation": "CLAIMS_TO_BE", "object_type": "ClaimedIdentity", "object": "government_agent",
     "detail": "Officer Brown", "evidence": "This is Officer Brown from the Social Security Administration",
     "inferred": False},
    {"relation": "MENTIONS_ORG", "object_type": "Organization", "object": "Social Security Administration",
     "detail": "", "evidence": "from the Social Security Administration", "inferred": False},
    {"relation": "USES_TACTIC", "object_type": "PressureTactic", "object": "threat", "detail": "arrest",
     "evidence": "To avoid arrest today", "inferred": False},
    {"relation": "USES_TACTIC", "object_type": "PressureTactic", "object": "urgency", "detail": "today",
     "evidence": "To avoid arrest today", "inferred": False},
    {"relation": "USES_TACTIC", "object_type": "PressureTactic", "object": "fear", "detail": "suspicious activity",
     "evidence": "suspended due to suspicious activity", "inferred": False},
    {"relation": "REQUESTS", "object_type": "RequestedAsset", "object": "ssn", "detail": "verify your number",
     "evidence": "you must verify your number", "inferred": True},
    {"relation": "REQUESTS", "object_type": "RequestedAsset", "object": "money", "detail": "fine",
     "evidence": "pay a fine", "inferred": False},
    {"relation": "PAY_VIA", "object_type": "PaymentChannel", "object": "gift_card", "detail": "Google Play",
     "evidence": "pay a fine using Google Play gift cards", "inferred": False},
    {"relation": "PROVIDES_NUMBER", "object_type": "CallbackNumber", "object": "1-888-555-0142", "detail": "",
     "evidence": "Call us back at 1-888-555-0142", "inferred": False},
]}

RETRY_MESSAGE = ("Your previous answer was not a valid JSON object of the form {\"triples\": [...]}. "
                 "Return ONLY that JSON object, with no explanation and no markdown.")

# CHANGE (TA comment 1): the retry used to send the same generic message whatever went wrong. The model now
# gets feedback for the specific failure, which fixes most errors on the first retry.
RETRY_WRAPPER_MESSAGE = ("Your previous answer was JSON but not an object with a \"triples\" list "
                         "(for example a bare list or a different key). Return ONLY {\"triples\": [...]}.")


def schema_retry_message(ontology: Ontology, problems: list[str]) -> str:
    """Feedback when the JSON parsed but none of its triples fit the ontology (unknown relations,
    wrong object types, values outside the allowed list, or missing fields)."""
    rels = ", ".join(f"{r.name} -> {r.target}" for r in ontology.relations.values())
    shown = "; ".join(problems[:5])
    return ("Your previous answer was valid JSON, but none of its triples fit the ontology "
            f"({shown}). Each triple needs the fields relation, object_type and object. "
            f"Use ONLY these relations (relation -> object_type): {rels}. "
            "For closed types use exactly one allowed value. If the caller states no such fact, return "
            "{\"triples\": []}. Return ONLY the JSON object.")

_RULES = """RULES
1. Extract facts about what the CALLER says or does, not the receiver.
2. Use ONLY the relations and entity types defined above. Every triple has the call as its subject, so give only the relation and the object.
3. Output a single JSON object {{"triples": [...]}} and nothing else. Each triple is:
   {{"relation": <relation name>, "object_type": <the relation's target type>, "object": <value>,
    "detail": <short qualifier such as "arrest" or "today", may be "">,
    "evidence": <short verbatim quote from the transcript, at most 20 words>,
    "inferred": <false if stated explicitly, true if you inferred it>}}
4. For types with allowed values, "object" must be exactly one allowed value; use "other" only if nothing fits.
5. Legitimate calls also have identities, organizations and requests - extract them the same way. Do not judge whether the call is a scam.
6. Do not invent facts. Use one triple per distinct fact; a relation may appear several times (e.g. several tactics).{example}"""

_OPEN_SYSTEM = """You are an information-extraction engine. Read the phone-call transcript and extract knowledge-graph triples about the CALLER.
Return a single JSON object {"triples": [{"relation": <predicate you choose, UPPER_SNAKE_CASE>, "object": <value>, "evidence": <short quote>, "inferred": <true|false>}]} and nothing else."""


def _system_prompt(ontology: Ontology, version: str) -> str:
    spec = PROMPT_VERSIONS[version]
    if not spec["ontology"]:
        return _OPEN_SYSTEM
    example = ""
    if spec["example"]:
        example = ("\n\nEXAMPLE (format only - extract from the new transcript, never copy the example)\n"
                   f"Transcript:\n{EXAMPLE_TRANSCRIPT}\nOutput:\n{json.dumps(EXAMPLE_OUTPUT)}")
    return ("You are a precise information-extraction engine that builds a knowledge graph of phone calls "
            "for fraud analysis.\n\nONTOLOGY\n" + ontology.prompt_block(descriptors=spec["descriptors"]) +
            "\n\n" + _RULES.format(example=example))


def build_extraction_messages(transcript: str, ontology: Ontology, version: str = "v1_full") -> list[dict]:
    if version not in PROMPT_VERSIONS:
        raise ValueError(f"Unknown prompt version {version!r}; choose from {sorted(PROMPT_VERSIONS)}")
    user = f"TRANSCRIPT:\n<<<\n{transcript}\n>>>\nReturn the JSON object now."
    return [{"role": "system", "content": _system_prompt(ontology, version)}, {"role": "user", "content": user}]


def prompt_fingerprint(ontology: Ontology, version: str) -> str:
    return sha256_text(version + "\n" + _system_prompt(ontology, version))[:12]


CLASSIFY_SYSTEM = """You are a fraud analyst. Read the phone-call transcript and decide whether the CALLER is running a scam.
Categories:
- fraudster: tries to obtain money, credentials, codes or personal data through deception or impersonation
- salesperson: sales, marketing or telemarketing without deception
- prankster: a joke or prank call
- other_legitimate: any other genuine call (appointments, deliveries, friends, real service calls)
Return ONLY a JSON object:
{"label": "scam" | "legitimate", "category": "fraudster" | "salesperson" | "prankster" | "other_legitimate",
 "scam_probability": <number between 0 and 1>, "reason": "<one short sentence>"}"""


def build_classification_messages(transcript: str) -> list[dict]:
    return [{"role": "system", "content": CLASSIFY_SYSTEM},
            {"role": "user", "content": f"TRANSCRIPT:\n<<<\n{transcript}\n>>>\nReturn the JSON object now."}]


CLASSIFY_PROMPT_VERSION = "cls_v1_" + sha256_text(CLASSIFY_SYSTEM)[:8]
