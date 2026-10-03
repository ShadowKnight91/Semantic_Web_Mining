"""Entity standardization: map free-text mentions onto canonical ontology nodes.

Examples: "SSA", "Social Security" and "the social security office" all become the
Organization node "Social Security Administration"; "Google Play gift cards" becomes the
PaymentChannel value "gift_card"; "(888) 555-0142" becomes CallbackNumber "888-555-0142".
"""
from __future__ import annotations

import difflib
import re
import unicodedata

from scamkg.ontology.schema import Ontology

NULL_VALUES = {"", "none", "null", "n a", "na", "unknown", "not mentioned", "not specified", "nothing", "no"}
GENERIC_ORGS = {"bank", "company", "government", "department", "office", "agency", "police", "store", "support",
                "customer service", "tech support", "the company", "caller", "receiver", "call center", "team"}
_ARTICLES = re.compile(r"^(?:the|a|an|your|my|our|their|his|her)\s+")
_WORD_DIGITS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
                "seven": "7", "eight": "8", "nine": "9"}


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower().strip()
    s = s.replace("&", " and ").replace("'", "").replace("’", "")
    s = re.sub(r"[_\-/]", " ", s)
    s = re.sub(r"\.(?!\d)", " ", s)
    s = re.sub(r"[^a-z0-9$. ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = _ARTICLES.sub("", s)
    return s


def _singular(word: str) -> str:
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss") and not word.endswith("us"):
        return word[:-1]
    return word


def match_key(s: str) -> str:
    """Normalization used for dictionary lookups (lowercase, no punctuation, singular words)."""
    return " ".join(_singular(w) for w in normalize_text(s).split())


def normalize_phone(s: str) -> str | None:
    """Canonical CallbackNumber: NNN-NNN-NNNN, raw digits, ext-N or ivr:press-N."""
    text = str(s).lower()
    m = re.search(r"press\s*(\d|zero|one|two|three|four|five|six|seven|eight|nine)\b", text)
    if m:
        d = _WORD_DIGITS.get(m.group(1), m.group(1))
        return f"ivr:press-{d}"
    ext = None
    m = re.search(r"\b(?:ext(?:ension)?\.?|x)\s*(\d{1,6})\b", text)
    if m:
        ext = m.group(1)
        text = text[:m.start()] + text[m.end():]
    digits = re.sub(r"\D", "", text)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) == 10:
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    if 7 <= len(digits) <= 15:
        return digits
    if ext:
        return f"ext-{ext}"
    return None


def clean_org_name(mention: str) -> str | None:
    s = normalize_text(mention)
    s = re.sub(r"\b(?:inc|llc|ltd|corp|corporation|co)\b", "", s).strip()
    if len(s) < 2 or s in GENERIC_ORGS:
        return None
    return " ".join(w.upper() if len(w) <= 3 and w.isalpha() and w not in {"of", "and", "the", "for"} else
                    (w if w in {"of", "and", "the", "for"} else w.capitalize()) for w in s.split())


class EntityStandardizer:
    """Maps (entity type, mention) -> canonical value using the ontology vocabularies."""

    def __init__(self, ontology: Ontology, fuzzy_cutoff: float = 0.88):
        self.ontology = ontology
        self.fuzzy_cutoff = fuzzy_cutoff
        self.lookup: dict[str, dict[str, str]] = {}
        self._patterns: dict[str, list[tuple[re.Pattern, str]]] = {}
        for t in ontology.extractable_types:
            table: dict[str, str] = {}
            if t.closed:
                for v, spec in t.values.items():
                    table[match_key(v)] = v
                for v, spec in t.values.items():
                    for syn in spec.synonyms:
                        table.setdefault(match_key(syn), v)
            else:
                for canon, aliases in t.aliases.items():
                    table[match_key(canon)] = canon
                for canon, aliases in t.aliases.items():
                    for a in aliases:
                        table.setdefault(match_key(a), canon)
            table.pop("", None)
            self.lookup[t.name] = table
            ordered = sorted(table.items(), key=lambda kv: -len(kv[0]))
            self._patterns[t.name] = [(re.compile(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])"), v)
                                      for k, v in ordered if k != "other"]

    def standardize(self, type_name: str, mention) -> str | None:
        """Return the canonical value, "other" for unmatched closed-vocabulary mentions, or None to drop."""
        if mention is None:
            return None
        t = self.ontology.entity_types.get(type_name)
        if t is None or not t.extractable:
            return None
        mention = str(mention).strip()
        if type_name == "CallbackNumber":
            return normalize_phone(mention)
        key = match_key(mention)
        if key in NULL_VALUES:
            return None
        table = self.lookup.get(type_name, {})
        if key in table:
            return table[key]
        for pat, value in self._patterns.get(type_name, []):
            if pat.search(key):
                return value
        close = difflib.get_close_matches(key, list(table), n=1, cutoff=self.fuzzy_cutoff)
        if close:
            return table[close[0]]
        if t.closed:
            return "other" if "other" in t.values else None
        return clean_org_name(mention)

    def infer_type(self, mention, exclude: str | None = None) -> tuple[str, str] | None:
        """Exact vocabulary match of a mention in *another* closed type (used to repair mistyped triples)."""
        key = match_key(str(mention))
        for tname, table in self.lookup.items():
            if tname == exclude or not self.ontology.is_closed(tname):
                continue
            value = table.get(key)
            if value and value != "other":
                return tname, value
        return None

    def is_exact_value(self, type_name: str, mention) -> bool:
        """True when a closed-type mention already is one of the allowed values (schema conformance)."""
        if not self.ontology.is_closed(type_name):
            return mention is not None and str(mention).strip() != ""
        v = str(mention).strip().lower().replace(" ", "_").replace("-", "_")
        return v in self.ontology.entity_types[type_name].values
