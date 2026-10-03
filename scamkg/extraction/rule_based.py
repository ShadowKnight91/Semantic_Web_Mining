"""Deterministic keyword/regex extractor that emits the same ontology triples as the LLM extractor.

Two roles: (1) an extraction baseline for RQ2 (LLM vs. rules), (2) an offline fallback so the whole
pipeline runs without an LLM server. It only reads the caller's turns when speaker tags exist.
"""
from __future__ import annotations

import re
import time

from scamkg.extraction.extractor import BaseExtractor, ExtractionRecord
from scamkg.ontology.schema import Ontology
from scamkg.ontology.standardize import EntityStandardizer, normalize_phone

RULES_VERSION = "1.0"
_I = re.IGNORECASE


def _rx(p: str) -> re.Pattern:
    return re.compile(p, _I)


# Ordered: the first matching identity wins for a self-introduction phrase.
IDENTITY_CUES: list[tuple[str, re.Pattern]] = [
    ("government_agent", _rx(r"social security|\bssa\b|\birs\b|internal revenue|medicare|government|federal (?:agency|agent|government|office)|"
                             r"department of (?:justice|treasury|homeland|revenue|labor|health)|customs|immigration|\buscis\b|treasury|census|\bdmv\b|tax (?:office|department|agency)")),
    ("law_enforcement", _rx(r"police|sheriff|detective|\bfbi\b|\bdea\b|marshal|law enforcement|officer|deputy|court|investigator")),
    ("tech_support", _rx(r"tech(?:nical)? support|\bit support|computer (?:support|department|technician)|help ?desk|microsoft|windows|"
                         r"apple support|geek squad|norton|mcafee|security technician|antivirus")),
    ("bank", _rx(r"\bbank\b|credit union|fraud (?:department|prevention|team)|card ?(?:member|holder)? services|account security|"
                 r"financial institution|chase|wells fargo|citi(?:bank)?|capital one")),
    ("prize_or_lottery", _rx(r"lottery|sweepstakes|prize|publishers clearing house|contest|giveaway|raffle")),
    ("insurance", _rx(r"insurance|warranty|state farm|geico")),
    ("utility_company", _rx(r"electric|utility|utilities|power (?:company|and light)|gas company|water (?:company|department)|energy")),
    ("delivery_service", _rx(r"deliver|courier|package|shipping|postal|post office|\busps\b|fedex|\bups\b|\bdhl\b")),
    ("healthcare_provider", _rx(r"doctor|\bdr\b\.?|clinic|hospital|pharmacy|dental|dentist|medical|health ?care|nurse|physician|optometr|walgreens|\bcvs\b")),
    ("employer_recruiter", _rx(r"recruit|hiring|human resources|\bhr\b|staffing|employment|job (?:offer|opportunity)|your (?:boss|manager|employer)")),
    ("charity", _rx(r"charity|foundation|non-?profit|fundrais|donation|relief fund|church|volunteer")),
    ("telemarketer_sales", _rx(r"\bsales\b|marketing|promotion|special offer|solar|timeshare|vacation|travel club|real estate|investment|"
                               r"mortgage|\bloan\b|interest rate|survey|market research")),
    ("retailer_customer_service", _rx(r"customer (?:service|support|care)|amazon|walmart|ebay|best buy|\btarget\b|order|billing|subscription|account services|store")),
    ("relative_or_friend", _rx(r"grand(?:son|daughter|child|ma|pa|mother|father)|your (?:son|daughter|nephew|niece|cousin|brother|sister|uncle|aunt)|"
                               r"it'?s me\b|\bmom\b|\bdad\b|old friend|your friend|a friend of|your neighbou?r")),
]

INTRO_RE = _rx(r"\b(?:this is|my name is|i am|i'm|calling from|calling on behalf of|on behalf of|i'?m with|i am with|"
               r"i work (?:for|with|at)|we are|we're|representing|it'?s me|it is me|here (?:at|from)|from)\s+(?P<who>[^.!?\n]{2,90})")

TACTIC_CUES: dict[str, list[re.Pattern]] = {
    "urgency": [_rx(r"\b(immediately|right now|right away|urgent(?:ly)?|within (?:the next )?(?:\d+|one|two|twenty[- ]four|an?) (?:hours?|minutes?|days?)|"
                    r"within the hour|as soon as possible|asap|time[- ]sensitive|act now|hurry|deadline|expires?|last chance|limited[- ]time|"
                    r"(?:by|before|until) (?:the end of )?(?:today|tonight|the end of the day))\b")],
    "threat": [_rx(r"\b(arrest(?:ed)?|warrant|lawsuit|legal action|sued|jail|prison|suspend(?:ed)?|suspension|deport(?:ed|ation)?|"
                   r"freez(?:e|ing)|frozen|blocked|cut off|disconnect(?:ed|ion)?|shut off|penalt(?:y|ies)|(?:pay|a|the) fines?|"
                   r"charges? (?:against|filed)|prosecut\w+|seiz(?:e|ed|ure)|legal consequences|account (?:closure|will be closed))\b")],
    "reward": [_rx(r"\b(you(?:'ve| have)? won|winner|prize|jackpot|sweepstakes|lottery|free (?:gift|vacation|cruise|trip)|cash back|"
                   r"bonus|reward|refund|compensation|guaranteed (?:return|profit)|selected to receive|eligible (?:for|to receive)|"
                   r"special offer|discount|lower (?:your )?interest rate)\b")],
    "fear": [_rx(r"\b(compromised|hacked|virus|malware|infected|stolen|identity theft|suspicious (?:activity|transactions?|charges?|login)|"
                 r"(?:fraudulent|illegal) (?:activity|charges?|transactions?)|illegal drugs|fraud alert|unauthori[sz]ed|security breach|"
                 r"emergency|in trouble|accident|danger)\b")],
    "secrecy": [_rx(r"\b(don'?t tell|do not tell|keep (?:this|it) (?:between us|confidential|secret|private|quiet)|confidential|"
                    r"stay on the line|don'?t hang up|do not hang up|don'?t (?:contact|call) (?:your|the) bank|do not discuss)\b")],
}
_TODAY = _rx(r"\b(today|tonight)\b")
_PRESSURE_WORDS = _rx(r"\b(must|need to|have to|avoid|or else|otherwise|immediately|arrest|before|deadline)\b")

REQUEST_VERBS = _rx(r"\b(verify|confirm|provide|give|need|read|tell|share|send|enter|update|what(?:'s| is)|buy|purchase|pay|"
                    r"install|download|grant|allow|let me|must|have to|require|go to|get)\b")

ASSET_CUES: dict[str, re.Pattern] = {
    "otp": _rx(r"\b(one[- ]time (?:pass(?:word|code)|code)|otp|verification code|security code|confirmation code|"
               r"authentication code|\d[- ]digit code|(?:six|6)[- ]digit|code (?:we|that was|i) (?:just )?sent|code sent)\b"),
    "ssn": _rx(r"\b(social security number|ssn|social insurance number|last four (?:digits )?of your social)\b"),
    "card_number": _rx(r"\b((?:credit|debit) card (?:number|details|information|info)|card number|cvv|expiration date|card details|"
                       r"number on (?:the front of )?your card)\b"),
    "bank_account": _rx(r"\b(account number|routing number|bank account (?:number|details|information)|banking (?:details|information|login)|online banking)\b"),
    "password": _rx(r"\b(password|passcode|pin(?: number)?|login credentials|username)\b"),
    "remote_access": _rx(r"\b(remote access|remote desktop|teamviewer|anydesk|screen ?shar\w*|access (?:to )?your (?:computer|device|pc|laptop)|"
                         r"install (?:this|a|the|our) (?:software|program|app|tool)|download (?:this|a|the|our) (?:software|program|app|tool))\b"),
    "money": _rx(r"\b(pay(?:ing|ment)?|fees?|fines?|penalt(?:y|ies)|\$\s?\d[\d,]*(?:\.\d{2})?|\d[\d,]* dollars|send (?:us )?(?:the )?money|"
                 r"transfer (?:the |your )?(?:money|funds|savings)|bail|donat(?:e|ion)|deposit|withdraw)\b"),
    "personal_info": _rx(r"\b(date of birth|birth ?date|dob|home address|mailing address|full name|mother'?s maiden name|"
                         r"driver'?s licen[cs]e|medicare (?:number|id)|member id|policy number|zip code|email address)\b"),
}
_SSN_MENTIONED = _rx(r"\b(ssn|social security number|social security)\b")
_VERIFY_NUMBER = _rx(r"\b(verify|confirm|provide|give|read)\b[^.?!]{0,30}\byour (?:number|social)\b")
# "we will never ask for your PIN" is a legitimate-bank phrase, not a request
_NEGATED = _rx(r"\b(never|will not|won'?t|do not|don'?t|not) (?:ever )?(?:ask|call|request|need)\b")

PAYMENT_CUES: dict[str, re.Pattern] = {
    "gift_card": _rx(r"\b(gift ?cards?|google play|itunes|steam cards?|prepaid cards?|apple cards?|ebay cards?)\b"),
    "wire_transfer": _rx(r"\b(wire (?:transfer|the money|it|the funds)|western union|money ?gram|wire)\b"),
    "crypto": _rx(r"\b(bitcoin|crypto(?:currency)?|btc|ethereum|usdt|coinbase|crypto wallet)\b"),
    "zelle": _rx(r"\bzelle\b"),
    "p2p_app": _rx(r"\b(venmo|cash ?app|paypal|apple pay|google pay)\b"),
    "bank_transfer": _rx(r"\b(bank transfer|safe account|secure (?:government )?account|transfer (?:the |your )?(?:funds|money|savings) (?:to|into)|direct deposit|ach)\b"),
    "cash": _rx(r"\b(in cash|cash pickup|pick up the cash|courier (?:will )?(?:pick|come)|mail (?:the )?cash|envelope of cash|withdraw (?:the )?cash)\b"),
    "check_or_money_order": _rx(r"\b(money order|cashier'?s check|certified check|send a check)\b"),
    "credit_card_payment": _rx(r"\b(pay (?:with|by|using) (?:your |a )?(?:credit|debit) card|card payment|pay (?:it )?over the phone)\b"),
}
PHONE_RE = re.compile(r"(?:\+?1[\s.-]?)?\(?\b\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b")
IVR_RE = _rx(r"\bpress (?:1|one|2|two|9|nine|0|zero)\b")
EXT_RE = _rx(r"\bextension (\d{2,6})\b")
_CALLER_LINE = re.compile(r"^\s*caller\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def caller_text(text: str) -> tuple[str, list[str]]:
    """(all caller text, caller turns). Falls back to the whole text when there are no speaker tags."""
    turns = [m.group(1).strip() for m in _CALLER_LINE.finditer(text)]
    if turns:
        return " ".join(turns), turns
    return text, [text[:600]]


class RuleBasedExtractor(BaseExtractor):
    name = "rule"

    def __init__(self, ontology: Ontology, standardizer: EntityStandardizer | None = None):
        self.ontology = ontology
        self.standardizer = standardizer or EntityStandardizer(ontology)
        org_type = ontology.entity_types.get("Organization")
        self._org_patterns: list[tuple[re.Pattern, str]] = []
        if org_type is not None:
            aliases = [(canon, a) for canon, al in org_type.aliases.items() for a in [canon, *al]]
            for canon, alias in sorted(aliases, key=lambda x: -len(x[1])):
                words = alias.split()
                if len(words) == 1 and len(alias) <= 4:
                    pat = re.compile(rf"\b{re.escape(alias.upper())}\b")            # acronyms: case-sensitive
                elif len(words) == 1:
                    pat = re.compile(rf"\b{re.escape(alias[0].upper() + alias[1:])}\b")  # proper names: capitalized
                else:
                    pat = re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)
                if canon == "Google":  # "Google Play" is a gift-card brand (PaymentChannel), not the company
                    pat = re.compile(pat.pattern + r"(?!\s+[Pp]lay)", pat.flags)
                self._org_patterns.append((pat, canon))

    def metadata(self) -> dict:
        return {"extractor": "rule", "rules_version": RULES_VERSION, "ontology_version": self.ontology.version,
                "ontology_hash": self.ontology.fingerprint}

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _classify_identity(phrase: str) -> tuple[str, str] | None:
        for value, pat in IDENTITY_CUES:
            m = pat.search(phrase)
            if m:
                return value, m.group(0)
        return None

    def _add(self, out: dict, rel: str, obj: str, mention: str, evidence: str, inferred: bool = False) -> None:
        target = self.ontology.target_of(rel)
        if target is None or target not in self.ontology.entity_types:
            return
        canon = obj if self.ontology.is_closed(target) else self.standardizer.standardize(target, obj)
        if canon is None:
            return
        key = (rel, canon)
        if key not in out:
            out[key] = {"relation": rel, "object_type": target, "object": canon, "mention": mention[:200],
                        "detail": mention.lower()[:120], "evidence": evidence.strip()[:300], "inferred": inferred,
                        "repaired": False}

    # ------------------------------------------------------------------ main
    def extract(self, call_id: str, text: str) -> ExtractionRecord:
        t0 = time.time()
        ctext, turns = caller_text(text)
        sentences = [s for s in _SENT_SPLIT.split(ctext) if s.strip()]
        out: dict[tuple[str, str], dict] = {}

        # ClaimedIdentity: self-introductions in the first caller turns, then a whole-text fallback
        intro_zone = " ".join(turns[:3])
        for m in INTRO_RE.finditer(intro_zone):
            hit = self._classify_identity(m.group("who"))
            if hit:
                self._add(out, "CLAIMS_TO_BE", hit[0], hit[1], m.group(0))
                break
        if not any(k[0] == "CLAIMS_TO_BE" for k in out):
            hit = self._classify_identity(intro_zone)
            if hit:
                self._add(out, "CLAIMS_TO_BE", hit[0], hit[1], intro_zone[:150], inferred=True)

        # Organizations
        for pat, canon in self._org_patterns:
            m = pat.search(ctext)
            if m:
                self._add(out, "MENTIONS_ORG", canon, m.group(0), ctext[max(0, m.start() - 60): m.end() + 60])

        # Pressure tactics
        for value, pats in TACTIC_CUES.items():
            for pat in pats:
                m = pat.search(ctext)
                if m:
                    self._add(out, "USES_TACTIC", value, m.group(0), ctext[max(0, m.start() - 60): m.end() + 60])
                    break
        for s in sentences:
            if _TODAY.search(s) and _PRESSURE_WORDS.search(s):
                self._add(out, "USES_TACTIC", "urgency", _TODAY.search(s).group(0), s)
                break

        # Requested assets (sentence must contain a request verb and not be negated)
        for s in sentences:
            if not REQUEST_VERBS.search(s) or _NEGATED.search(s):
                continue
            for value, pat in ASSET_CUES.items():
                m = pat.search(s)
                if m:
                    self._add(out, "REQUESTS", value, m.group(0), s)
        if _SSN_MENTIONED.search(ctext):
            m = _VERIFY_NUMBER.search(ctext)
            if m:
                self._add(out, "REQUESTS", "ssn", m.group(0), m.group(0), inferred=True)

        # Payment channels
        for value, pat in PAYMENT_CUES.items():
            m = pat.search(ctext)
            if m:
                self._add(out, "PAY_VIA", value, m.group(0), ctext[max(0, m.start() - 60): m.end() + 60])

        # Callback numbers / extensions / IVR options
        for m in PHONE_RE.finditer(ctext):
            self._add(out, "PROVIDES_NUMBER", m.group(0), m.group(0), ctext[max(0, m.start() - 40): m.end() + 10])
        for m in EXT_RE.finditer(ctext):
            self._add(out, "PROVIDES_NUMBER", f"ext {m.group(1)}", m.group(0), m.group(0))
        m = IVR_RE.search(ctext)
        if m:
            self._add(out, "PROVIDES_NUMBER", m.group(0), m.group(0), ctext[max(0, m.start() - 60): m.end() + 20])

        triples = list(out.values())
        return ExtractionRecord(call_id=call_id, extractor=self.name, triples=triples, json_valid=True,
                                schema_conformant=True, attempts=1, n_raw=len(triples),
                                latency_s=round(time.time() - t0, 4))


__all__ = ["RuleBasedExtractor", "caller_text", "normalize_phone"]
