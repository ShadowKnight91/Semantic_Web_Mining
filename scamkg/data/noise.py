"""Simulated speech-to-text noise, for measuring how recall degrades on noisy calls.

Phone transcripts come out of speech recognition: no punctuation or capitals, spoken digits, a missed first
second of the recording, and word errors. `add_asr_noise` applies those at a target word error rate (WER). It is
a stress test with fixed seeds, not a model of one recognizer. Noise is deterministic per (call, level, seed).
"""
from __future__ import annotations

import random
import re

from scamkg.data.preprocess import clean_transcript

# level: word error rate, and the share of words lost at the start of the recording
NOISE_LEVELS = {"clean": (0.0, 0.0), "light": (0.05, 0.0), "medium": (0.15, 0.05), "heavy": (0.30, 0.10)}

# errors are mostly substitutions; deletions and insertions are less common
_SUB, _DEL = 0.6, 0.3
_FILLERS = ("uh", "um", "the", "and", "you")
# phonetically close words, weighted towards the vocabulary that scam calls depend on
CONFUSIONS = {
    "social": "special", "security": "secure", "number": "amber", "card": "car", "cards": "cars", "gift": "give",
    "account": "count", "accounts": "counts", "press": "dress", "pay": "play", "payment": "pavement", "fraud": "frog",
    "warranty": "warrant", "amazon": "amazing", "order": "older", "bank": "back", "call": "tall", "verify": "very",
    "officer": "offer", "arrest": "rest", "warrant": "warren", "loan": "lone", "credit": "credits", "debt": "dead",
    "federal": "feral", "government": "governor", "police": "please", "legal": "regal", "immediately": "immediate",
    "urgent": "agent", "suspended": "suspected", "suspicious": "suspicion", "transfer": "transfers", "wire": "wider",
    "password": "passport", "claim": "clam", "refund": "refunded", "insurance": "assurance", "medicare": "medical",
    "microsoft": "micro", "computer": "commuter", "virus": "iris", "prize": "price", "free": "three", "agent": "aging",
    "representative": "represented", "department": "apartment", "expire": "expired", "expired": "expire",
}
_DIGITS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven",
           "8": "eight", "9": "nine"}
_SPEAKER = re.compile(r"^\s*(caller|receiver)\s*:\s*", re.I)
_PUNCT = re.compile(r"[^\w\s'-]")


def _mishear(word: str, rng: random.Random) -> str:
    if word in CONFUSIONS:
        return CONFUSIONS[word]
    if len(word) > 4 and word.endswith(("s", "d", "g")):
        return word[:-1]
    return rng.choice(_FILLERS)


def _noisy_words(words: list[str], wer: float, rng: random.Random) -> list[str]:
    out = []
    for w in words:
        if rng.random() >= wer:
            out.append(w)
            continue
        kind = rng.random()
        if kind < _SUB:
            out.append(_mishear(w, rng))
        elif kind >= _SUB + _DEL:
            out += [w, rng.choice(_FILLERS)]
    return out


def add_asr_noise(text: str, level: str, key: str = "", seed: int = 42) -> str:
    """`text` as a speech recognizer might have returned it. Speaker tags are kept, so turns stay attributable."""
    wer, head_cut = NOISE_LEVELS[level]
    if wer == 0:
        return text
    rng = random.Random(f"{seed}:{key}:{level}")
    lines = []
    for i, line in enumerate(str(text).split("\n")):
        tag = _SPEAKER.match(line)
        body = line[tag.end():] if tag else line
        words = _PUNCT.sub(" ", body.lower()).split()
        words = [_DIGITS.get(w, w) for w in words]            # "press 1" -> "press one"
        if i == 0 and head_cut:                               # recording started late
            words = words[int(len(words) * head_cut):]
        words = _noisy_words(words, wer, rng)
        lines.append((f"{tag.group(1).lower()}: " if tag else "") + " ".join(words))
    return "\n".join(lines)


def noisy_calls(calls, ids, level: str, seed: int = 42):
    """Copy of the calls table with noise on `ids` (text, text_plain, n_words); every other call is unchanged."""
    out = calls.copy()
    if NOISE_LEVELS[level][0] == 0:
        return out
    for cid in ids:
        text, plain, _ = clean_transcript(add_asr_noise(out.at[cid, "text"], level, cid, seed))
        out.at[cid, "text"], out.at[cid, "text_plain"] = text, plain
        out.at[cid, "n_words"] = len(plain.split())
    return out
