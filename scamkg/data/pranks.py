"""Synthetic prank calls for the caller-type task (fraudster / salesperson / prankster / other legitimate).

None of the data sets has prank calls (BothBosu: scams, sales, deliveries, wrong numbers; Kaggle and NCSU:
scams, sales and service calls), so the proposal's prankster class needs generated data. BothBosu is itself
LLM-generated, so the pranks are generated the same way, in its "caller: ... / receiver: ..." format.

Two record sets go to one JSONL file:
  pranks          prank calls (label 0, label3 "prankster"); split into train / val / test like BothBosu
  pranks_control  fraud and sales calls written by the SAME model and prompt template (test only). If the
                  classifier calls these "prankster", it learned the generator's writing style, not the content.

Every request is cached (data/cache/llm) and seeded, so `generate-pranks` replays the same calls.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from scamkg.utils import get_logger, write_jsonl

log = get_logger(__name__)

# what the prankster does; several imitate a scam or sales opening on purpose, so the class is not
# separable by the opening line alone
PRANKS = {
    "running_fridge": "asks whether the receiver's refrigerator is running, then tells them they had better go catch it",
    "radio_prize": "pretends to be a radio host announcing that the receiver won an absurd prize, such as a lifetime "
                   "supply of rubber ducks, and asks them to sing the station jingle to claim it",
    "snack_survey": "pretends to run a national survey and asks more and more absurd questions about breakfast cereal",
    "pizza_order": "calls as if the receiver were a pizzeria and orders a pizza with ridiculous toppings",
    "lost_twin": "claims to be the receiver's long-lost identical twin and asks about their shared childhood",
    "time_traveler": "claims to be calling from the future to warn the receiver about something trivial",
    "talking_pet": "pretends to be the receiver's dog or cat who learned to use the phone",
    "pun_name": "keeps asking for a person whose name is a silly pun and acts surprised the person is not there",
    "zoo_escape": "pretends to be from the city zoo and says an escaped penguin was seen in the receiver's yard",
    "absurd_job": "pretends to hold a job interview for a ridiculous job, such as professional bubble-wrap popper",
    "fake_boss": "pretends to be the receiver's boss and gives silly instructions for tomorrow",
    "silly_bet": "asks the receiver to settle a silly bet between the caller and their friends",
    "fortune_teller": "pretends to be a fortune teller predicting absurd events in the receiver's week",
    "cheese_hotline": "pretends to be an emergency hotline for cheese and asks about the receiver's cheese situation",
    "average_award": "announces that the receiver was voted the most average person in town",
    "echo": "repeats everything the receiver says back to them",
    "clown_booking": "asks whether the receiver ordered a clown for a party and describes the clown's act",
    "sock_census": "pretends to take the national sock census and asks how many socks the receiver owns",
    "alien": "pretends to be an alien asking for directions to the leader of Earth",
    "ghost": "pretends to be a ghost living in the receiver's attic who wants the music turned down",
    "game_show": "pretends to be a game-show host, asks easy trivia questions and awards imaginary points",
    "llama_wrong_number": "starts like a wrong number and turns into a long story about the caller's lost llama",
    "chicken_delivery": "pretends to be a delivery driver with 500 live chickens for the receiver",
    "loud_thinking": "complains that the receiver is thinking too loudly",
    "poem_review": "reads a silly poem and asks the receiver to rate it",
    "fake_tech_joke": "pretends to be computer support and asks the receiver to turn the toaster off and on again",
    "celebrity_voice": "pretends to be a famous singer (no real name) and invites the receiver to sing a duet",
    "mystery_caller": "refuses to say who is calling and makes the receiver guess, giving silly hints",
}
CALLERS = ["a bored teenager whose friends can be heard laughing", "an adult using an exaggerated formal voice",
           "a college student filming the call for a social media video", "a kid who keeps trying not to laugh"]
REACTIONS = ["confused", "amused and plays along", "annoyed", "suspicious, asks whether this is a scam",
             "polite but impatient", "quick to see through it"]
ENDINGS = ["The caller admits it was a prank near the end.",
           "The caller never admits it is a prank; the call ends when the receiver hangs up or gives up."]

CONTROLS = {   # (label, label3, what the caller does) - mirrors BothBosu's scam and sales types
    "ssn": (1, "fraudster", "a scammer posing as a government agent who says the receiver's social security number "
                            "was suspended and asks them to confirm it"),
    "refund": (1, "fraudster", "a scammer posing as an online store's refund department who asks for bank details "
                               "to send a refund"),
    "support": (1, "fraudster", "a scammer posing as computer technical support who says the computer has a virus "
                                "and asks for remote access"),
    "reward": (1, "fraudster", "a scammer announcing a gift-card prize who asks for personal details to deliver it"),
    "insurance": (0, "salesperson", "an honest insurance agent offering a discounted policy"),
    "telemarketing": (0, "salesperson", "an honest telemarketer offering a product or service at a special price"),
}

SYSTEM = "You write realistic phone call transcripts for a research data set on phone call classification."
SPEAKER = re.compile(r"^\W*(caller|receiver)\s*:\s*(.*)$", re.IGNORECASE)


def prank_prompt(scenario: str, caller: str, reaction: str, ending: str, n_lines: int) -> str:
    return (f"Write a phone call transcript in which the CALLER makes a prank call.\n"
            f"The prank: the caller {scenario}.\nThe caller is {caller}. The receiver is {reaction}.\n"
            f"{ending}\n" + _FORMAT.format(n=n_lines) +
            "The caller never asks for money, bank or card details, passwords or identity numbers.\n")


def control_prompt(behaviour: str, reaction: str, n_lines: int) -> str:
    return (f"Write a phone call transcript in which the CALLER is {behaviour}.\nThe receiver is {reaction}.\n"
            + _FORMAT.format(n=n_lines))


_FORMAT = ("Format: exactly {n} lines, one per turn, alternating and starting with the caller. Every line starts "
           "with \"caller: \" or \"receiver: \". Plain spoken English, no names of real people, no stage "
           "directions, sound effects or narration. Return only the transcript.\n")


def parse_dialogue(text: str | None, min_lines: int = 6) -> str | None:
    """Normalize a generated transcript to 'caller: ...' / 'receiver: ...' lines; None if it is not one."""
    lines = []
    for raw in str(text or "").splitlines():
        s = raw.strip().replace("**", "")
        if not s:
            continue
        m = SPEAKER.match(s)
        if m:
            utterance = re.sub(r"\s*[\(\[][^\)\]]{0,60}[\)\]]\s*", " ", m.group(2)).strip()   # stage directions
            if utterance:
                lines.append((m.group(1).lower(), utterance))
        elif lines:                  # a wrapped continuation of the previous turn
            lines[-1] = (lines[-1][0], f"{lines[-1][1]} {s}")
    if len(lines) < min_lines or lines[0][0] != "caller" or {s for s, _ in lines} != {"caller", "receiver"}:
        return None
    return "\n".join(f"{s}: {u}" for s, u in lines)


def plan(n_pranks: int, n_controls: int, seed: int) -> list[dict]:
    """Balanced, seeded plan: every prank scenario and control type is used about equally often."""
    rng = np.random.default_rng(seed)
    out = []
    names = list(PRANKS)
    for i in range(n_pranks):
        name = names[i % len(names)]
        out.append({"dataset": "pranks", "label": 0, "label3": "prankster", "subtype": name,
                    "prompt": prank_prompt(PRANKS[name], str(rng.choice(CALLERS)), str(rng.choice(REACTIONS)),
                                           str(rng.choice(ENDINGS)), int(rng.integers(8, 17)))})
    cnames = list(CONTROLS)
    for i in range(n_controls):
        name = cnames[i % len(cnames)]
        label, label3, behaviour = CONTROLS[name]
        out.append({"dataset": "pranks_control", "label": label, "label3": label3, "subtype": name,
                    "prompt": control_prompt(behaviour, str(rng.choice(REACTIONS)), int(rng.integers(8, 17)))})
    return out


def generate(cfg: dict, out_path: Path, cache_dir: Path, n_pranks: int = 120, n_controls: int = 36,
             model: str | None = None, provider: str | None = None, seed: int = 42, attempts: int = 3) -> Path:
    from scamkg.extraction.llm_client import LLMError, make_client
    gen_cfg = {**cfg, "llm": {**cfg["llm"], "temperature": 0.9, "json_mode": False, "max_tokens": 1200}}
    rows, seen, failed = [], set(), 0
    items = plan(n_pranks, n_controls, seed)
    for i, item in enumerate(items):
        text = None
        for a in range(attempts):        # a new seed per attempt: a cached bad answer is not replayed
            gen_cfg["llm"]["seed"] = seed * 1000 + i * 10 + a
            client = make_client(gen_cfg, cache_dir=cache_dir, model=model, provider=provider)
            try:
                text = parse_dialogue(client.chat([{"role": "system", "content": SYSTEM},
                                                   {"role": "user", "content": item["prompt"]}]))
            except LLMError as e:
                log.warning("Generation %d failed: %s", i, e)
                text = None
            if text and text.lower() not in seen:
                break
            text = None
        if text is None:
            failed += 1
            continue
        seen.add(text.lower())
        rows.append({"source_id": f"{item['dataset']}_{i:04d}", "dataset": item["dataset"], "raw_text": text,
                     "label": item["label"], "label3": item["label3"], "subtype": item["subtype"],
                     "generator": f"{client.provider}:{client.model}", "seed": gen_cfg["llm"]["seed"]})
        if (i + 1) % 20 == 0:
            log.info("Generated %d / %d calls", i + 1, len(items))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, rows)
    log.info("Wrote %d calls to %s (%d pranks, %d controls; %d failed)", len(rows), out_path,
             sum(r["dataset"] == "pranks" for r in rows), sum(r["dataset"] == "pranks_control" for r in rows), failed)
    return out_path
