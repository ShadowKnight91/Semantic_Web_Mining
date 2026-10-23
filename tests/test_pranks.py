"""Generated prank calls: transcript parsing, the generation plan, loading and splitting (no LLM needed)."""
import json

import pandas as pd

from scamkg.data.pranks import CONTROLS, PRANKS, parse_dialogue, plan


def test_parse_dialogue_normalizes_generated_text():
    raw = ("Here is the call:\n**Caller:** Is your fridge running? (giggles)\nReceiver: What?\nwho is this\n"
           "CALLER: Go catch it!\nreceiver: Very funny.\ncaller: Bye!\nreceiver: Bye.")
    assert parse_dialogue(raw) == ("caller: Is your fridge running?\nreceiver: What? who is this\n"
                                   "caller: Go catch it!\nreceiver: Very funny.\ncaller: Bye!\nreceiver: Bye.")


def test_parse_dialogue_rejects_non_dialogues():
    assert parse_dialogue("Sorry, I can't write that.") is None
    assert parse_dialogue("receiver: hi\ncaller: hello\n" * 4) is None            # must open with the caller
    assert parse_dialogue("\n".join(f"caller: line {i}" for i in range(8))) is None  # needs both speakers
    assert parse_dialogue("caller: hi\nreceiver: hello") is None                    # too short


def test_plan_is_seeded_and_balanced():
    p = plan(len(PRANKS) * 2, len(CONTROLS), seed=1)
    assert p == plan(len(PRANKS) * 2, len(CONTROLS), seed=1)
    pranks = [x for x in p if x["dataset"] == "pranks"]
    assert pd.Series([x["subtype"] for x in pranks]).value_counts().eq(2).all()
    assert all(x["label"] == 0 and x["label3"] == "prankster" for x in pranks)
    ctrl = [x for x in p if x["dataset"] == "pranks_control"]
    assert {x["subtype"] for x in ctrl} == set(CONTROLS)
    assert all((x["label"], x["label3"]) == CONTROLS[x["subtype"]][:2] for x in ctrl)


def test_pranks_get_their_own_split_and_the_primary_split_is_unchanged():
    from scamkg.data.preprocess import assign_splits
    prim = pd.DataFrame({"dataset": "p", "label": [0, 1] * 50}, index=[f"p_{i}" for i in range(100)])
    pranks = pd.DataFrame({"dataset": "pranks", "label": 0}, index=[f"k_{i}" for i in range(40)])
    ctrl = pd.DataFrame({"dataset": "pranks_control", "label": [0, 1] * 5}, index=[f"c_{i}" for i in range(10)])
    alone = assign_splits(prim, "p", 0.15, 0.15, 7)
    both = assign_splits(pd.concat([prim, pranks, ctrl]), "p", 0.15, 0.15, 7, also=["pranks"])
    assert both.loc[prim.index, "split"].equals(alone["split"])
    assert set(both.loc[pranks.index, "split"]) == {"train", "val", "test"}
    assert set(both.loc[ctrl.index, "split"]) == {"external"}


def test_load_pranks(tmp_path):
    from scamkg.config import get_paths, load_config
    from scamkg.data.loaders import load_pranks
    rows = [{"source_id": "pranks_0000", "dataset": "pranks", "raw_text": "caller: hi\nreceiver: hello", "label": 0,
             "label3": "prankster", "subtype": "echo"},
            {"source_id": "pranks_control_0001", "dataset": "pranks_control", "raw_text": "caller: ssn\nreceiver: no",
             "label": 1, "label3": "fraudster", "subtype": "ssn"}]
    (tmp_path / "pranks.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    cfg = load_config(overrides={"data": {"sources": {"pranks": {"local_dir": str(tmp_path)}}}})
    df = load_pranks(cfg, get_paths(cfg)).set_index("dataset")
    assert df.at["pranks", "label3"] == "prankster" and df.at["pranks", "subtype"] == "echo"
    assert (df.at["pranks_control", "label"], df.at["pranks_control", "label3"]) == (1, "fraudster")
