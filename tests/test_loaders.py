import pandas as pd

from scamkg.config import get_paths, load_config
from scamkg.context import structural_cache_path
from scamkg.data.loaders import load_bothbosu


def test_bothbosu_types_give_three_way_labels(tmp_path):
    pd.DataFrame({"dialogue": ["caller: hello there how are you"] * 5,
                  "type": ["ssn", "telemarketing", "wrong", "insurance", "delivery"],
                  "label": [1, 0, 0, 0, 0]}).to_csv(tmp_path / "bothbosu.csv", index=False)
    cfg = {"data": {"sources": {"bothbosu": {"local_dir": str(tmp_path)}}}}
    df = load_bothbosu(cfg, get_paths(load_config()))
    assert df["label3"].tolist() == ["fraudster", "salesperson", "other_legitimate", "salesperson",
                                     "other_legitimate"]


def test_structural_cache_tracks_run_content(tmp_path):
    cfg = load_config()
    paths = get_paths(cfg)
    run = tmp_path / "myrun.jsonl"
    run.write_text('{"call_id": "a", "triples": []}\n', encoding="utf-8")
    before = structural_cache_path(paths, run, cfg, ["a"])
    run.write_text('{"call_id": "a", "triples": [{"relation": "PAY_VIA", "object": "zelle"}]}\n', encoding="utf-8")
    after = structural_cache_path(paths, run, cfg, ["a"])
    assert before != after and after.name.startswith("structural__myrun__")


def test_kaggle_text_files(tmp_path):
    from scamkg.data.loaders import load_kaggle
    (tmp_path / "English_Scam.txt").write_bytes(
        "1.\t[Greetings], this is the IRS. Pay the fine today with gift cards.\r\n\r\n"
        "2.\tYou won a prize, send the fee now.\r\n".encode())
    (tmp_path / "English_NonScam.txt").write_bytes(
        "[Greetings], this is [Company] Pharmacy. Your refill is ready.\r\n\r\n".encode())
    (tmp_path / "Malay_Scam.txt").write_text("1.\tTidak.\n", encoding="utf-8")       # other language: ignored
    df = load_kaggle({"data": {"sources": {"kaggle": {"local_dir": str(tmp_path)}}}}, get_paths(load_config()))
    assert sorted(zip(df["label"], df["raw_text"].str[:12])) == [
        (0, "[Greetings],"), (1, "You won a pr"), (1, "[Greetings],")]


def test_template_placeholders_are_removed():
    from scamkg.data.preprocess import clean_transcript
    _, plain, _ = clean_transcript("[Greetings], this is [Company] Bank. Your card ending in [Number] is blocked.")
    assert plain == "this is Bank. Your card ending in is blocked."
    # bracketed noise inside a dialogue keeps its old treatment
    _, plain, _ = clean_transcript("caller: email us at [support@x.com]. receiver: ok")
    assert plain == "email us at . ok"


def test_call_ids_do_not_shift_when_a_data_set_is_added(tmp_path, monkeypatch):
    import scamkg.data.preprocess as pre
    from scamkg.data.loaders import _frame
    prim = _frame("prim", [f"caller: primary call number {i} about topic {i % 7}" for i in range(40)], [i % 2 for i in range(40)])
    ext = _frame("ext", [f"external robocall {w} please press one now" for w in "abcdefgh"], [1] * 8)
    mid = _frame("mid", ["external robocall c please press one now", "something else entirely here"], [1, 0])
    frames = {"prim": prim, "ext": ext, "mid": mid}
    monkeypatch.setattr(pre, "load_source", lambda name, cfg, paths: frames[name])
    ids = {}
    for load in (["prim", "ext"], ["prim", "mid", "ext"]):
        cfg = load_config(overrides={"paths": {k: str(tmp_path / k) for k in ("data_processed", "data_raw")},
                                     "data": {"load": load, "primary": "prim"}})
        df = pre.preprocess(cfg, get_paths(cfg))
        ids[len(load)] = dict(zip(df.loc[df["dataset"] == "ext", "text_plain"], df.loc[df["dataset"] == "ext", "call_id"]))
    # "mid" duplicates one external transcript: that call disappears, every other external id stays put
    assert len(ids[3]) == 7 and all(ids[2][t] == i for t, i in ids[3].items())
