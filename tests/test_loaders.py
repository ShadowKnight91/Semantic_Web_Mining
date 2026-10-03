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
