"""Recall under simulated speech-to-text noise: classical models here, GraphSAGE in a spawned worker (torch)."""
from __future__ import annotations

import multiprocessing

import pandas as pd

from scamkg.utils import get_logger

log = get_logger(__name__)


SEEDS = (42, 1, 2, 3, 4)


def _gnn_worker(cfg: dict, run: str, levels: list[str], out_csv: str) -> None:
    import copy

    import numpy as np
    import torch

    from scamkg.context import Context
    from scamkg.evaluation.evaluator import Evaluator
    from scamkg.evaluation.monitor import noise_rows, noisy_context
    from scamkg.models.gnn import (VARIANT_NAMES, _train_variant, kg_block, prepare_graphs, score_variant, text_block,
                                   topology_block, type_block)
    from scamkg.models.transformer import sbert_encode
    from scamkg.utils import has_module, set_seed

    set_seed(int(cfg["project"]["seed"]))
    ctx = Context.load(cfg, run=run)
    gcfg = cfg["models"]["graphsage"]
    use_pyg = has_module("torch_geometric")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ext = ctx.calls.index[ctx.calls["dataset"].isin(ctx.external_datasets)].tolist()

    def blocks_for(nctx, level):
        # noisy triples change the entity set, so this block is rebuilt. Call rows come from the cache of clean
        # transcripts (training calls stay clean); the external calls are re-encoded from their noisy text.
        text = text_block(nctx, gcfg.get("node_features", "sbert"))
        if level != "clean":
            text[[nctx.kg.call_index[c] for c in ext]] = sbert_encode(
                nctx.texts(ext), cfg["models"]["sbert"]["model_name"], int(cfg["models"]["sbert"].get("chunk_words", 180)),
                device=cfg["models"]["transformer"].get("device", "auto"))
        return {"text": text, "kg": kg_block(text, nctx.kg.n_calls), "topo": topology_block(nctx)[0],
                "type": type_block(nctx)}

    views = {}
    for level in levels:
        nctx = noisy_context(ctx, level)
        graphs, _, _, n_c, _ = prepare_graphs(nctx, use_pyg, device)
        views[level] = (nctx, blocks_for(nctx, level), graphs, n_c)
    clean, clean_blocks, clean_graphs, n_c = views["clean"]
    rows = []
    for seed in SEEDS:
        run_ctx = copy.copy(clean)
        run_ctx.seed = seed
        thresholds: dict = {}
        for variant in ("text+topo", "topo"):
            # trained once on clean calls; each noise level only runs these weights
            trained = _train_variant(run_ctx, variant, clean_blocks, clean_graphs, n_c, use_pyg, device)
            for level in levels:
                nctx, blocks, graphs, n = views[level]
                eval_ids = Evaluator(nctx).eval_ids()
                idx = np.array([nctx.kg.call_index[c] for c in eval_ids])
                prob = score_variant(nctx, variant, trained, blocks, graphs["full"], n, use_pyg, device)
                rows += [{**r, "seed": seed} for r in noise_rows(nctx, f"E4__{VARIANT_NAMES[variant]}", level,
                                                                 pd.Series(prob[idx], index=eval_ids), thresholds)]
        log.info("Noise: GraphSAGE seed %s done", seed)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def noise_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread over seeds (GraphSAGE); classical models are deterministic, so their spread is empty."""
    metrics = ["recall_default", "false_alarm_default", "recall_clean_threshold", "false_alarm_clean_threshold",
               "recall_at_target_fpr"]
    grouped = df.groupby(["model", "stage", "noise"], sort=False)[metrics]
    out = grouped.mean().join(grouped.std(ddof=1).add_suffix("_sd"))
    return out.reset_index()


def run_noise_robustness(cfg: dict, ctx, noise_run: str = "rule", deep: bool = True) -> pd.DataFrame:
    """Every model scores the same noisy transcripts; triples are re-extracted by rules (see noisy_context)."""
    from scamkg.config import deep_merge
    from scamkg.context import Context
    from scamkg.evaluation.monitor import monitor_config, plot_noise, run_noise_classical
    from scamkg.extraction.benchmark import feedback_dir
    from scamkg.utils import has_module

    levels = list(monitor_config(cfg)["noise_levels"])
    rcfg = deep_merge(cfg, {"extraction": {"default_run": noise_run}})
    nctx = Context.load(rcfg, run=noise_run)
    df = run_noise_classical(nctx, levels)
    fb = feedback_dir(ctx)
    if deep and has_module("torch"):
        tmp = fb / "_noise_gnn.csv"
        p = multiprocessing.get_context("spawn").Process(target=_gnn_worker, args=(rcfg, noise_run, levels, str(tmp)))
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"GraphSAGE noise worker failed (exit code {p.exitcode})")
        df = pd.concat([df, pd.read_csv(tmp)], ignore_index=True)
        tmp.unlink()
    summary = noise_summary(df)
    summary.to_csv(fb / "recall_noise.csv", index=False)
    plot_noise(summary, fb / "figures" / "recall_noise.png")
    return summary[summary["stage"].str.contains("robocalls")].pivot_table(
        index="model", columns="noise", values="recall_clean_threshold")[levels].reset_index()
