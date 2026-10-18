"""E3 - transformer text models (strong text-only deep baselines).

* sbert_mlp  : Sentence-BERT embeddings (mean over ~180-word chunks) -> MLP
* distilbert : fine-tuned DistilBERT (head + tail truncation, early stopping on validation PR-AUC)
Requires `pip install -r requirements-deep.txt`.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier

from scamkg.evaluation.evaluator import Evaluator
from scamkg.utils import get_logger, has_module, set_seed, slugify

log = get_logger(__name__)


def require(*modules: str) -> None:
    missing = [m for m in modules if not has_module(m)]
    if missing:
        raise RuntimeError(f"Missing optional dependencies {missing}: pip install -r requirements-deep.txt")


def _device(preference: str = "auto"):
    """`auto`: CUDA, else Apple-silicon MPS, else CPU. Set models.transformer.device: cpu for bit-reproducible runs."""
    import torch
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def sbert_encode(texts: list[str], model_name: str, chunk_words: int = 180, batch_size: int = 32,
                 device: str = "auto") -> np.ndarray:
    """Mean-pooled, L2-normalized sentence embeddings over word chunks (handles long transcripts)."""
    require("sentence_transformers")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device=str(_device(device)))
    chunks, owners = [], []
    for i, t in enumerate(texts):
        words = str(t).split() or [""]
        for j in range(0, len(words), chunk_words):
            chunks.append(" ".join(words[j:j + chunk_words]))
            owners.append(i)
    emb = model.encode(chunks, batch_size=batch_size, show_progress_bar=len(chunks) > 200, normalize_embeddings=True)
    out = np.zeros((len(texts), emb.shape[1]), dtype=np.float32)
    counts = np.zeros(len(texts))
    np.add.at(out, np.asarray(owners), emb)
    np.add.at(counts, np.asarray(owners), 1)
    out /= np.maximum(counts, 1)[:, None]
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
    return out


def call_embeddings(ctx, ids: list[str]) -> np.ndarray:
    """SBERT embeddings for calls, cached in data/cache/embeddings/<model>.npz."""
    scfg = ctx.cfg["models"]["sbert"]
    path = Path(ctx.paths.cache) / "embeddings" / f"{slugify(scfg['model_name'])}.npz"
    cache: dict[str, np.ndarray] = {}
    if path.exists():
        z = np.load(path, allow_pickle=False)
        cache = dict(zip(z["ids"].tolist(), z["emb"]))
    missing = [c for c in ids if c not in cache]
    if missing:
        log.info("Encoding %d calls with %s", len(missing), scfg["model_name"])
        emb = sbert_encode(ctx.texts(missing), scfg["model_name"], int(scfg.get("chunk_words", 180)),
                           device=ctx.cfg["models"]["transformer"].get("device", "auto"))
        cache.update(zip(missing, emb))
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(cache)
        np.savez_compressed(path, ids=np.array(keys), emb=np.stack([cache[k] for k in keys]))
    return np.stack([cache[c] for c in ids])


def run_sbert_mlp(ctx) -> None:
    ev = Evaluator(ctx)
    scfg = ctx.cfg["models"]["sbert"]
    fit_ids = ctx.train_ids + ctx.val_ids
    eval_ids = ev.eval_ids()
    Xfit, Xev = call_embeddings(ctx, fit_ids), call_embeddings(ctx, eval_ids)
    mlp = MLPClassifier(hidden_layer_sizes=tuple(scfg.get("mlp_hidden", [256])), alpha=1e-4, early_stopping=True,
                        validation_fraction=0.15, max_iter=500, random_state=ctx.seed)
    mlp.fit(Xfit, ctx.labels(fit_ids))
    ev.evaluate("E3", "sbert_mlp", pd.Series(mlp.predict_proba(Xev)[:, 1], index=eval_ids))


def _encode_head_tail(tokenizer, texts: list[str], max_length: int):
    """Tokenize keeping the first ~half and the last ~half of long transcripts."""
    import torch
    head = (max_length - 2) // 2
    tail = max_length - 2 - head
    ids_list = []
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False, truncation=False)["input_ids"]
        if len(ids) > max_length - 2:
            ids = ids[:head] + ids[-tail:]
        ids_list.append([tokenizer.cls_token_id] + ids + [tokenizer.sep_token_id])
    L = max(len(x) for x in ids_list)
    input_ids = torch.full((len(texts), L), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros((len(texts), L), dtype=torch.long)
    for i, x in enumerate(ids_list):
        input_ids[i, :len(x)] = torch.tensor(x)
        mask[i, :len(x)] = 1
    return input_ids, mask


def _predict(model, input_ids, mask, batch_size: int, device) -> np.ndarray:
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(input_ids), batch_size):
            logits = model(input_ids=input_ids[i:i + batch_size].to(device),
                           attention_mask=mask[i:i + batch_size].to(device)).logits
            out.append(torch.softmax(logits, dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def run_distilbert(ctx) -> None:
    require("torch", "transformers")
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    tcfg = ctx.cfg["models"]["transformer"]
    ev = Evaluator(ctx)
    device = _device(tcfg.get("device", "auto"))
    log.info("DistilBERT on %s", device)
    tok = AutoTokenizer.from_pretrained(tcfg["model_name"])
    L, bs = int(tcfg["max_length"]), int(tcfg["batch_size"])
    tr, va, eval_ids = ctx.train_ids, ctx.val_ids, ev.eval_ids()
    Xtr, Mtr = _encode_head_tail(tok, ctx.texts(tr), L)
    Xva, Mva = _encode_head_tail(tok, ctx.texts(va), L)
    Xev, Mev = _encode_head_tail(tok, ctx.texts(eval_ids), L)
    ytr, yva = torch.tensor(ctx.labels(tr)), ctx.labels(va)
    for k, seed in enumerate(tcfg.get("seeds", [42])):
        set_seed(int(seed))
        model = AutoModelForSequenceClassification.from_pretrained(tcfg["model_name"], num_labels=2).to(device)
        loader = DataLoader(TensorDataset(Xtr, Mtr, ytr), batch_size=bs, shuffle=True,
                            generator=torch.Generator().manual_seed(int(seed)))
        opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg.get("weight_decay", 0.01)))
        steps = len(loader) * int(tcfg["epochs"])
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        best_ap, best_state = -1.0, None
        for epoch in range(int(tcfg["epochs"])):
            model.train()
            total = 0.0
            for xb, mb, yb in loader:
                opt.zero_grad()
                loss = model(input_ids=xb.to(device), attention_mask=mb.to(device), labels=yb.to(device)).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                total += loss.item() * len(yb)
            ap = float(average_precision_score(yva, _predict(model, Xva, Mva, bs * 2, device)))
            log.info("DistilBERT seed %s epoch %d: train loss %.4f, val PR-AUC %.4f", seed, epoch + 1,
                     total / len(tr), ap)
            if ap > best_ap:
                best_ap, best_state = ap, copy.deepcopy({k_: v.cpu() for k_, v in model.state_dict().items()})
        model.load_state_dict(best_state)
        scores = pd.Series(_predict(model, Xev, Mev, bs * 2, device), index=eval_ids)
        ev.evaluate("E3", "distilbert" if k == 0 else f"distilbert_s{seed}", scores, extra={"val_pr_auc": best_ap})
        if k == 0:
            out = ctx.paths.models / "distilbert"
            model.save_pretrained(out)
            tok.save_pretrained(out)
