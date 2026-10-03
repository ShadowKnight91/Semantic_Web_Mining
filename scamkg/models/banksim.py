"""Optional reference baseline from prior course projects: BankSim transactions + graph features.

Customer-merchant graph -> PageRank, degree, Louvain community size (label-free) + tabular
features -> Random Forest / XGBoost. Download bs140513_032310.csv from
https://www.kaggle.com/datasets/ealaxi/banksim1 into data/raw/banksim/.
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from scamkg.evaluation.metrics import binary_metrics
from scamkg.models.classical import make_model
from scamkg.utils import get_logger

log = get_logger(__name__)


def run_banksim(cfg: dict, paths, csv_path: str | Path | None = None) -> pd.DataFrame:
    csv_path = Path(csv_path) if csv_path else paths.raw / "banksim" / "bs140513_032310.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found - download BankSim from Kaggle (ealaxi/banksim1)")
    seed = int(cfg["project"]["seed"])
    df = pd.read_csv(csv_path)
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].str.strip("'")
    G = nx.Graph()
    edges = df.groupby(["customer", "merchant"]).size().reset_index(name="w")
    G.add_weighted_edges_from(("C:" + edges["customer"], "M:" + edges["merchant"], edges["w"]))
    pr = nx.pagerank(G, weight="weight")
    deg = dict(G.degree())
    comm = nx.community.louvain_communities(G, weight="weight", seed=seed)
    size = {v: len(c) for c in comm for v in c}
    comm_id = {v: i for i, c in enumerate(comm) for v in c}
    feats = pd.DataFrame({
        "amount": df["amount"], "step": df["step"],
        "customer_pagerank": ("C:" + df["customer"]).map(pr), "merchant_pagerank": ("M:" + df["merchant"]).map(pr),
        "customer_degree": ("C:" + df["customer"]).map(deg), "merchant_degree": ("M:" + df["merchant"]).map(deg),
        "customer_community_size": ("C:" + df["customer"]).map(size),
        "same_community": [int(comm_id["C:" + c] == comm_id["M:" + m]) for c, m in zip(df["customer"], df["merchant"])],
    })
    feats = pd.concat([feats, pd.get_dummies(df[["category", "age", "gender"]].astype(str), dtype=float)], axis=1)
    y = df["fraud"].astype(int).to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(feats.to_numpy(dtype=float), y, test_size=0.3, stratify=y, random_state=seed)
    rows = []
    for name, model in (("rf", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=-1,
                                                      random_state=seed)),
                        ("xgb", make_model("xgb", {}, seed, ytr))):
        model.fit(Xtr, ytr)
        rows.append({"model": name, **binary_metrics(yte, model.predict_proba(Xte)[:, 1])})
        log.info("BankSim %s: PR-AUC %.4f recall %.4f", name, rows[-1]["pr_auc"], rows[-1]["recall"])
    out = pd.DataFrame(rows)
    out.to_csv(paths.results / "banksim_baseline.csv", index=False)
    return out
