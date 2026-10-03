"""E5 - unsupervised scam-script discovery (RQ4).

K-Means (k chosen by silhouette, plus k=2) and DBSCAN (eps from the k-distance curve) on
label-free graph features + one-hot ontology entities. Labels are used only to score the
clusters (purity, NMI, ARI) and to describe them; every cluster is summarized by its dominant
identity -> tactic -> asset -> payment "script signature".
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import DBSCAN, KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import normalized_mutual_info_score, silhouette_score  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from scamkg.evaluation.metrics import clustering_metrics  # noqa: E402
from scamkg.graph.build import split_key  # noqa: E402
from scamkg.graph.features import LABEL_BASED  # noqa: E402
from scamkg.utils import get_logger, md_table, save_json  # noqa: E402

log = get_logger(__name__)
SIGNATURE_TYPES = ["ClaimedIdentity", "PressureTactic", "RequestedAsset", "PaymentChannel"]


def cluster_matrix(ctx, ids: list[str], pca_dim: int = 20) -> tuple[np.ndarray, list[str], np.ndarray]:
    s = ctx.fb.structural.loc[ids].drop(columns=["community_id"])
    s = s[[c for c in s.columns if c not in LABEL_BASED]]
    names = ctx.fb.onehot_names()
    oh = ctx.fb.onehot_matrix(ids, names).toarray()
    X = np.hstack([StandardScaler().fit_transform(s.to_numpy(dtype=float)), oh * 2.0])
    X = np.nan_to_num(X)
    if X.shape[1] > pca_dim and X.shape[0] > pca_dim:
        X = PCA(n_components=pca_dim, random_state=ctx.seed).fit_transform(X)
    return X, names, oh


def describe_clusters(ids: list[str], labels: np.ndarray, clusters: np.ndarray, names: list[str], oh: np.ndarray,
                      subtypes: pd.Series | None = None) -> list[dict]:
    overall = oh.mean(axis=0) + 1e-9
    out = []
    for c in sorted(set(clusters)):
        mask = clusters == c
        freq = oh[mask].mean(axis=0) if mask.any() else np.zeros(oh.shape[1])
        lift = freq / overall
        sig, top = [], []
        for t in SIGNATURE_TYPES:
            cand = [(freq[j], names[j]) for j in range(len(names)) if split_key(names[j])[0] == t and freq[j] >= 0.3]
            if cand:
                sig.append("+".join(split_key(n)[1] for _, n in sorted(cand, reverse=True)[:2]))
        for j in np.argsort(-(lift * (freq >= 0.2)))[:6]:
            if freq[j] >= 0.2:
                top.append({"entity": names[j], "freq": round(float(freq[j]), 3), "lift": round(float(lift[j]), 2)})
        lab = labels[mask]
        entry = {"cluster": int(c), "size": int(mask.sum()),
                 "scam_rate": float(lab[lab >= 0].mean()) if (lab >= 0).any() else None,
                 "signature": " -> ".join(sig) if sig else "(no dominant entities)", "top_entities": top}
        if subtypes is not None:
            vc = subtypes[mask].value_counts()
            if len(vc) and vc.index[0]:
                entry["dominant_subtype"] = f"{vc.index[0]} ({vc.iloc[0] / mask.sum():.0%})"
        out.append(entry)
    return sorted(out, key=lambda e: -e["size"])


def run_clustering(ctx) -> dict:
    ctx.require_graph()
    ccfg = ctx.cfg["models"]["clustering"]
    ids = ctx.ids(dataset=ctx.primary, labeled=True)
    y = ctx.labels(ids)
    X, names, oh = cluster_matrix(ctx, ids, int(ccfg.get("pca_dim", 20)))
    n = len(ids)
    sample = min(3000, n)
    sweep = []
    for k in range(int(ccfg.get("k_min", 2)), min(int(ccfg.get("k_max", 12)), n - 1) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=ctx.seed).fit(X)
        try:
            sil = float(silhouette_score(X, km.labels_, sample_size=sample, random_state=ctx.seed))
        except ValueError:  # fewer distinct clusters than k (many identical points)
            sil = -1.0
        sweep.append({"k": k, "silhouette": sil, "inertia": float(km.inertia_), **clustering_metrics(y, km.labels_)})
    best_k = max(sweep, key=lambda r: r["silhouette"])["k"]
    km_best = KMeans(n_clusters=best_k, n_init=10, random_state=ctx.seed).fit(X).labels_
    km2 = KMeans(n_clusters=2, n_init=10, random_state=ctx.seed).fit(X).labels_

    subtypes = ctx.calls.loc[ids, "subtype"].reset_index(drop=True)
    results = {"n_calls": n, "kmeans_sweep": sweep,
               "kmeans_best": {"k": best_k, **clustering_metrics(y, km_best)},
               "kmeans_k2": {"k": 2, **clustering_metrics(y, km2)}}
    if subtypes.nunique() > 1 and (subtypes != "").all():
        results["kmeans_best"]["nmi_vs_subtype"] = float(normalized_mutual_info_score(subtypes, km_best))

    # DBSCAN: eps candidates from the k-distance distribution, choose best silhouette (noise excluded)
    ms = int(ccfg.get("dbscan_min_samples", 5))
    dist = NearestNeighbors(n_neighbors=ms).fit(X).kneighbors(X)[0][:, -1]
    best_db = None
    for q in (50, 60, 70, 80, 90, 95):
        eps = float(np.percentile(dist, q))
        if eps <= 0:
            continue
        lab = DBSCAN(eps=eps, min_samples=ms).fit(X).labels_
        core = lab != -1
        n_cl = len(set(lab[core]))
        if n_cl < 2 or core.sum() < 10:
            continue
        try:
            sil = float(silhouette_score(X[core], lab[core], sample_size=min(sample, int(core.sum())),
                                         random_state=ctx.seed))
        except ValueError:
            continue
        if best_db is None or sil > best_db["silhouette"]:
            best_db = {"eps": eps, "eps_percentile": q, "silhouette": sil, "labels": lab}
    if best_db is not None:
        lab = best_db.pop("labels")
        results["dbscan"] = {**best_db, **clustering_metrics(y, lab)}
    else:
        lab = np.full(n, -1)
        results["dbscan"] = {"note": "no eps produced >= 2 clusters"}

    results["scripts"] = describe_clusters(ids, y, km_best, names, oh, subtypes)
    assign = pd.DataFrame({"call_id": ids, "label": y, "kmeans": km_best, "kmeans_k2": km2, "dbscan": lab,
                           "subtype": subtypes})
    assign.to_csv(ctx.paths.results / "cluster_assignments.csv", index=False)
    save_json(ctx.paths.results / "clustering.json", results)

    # 2-D view
    P = PCA(n_components=2, random_state=ctx.seed).fit_transform(X) if X.shape[1] >= 2 else np.c_[X, np.zeros(n)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(P[:, 0], P[:, 1], c=km_best, cmap="tab20", s=10)
    axes[0].set_title(f"K-Means clusters (k={best_k})")
    axes[1].scatter(P[:, 0], P[:, 1], c=np.where(y == 1, "#d62728", "#1f77b4"), s=10)
    axes[1].set_title("Gold labels (red = scam)")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(ctx.paths.figures / "clusters_pca.png", dpi=130)
    plt.close(fig)

    table = pd.DataFrame([{k: v for k, v in s.items() if k != "top_entities"} for s in results["scripts"]])
    (ctx.paths.results / "scam_scripts.md").write_text("# Discovered call scripts (K-Means)\n\n" + md_table(table),
                                                       encoding="utf-8")
    kb = results["kmeans_best"]
    log.info("E5: K-Means k=%d purity %.3f NMI %.3f | k=2 purity %.3f | DBSCAN %s", best_k, kb["purity"], kb["nmi"],
             results["kmeans_k2"]["purity"],
             f"purity {results['dbscan'].get('purity', float('nan')):.3f}" if "purity" in results["dbscan"] else "n/a")
    return results
