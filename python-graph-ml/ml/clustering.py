"""
Unsupervised Clustering
========================
Clusters birds in embedding / graph-metric space using:
  - KMeans (elbow + silhouette selection)
  - HDBSCAN (density-based, handles noise/outliers)
  - GMM (soft assignment)

Each method returns a labels array and a summary dict.

Typical usage
-------------
    from ml.features import build_feature_matrix
    from ml.clustering import kmeans_sweep, hdbscan_cluster, evaluate_clustering

    X, names, meta = build_feature_matrix(2016, "SPRA", "daytime")
    labels, summary = kmeans_sweep(X, k_range=range(2, 9))
    print(summary)

    # Compare cluster labels to known sex
    evaluate_clustering(labels, meta["sex"])
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.decomposition import PCA

from config import RANDOM_STATE, KMEANS_K_RANGE, HDBSCAN_MIN_SAMPLES


# ── KMeans elbow sweep ────────────────────────────────────────────────────────

def kmeans_sweep(
    X: np.ndarray,
    k_range=KMEANS_K_RANGE,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, Dict]:
    """
    Fit KMeans for k in k_range, select best k by silhouette score.

    Returns:
        labels   — cluster labels for best k
        summary  — dict with k_scores, best_k, silhouette, inertias
    """
    results = {}
    for k in k_range:
        if k >= len(X):
            break
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        lbl = km.fit_predict(X)
        sil = silhouette_score(X, lbl) if len(set(lbl)) > 1 else -1.0
        dbi = davies_bouldin_score(X, lbl) if len(set(lbl)) > 1 else np.inf
        results[k] = {
            "silhouette":       round(sil, 4),
            "davies_bouldin":   round(dbi, 4),
            "inertia":          round(km.inertia_, 2),
            "labels":           lbl,
        }

    best_k = max(results, key=lambda k: results[k]["silhouette"])
    best   = results[best_k]

    summary = {
        "best_k":      best_k,
        "silhouette":  best["silhouette"],
        "davies_bouldin": best["davies_bouldin"],
        "inertia":     best["inertia"],
        "k_scores":    {k: v["silhouette"] for k, v in results.items()},
        "k_inertias":  {k: v["inertia"]    for k, v in results.items()},
    }
    return best["labels"], summary


# ── HDBSCAN ───────────────────────────────────────────────────────────────────

def hdbscan_cluster(
    X: np.ndarray,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
    min_cluster_size: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Density-based clustering. Returns -1 for noise points.

    Args:
        X:                feature matrix
        min_samples:      core point threshold
        min_cluster_size: defaults to max(2, len(X)//10)

    Returns:
        labels   — cluster labels (-1 = noise)
        summary  — n_clusters, n_noise, silhouette (if >1 cluster)
    """
    try:
        import hdbscan
    except ImportError:
        raise ImportError("pip install hdbscan")

    if min_cluster_size is None:
        min_cluster_size = max(2, len(X) // 10)

    clusterer = hdbscan.HDBSCAN(
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
    )
    labels = clusterer.fit_predict(X)

    n_clusters = len(set(labels) - {-1})
    n_noise    = int((labels == -1).sum())

    sil = -1.0
    if n_clusters > 1:
        mask = labels != -1
        if mask.sum() > n_clusters:
            sil = round(silhouette_score(X[mask], labels[mask]), 4)

    summary = {
        "n_clusters": n_clusters,
        "n_noise":    n_noise,
        "silhouette": sil,
    }
    return labels, summary


# ── GMM clustering ────────────────────────────────────────────────────────────

def gmm_cluster(
    X: np.ndarray,
    k_range=KMEANS_K_RANGE,
    random_state: int = RANDOM_STATE,
) -> Tuple[np.ndarray, Dict]:
    """
    Fit GMM (BIC selection) and return hard cluster assignments.
    """
    models, bics = [], []
    for k in k_range:
        if k >= len(X):
            break
        gm = GaussianMixture(n_components=k, random_state=random_state)
        gm.fit(X)
        models.append(gm)
        bics.append(gm.bic(X))

    best_model = models[int(np.argmin(bics))]
    labels     = best_model.predict(X)
    proba      = best_model.predict_proba(X)

    sil = silhouette_score(X, labels) if len(set(labels)) > 1 else -1.0

    summary = {
        "best_k":          best_model.n_components,
        "bic":             round(min(bics), 2),
        "silhouette":      round(sil, 4),
        "mean_confidence": round(float(proba.max(axis=1).mean()), 4),
    }
    return labels, summary


# ── External label comparison ─────────────────────────────────────────────────

def evaluate_clustering(
    labels: np.ndarray,
    true_labels: pd.Series,
) -> Dict:
    """
    Compare cluster assignments to a known label (sex, community, etc.).
    Drops NaN entries before scoring.

    Returns dict with ARI and NMI.
    """
    mask = true_labels.notna()
    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        return {"ari": np.nan, "nmi": np.nan, "n_labelled": int(mask.sum())}

    enc, _ = pd.factorize(true_labels[mask])
    return {
        "ari":        round(adjusted_rand_score(enc, labels[mask]), 4),
        "nmi":        round(normalized_mutual_info_score(enc, labels[mask]), 4),
        "n_labelled": int(mask.sum()),
    }


# ── Colony ARI: Louvain communities vs field colonies ─────────────────────────

def colony_ari(
    nodes_df: pd.DataFrame,
    colony_labels: pd.DataFrame,
) -> Dict:
    """
    Compare Louvain community assignments to field-observed colony labels.

    Args:
        nodes_df:       nodes CSV DataFrame (must have bird_id, community columns)
        colony_labels:  DataFrame from load_colony_labels() (bird_id, field_colony)

    Returns:
        dict with ari, nmi, n_matched, n_louvain_communities, n_field_colonies
    """
    merged = nodes_df[["bird_id", "community"]].merge(
        colony_labels[["bird_id", "field_colony"]], on="bird_id", how="inner"
    )
    if len(merged) < 5:
        return {
            "ari": np.nan, "nmi": np.nan,
            "n_matched": len(merged),
            "n_louvain_communities": np.nan,
            "n_field_colonies": np.nan,
        }

    louvain_enc, _ = pd.factorize(merged["community"])
    colony_enc, _  = pd.factorize(merged["field_colony"])

    return {
        "ari":                    round(adjusted_rand_score(colony_enc, louvain_enc), 4),
        "nmi":                    round(normalized_mutual_info_score(colony_enc, louvain_enc), 4),
        "n_matched":              len(merged),
        "n_louvain_communities":  int(merged["community"].nunique()),
        "n_field_colonies":       int(merged["field_colony"].nunique()),
    }


# ── UMAP / PCA helper for visualisation ──────────────────────────────────────

def reduce_for_plot(X: np.ndarray, method: str = "umap", n_components: int = 2) -> np.ndarray:
    """
    Reduce X to 2D (or n_components) for scatter-plot visualisation.

    method: 'umap' (requires pip install umap-learn) or 'pca'
    """
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=n_components, random_state=RANDOM_STATE)
            return reducer.fit_transform(X)
        except ImportError:
            print("[warning] umap-learn not installed, falling back to PCA")
            method = "pca"

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    return pca.fit_transform(X)
