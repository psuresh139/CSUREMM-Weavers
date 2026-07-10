"""
ML Runner
=========
Orchestrates node2vec → feature assembly → classification → clustering
for all 26 completed contexts, one at a time to stay within laptop RAM.

Outputs (saved to workspace/ml/):
    classification_results.csv  — per-context ROC-AUC / accuracy / F1 per model
    clustering_results.csv      — per-context KMeans + HDBSCAN summaries
    feature_importance.csv      — RF feature importances pooled across contexts

Usage:
    cd birds_new/
    python ml/run_ml.py                  # full run
    python ml/run_ml.py --skip-n2v       # skip node2vec if embeddings exist
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from config import YEARS, PLOTS, PERIODS, OUTPUT_ROOT
from ml.features import build_feature_matrix, load_sex_labels
from ml.classification import train_and_evaluate
from ml.clustering import kmeans_sweep, hdbscan_cluster, evaluate_clustering


ML_OUT = Path(__file__).parent.parent / "workspace" / "ml"


def run_node2vec_context(year, plot, period):
    """Run node2vec for one context. Returns True on success."""
    from embeddings.node2vec_pipeline import run_context
    label = f"{year}_{plot}_{period}"
    emb_path = OUTPUT_ROOT / label / f"embeddings_{label}.csv"
    if emb_path.exists():
        print(f"  [n2v] {label} — embeddings already exist, skipping")
        return True
    try:
        run_context(year, plot, period)
        return True
    except FileNotFoundError:
        print(f"  [n2v] {label} — no edges file, skipping")
        return False
    except Exception as e:
        print(f"  [n2v] {label} — error: {e}")
        return False


def run_classification_context(year, plot, period, sex_labels):
    """Run sex classification for one context. Returns summary row dict or None."""
    label = f"{year}_{plot}_{period}"
    try:
        X, names, meta = build_feature_matrix(year, plot, period)
    except FileNotFoundError as e:
        print(f"  [cls] {label} — missing file: {e}")
        return None, None

    if "sex" not in meta.columns or meta["sex"].notna().sum() < 10:
        print(f"  [cls] {label} — too few sex labels ({meta.get('sex', pd.Series()).notna().sum()})")
        return None, None

    print(f"  [cls] {label} — {len(X)} birds, {meta['sex'].notna().sum()} labelled")
    try:
        results = train_and_evaluate(X, meta["sex"], feature_names=names)
    except ValueError as e:
        print(f"  [cls] {label} — skipped: {e}")
        return None, None

    rows = []
    for _, row in results["summary"].iterrows():
        rows.append({
            "year": year, "plot": plot, "period": period,
            "model":    row["model"],
            "roc_auc":  row["roc_auc"],
            "accuracy": row["accuracy"],
            "macro_f1": row["macro_f1"],
            "n_birds":  len(X),
            "n_labelled": int(meta["sex"].notna().sum()),
        })

    fi = results.get("feature_importance")
    if fi is not None:
        fi = fi.copy()
        fi["year"] = year
        fi["plot"] = plot
        fi["period"] = period

    return rows, fi


def run_clustering_context(year, plot, period):
    """Run KMeans + HDBSCAN for one context. Returns summary row dict or None."""
    label = f"{year}_{plot}_{period}"
    try:
        X, names, meta = build_feature_matrix(year, plot, period)
    except FileNotFoundError:
        return None

    if len(X) < 5:
        return None

    row = {"year": year, "plot": plot, "period": period, "n_birds": len(X)}

    # KMeans
    try:
        km_labels, km_summary = kmeans_sweep(X)
        row.update({
            "kmeans_best_k":   km_summary["best_k"],
            "kmeans_sil":      km_summary["silhouette"],
            "kmeans_dbi":      km_summary["davies_bouldin"],
        })
        # Compare to sex if available
        if "sex" in meta.columns and meta["sex"].notna().sum() > 2:
            ev = evaluate_clustering(km_labels, meta["sex"])
            row["kmeans_sex_ari"] = ev["ari"]
            row["kmeans_sex_nmi"] = ev["nmi"]
    except Exception as e:
        print(f"  [clust] {label} kmeans error: {e}")

    # HDBSCAN
    try:
        hdb_labels, hdb_summary = hdbscan_cluster(X)
        row.update({
            "hdbscan_n_clusters": hdb_summary["n_clusters"],
            "hdbscan_n_noise":    hdb_summary["n_noise"],
            "hdbscan_sil":        hdb_summary["silhouette"],
        })
        if "sex" in meta.columns and meta["sex"].notna().sum() > 2 and hdb_summary["n_clusters"] > 1:
            mask = hdb_labels != -1
            if mask.sum() > 2:
                ev = evaluate_clustering(hdb_labels[mask], meta["sex"][mask])
                row["hdbscan_sex_ari"] = ev["ari"]
                row["hdbscan_sex_nmi"] = ev["nmi"]
    except Exception as e:
        print(f"  [clust] {label} hdbscan error: {e}")

    return row


def main(skip_n2v=False):
    ML_OUT.mkdir(parents=True, exist_ok=True)

    sex_labels = None
    try:
        sex_labels = load_sex_labels()
        print(f"Sex labels loaded: {len(sex_labels)} birds\n")
    except Exception as e:
        print(f"[warn] could not load sex labels: {e}\n")

    cls_rows   = []
    fi_frames  = []
    clust_rows = []

    for year in YEARS:
        for plot in PLOTS:
            for period in PERIODS:
                label = f"{year}_{plot}_{period}"
                edge_path = OUTPUT_ROOT / label / f"edges_{label}.csv"
                if not edge_path.exists():
                    continue

                print(f"\n{'='*50}\n  {label}\n{'='*50}")

                # ── 1. node2vec ──────────────────────────────────────
                if not skip_n2v:
                    ok = run_node2vec_context(year, plot, period)
                    if not ok:
                        continue

                # ── 2. Classification ────────────────────────────────
                rows, fi = run_classification_context(year, plot, period, sex_labels)
                if rows:
                    cls_rows.extend(rows)
                if fi is not None:
                    fi_frames.append(fi)

                # ── 3. Clustering ────────────────────────────────────
                crow = run_clustering_context(year, plot, period)
                if crow:
                    clust_rows.append(crow)

    # ── Save ─────────────────────────────────────────────────────────────────
    if cls_rows:
        cls_df = pd.DataFrame(cls_rows)
        cls_df.to_csv(ML_OUT / "classification_results.csv", index=False)
        print(f"\nClassification results → {ML_OUT / 'classification_results.csv'}")
        print(cls_df.groupby("model")[["roc_auc", "accuracy", "macro_f1"]].mean().round(3))

    if fi_frames:
        fi_all = pd.concat(fi_frames, ignore_index=True)
        fi_agg = (
            fi_all.groupby("feature")["importance"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        fi_agg.to_csv(ML_OUT / "feature_importance.csv", index=False)
        print(f"\nTop features:\n{fi_agg.head(10).to_string(index=False)}")

    if clust_rows:
        clust_df = pd.DataFrame(clust_rows)
        clust_df.to_csv(ML_OUT / "clustering_results.csv", index=False)
        print(f"\nClustering results → {ML_OUT / 'clustering_results.csv'}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-n2v", action="store_true",
                        help="Skip node2vec if embeddings already exist")
    args = parser.parse_args()
    main(skip_n2v=args.skip_n2v)
