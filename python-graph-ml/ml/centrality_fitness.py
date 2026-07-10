"""
Does social centrality predict breeding success?
=================================================
We established that centrality (degree, strength) is a stable individual
trait. The natural next question — the field's question 3 — is whether that
trait carries a fitness consequence: do more central birds fledge more young?

Data
----
    Unit    : (bird, year, plot) with daytime network centrality + that
              bird's fledge / hatch outcome (ml/breeding_labels.py).
    Outcome : fledged (0/1) and hatched (0/1).

Confound guard
--------------
A bird detected more often can look more central AND be more likely to have
its nest identified as fledged. We therefore:
    - report the detection-effort (n_hits) ~ centrality and n_hits ~ fledge
      correlations,
    - include a detection-effort-only model as a baseline, and
    - test whether centrality adds ROC-AUC *over* effort alone.
If centrality only predicts fledging as well as raw detection count does, the
"centrality → fitness" story is a detection artefact, and we say so.

Tests
-----
    - Mann-Whitney U: centrality of breeders vs non-breeders (per metric).
    - Logistic / RF: predict outcome from centrality percentiles, grouped CV
      by bird, ROC-AUC vs effort-only and shuffled-null baselines.

Outputs (workspace/ml/):
    centrality_fitness_tests.csv     — per metric/outcome group comparison
    centrality_fitness_models.csv    — model ROC-AUCs vs baselines

Usage
-----
    cd birds_new/
    python ml/centrality_fitness.py
"""

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pointbiserialr

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

from config import OUTPUT_ROOT, YEARS, PLOTS, RANDOM_STATE
from ml.breeding_labels import load_breeding_labels

ML_OUT = Path(__file__).parent.parent / "workspace" / "ml"
METRICS = ["degree", "strength", "betweenness", "eigenvector", "clustering"]


def build_table(period="daytime") -> pd.DataFrame:
    """Pool (bird, year, plot) centrality + breeding outcome + detection effort."""
    # detection effort comes from the label table's n_hits
    from ml.breeding_labels import LABEL_FILE, build_breeding_labels
    if not LABEL_FILE.exists():
        build_breeding_labels(save=True)
    labels = pd.read_csv(LABEL_FILE)

    rows = []
    for year in YEARS:
        for plot in PLOTS:
            lab = f"{year}_{plot}_{period}"
            p = OUTPUT_ROOT / lab / f"nodes_{lab}.csv"
            if not p.exists():
                continue
            nodes = pd.read_csv(p)
            for c in METRICS:
                nodes[c + "_pct"] = nodes[c].rank(pct=True)
            ly = labels[(labels["year"] == year) & (labels["plot"] == plot)]
            m = nodes.merge(ly[["bird_id", "fledged", "hatched", "n_hits"]],
                            on="bird_id", how="inner")
            m["year"] = year
            m["plot"] = plot
            rows.append(m)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def group_tests(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    out = []
    y = df[outcome]
    grp1 = df[y == 1]
    grp0 = df[y == 0]
    for c in METRICS:
        a = grp1[c + "_pct"].dropna()
        b = grp0[c + "_pct"].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        out.append({
            "outcome": outcome, "metric": c,
            "n_pos": len(a), "n_neg": len(b),
            "median_breeder": round(float(a.median()), 3),
            "median_nonbreeder": round(float(b.median()), 3),
            "mannwhitney_p": round(float(p), 5),
        })
    return pd.DataFrame(out)


def _auc_cv(X, y, groups, model, seed=RANDOM_STATE) -> float:
    gkf = GroupKFold(n_splits=5)
    try:
        proba = cross_val_predict(model, X, y, cv=gkf, groups=groups,
                                  method="predict_proba")[:, 1]
    except Exception:
        return float("nan")
    if len(np.unique(y)) < 2:
        return float("nan")
    return round(float(roc_auc_score(y, proba)), 4)


def model_eval(df: pd.DataFrame, outcome: str, seed=RANDOM_STATE) -> dict:
    d = df.dropna(subset=[outcome]).copy()
    y = d[outcome].astype(int).values
    groups = d["bird_id"].values
    feat = [c + "_pct" for c in METRICS]
    Xc = d[feat].fillna(0.5).values
    Xe = d[["n_hits"]].fillna(0).values
    Xce = np.hstack([Xc, np.log1p(d[["n_hits"]].fillna(0).values)])

    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=seed, n_jobs=-1)

    rng = np.random.default_rng(seed)
    y_shuf = rng.permutation(y)

    return {
        "outcome": outcome, "n": len(d), "n_pos": int(y.sum()),
        "base_rate": round(float(y.mean()), 3),
        "auc_effort_only_lr": _auc_cv(Xe, y, groups, lr),
        "auc_centrality_lr": _auc_cv(Xc, y, groups, lr),
        "auc_centrality_rf": _auc_cv(Xc, y, groups, rf),
        "auc_centrality+effort_rf": _auc_cv(Xce, y, groups, rf),
        "auc_shuffled_null": _auc_cv(Xc, y_shuf, groups, lr),
    }


def run(period="daytime", years=None, save=True):
    df = build_table(period)
    if years:
        df = df[df["year"].isin(years)]
    tag = "" if not years else f" (years={years})"
    print(f"\n=== Centrality → breeding success ({period}){tag} ===")
    print(f"Pooled bird-years: {len(df)}  |  unique birds: {df['bird_id'].nunique()}")
    print(f"fledge base rate: {df['fledged'].mean():.3f}  hatch base rate: {df['hatched'].mean():.3f}")

    # Effort confound snapshot
    r_eff_deg = pointbiserialr(df["n_hits"], df["degree_pct"])[0]
    r_eff_fl = pointbiserialr(df["fledged"], np.log1p(df["n_hits"]))[0]
    print(f"\nConfound check: corr(n_hits, degree_pct)={r_eff_deg:+.3f} | "
          f"corr(fledged, log n_hits)={r_eff_fl:+.3f}")

    tests = pd.concat([group_tests(df, "fledged"), group_tests(df, "hatched")],
                      ignore_index=True)
    print("\n-- Breeder vs non-breeder centrality (Mann-Whitney) --")
    for _, r in tests.iterrows():
        sig = "*" if r["mannwhitney_p"] < 0.05 else " "
        print(f" {sig} {r['outcome']:8s} {r['metric']:12s} "
              f"breeder med={r['median_breeder']:.2f} vs {r['median_nonbreeder']:.2f}  "
              f"p={r['mannwhitney_p']:.4f}")

    models = pd.DataFrame([model_eval(df, "fledged"), model_eval(df, "hatched")])
    print("\n-- Predicting outcome (ROC-AUC, grouped CV) --")
    for _, r in models.iterrows():
        print(f"  {r['outcome']:8s} (n={r['n']}, base={r['base_rate']:.2f}): "
              f"effort-only={r['auc_effort_only_lr']:.3f} | "
              f"centrality-LR={r['auc_centrality_lr']:.3f} | "
              f"centrality-RF={r['auc_centrality_rf']:.3f} | "
              f"cent+effort={r['auc_centrality+effort_rf']:.3f} | "
              f"null={r['auc_shuffled_null']:.3f}")

    if save:
        ML_OUT.mkdir(parents=True, exist_ok=True)
        tests.to_csv(ML_OUT / "centrality_fitness_tests.csv", index=False)
        models.to_csv(ML_OUT / "centrality_fitness_models.csv", index=False)
        print(f"\nSaved → {ML_OUT}/centrality_fitness_*.csv")
    return df, tests, models


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="daytime", choices=["daytime", "nighttime"])
    ap.add_argument("--years", type=int, nargs="*", default=None,
                    help="restrict to years, e.g. --years 2015 2016")
    args = ap.parse_args()
    run(period=args.period, years=args.years)
