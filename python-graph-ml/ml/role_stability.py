"""
Cross-year role stability — is social position a persistent individual trait?
=============================================================================
The one strong signal in this dataset is that individual network position is
consistent across contexts (day/night degree ρ up to 0.99). This module tests
whether that consistency holds *across years*, and whether it is predictable.

Question
--------
Given a bird's network position in year Y, can we predict its position in
year Y+1 (same plot)? If social role is a stable individual trait, yes — and
the naive "unchanged since last year" persistence baseline will already be
strong. The ML question is whether a multi-feature model beats that baseline.

Design
------
    Unit      : a (bird, Y → Y+1) transition — bird present in both years,
                same plot & period. 415 such pairs pooled (daytime).
    Features  : year-Y metrics, percentile-ranked WITHIN each context so they
                are comparable across graphs of different sizes:
                degree, strength, betweenness, eigenvector, clustering (+sex).
    Targets   : (regression)     next-year degree percentile [0,1]
                (classification) next-year network_position (Core/Interm/Periph)
    Model     : RandomForest, evaluated with GroupKFold grouped by bird_id
                (a bird never appears in both train and test).
    Baselines : persistence (predict next = this-year same metric / position)
                and a shuffled-target null (chance floor).

Outputs (workspace/ml/):
    role_stability_regression.csv       — per split R², Spearman, vs baselines
    role_stability_by_metric.csv        — raw year→year+1 stability per metric

Usage
-----
    cd birds_new/
    python ml/role_stability.py                 # daytime, pooled
    python ml/role_stability.py --period nighttime
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, accuracy_score, f1_score

from config import OUTPUT_ROOT, YEARS, PLOTS, RANDOM_STATE
from ml.features import load_sex_labels

ML_OUT = Path(__file__).parent.parent / "workspace" / "ml"

METRICS = ["degree", "strength", "betweenness", "eigenvector", "clustering"]


# ── Load + build transition table ────────────────────────────────────────────────

def _load_nodes(year: int, plot: str, period: str) -> Optional[pd.DataFrame]:
    lab = f"{year}_{plot}_{period}"
    p = OUTPUT_ROOT / lab / f"nodes_{lab}.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def _pct_rank(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Percentile-rank the given columns within this context ([0,1])."""
    out = df.copy()
    for c in cols:
        out[c + "_pct"] = df[c].rank(pct=True)
    return out


def build_transitions(period: str) -> pd.DataFrame:
    """
    One row per (bird, Y→Y+1) persistence within a plot.
    Columns: bird_id, plot, year, this-year *_pct features, sex,
             next-year degree_pct (regression target), next-year
             network_position (classification target).
    """
    try:
        sex_map = load_sex_labels().set_index("bird_id")["sex"].to_dict()
    except Exception:
        sex_map = {}

    rows = []
    for plot in PLOTS:
        for y in YEARS[:-1]:
            a = _load_nodes(y, plot, period)
            b = _load_nodes(y + 1, plot, period)
            if a is None or b is None:
                continue
            a = _pct_rank(a, METRICS)
            b = _pct_rank(b, METRICS)
            b_idx = b.set_index("bird_id")
            for _, r in a.iterrows():
                bid = r["bird_id"]
                if bid not in b_idx.index:
                    continue
                nxt = b_idx.loc[bid]
                if isinstance(nxt, pd.DataFrame):     # dup id guard
                    nxt = nxt.iloc[0]
                row = {"bird_id": bid, "plot": plot, "year": y,
                       "sex": sex_map.get(bid, "U")}
                for c in METRICS:
                    row[c + "_pct"] = r[c + "_pct"]
                    row["next_" + c + "_pct"] = nxt[c + "_pct"]
                row["next_network_position"] = nxt["network_position"]
                row["this_network_position"] = r["network_position"]
                rows.append(row)
    return pd.DataFrame(rows)


# ── Raw stability (correlation this-year vs next-year, per metric) ────────────────

def stability_by_metric(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in METRICS:
        x = df[c + "_pct"].values
        yv = df["next_" + c + "_pct"].values
        rho, p = spearmanr(x, yv)
        rows.append({"metric": c, "n": len(df),
                     "spearman_rho": round(float(rho), 4),
                     "p_value": round(float(p), 6)})
    return pd.DataFrame(rows)


# ── Regression: predict next-year degree percentile ──────────────────────────────

def _feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    feat_cols = [c + "_pct" for c in METRICS]
    X = df[feat_cols].fillna(0.5).values
    # sex one-hot (M/F/U)
    sex = pd.get_dummies(df["sex"].fillna("U"), prefix="sex")
    X = np.hstack([X, sex.values.astype(float)])
    names = feat_cols + list(sex.columns)
    return X, names


def regression_cv(df: pd.DataFrame, target="degree", n_splits=5,
                  seed=RANDOM_STATE) -> Dict:
    tgt = f"next_{target}_pct"
    X, names = _feature_matrix(df)
    y = df[tgt].values
    groups = df["bird_id"].values
    persistence = df[f"{target}_pct"].values   # baseline: next = this

    gkf = GroupKFold(n_splits=min(n_splits, df["bird_id"].nunique()))
    rf_pred = np.zeros(len(df))
    for tr, te in gkf.split(X, y, groups):
        rf = RandomForestRegressor(n_estimators=300, random_state=seed,
                                   min_samples_leaf=3, n_jobs=-1)
        rf.fit(X[tr], y[tr])
        rf_pred[te] = rf.predict(X[te])

    # Shuffled null
    rng = np.random.default_rng(seed)
    y_shuf = rng.permutation(y)

    def summarize(pred):
        return (round(r2_score(y, pred), 4),
                round(float(spearmanr(y, pred).correlation), 4),
                round(float(np.mean(np.abs(y - pred))), 4))

    rf_r2, rf_rho, rf_mae = summarize(rf_pred)
    ps_r2, ps_rho, ps_mae = summarize(persistence)
    nl_r2, nl_rho, nl_mae = summarize(y_shuf)

    # Feature importance on full fit
    rf_full = RandomForestRegressor(n_estimators=300, random_state=seed,
                                    min_samples_leaf=3, n_jobs=-1).fit(X, y)
    fi = sorted(zip(names, rf_full.feature_importances_),
                key=lambda t: -t[1])

    return {
        "target": target, "n": len(df),
        "rf_r2": rf_r2, "rf_spearman": rf_rho, "rf_mae": rf_mae,
        "persistence_r2": ps_r2, "persistence_spearman": ps_rho, "persistence_mae": ps_mae,
        "null_r2": nl_r2, "null_spearman": nl_rho,
        "top_features": fi[:5],
    }


# ── Classification: next-year position category ──────────────────────────────────

def classification_cv(df: pd.DataFrame, n_splits=5, seed=RANDOM_STATE) -> Dict:
    X, _ = _feature_matrix(df)
    y = df["next_network_position"].values
    groups = df["bird_id"].values
    persistence = df["this_network_position"].values  # baseline: next = this

    gkf = GroupKFold(n_splits=min(n_splits, df["bird_id"].nunique()))
    rf_pred = np.empty(len(df), dtype=object)
    for tr, te in gkf.split(X, y, groups):
        rf = RandomForestClassifier(n_estimators=300, random_state=seed,
                                    class_weight="balanced", n_jobs=-1)
        rf.fit(X[tr], y[tr])
        rf_pred[te] = rf.predict(X[te])

    return {
        "n": len(df),
        "rf_accuracy": round(accuracy_score(y, rf_pred), 4),
        "rf_macro_f1": round(f1_score(y, rf_pred, average="macro"), 4),
        "persistence_accuracy": round(accuracy_score(y, persistence), 4),
        "persistence_macro_f1": round(f1_score(y, persistence, average="macro"), 4),
    }


# ── Runner ───────────────────────────────────────────────────────────────────────

def run(period="daytime", save=True) -> Dict:
    df = build_transitions(period)
    print(f"\n=== Cross-year role stability ({period}) ===")
    print(f"Pooled transitions: {len(df)}  |  unique birds: {df['bird_id'].nunique()}  "
          f"|  plots: {sorted(df['plot'].unique())}")

    stab = stability_by_metric(df)
    print("\n-- Raw stability (this-year vs next-year percentile) --")
    for _, r in stab.iterrows():
        print(f"  {r['metric']:12s} ρ={r['spearman_rho']:+.3f}  p={r['p_value']:.1e}")

    print("\n-- Regression: predict NEXT-year degree percentile --")
    reg = regression_cv(df, target="degree")
    print(f"  RF          : R²={reg['rf_r2']:+.3f}  ρ={reg['rf_spearman']:+.3f}  MAE={reg['rf_mae']:.3f}")
    print(f"  persistence : R²={reg['persistence_r2']:+.3f}  ρ={reg['persistence_spearman']:+.3f}  MAE={reg['persistence_mae']:.3f}")
    print(f"  shuffled null: R²={reg['null_r2']:+.3f}  ρ={reg['null_spearman']:+.3f}")
    print(f"  top features : {', '.join(f'{n}={v:.2f}' for n,v in reg['top_features'])}")

    print("\n-- Classification: predict NEXT-year network position --")
    clf = classification_cv(df)
    print(f"  RF          : acc={clf['rf_accuracy']:.3f}  macro-F1={clf['rf_macro_f1']:.3f}")
    print(f"  persistence : acc={clf['persistence_accuracy']:.3f}  macro-F1={clf['persistence_macro_f1']:.3f}")

    if save:
        ML_OUT.mkdir(parents=True, exist_ok=True)
        stab.to_csv(ML_OUT / f"role_stability_by_metric_{period}.csv", index=False)
        reg_row = {k: v for k, v in reg.items() if k != "top_features"}
        reg_row["period"] = period
        reg_row.update({f"top_feat_{i+1}": f"{n}:{v:.3f}"
                        for i, (n, v) in enumerate(reg["top_features"])})
        pd.DataFrame([reg_row]).to_csv(
            ML_OUT / f"role_stability_regression_{period}.csv", index=False)
        clf_row = dict(clf); clf_row["period"] = period
        pd.DataFrame([clf_row]).to_csv(
            ML_OUT / f"role_stability_classification_{period}.csv", index=False)
        print(f"\nSaved → {ML_OUT}/role_stability_*_{period}.csv")

    return {"stability": stab, "regression": reg, "classification": clf, "df": df}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cross-year role stability")
    ap.add_argument("--period", type=str, default="daytime",
                    choices=["daytime", "nighttime"])
    args = ap.parse_args()
    run(period=args.period)
