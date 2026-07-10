"""
Relatedness → SRI Analysis
===========================
Tests whether genetic relatedness predicts social association strength (SRI)
across all 26 contexts using:
  - OLS regression: SRI ~ Wang relatedness coefficient
  - Mantel test: matrix correlation between relatedness and SRI distance matrices

For each context, only dyads where both birds are present in the network
AND have a known relatedness value are included.

Output:
    workspace/ml/relatedness_sri.csv  — per-context stats
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from config import YEARS, PLOTS, PERIODS, OUTPUT_ROOT
from ml.features import load_relatedness, load_nodes


# ── Mantel test ───────────────────────────────────────────────────────────────

def mantel_test(x_vec: np.ndarray, y_vec: np.ndarray, n_perm: int = 999, seed: int = 42) -> dict:
    """
    Mantel test: permutation-based correlation between two distance vectors.
    x_vec, y_vec are flattened upper triangles of symmetric distance matrices.
    """
    rng = np.random.default_rng(seed)
    r_obs, _ = stats.pearsonr(x_vec, y_vec)

    count = 0
    n = int(round((1 + np.sqrt(1 + 8 * len(x_vec))) / 2))  # recover matrix size

    # Rebuild matrix for permutation
    X_mat = np.zeros((n, n))
    idx = np.triu_indices(n, k=1)
    X_mat[idx] = x_vec
    X_mat = X_mat + X_mat.T

    for _ in range(n_perm):
        perm = rng.permutation(n)
        X_perm = X_mat[np.ix_(perm, perm)]
        x_perm = X_perm[idx]
        r_perm, _ = stats.pearsonr(x_perm, y_vec)
        if abs(r_perm) >= abs(r_obs):
            count += 1

    return {
        "r":      round(float(r_obs), 4),
        "p_mantel": round((count + 1) / (n_perm + 1), 4),
    }


# ── Per-context analysis ──────────────────────────────────────────────────────

def analyse_context(year, plot, period, relatedness_df):
    label = f"{year}_{plot}_{period}"
    edge_path = OUTPUT_ROOT / label / f"edges_{label}.csv"
    if not edge_path.exists():
        return None

    edges = pd.read_csv(edge_path)
    nodes = load_nodes(year, plot, period)
    bird_set = set(nodes["bird_id"].astype(str))

    # Filter relatedness to birds in this network
    rel = relatedness_df[
        relatedness_df["bird_a"].isin(bird_set) &
        relatedness_df["bird_b"].isin(bird_set)
    ].copy()

    if len(rel) < 20:
        print(f"  [skip] {label}: only {len(rel)} relatedness pairs in network")
        return None

    # Build SRI lookup (symmetric)
    sri_lookup = {}
    for _, row in edges.iterrows():
        a, b = str(row["id1"]), str(row["id2"])
        sri_lookup[(a, b)] = row["association"]
        sri_lookup[(b, a)] = row["association"]

    # Attach SRI to relatedness pairs (0.0 if pair not connected)
    rel["sri"] = rel.apply(
        lambda r: sri_lookup.get((r["bird_a"], r["bird_b"]), 0.0), axis=1
    )

    # ── OLS regression: SRI ~ wang ────────────────────────────────────────────
    slope, intercept, r_ols, p_ols, se = stats.linregress(rel["wang"], rel["sri"])

    # ── Subset: only dyads with SRI > 0 (detected associations) ──────────────
    associated = rel[rel["sri"] > 0]
    if len(associated) >= 10:
        slope_a, _, r_a, p_a, _ = stats.linregress(associated["wang"], associated["sri"])
    else:
        slope_a, r_a, p_a = np.nan, np.nan, np.nan

    # ── Mantel test ────────────────────────────────────────────────────────────
    # Build symmetric matrices over the birds in both the network and relatedness
    birds = sorted(bird_set & (set(rel["bird_a"]) | set(rel["bird_b"])))
    if len(birds) < 5:
        mantel = {"r": np.nan, "p_mantel": np.nan}
    else:
        b_idx = {b: i for i, b in enumerate(birds)}
        n = len(birds)
        sri_mat = np.zeros((n, n))
        rel_mat = np.zeros((n, n))

        for _, row in rel.iterrows():
            i, j = b_idx.get(row["bird_a"]), b_idx.get(row["bird_b"])
            if i is None or j is None:
                continue
            sri_mat[i, j] = sri_mat[j, i] = row["sri"]
            rel_mat[i, j] = rel_mat[j, i] = row["wang"]

        idx = np.triu_indices(n, k=1)
        mantel = mantel_test(rel_mat[idx], sri_mat[idx], n_perm=999)

    n_pairs    = len(rel)
    n_assoc    = int((rel["sri"] > 0).sum())
    mean_wang  = round(float(rel["wang"].mean()), 4)
    mean_sri   = round(float(rel["sri"].mean()), 4)

    print(f"  {label}: {n_pairs} pairs, {n_assoc} associated | "
          f"OLS r={r_ols:.3f} p={p_ols:.3f} | Mantel r={mantel['r']} p={mantel['p_mantel']}")

    return {
        "year": year, "plot": plot, "period": period,
        "n_pairs":         n_pairs,
        "n_associated":    n_assoc,
        "mean_wang":       mean_wang,
        "mean_sri":        mean_sri,
        "ols_slope":       round(float(slope), 6),
        "ols_r":           round(float(r_ols), 4),
        "ols_p":           round(float(p_ols), 4),
        "ols_r_assoc_only": round(float(r_a), 4) if not np.isnan(r_a) else np.nan,
        "ols_p_assoc_only": round(float(p_a), 4) if not np.isnan(p_a) else np.nan,
        "mantel_r":        mantel["r"],
        "mantel_p":        mantel["p_mantel"],
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    out_dir = Path(__file__).parent.parent / "workspace" / "ml"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading relatedness…")
    rel_df = load_relatedness()
    print(f"  {len(rel_df)} resolved pairs\n")

    rows = []
    for year in YEARS:
        for plot in PLOTS:
            for period in PERIODS:
                row = analyse_context(year, plot, period, rel_df)
                if row:
                    rows.append(row)

    if not rows:
        print("No results.")
        return

    df = pd.DataFrame(rows)
    out_path = out_dir / "relatedness_sri.csv"
    df.to_csv(out_path, index=False)

    print(f"\n=== SUMMARY ===")
    print(df[["year","plot","period","n_pairs","n_associated",
              "ols_r","ols_p","mantel_r","mantel_p"]].to_string(index=False))

    sig = df[df["mantel_p"] < 0.05]
    print(f"\n{len(sig)}/{len(df)} contexts show significant relatedness-SRI correlation (Mantel p<0.05)")
    if len(sig):
        print(sig[["year","plot","period","mantel_r","mantel_p"]].to_string(index=False))

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
