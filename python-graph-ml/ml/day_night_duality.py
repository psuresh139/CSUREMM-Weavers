"""
Day-Night Role Duality
=======================
For each (year, plot), correlates each bird's centrality rank in the
daytime foraging network with its rank in the nighttime roosting network.

Metrics compared: degree, strength, betweenness, eigenvector, clustering

Key questions:
  - Do birds that are central during foraging (day) also dominate roosting (night)?
  - Or are there specialist foragers vs specialist roosters?
  - Does this pattern vary by sex, year, or plot?

Output:
    workspace/ml/day_night_duality.csv     — per-(year,plot) correlations
    workspace/ml/day_night_bird_roles.csv  — per-bird role scores across all contexts
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from config import YEARS, PLOTS, OUTPUT_ROOT
from ml.features import load_nodes, load_sex_labels


METRICS = ["degree", "strength", "betweenness", "eigenvector", "clustering"]


def analyse_year_plot(year, plot, sex_labels):
    label_d = f"{year}_{plot}_daytime"
    label_n = f"{year}_{plot}_nighttime"

    try:
        day = load_nodes(year, plot, "daytime")
        ngt = load_nodes(year, plot, "nighttime")
    except FileNotFoundError:
        return None, None

    # Inner join: birds present in both networks
    merged = day.merge(ngt, on="bird_id", suffixes=("_day", "_night"), how="inner")

    if len(merged) < 10:
        print(f"  [skip] {year}_{plot}: only {len(merged)} birds in both networks")
        return None, None

    # Attach sex
    merged = merged.merge(sex_labels, on="bird_id", how="left")

    print(f"  {year}_{plot}: {len(merged)} birds in both networks "
          f"(day={len(day)}, night={len(ngt)})")

    # ── Per-metric Spearman correlations ──────────────────────────────────────
    corr_row = {"year": year, "plot": plot, "n_shared": len(merged)}

    for m in METRICS:
        col_d = f"{m}_day"
        col_n = f"{m}_night"
        if col_d not in merged.columns or col_n not in merged.columns:
            continue
        r, p = stats.spearmanr(merged[col_d], merged[col_n])
        corr_row[f"{m}_rho"]   = round(float(r), 4)
        corr_row[f"{m}_p"]     = round(float(p), 4)

    # ── Sex-stratified correlations (degree only) ─────────────────────────────
    for sex in ["M", "F"]:
        sub = merged[merged["sex"] == sex]
        if len(sub) >= 8 and "degree_day" in merged.columns:
            r, p = stats.spearmanr(sub["degree_day"], sub["degree_night"])
            corr_row[f"degree_rho_{sex}"] = round(float(r), 4)
            corr_row[f"degree_p_{sex}"]   = round(float(p), 4)

    # ── Per-bird role score: mean rank percentile across both networks ─────────
    bird_rows = []
    for _, row in merged.iterrows():
        if "degree_day" not in merged.columns:
            break
        day_pct  = (merged["degree_day"]  <= row["degree_day"]).mean()
        ngt_pct  = (merged["degree_night"] <= row["degree_night"]).mean()
        role_gap = float(day_pct - ngt_pct)   # +ve = higher rank in day, -ve = higher rank in night
        bird_rows.append({
            "year": year, "plot": plot,
            "bird_id":      row["bird_id"],
            "sex":          row.get("sex", np.nan),
            "degree_day":   row.get("degree_day"),
            "degree_night": row.get("degree_night"),
            "day_pct":      round(day_pct, 3),
            "night_pct":    round(ngt_pct, 3),
            "role_gap":     round(role_gap, 3),  # +ve = day specialist, -ve = night specialist
        })

    bird_df = pd.DataFrame(bird_rows) if bird_rows else None
    return corr_row, bird_df


def main():
    out_dir = Path(__file__).parent.parent / "workspace" / "ml"
    out_dir.mkdir(parents=True, exist_ok=True)

    sex_labels = load_sex_labels()

    corr_rows  = []
    bird_frames = []

    for year in YEARS:
        for plot in PLOTS:
            corr_row, bird_df = analyse_year_plot(year, plot, sex_labels)
            if corr_row:
                corr_rows.append(corr_row)
            if bird_df is not None:
                bird_frames.append(bird_df)

    if not corr_rows:
        print("No results.")
        return

    corr_df = pd.DataFrame(corr_rows)
    corr_path = out_dir / "day_night_duality.csv"
    corr_df.to_csv(corr_path, index=False)

    if bird_frames:
        bird_df_all = pd.concat(bird_frames, ignore_index=True)
        bird_path = out_dir / "day_night_bird_roles.csv"
        bird_df_all.to_csv(bird_path, index=False)

    # ── Print results ──────────────────────────────────────────────────────────
    rho_cols = [c for c in corr_df.columns if c.endswith("_rho")]
    print(f"\n=== DAY-NIGHT CENTRALITY CORRELATIONS (Spearman ρ) ===")
    print(corr_df[["year", "plot", "n_shared"] + rho_cols].to_string(index=False))

    print(f"\n=== MEAN ρ ACROSS ALL CONTEXTS ===")
    print(corr_df[rho_cols].mean().round(3).to_string())

    print(f"\n=== SEX-STRATIFIED DEGREE CORRELATION ===")
    sex_cols = [c for c in corr_df.columns if "degree_rho_" in c or "degree_p_" in c]
    if sex_cols:
        print(corr_df[["year", "plot"] + sex_cols].to_string(index=False))

    # Specialists: birds with large role_gap
    if bird_frames:
        bd = pd.concat(bird_frames)
        print(f"\n=== TOP 10 DAY SPECIALISTS (high role_gap) ===")
        print(bd.nlargest(10, "role_gap")[["bird_id","year","plot","sex","day_pct","night_pct","role_gap"]].to_string(index=False))
        print(f"\n=== TOP 10 NIGHT SPECIALISTS (low role_gap) ===")
        print(bd.nsmallest(10, "role_gap")[["bird_id","year","plot","sex","day_pct","night_pct","role_gap"]].to_string(index=False))

    print(f"\nSaved → {corr_path}")


if __name__ == "__main__":
    main()
