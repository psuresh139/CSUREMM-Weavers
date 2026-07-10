"""
Day vs. Night Network Comparison
==================================
For each (year, plot) pair, loads the daytime and nighttime graphs and
computes paired structural statistics to answer:

  - Are social networks denser / more modular during the day or night?
  - Do the same birds act as connectors in both periods?
  - How much do individual centrality ranks shift between periods?
  - Which birds are only active in one period?

Outputs
-------
    analysis/outputs/day_night_summary.csv   — one row per (year, plot) pair
    analysis/outputs/day_night_nodes.csv     — per-bird metrics for both periods
    analysis/outputs/connector_overlap.csv   — connector stability across periods

Usage
-----
    python analysis/day_night_comparison.py
    # or import and call directly:
    from analysis.day_night_comparison import run_comparison
    summary, node_df = run_comparison()
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import YEARS, PLOTS, OUTPUT_ROOT

OUTPUT_DIR = Path(__file__).parent / "outputs"


# ── Loaders ────────────────────────────────────────────────────────────────────

def _load(year: int, plot: str, period: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load nodes + stats for one context. Returns (nodes, stats_dict)."""
    label    = f"{year}_{plot}_{period}"
    ctx_dir  = OUTPUT_ROOT / label
    node_path = ctx_dir / f"nodes_{label}.csv"
    stat_path = ctx_dir / f"stats_{label}.csv"

    if not node_path.exists():
        return None, None

    nodes = pd.read_csv(node_path)
    nodes["period"] = period

    stats = {}
    if stat_path.exists():
        row = pd.read_csv(stat_path).iloc[0].to_dict()
        stats = row

    return nodes, stats


# ── Per-pair comparison ────────────────────────────────────────────────────────

def compare_pair(year: int, plot: str) -> dict:
    """
    Compare day vs. night graphs for one (year, plot).
    Returns a dict of comparison metrics.
    """
    nodes_d, stats_d = _load(year, plot, "daytime")
    nodes_n, stats_n = _load(year, plot, "nighttime")

    row = {"year": year, "plot": plot}

    # ── Network-level stats ────────────────────────────────────────────────
    for key in ("n_nodes", "n_edges", "edge_density", "avg_degree",
                "avg_clustering", "n_components"):
        row[f"day_{key}"]   = stats_d.get(key, np.nan) if stats_d else np.nan
        row[f"night_{key}"] = stats_n.get(key, np.nan) if stats_n else np.nan

    if stats_d and stats_n:
        row["density_ratio"] = (
            row["day_edge_density"] / row["night_edge_density"]
            if row["night_edge_density"] > 0 else np.nan
        )

    if nodes_d is None or nodes_n is None:
        return row

    # ── Node overlap ──────────────────────────────────────────────────────
    birds_d = set(nodes_d["bird_id"])
    birds_n = set(nodes_n["bird_id"])
    shared  = birds_d & birds_n
    row["n_birds_day"]        = len(birds_d)
    row["n_birds_night"]      = len(birds_n)
    row["n_birds_shared"]     = len(shared)
    row["n_birds_dayonly"]    = len(birds_d - birds_n)
    row["n_birds_nightonly"]  = len(birds_n - birds_d)
    row["jaccard_birds"]      = round(len(shared) / len(birds_d | birds_n), 4)

    # ── Centrality rank correlation (Spearman) ────────────────────────────
    if len(shared) >= 5:
        d_idx = nodes_d.set_index("bird_id")
        n_idx = nodes_n.set_index("bird_id")
        for metric in ("degree", "betweenness", "eigenvector"):
            if metric not in d_idx.columns or metric not in n_idx.columns:
                continue
            common = sorted(shared)
            d_vals = d_idx.loc[common, metric].values
            n_vals = n_idx.loc[common, metric].values
            rho, pval = spearmanr(d_vals, n_vals)
            row[f"spearman_{metric}"]    = round(rho, 4)
            row[f"spearman_{metric}_p"]  = round(pval, 4)

    # ── Strength distribution comparison (Mann-Whitney) ───────────────────
    if "strength" in nodes_d.columns and "strength" in nodes_n.columns:
        stat, pval = mannwhitneyu(
            nodes_d["strength"].dropna(),
            nodes_n["strength"].dropna(),
            alternative="two-sided",
        )
        row["mw_strength_stat"] = round(stat, 2)
        row["mw_strength_p"]    = round(pval, 4)

    # ── Connector overlap ────────────────────────────────────────────────
    if "is_connector" in nodes_d.columns and "is_connector" in nodes_n.columns:
        conn_d = set(nodes_d.loc[nodes_d["is_connector"] == True, "bird_id"])
        conn_n = set(nodes_n.loc[nodes_n["is_connector"] == True, "bird_id"])
        if conn_d | conn_n:
            row["connector_jaccard"] = round(
                len(conn_d & conn_n) / len(conn_d | conn_n), 4
            )
        row["n_connectors_day"]   = len(conn_d)
        row["n_connectors_night"] = len(conn_n)

    return row


# ── Node-level paired frame ───────────────────────────────────────────────────

def build_node_comparison(year: int, plot: str) -> pd.DataFrame:
    """
    Build a per-bird DataFrame with day and night metrics side-by-side,
    limited to birds present in both periods.
    """
    nodes_d, _ = _load(year, plot, "daytime")
    nodes_n, _ = _load(year, plot, "nighttime")

    if nodes_d is None or nodes_n is None:
        return pd.DataFrame()

    metric_cols = ["degree", "strength", "betweenness", "eigenvector",
                   "clustering", "network_position", "is_connector", "community"]
    avail_d = [c for c in metric_cols if c in nodes_d.columns]
    avail_n = [c for c in metric_cols if c in nodes_n.columns]

    d = nodes_d[["bird_id"] + avail_d].rename(
        columns={c: f"day_{c}" for c in avail_d}
    )
    n = nodes_n[["bird_id"] + avail_n].rename(
        columns={c: f"night_{c}" for c in avail_n}
    )
    merged = d.merge(n, on="bird_id", how="inner")
    merged["year"] = year
    merged["plot"] = plot
    return merged


# ── Main runner ───────────────────────────────────────────────────────────────

def run_comparison(
    years=None, plots=None, save: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run day-vs-night comparison across all (year, plot) pairs.

    Returns:
        summary_df  — one row per (year, plot)
        node_df     — per-bird metrics for all shared birds
    """
    years = years or YEARS
    plots = plots or PLOTS

    summary_rows = []
    node_frames  = []

    for year in years:
        for plot in plots:
            row = compare_pair(year, plot)
            summary_rows.append(row)

            node_frame = build_node_comparison(year, plot)
            if not node_frame.empty:
                node_frames.append(node_frame)

    summary = pd.DataFrame(summary_rows)
    nodes   = pd.concat(node_frames, ignore_index=True) if node_frames else pd.DataFrame()

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        summary.to_csv(OUTPUT_DIR / "day_night_summary.csv",  index=False)
        nodes.to_csv(  OUTPUT_DIR / "day_night_nodes.csv",    index=False)
        print(f"Saved to {OUTPUT_DIR}")

    return summary, nodes


if __name__ == "__main__":
    summary, nodes = run_comparison()
    print("\nDay-vs-Night Summary:")
    print(summary[["year", "plot", "day_n_nodes", "night_n_nodes",
                    "day_edge_density", "night_edge_density",
                    "jaccard_birds"]].to_string(index=False))
