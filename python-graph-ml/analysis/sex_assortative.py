"""
Sex-Assortative Association Analysis
======================================
Tests whether weaver birds preferentially associate with same-sex individuals
(sex-assortative) or opposite-sex (sex-disassortative).

Approach (Farine 2015, J Animal Ecology):
  1. Load SRI edge list for a context.
  2. Annotate each dyad as M-M, F-F, or M-F using sex labels.
  3. Compare SRI distributions across dyad types with Kruskal-Wallis + Dunn's.
  4. Compute mean SRI per dyad type and the assortativity coefficient.

Usage
-----
    from analysis.sex_assortative import sex_assortative_analysis
    results = sex_assortative_analysis(2016, "SPRA", "daytime")
    print(results["summary"])
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import networkx as nx

from config import OUTPUT_ROOT
from ml.features import load_sex_labels


# ── Core analysis ─────────────────────────────────────────────────────────────

def annotate_dyads(
    edges: pd.DataFrame,
    sex_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a dyad_type column (MM / FF / MF) to an edge list.

    Args:
        edges:   DataFrame with id1, id2, association
        sex_df:  DataFrame with bird_id, sex (M/F)

    Returns:
        edges with added sex_a, sex_b, dyad_type columns.
        Dyads where either bird has unknown sex are dropped.
    """
    sex_map = sex_df.set_index("bird_id")["sex"].to_dict()

    out = edges.copy()
    out["sex_a"] = out["id1"].map(sex_map)
    out["sex_b"] = out["id2"].map(sex_map)

    # Drop unknown sex
    out = out.dropna(subset=["sex_a", "sex_b"]).copy()

    def _dtype(row):
        pair = tuple(sorted([row["sex_a"], row["sex_b"]]))
        return "FF" if pair == ("F", "F") else "MM" if pair == ("M", "M") else "MF"

    out["dyad_type"] = out.apply(_dtype, axis=1)
    return out


def kruskal_dunn(groups: Dict[str, np.ndarray]) -> Dict:
    """
    Kruskal-Wallis test across groups, followed by pairwise Dunn's test
    (Bonferroni correction).

    Returns dict with H statistic, p-value, and pairwise comparisons.
    """
    keys = list(groups.keys())
    arrays = [groups[k] for k in keys]

    H, p_kw = stats.kruskal(*arrays)
    result = {
        "H":       round(float(H), 4),
        "p_kruskal": round(float(p_kw), 6),
    }

    # Pairwise Mann-Whitney U with Bonferroni correction
    n_pairs = len(keys) * (len(keys) - 1) // 2
    pairwise = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            u, p = stats.mannwhitneyu(groups[a], groups[b], alternative="two-sided")
            p_bonf = min(p * n_pairs, 1.0)
            pairwise[f"{a}_vs_{b}"] = {
                "U":       round(float(u), 2),
                "p_raw":   round(float(p), 6),
                "p_bonf":  round(float(p_bonf), 6),
                "significant": p_bonf < 0.05,
            }
    result["pairwise"] = pairwise
    return result


def sex_assortativity(G: nx.Graph, sex_map: Dict[str, str]) -> float:
    """
    Compute Newman's assortativity coefficient for sex as a categorical attribute.
    Returns NaN if sex info is insufficient.
    """
    try:
        nx.set_node_attributes(G, sex_map, name="sex")
        return round(nx.attribute_assortativity_coefficient(G, "sex"), 4)
    except Exception:
        return float("nan")


# ── Public API ────────────────────────────────────────────────────────────────

def sex_assortative_analysis(
    year: int,
    plot: str,
    period: str,
    sri_threshold: float = 0.0,
) -> Dict:
    """
    Run full sex-assortative association analysis for one context.

    Args:
        year, plot, period: context identifiers
        sri_threshold:      only include edges above this SRI (default: all)

    Returns dict with:
        summary       — mean SRI per dyad type + sample sizes
        stats         — Kruskal-Wallis + Dunn's results
        assortativity — Newman's r for sex
        annotated_edges — full annotated edge DataFrame
    """
    label    = f"{year}_{plot}_{period}"
    edge_file = OUTPUT_ROOT / label / f"edges_{label}.csv"
    if not edge_file.exists():
        raise FileNotFoundError(f"Edge file not found: {edge_file}. Run pipeline first.")

    edges = pd.read_csv(edge_file)
    if sri_threshold > 0:
        edges = edges[edges["association"] >= sri_threshold]

    sex_df = load_sex_labels()
    annotated = annotate_dyads(edges, sex_df)

    if annotated.empty:
        raise ValueError("No dyads with known sex for both birds — check sex label alignment.")

    # Per-dyad-type SRI distributions
    groups = {
        dtype: grp["association"].values
        for dtype, grp in annotated.groupby("dyad_type")
        if len(grp) >= 3
    }

    # Summary table
    summary_rows = []
    for dtype, arr in groups.items():
        summary_rows.append({
            "dyad_type":  dtype,
            "n_dyads":    len(arr),
            "mean_SRI":   round(float(arr.mean()), 5),
            "median_SRI": round(float(np.median(arr)), 5),
            "sd_SRI":     round(float(arr.std()), 5),
        })
    summary = pd.DataFrame(summary_rows).sort_values("dyad_type")

    # Statistical test
    kw_result = kruskal_dunn(groups) if len(groups) >= 2 else {}

    # Network assortativity
    G = nx.Graph()
    for _, row in edges.iterrows():
        G.add_edge(str(row["id1"]), str(row["id2"]), weight=float(row["association"]))
    sex_map = sex_df.set_index("bird_id")["sex"].to_dict()
    assort = sex_assortativity(G, sex_map)

    return {
        "context":         label,
        "summary":         summary,
        "stats":           kw_result,
        "assortativity":   assort,
        "annotated_edges": annotated,
        "n_sex_known":     len(annotated),
        "n_sex_unknown":   len(edges) - len(annotated),
    }


# ── Batch: run across multiple contexts ──────────────────────────────────────

def batch_sex_assortative(
    years=None, plots=None, periods=None,
) -> pd.DataFrame:
    """
    Run sex_assortative_analysis for all available contexts.
    Returns a summary DataFrame (one row per context).
    """
    from config import YEARS, PLOTS, PERIODS
    years   = years   or YEARS
    plots   = plots   or PLOTS
    periods = periods or PERIODS

    rows = []
    for year in years:
        for plot in plots:
            for period in periods:
                label = f"{year}_{plot}_{period}"
                edge_file = OUTPUT_ROOT / label / f"edges_{label}.csv"
                if not edge_file.exists():
                    continue
                try:
                    r = sex_assortative_analysis(year, plot, period)
                    row = {"context": label, "year": year, "plot": plot, "period": period,
                           "assortativity": r["assortativity"],
                           "n_sex_known": r["n_sex_known"]}
                    for _, s in r["summary"].iterrows():
                        row[f"mean_SRI_{s['dyad_type']}"] = s["mean_SRI"]
                    row["p_kruskal"] = r["stats"].get("p_kruskal", np.nan)
                    rows.append(row)
                except Exception as e:
                    print(f"  [skip] {label}: {e}")

    return pd.DataFrame(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sex-assortative association analysis")
    parser.add_argument("--year",   type=int, required=True)
    parser.add_argument("--plot",   type=str, required=True)
    parser.add_argument("--period", type=str, required=True)
    args = parser.parse_args()

    result = sex_assortative_analysis(args.year, args.plot, args.period)
    print(f"\nContext: {result['context']}")
    print(f"Newman assortativity for sex: {result['assortativity']}")
    print(f"\nSRI by dyad type:")
    print(result["summary"].to_string(index=False))
    if result["stats"]:
        print(f"\nKruskal-Wallis H={result['stats']['H']}, p={result['stats']['p_kruskal']}")
        for pair, pw in result["stats"]["pairwise"].items():
            sig = "  *** SIGNIFICANT" if pw["significant"] else ""
            print(f"  {pair}: p_bonf={pw['p_bonf']}{sig}")
