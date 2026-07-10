"""
Full Pipeline Orchestrator
===========================
Runs the complete pipeline for every (year, plot, period) combination:

  Logger data → GMM events → SRI edges → Graph metrics

Outputs for each context land in:
    workspace/graphs/{year}_{plot}_{period}/
        nodes_{year}_{plot}_{period}.csv   ← per-bird graph features
        edges_{year}_{plot}_{period}.csv   ← SRI-weighted edge list
        stats_{year}_{plot}_{period}.csv   ← network-level statistics
        community_info_{...}.csv

Usage
-----
    python run_pipeline.py                     # all 30 contexts
    python run_pipeline.py --year 2016 --plot SPRA --period daytime
    python run_pipeline.py --year 2016         # all plots & periods for 2016
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# Make sure project root is on the path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import YEARS, PLOTS, PERIODS, LOGGER_FILE, OUTPUT_ROOT
from pipeline.gmm_events import (
    load_logger, filter_by_year, filter_by_plot, filter_by_period, detect_events,
)
from pipeline.sri_construction import compute_sri, sri_to_edge_list
from pipeline.graph_construction import analyse_graph


# ── Logger cache (load once, reuse for all contexts) ──────────────────────────
_LOGGER_CACHE: pd.DataFrame | None = None


def get_logger() -> pd.DataFrame:
    global _LOGGER_CACHE
    if _LOGGER_CACHE is None:
        print(f"Loading logger: {LOGGER_FILE}")
        _LOGGER_CACHE = load_logger(LOGGER_FILE)
        print(f"  {len(_LOGGER_CACHE):,} rows loaded")
    return _LOGGER_CACHE


# ── Single context ────────────────────────────────────────────────────────────

def run_one(year: int, plot: str, period: str, verbose: bool = True) -> bool:
    """
    Run the full pipeline for one (year, plot, period) context.
    Returns True on success, False if insufficient data.
    """
    label    = f"{year}_{plot}_{period}"
    out_dir  = OUTPUT_ROOT / label
    prefix   = label

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

    logger = get_logger()

    # ── 1. Filter ──────────────────────────────────────────────────────────
    df = filter_by_year(logger, year)
    df = filter_by_plot(df, plot)
    df = filter_by_period(df, period)

    if verbose:
        print(f"  [{1}] Filtered: {len(df):,} rows")

    if len(df) < 10:
        print(f"  [skip] too few rows ({len(df)}) for {label}")
        return False

    # ── 2. GMM event detection ─────────────────────────────────────────────
    events, membership = detect_events(df)

    if verbose:
        print(f"  [2] Events: {len(events):,}  |  Membership rows: {len(membership):,}")

    if events.empty:
        print(f"  [skip] no events detected for {label}")
        return False

    # ── 3. SRI calculation ─────────────────────────────────────────────────
    # Pass df (year+period filtered) as the sampling frame so the SRI
    # denominator only counts presence during the relevant context.
    # Passing the full logger would bleed all 5 years into every graph.
    sri = compute_sri(events, membership, df, plot=plot)
    edges = sri_to_edge_list(sri)

    if verbose:
        print(f"  [3] SRI edges: {len(edges):,}")

    if edges.empty:
        print(f"  [skip] no SRI edges for {label}")
        return False

    # ── 4. Graph metrics ───────────────────────────────────────────────────
    results = analyse_graph(edges, output_dir=out_dir, prefix=prefix)

    if verbose:
        stats = results["stats"].iloc[0]
        print(f"  [4] Nodes: {int(stats['n_nodes'])}  "
              f"Edges: {int(stats['n_edges'])}  "
              f"Density: {stats['edge_density']:.4f}")
        comm  = results["community_info"].iloc[0]
        print(f"      Communities: {comm['n_communities']}  "
              f"Modularity: {comm['modularity']:.4f}")
        print(f"  → saved to {out_dir}")

    return True


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_all(years=None, plots=None, periods=None, verbose=True):
    years   = years   or YEARS
    plots   = plots   or PLOTS
    periods = periods or PERIODS

    total   = len(years) * len(plots) * len(periods)
    success = 0
    skipped = 0
    t0      = time.time()

    for year in years:
        for plot in plots:
            for period in periods:
                ok = run_one(year, plot, period, verbose=verbose)
                if ok:
                    success += 1
                else:
                    skipped += 1

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done — {success}/{total} succeeded, {skipped} skipped")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Output root: {OUTPUT_ROOT}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weaver bird pipeline")
    parser.add_argument("--year",   type=int,   default=None)
    parser.add_argument("--plot",   type=str,   default=None)
    parser.add_argument("--period", type=str,   default=None,
                        choices=["daytime", "nighttime"])
    parser.add_argument("--quiet",  action="store_true")
    args = parser.parse_args()

    years   = [args.year]   if args.year   else None
    plots   = [args.plot]   if args.plot   else None
    periods = [args.period] if args.period else None

    run_all(years=years, plots=plots, periods=periods, verbose=not args.quiet)
