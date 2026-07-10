"""
Simple Ratio Index (SRI) Construction
======================================
Computes pairwise social association strengths from GMM event data.

Sampling interval = 1 day within (plot, colony):
    X_AB  += 1  birds A & B share ≥1 event that day
    YAB   += 1  both present but never co-occurring
    YA    += 1  only A present
    YB    += 1  only B present

    SRI(A,B) = X / (X + YA + YB + YAB)

Reference: Whitehead (2008) Analyzing Animal Societies, Ch. 3
"""

from collections import Counter
from typing import Optional

import pandas as pd

from config import (
    SRI_MIN_EVENTS_PER_BIRD, SRI_MIN_DAYS_PER_BIRD,
    SRI_MIN_SHARED_DAYS, SRI_MIN_VALUE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _daily_presence(logger_df: pd.DataFrame,
                    plot: Optional[str] = None,
                    colony: Optional[str] = None) -> dict:
    """
    Build a dict: (plot, colony, date) → set of bird_ids present that day.
    Uses the raw logger (all detections) as the sampling frame.
    """
    lg = logger_df.copy()
    if plot is not None:
        lg = lg[lg["plot"] == plot]
    if colony is not None:
        lg = lg[lg["colony"] == colony]

    presence = (
        lg.groupby(["plot", "colony", "date"])["bird_id"]
        .apply(lambda s: set(s.astype(str)))
        .to_dict()
    )
    return presence


def _drop_transients(membership: pd.DataFrame) -> pd.DataFrame:
    """Remove birds that appear in too few events or days to be residents."""
    event_counts = membership.groupby("bird_id")["event_id"].nunique()
    day_counts   = membership.groupby("bird_id")["date"].nunique()
    keep = (
        set(event_counts[event_counts >= SRI_MIN_EVENTS_PER_BIRD].index)
        & set(day_counts[day_counts   >= SRI_MIN_DAYS_PER_BIRD].index)
    )
    return membership[membership["bird_id"].isin(keep)].copy()


# ── Core SRI logic ────────────────────────────────────────────────────────────

def compute_sri(
    events: pd.DataFrame,
    membership: pd.DataFrame,
    logger_df: pd.DataFrame,
    plot: Optional[str] = None,
    colony: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute SRI edge list for one (plot [, colony]) context.

    Args:
        events:     GMM event table
        membership: bird_id × event_id table (from detect_events)
        logger_df:  raw logger DataFrame (defines daily presence frame)
        plot:       if given, restrict to this plot
        colony:     if given, restrict to this colony

    Returns:
        DataFrame with columns:
            bird_a, bird_b, X, YA, YB, YAB, SRI
    """
    # Optionally focus
    ev  = events.copy()
    mem = membership.copy()
    if plot is not None:
        ev  = ev[ev["plot"]  == plot]
        mem = mem[mem["plot"] == plot]
    if colony is not None:
        ev  = ev[ev["colony"]  == colony]
        mem = mem[mem["colony"] == colony]

    # Align membership to valid events
    key = ["plot", "colony", "date", "nest_id", "event_id"]
    mem = mem.merge(ev[key], on=key, how="inner")

    # Drop transient birds
    mem = _drop_transients(mem)

    # Daily presence from raw logger
    presence_by_day = _daily_presence(logger_df, plot=plot, colony=colony)

    X   = Counter()
    YA  = Counter()
    YB  = Counter()
    YAB = Counter()

    day_keys = set(
        tuple(k) for k in mem[["plot", "colony", "date"]]
        .drop_duplicates().itertuples(index=False, name=None)
    )
    day_keys |= set(presence_by_day.keys())

    for (plt, col, date) in sorted(day_keys):
        birds_day = presence_by_day.get((plt, col, date), set())
        if len(birds_day) < 2:
            continue

        gmem = mem[(mem["plot"] == plt) & (mem["colony"] == col) & (mem["date"] == date)]
        evs_by_bird = (
            gmem.groupby("bird_id")["event_id"].apply(set).to_dict()
            if not gmem.empty else {}
        )

        birds = sorted(birds_day)
        for i in range(len(birds)):
            for j in range(i + 1, len(birds)):
                A, B = birds[i], birds[j]
                A_evs = evs_by_bird.get(A, set())
                B_evs = evs_by_bird.get(B, set())
                shared = A_evs & B_evs
                pair = (A, B)
                if shared:
                    X[pair] += 1
                else:
                    if A in birds_day and B in birds_day:
                        YAB[pair] += 1
                    elif A in birds_day:
                        YA[pair] += 1
                    elif B in birds_day:
                        YB[pair] += 1

    rows = []
    for pair in set(X) | set(YA) | set(YB) | set(YAB):
        A, B = pair
        x, ya, yb, yab = X[pair], YA[pair], YB[pair], YAB[pair]
        denom = x + ya + yb + yab
        rows.append({
            "bird_a": A, "bird_b": B,
            "X": x, "YA": ya, "YB": yb, "YAB": yab,
            "SRI": x / denom if denom > 0 else 0.0,
        })

    if not rows:
        return pd.DataFrame(columns=["bird_a", "bird_b", "X", "YA", "YB", "YAB", "SRI"])

    df = pd.DataFrame(rows).sort_values(["SRI", "X"], ascending=[False, False])
    df = df[(df["SRI"] >= SRI_MIN_VALUE) &
            ((df["X"] + df["YAB"]) >= SRI_MIN_SHARED_DAYS)]
    return df.reset_index(drop=True)


def sri_to_edge_list(sri_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert SRI table to the standard edge-list format expected by
    graph_construction (columns: id1, id2, association).
    """
    return (
        sri_df[["bird_a", "bird_b", "SRI"]]
        .rename(columns={"bird_a": "id1", "bird_b": "id2", "SRI": "association"})
        .reset_index(drop=True)
    )
