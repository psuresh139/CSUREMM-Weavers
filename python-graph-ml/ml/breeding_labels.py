"""
Breeding-success labels — bird → fledge outcome
================================================
Resolves the logger↔breeding join that was previously assumed impossible.

The key realisation: the colony/nest namespaces DO align once normalised.
    logger colony 'MSTO_01'  →  breeding colony 'MSTO01'   (strip underscore)
    logger nest   '118'      →  breeding nest    118        (numeric cast)
Every breeding colony appears in the logger; only the logger's 'F##' sensor
codes (non-breeding stations) fail to match. This yields a fledge label for
~74% of bird-years.

Method
------
    1. breeding: aggregate daily_breeding.xlsx fledge events to
       (year, plot, colony, nest) → fledged (max over the season).
    2. logger  : each bird's DOMINANT breeding-season (Apr-Aug) nest =
       the (colony, nest) where it was detected most that year.
    3. join     the dominant nest to its fledge outcome.

Proxy caveat: the dominant *detection* nest is a proxy for a bird's own
breeding nest — loggers are not on every nest, and a bird may be detected at
a colony-mate's nest. Treat the label as "breeding success at the bird's
primary nest site", not a certified individual reproductive record.

Output
------
    workspace/ml/bird_fledge_labels.csv
        bird_id, year, plot, colony, nest, fledged (0/1), hatched (0/1),
        n_hits (detections at that nest), n_nest_days (breeding records)

Usage
-----
    cd birds_new/
    python ml/breeding_labels.py            # build the label table
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import LOGGER_FILE, BREEDING_FILE, TEST_BIRD_IDS

ML_OUT = Path(__file__).parent.parent / "workspace" / "ml"
LABEL_FILE = ML_OUT / "bird_fledge_labels.csv"

BREEDING_MONTHS = (4, 8)   # Apr–Aug breeding season


def _norm_colony(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("_", "", regex=False).str.upper()


def build_breeding_labels(save: bool = True) -> pd.DataFrame:
    """Compute the bird-year → fledge outcome table. Reads the big logger file."""
    # ── 1. Breeding outcomes per (year, plot, colony, nest) ──────────────────
    brd = pd.read_excel(
        BREEDING_FILE, usecols=["year", "plot", "colony", "nest", "fledge", "hatch"]
    )
    brd["colony_norm"] = _norm_colony(brd["colony"])
    brd["nest_int"] = pd.to_numeric(brd["nest"], errors="coerce")
    brd = brd.dropna(subset=["nest_int"])

    nest_out = (
        brd.groupby(["year", "plot", "colony_norm", "nest_int"])
        .agg(fledged=("fledge", "max"),
             hatched=("hatch", "max"),
             n_nest_days=("fledge", "size"))
        .reset_index()
    )
    nest_out["fledged"] = (nest_out["fledged"] > 0).astype(int)
    nest_out["hatched"] = (nest_out["hatched"] > 0).astype(int)

    # ── 2. Each bird's dominant breeding-season nest per year ────────────────
    log = pd.read_excel(LOGGER_FILE, usecols=["Kring", "plot", "colony", "nest", "Date"])
    log = log[~log["Kring"].isin(TEST_BIRD_IDS)]
    dt = pd.to_datetime(log["Date"])
    log["year"] = dt.dt.year
    m = dt.dt.month
    log = log[(m >= BREEDING_MONTHS[0]) & (m <= BREEDING_MONTHS[1])]
    log["colony_norm"] = _norm_colony(log["colony"])
    log["nest_int"] = pd.to_numeric(log["nest"], errors="coerce")
    log = log.dropna(subset=["nest_int"])

    hits = (
        log.groupby(["Kring", "year", "plot", "colony_norm", "nest_int"])
        .size().reset_index(name="n_hits")
    )
    dom = (hits.sort_values("n_hits")
              .drop_duplicates(["Kring", "year"], keep="last"))

    # ── 3. Join dominant nest → outcome ──────────────────────────────────────
    out = dom.merge(nest_out, on=["year", "plot", "colony_norm", "nest_int"], how="left")
    out = out.rename(columns={"Kring": "bird_id", "colony_norm": "colony",
                              "nest_int": "nest"})
    out["nest"] = out["nest"].astype(int)
    out = out[["bird_id", "year", "plot", "colony", "nest",
               "fledged", "hatched", "n_hits", "n_nest_days"]]
    out = out.dropna(subset=["fledged"])           # keep only labelled bird-years
    out["fledged"] = out["fledged"].astype(int)
    out["hatched"] = out["hatched"].astype(int)

    if save:
        ML_OUT.mkdir(parents=True, exist_ok=True)
        out.to_csv(LABEL_FILE, index=False)
        print(f"Saved → {LABEL_FILE}  ({len(out)} labelled bird-years)")
        print(f"Overall fledge rate: {out['fledged'].mean():.3f}")
        print("Per-year (labelled n, fledge rate):")
        for y, g in out.groupby("year"):
            print(f"  {y}: n={len(g):4d}  fledge={g['fledged'].mean():.2f}  hatch={g['hatched'].mean():.2f}")
    return out


def load_breeding_labels(year: int, plot: str,
                         rebuild: bool = False) -> pd.DataFrame:
    """
    Per-bird fledge label for a (year, plot). Returns bird_id, fledged, hatched.
    Builds and caches the full table on first call.
    """
    if rebuild or not LABEL_FILE.exists():
        build_breeding_labels(save=True)
    df = pd.read_csv(LABEL_FILE)
    sub = df[(df["year"] == year) & (df["plot"] == plot)]
    return sub[["bird_id", "fledged", "hatched"]].reset_index(drop=True)


if __name__ == "__main__":
    build_breeding_labels(save=True)
