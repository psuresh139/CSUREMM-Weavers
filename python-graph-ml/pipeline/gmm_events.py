"""
GMM Event Detection
===================
Detects temporal gathering events from logger detection data using Gaussian
Mixture Models. BIC selects the optimal number of components automatically.

For each (plot, colony, date, nest) group:
  1. Fit GMM to per-second detection timestamps
  2. Merge clusters within MIN_GAP_S seconds of each other
  3. Drop events with fewer than MIN_HITS detections or longer than MAX_EVENT_S

Returns:
    events DataFrame     — one row per detected event window
    membership DataFrame — bird_id × event_id associations
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.mixture import GaussianMixture
from typing import Tuple

from config import (
    GMM_KMAX, GMM_REG_COVAR, GMM_RANDOM_SEED,
    GMM_MIN_GAP_S, GMM_MIN_HITS, GMM_MAX_EVENT_S,
    DAYTIME_START, DAYTIME_END, NIGHTTIME_START, NIGHTTIME_END,
    TEST_BIRD_IDS,
)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_logger(path: Path) -> pd.DataFrame:
    """
    Load a logger file (XLSX or CSV) and normalise to the internal schema.

    Returns a DataFrame with columns:
        plot, colony, date, nest_id, bird_id, _sec (seconds since midnight)
    """
    ext = path.suffix.lower()
    df = pd.read_excel(path) if ext in (".xlsx", ".xls") else pd.read_csv(path)

    need = ["Date", "Time", "Kring", "plot", "colony", "nest"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Logger missing columns: {miss}")

    td = pd.to_timedelta(df["Time"].astype(str), errors="coerce")
    if td.isna().any():
        raise ValueError(f"Unparseable Time values: {df['Time'][td.isna()].head().tolist()}")

    sec = (
        td.dt.components.hours * 3600
        + td.dt.components.minutes * 60
        + td.dt.components.seconds
    ).astype(int)

    out = pd.DataFrame({
        "plot":    df["plot"].astype(str),
        "colony":  df["colony"].astype(str),
        "date":    df["Date"].astype(str).str.slice(0, 10),
        "nest_id": df["nest"].astype(str),
        "bird_id": df["Kring"].astype(str),
        "_sec":    sec,
    })
    return out[~out["bird_id"].isin(TEST_BIRD_IDS)].reset_index(drop=True)


def filter_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Return only rows matching 'daytime' or 'nighttime'."""
    if period == "daytime":
        mask = (df["_sec"] >= DAYTIME_START) & (df["_sec"] < DAYTIME_END)
    elif period == "nighttime":
        mask = (df["_sec"] >= NIGHTTIME_START) | (df["_sec"] < NIGHTTIME_END)
    else:
        raise ValueError(f"period must be 'daytime' or 'nighttime', got '{period}'")
    return df[mask].copy()


def filter_by_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    return df[df["date"].str.startswith(str(year))].copy()


def filter_by_plot(df: pd.DataFrame, plot: str) -> pd.DataFrame:
    return df[df["plot"] == plot].copy()


# ── GMM core ──────────────────────────────────────────────────────────────────

def _gmm_labels(sec: np.ndarray) -> np.ndarray:
    """Fit GMM with BIC-optimal k to a 1-D time array. Returns component labels."""
    x = sec.reshape(-1, 1)
    if len(x) < 2:
        return np.zeros(len(x), dtype=int)
    maxk = min(GMM_KMAX, len(x))
    models, bics = [], []
    for k in range(1, maxk + 1):
        gm = GaussianMixture(
            n_components=k, covariance_type="full",
            reg_covar=GMM_REG_COVAR, random_state=GMM_RANDOM_SEED
        )
        gm.fit(x)
        models.append(gm)
        bics.append(gm.bic(x))
    return models[int(np.argmin(bics))].predict(x)


def _windows_from_labels(block: pd.DataFrame) -> pd.DataFrame:
    """
    Merge nearby GMM clusters into event windows.
    block must have columns: _abs, _label
    Returns DataFrame with columns: start_abs, end_abs, n_detections
    """
    chunks = []
    for _, g in block.groupby("_label"):
        if len(g) < GMM_MIN_HITS:
            continue
        s, e = int(g["_abs"].min()), int(g["_abs"].max())
        if (e - s) > GMM_MAX_EVENT_S:
            continue
        chunks.append([s - 1, e + 1, len(g)])

    if not chunks:
        return pd.DataFrame(columns=["start_abs", "end_abs", "n_detections"])

    chunks.sort(key=lambda z: z[0])
    merged = [chunks[0]]
    for s, e, c in chunks[1:]:
        if s - merged[-1][1] <= GMM_MIN_GAP_S:
            merged[-1][1] = max(merged[-1][1], e)
            merged[-1][2] += c
        else:
            merged.append([s, e, c])
    return pd.DataFrame(merged, columns=["start_abs", "end_abs", "n_detections"])


def _abs_to_time(abs_s: int, date_index: pd.Index) -> Tuple[str, str]:
    """Convert absolute seconds back to (date_str, HH:MM:SS)."""
    di  = max(0, min(abs_s // 86400, len(date_index) - 1))
    sod = abs_s % 86400
    h, m, s = sod // 3600, (sod % 3600) // 60, sod % 60
    return date_index[di], f"{h:02d}:{m:02d}:{s:02d}"


# ── Public API ────────────────────────────────────────────────────────────────

def detect_events(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run GMM event detection on an already-filtered logger DataFrame.

    Args:
        df: rows for a single (year, plot, period) slice with columns
            plot, colony, date, nest_id, bird_id, _sec

    Returns:
        (events, membership)
        events     — plot, colony, date, nest_id, event_id,
                     start_date, start_time, end_date, end_time, n_detections
        membership — plot, colony, date, nest_id, event_id, bird_id
    """
    date_order = pd.Index(sorted(df["date"].unique()))
    day_idx    = pd.factorize(df["date"], sort=True)[0].astype(int)
    df         = df.copy()
    df["_abs"] = day_idx * 86400 + df["_sec"]

    ev_rows, mem_rows = [], []
    for keys, g in df.groupby(["plot", "colony", "date", "nest_id"], sort=False):
        labels = _gmm_labels(g["_sec"].to_numpy())
        gg     = g.assign(_label=labels)

        ev = _windows_from_labels(gg[["_abs", "_label"]])
        if ev.empty:
            continue

        ev = ev.sort_values("start_abs").reset_index(drop=True)
        ev["event_id"] = ev.index

        # Membership: which birds were detected inside each window
        det = gg[["bird_id", "_abs"]].drop_duplicates().copy()
        det["_key"] = 1
        E = ev[["event_id", "start_abs", "end_abs"]].copy()
        E["_key"] = 1
        m = det.merge(E, on="_key").drop(columns="_key")
        m = m[(m["_abs"] >= m["start_abs"]) & (m["_abs"] <= m["end_abs"])]
        m = m[["event_id", "bird_id"]].drop_duplicates()

        # Pretty-print times
        times = ev.apply(
            lambda r: pd.Series({
                "start_date": _abs_to_time(int(r["start_abs"]), date_order)[0],
                "start_time": _abs_to_time(int(r["start_abs"]), date_order)[1],
                "end_date":   _abs_to_time(int(r["end_abs"]),   date_order)[0],
                "end_time":   _abs_to_time(int(r["end_abs"]),   date_order)[1],
            }), axis=1,
        )
        ev_out = pd.concat([ev[["event_id", "n_detections"]], times], axis=1)
        ev_out["plot"], ev_out["colony"], ev_out["date"], ev_out["nest_id"] = keys
        ev_rows.append(ev_out)

        m["plot"], m["colony"], m["date"], m["nest_id"] = keys
        mem_rows.append(m)

    _ev_cols  = ["plot", "colony", "date", "nest_id", "event_id",
                 "start_date", "start_time", "end_date", "end_time", "n_detections"]
    _mem_cols = ["plot", "colony", "date", "nest_id", "event_id", "bird_id"]

    if not ev_rows:
        return pd.DataFrame(columns=_ev_cols), pd.DataFrame(columns=_mem_cols)

    events = (
        pd.concat(ev_rows, ignore_index=True)
        .sort_values(["plot", "colony", "date", "nest_id", "start_date", "start_time"])
        .reset_index(drop=True)
    )
    membership = (
        pd.concat(mem_rows, ignore_index=True)
        .sort_values(["plot", "colony", "date", "nest_id", "event_id", "bird_id"])
        .reset_index(drop=True)
    )
    return events, membership
