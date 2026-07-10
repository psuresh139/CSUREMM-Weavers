"""
Feature Engineering
====================
Merges graph-theoretic node metrics with node2vec embeddings into a single
feature matrix ready for clustering / classification.

Also loads external label files (sex, breeding status, colony) for supervised tasks.

Label loading summary
---------------------
    load_sex_labels()        → bird_id, sex (M/F) from weaver_sex.xlsx
    load_breeding_labels()   → bird_id, fledge_success (0/1) derived by
                               matching each bird's dominant nest in the logger
                               to daily_breeding.xlsx fledge outcomes
    load_colony_labels()     → bird_id, field_colony from banding file
    load_relatedness()       → (ind1_kring, ind2_kring, wang, dyadml) joining
                               Relateness_weaver.csv combo IDs → K-ring IDs
"""

from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

from config import (
    OUTPUT_ROOT, SEX_FILE, BREEDING_FILE, BANDING_FILE,
    LOGGER_FILE, RELATEDNESS_FILE,
)


# ── Load helpers ──────────────────────────────────────────────────────────────

def load_nodes(year: int, plot: str, period: str) -> pd.DataFrame:
    label = f"{year}_{plot}_{period}"
    path  = OUTPUT_ROOT / label / f"nodes_{label}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Node file not found: {path}")
    return pd.read_csv(path)


def load_embeddings(year: int, plot: str, period: str) -> pd.DataFrame:
    label = f"{year}_{plot}_{period}"
    path  = OUTPUT_ROOT / label / f"embeddings_{label}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Embeddings not found: {path}\n"
            f"Run embeddings/node2vec_pipeline.py first."
        )
    return pd.read_csv(path)


# ── Label loading ─────────────────────────────────────────────────────────────

def load_sex_labels() -> pd.DataFrame:
    """
    Load sex labels.  Returns DataFrame with bird_id, sex columns.
    weaver_sex.xlsx uses column 'metal' for the K-ring bird ID.
    """
    df = pd.read_excel(SEX_FILE)
    df.columns = [c.strip().lower() for c in df.columns]

    # 'metal' is the K-ring bird ID (format e.g. K#####); fall back to other names
    id_col = next(
        (c for c in df.columns if c in ("metal", "kring", "bird_id", "id")),
        None,
    )
    sex_col = next((c for c in df.columns if "sex" in c), None)
    if id_col is None or sex_col is None:
        raise ValueError(
            f"Cannot find id/sex columns in {SEX_FILE}. "
            f"Available: {df.columns.tolist()}"
        )

    out = df[[id_col, sex_col]].rename(columns={id_col: "bird_id", sex_col: "sex"})
    out["bird_id"] = out["bird_id"].astype(str).str.strip()
    out["sex"]     = out["sex"].astype(str).str.strip().str.upper()
    # Keep only M/F rows
    out = out[out["sex"].isin(["M", "F"])]
    return out.drop_duplicates("bird_id").reset_index(drop=True)


def load_breeding_labels(
    year: int,
    plot: str,
    logger_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Per-bird breeding success for a given (year, plot).

    The logger↔breeding join IS solvable: the colony/nest namespaces align
    once normalised ('MSTO_01'→'MSTO01', logger nest '118'→breeding nest 118).
    Each bird is assigned its dominant breeding-season nest, joined to that
    nest's fledge outcome. See ml/breeding_labels.py for the full method and
    the detection-proxy caveat. ~74% of bird-years get a label.

    Returns DataFrame with columns: bird_id, fledged (0/1), hatched (0/1).
    Renames 'fledged' → 'fledge_success' for backward compatibility.
    """
    from ml.breeding_labels import load_breeding_labels as _load
    df = _load(year, plot)
    return df.rename(columns={"fledged": "fledge_success"})


def load_colony_labels() -> pd.DataFrame:
    """
    Load field-colony assignment from banding file.
    Returns DataFrame with bird_id (K-ring), field_colony.
    """
    df = pd.read_excel(BANDING_FILE)
    df.columns = [c.strip().lower() for c in df.columns]

    id_col     = next(c for c in df.columns if c in ("metal", "kring", "bird_id"))
    colony_col = next(c for c in df.columns if c == "colony")

    out = (
        df[[id_col, colony_col]]
        .rename(columns={id_col: "bird_id", colony_col: "field_colony"})
    )
    out["bird_id"]      = out["bird_id"].astype(str).str.strip()
    out["field_colony"] = out["field_colony"].astype(str).str.strip()
    # Exclude test tags and rows without a colony
    out = out[
        (out["bird_id"] != "K00000")
        & out["field_colony"].notna()
        & (out["field_colony"] != "nan")
    ]
    return out.drop_duplicates("bird_id").reset_index(drop=True)


def load_relatedness() -> pd.DataFrame:
    """
    Load pairwise relatedness, joining combo IDs → K-ring IDs via banding file.

    Returns DataFrame with columns:
        bird_a, bird_b, wang, dyadml
    Only pairs where both IDs are found in banding are returned.
    """
    rel = pd.read_csv(RELATEDNESS_FILE)
    rel.columns = [c.strip().lower().replace(".", "_") for c in rel.columns]

    band = pd.read_excel(BANDING_FILE)
    band.columns = [c.strip().lower() for c in band.columns]
    combo_col = next(c for c in band.columns if c in ("combo", "colour"))
    metal_col = next(c for c in band.columns if c in ("metal", "kring"))

    # Build combo → kring lookup
    lookup = (
        band[[metal_col, combo_col]]
        .dropna(subset=[combo_col])
        .drop_duplicates(combo_col)
        .set_index(combo_col)[metal_col]
        .to_dict()
    )

    out = rel.copy()
    # Relatedness IDs have a plot prefix: 'AA_MYYW' → strip to 'MYYW'
    out["_combo1"] = out["ind1_id"].str.replace(r'^[A-Z]+_', '', regex=True)
    out["_combo2"] = out["ind2_id"].str.replace(r'^[A-Z]+_', '', regex=True)
    out["bird_a"] = out["_combo1"].map(lookup)
    out["bird_b"] = out["_combo2"].map(lookup)
    out = out.drop(columns=["_combo1", "_combo2"])
    out = out.dropna(subset=["bird_a", "bird_b"])

    keep_cols = ["bird_a", "bird_b"]
    for c in ("wang", "dyadml", "quellergt"):
        if c in out.columns:
            keep_cols.append(c)

    return out[keep_cols].reset_index(drop=True)


# ── Feature matrix builder ────────────────────────────────────────────────────

GRAPH_FEATURE_COLS = [
    "degree", "strength", "betweenness", "eigenvector", "clustering",
]


def build_feature_matrix(
    year: int,
    plot: str,
    period: str,
    use_graph_metrics: bool = True,
    use_embeddings: bool = True,
    normalise: bool = True,
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Build feature matrix X for one (year, plot, period) context.

    Args:
        use_graph_metrics: include graph-theoretic features
        use_embeddings:    include node2vec embedding dimensions
        normalise:         z-score normalise numeric features

    Returns:
        X              — (n_birds, n_features) float array
        feature_names  — list of column names
        meta           — DataFrame with bird_id + available label columns
    """
    nodes = load_nodes(year, plot, period)

    parts = []
    names = []

    if use_embeddings:
        emb = load_embeddings(year, plot, period)
        nodes = nodes.merge(emb, on="bird_id", how="inner")
        dim_cols = [c for c in emb.columns if c.startswith("dim_")]
        parts.append(nodes[dim_cols].values.astype(float))
        names.extend(dim_cols)

    if use_graph_metrics:
        # Extract AFTER the embedding join so row count is consistent
        available = [c for c in GRAPH_FEATURE_COLS if c in nodes.columns]
        parts.append(nodes[available].values.astype(float))
        names.extend(available)

    if not parts:
        raise ValueError("No features selected — enable at least one of use_graph_metrics / use_embeddings")

    X = np.hstack(parts)

    # Replace NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if normalise:
        mu  = X.mean(axis=0, keepdims=True)
        sig = X.std(axis=0, keepdims=True)
        sig[sig == 0] = 1.0
        X = (X - mu) / sig

    # Attach available labels to meta
    meta = nodes[["bird_id"]].copy()

    try:
        sex = load_sex_labels()
        meta = meta.merge(sex, on="bird_id", how="left")
    except Exception as e:
        print(f"  [warn] sex labels unavailable: {e}")

    try:
        breed = load_breeding_labels(year=year, plot=plot)
        meta = meta.merge(breed, on="bird_id", how="left")
    except Exception as e:
        print(f"  [warn] breeding labels unavailable: {e}")

    try:
        colony = load_colony_labels()
        meta = meta.merge(colony, on="bird_id", how="left")
    except Exception as e:
        print(f"  [warn] colony labels unavailable: {e}")

    # Network position is already in nodes.csv — add it to meta
    if "network_position" in nodes.columns:
        meta = meta.merge(nodes[["bird_id", "network_position"]], on="bird_id", how="left")

    return X, names, meta


# ── Convenience: load all contexts into a combined matrix ─────────────────────

def build_combined_matrix(
    years=None, plots=None, periods=None,
    use_graph_metrics=True,
    use_embeddings=True,
    normalise=True,
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Build and vertically stack feature matrices from multiple contexts.
    Adds context columns (year, plot, period) to meta.
    """
    from config import YEARS, PLOTS, PERIODS
    years   = years   or YEARS
    plots   = plots   or PLOTS
    periods = periods or PERIODS

    X_list, meta_list = [], []
    feature_names = None

    for year in years:
        for plot in plots:
            for period in periods:
                try:
                    X, names, meta = build_feature_matrix(
                        year, plot, period,
                        use_graph_metrics=use_graph_metrics,
                        use_embeddings=use_embeddings,
                        normalise=False,
                    )
                    meta["year"]   = year
                    meta["plot"]   = plot
                    meta["period"] = period
                    X_list.append(X)
                    meta_list.append(meta)
                    if feature_names is None:
                        feature_names = names
                except FileNotFoundError:
                    pass

    if not X_list:
        raise ValueError("No contexts found. Run the pipeline first.")

    X_all    = np.vstack(X_list)
    meta_all = pd.concat(meta_list, ignore_index=True)

    if normalise:
        mu  = X_all.mean(axis=0, keepdims=True)
        sig = X_all.std(axis=0, keepdims=True)
        sig[sig == 0] = 1.0
        X_all = (X_all - mu) / sig

    return X_all, feature_names, meta_all
