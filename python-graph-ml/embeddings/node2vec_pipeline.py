"""
Node2Vec Embeddings — Apple Silicon Compatible
================================================
Generates node embeddings from a bird social network using the `node2vec`
package (pip install node2vec), which is backed by gensim's Word2Vec and
has ARM64-native wheels. No TensorFlow / StellarGraph required.

Typical usage
-------------
    from embeddings.node2vec_pipeline import embed_graph, load_graph

    edges = pd.read_csv("workspace/graphs/2016_SPRA_daytime/edges_2016_SPRA_daytime.csv")
    model = embed_graph(edges)
    embeddings = get_embeddings(model, list(edges["id1"].unique()))

Output
------
    embeddings_{prefix}.csv  — bird_id + 128 numeric columns (dim_0 … dim_127)
"""

import sys
from pathlib import Path
from typing import Optional

# Ensure project root is on path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import networkx as nx

# node2vec package: pip install node2vec
# Uses gensim Word2Vec internally — no TF dependency, ARM64-native
from node2vec import Node2Vec

from config import (
    N2V_DIMENSIONS, N2V_WALK_LENGTH, N2V_NUM_WALKS,
    N2V_WINDOW, N2V_P, N2V_Q, N2V_WORKERS,
    OUTPUT_ROOT,
)


# ── Graph loading ─────────────────────────────────────────────────────────────

def load_graph(edges: pd.DataFrame, weight_col: str = "association") -> nx.Graph:
    """Build a weighted NetworkX graph from an edge-list DataFrame.
    Zero-weight edges are dropped — they crash node2vec's walk normalisation
    and represent dyads with no detected association (SRI numerator = 0).
    """
    edges = edges[edges[weight_col] > 0].copy()
    G = nx.Graph()
    for _, row in edges.iterrows():
        G.add_edge(str(row["id1"]), str(row["id2"]), weight=float(row[weight_col]))
    return G


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_graph(
    edges: pd.DataFrame,
    weight_col: str = "association",
    dimensions: int = N2V_DIMENSIONS,
    walk_length: int = N2V_WALK_LENGTH,
    num_walks: int = N2V_NUM_WALKS,
    window: int = N2V_WINDOW,
    p: float = N2V_P,
    q: float = N2V_Q,
    workers: int = N2V_WORKERS,
    seed: int = 42,
):
    """
    Train node2vec on a bird association graph.

    Args:
        edges:      DataFrame with id1, id2, weight_col
        dimensions: embedding dimension (default 128)
        walk_length: length of each random walk
        num_walks:  number of walks per node
        window:     Word2Vec context window
        p:          return parameter (1 = no bias)
        q:          in-out parameter (1 = random walk)
        workers:    parallel workers (set to 1 if gensim complains)
        seed:       reproducibility

    Returns:
        Trained Node2Vec model (access embeddings via model.wv)
    """
    G = load_graph(edges, weight_col)

    n2v = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=workers,
        seed=seed,
        quiet=True,
    )
    model = n2v.fit(window=window, min_count=1, batch_words=4)
    return model


def get_embeddings(model, nodes: list) -> pd.DataFrame:
    """
    Extract embedding vectors for a list of nodes.

    Returns a DataFrame with columns [bird_id, dim_0, dim_1, ..., dim_{d-1}].
    Nodes not present in the model vocabulary are filled with zeros.
    """
    dim = model.wv.vector_size
    rows = []
    for node in nodes:
        key = str(node)
        vec = model.wv[key] if key in model.wv else np.zeros(dim)
        rows.append([node] + vec.tolist())

    cols = ["bird_id"] + [f"dim_{i}" for i in range(dim)]
    return pd.DataFrame(rows, columns=cols)


# ── Convenience: run for one context ─────────────────────────────────────────

def run_context(
    year: int,
    plot: str,
    period: str,
    output_dir: Optional[Path] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Load edges from workspace, train embeddings, save & return embedding DataFrame.

    Args:
        year, plot, period: context identifiers
        output_dir:         where to save; defaults to workspace/graphs/{context}/
        **kwargs:           forwarded to embed_graph (override dimensions, p, q, etc.)

    Returns:
        embeddings DataFrame (bird_id + dim_* columns)
    """
    label    = f"{year}_{plot}_{period}"
    ctx_dir  = OUTPUT_ROOT / label
    edge_file = ctx_dir / f"edges_{label}.csv"

    if not edge_file.exists():
        raise FileNotFoundError(
            f"Edge file not found: {edge_file}\n"
            f"Run pipeline/run_pipeline.py first."
        )

    edges = pd.read_csv(edge_file)
    print(f"[{label}] {len(edges)} edges, fitting node2vec…")

    model = embed_graph(edges, **kwargs)
    nodes = sorted(set(edges["id1"]) | set(edges["id2"]))
    emb   = get_embeddings(model, nodes)

    save_dir = output_dir or ctx_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"embeddings_{label}.csv"
    emb.to_csv(out_path, index=False)
    print(f"[{label}] embeddings saved → {out_path}  shape={emb.shape}")

    return emb


# ── Batch run ──────────────────────────────────────────────────────────────────

def run_all_contexts(years=None, plots=None, periods=None, **kwargs):
    """Run node2vec for every available context in the workspace."""
    from config import YEARS, PLOTS, PERIODS
    years   = years   or YEARS
    plots   = plots   or PLOTS
    periods = periods or PERIODS

    results = {}
    for year in years:
        for plot in plots:
            for period in periods:
                label = f"{year}_{plot}_{period}"
                ctx_dir = OUTPUT_ROOT / label / f"edges_{label}.csv"
                if not ctx_dir.exists():
                    print(f"[skip] no edges for {label}")
                    continue
                try:
                    emb = run_context(year, plot, period, **kwargs)
                    results[label] = emb
                except Exception as e:
                    print(f"[error] {label}: {e}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    parser = argparse.ArgumentParser(description="Node2Vec embeddings")
    parser.add_argument("--year",   type=int, default=None)
    parser.add_argument("--plot",   type=str, default=None)
    parser.add_argument("--period", type=str, default=None)
    parser.add_argument("--dim",    type=int, default=N2V_DIMENSIONS)
    parser.add_argument("--p",      type=float, default=N2V_P)
    parser.add_argument("--q",      type=float, default=N2V_Q)
    args = parser.parse_args()

    if args.year and args.plot and args.period:
        run_context(args.year, args.plot, args.period,
                    dimensions=args.dim, p=args.p, q=args.q)
    else:
        run_all_contexts(
            years=[args.year] if args.year else None,
            plots=[args.plot] if args.plot else None,
            periods=[args.period] if args.period else None,
            dimensions=args.dim, p=args.p, q=args.q,
        )
