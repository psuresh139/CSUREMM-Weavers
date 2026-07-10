"""
Graph Construction & Network Metrics
======================================
Builds a NetworkX graph from an SRI edge list and computes a full suite of
graph-theoretic node features used downstream by node2vec and ML modules.

Node features produced
----------------------
degree              — raw number of neighbours
strength            — weighted degree (sum of SRI weights)
betweenness         — normalised betweenness centrality
eigenvector         — eigenvector centrality
clustering          — weighted clustering coefficient
community           — Louvain community id (int)
network_position    — Core / Intermediate / Peripheral
is_connector        — bool, top-10% betweenness

Network-level stats (returned separately)
------------------------------------------
n_nodes, n_edges, edge_density, avg_degree,
avg_clustering, avg_path_length, diameter, n_components
"""

import warnings
from typing import Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

warnings.filterwarnings("ignore")

from config import CONNECTOR_TOP_PERCENTILE


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(edges: pd.DataFrame, weight_col: str = "association") -> nx.Graph:
    """Build weighted undirected graph from edge list (id1, id2, weight_col)."""
    G = nx.Graph()
    for _, row in edges.iterrows():
        G.add_edge(row["id1"], row["id2"], weight=float(row[weight_col]))
    return G


# ── Node metrics ──────────────────────────────────────────────────────────────

def compute_node_metrics(G: nx.Graph) -> pd.DataFrame:
    """
    Compute per-node graph metrics.
    Returns DataFrame indexed by bird_id.
    """
    nodes = sorted(G.nodes())
    if not nodes:
        return pd.DataFrame()

    degree = dict(G.degree())
    strength = {n: sum(d["weight"] for _, d in G[n].items()) for n in nodes}

    try:
        # weight=None → hop-based betweenness, standard in animal social networks
        # (Farine 2015). Using SRI as a distance causes float overflow because
        # very small SRI values (~0.001) sum to astronomically large path costs.
        betweenness = nx.betweenness_centrality(G, weight=None, normalized=True)
    except Exception:
        betweenness = {n: 0.0 for n in nodes}

    try:
        eigenvector = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        eigenvector = {n: 0.0 for n in nodes}

    try:
        clustering = nx.clustering(G, weight="weight")
    except Exception:
        clustering = {n: 0.0 for n in nodes}

    return pd.DataFrame({
        "bird_id":    nodes,
        "degree":     [degree.get(n, 0) for n in nodes],
        "strength":   [strength.get(n, 0.0) for n in nodes],
        "betweenness":[betweenness.get(n, 0.0) for n in nodes],
        "eigenvector":[eigenvector.get(n, 0.0) for n in nodes],
        "clustering": [clustering.get(n, 0.0) for n in nodes],
    })


def classify_positions(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Add degree_category, betweenness_category, network_position columns.
    Core   = top quartile on BOTH degree and betweenness
    Peripheral = bottom quartile on EITHER
    Intermediate = everything else
    """
    df = metrics.copy()
    dq25, dq75 = df["degree"].quantile([0.25, 0.75])
    bq25, bq75 = df["betweenness"].quantile([0.25, 0.75])

    df["degree_category"] = pd.cut(
        df["degree"], bins=[-np.inf, dq25, dq75, np.inf],
        labels=["Low", "Medium", "High"],
    )
    df["betweenness_category"] = pd.cut(
        df["betweenness"], bins=[-np.inf, bq25, bq75, np.inf],
        labels=["Low", "Medium", "High"],
    )

    pos = pd.Series("Intermediate", index=df.index)
    pos[(df["degree"] >= dq75) & (df["betweenness"] >= bq75)] = "Core"
    pos[(df["degree"] < dq25)  | (df["betweenness"] < bq25)]  = "Peripheral"
    df["network_position"] = pos
    return df


def add_connectors(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mark top-10% betweenness nodes as connectors."""
    df = metrics.copy()
    threshold = df["betweenness"].quantile(CONNECTOR_TOP_PERCENTILE)
    df["is_connector"] = df["betweenness"] >= threshold
    return df


def detect_communities(G: nx.Graph, nodes: list) -> Tuple[pd.DataFrame, Dict]:
    """
    Louvain community detection with NetworkX greedy_modularity fallback.
    Returns (community_df, info_dict).
    """
    try:
        import community.community_louvain as cl
        partition = cl.best_partition(G, weight="weight")
        modularity = cl.modularity(partition, G, weight="weight")
        communities = {}
        for n, c in partition.items():
            communities.setdefault(c, []).append(n)
    except ImportError:
        comms = nx.community.greedy_modularity_communities(G, weight="weight")
        partition = {n: i for i, c in enumerate(comms) for n in c}
        modularity = nx.community.modularity(G, comms, weight="weight")
        communities = {i: list(c) for i, c in enumerate(comms)}

    comm_df = pd.DataFrame({
        "bird_id":   nodes,
        "community": [partition.get(n, -1) for n in nodes],
    })
    info = {
        "n_communities": len(communities),
        "modularity":    round(modularity, 4),
        "community_sizes": {k: len(v) for k, v in communities.items()},
    }
    return comm_df, info


# ── Network-level stats ───────────────────────────────────────────────────────

def network_stats(G: nx.Graph) -> Dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    stats = {
        "n_nodes":      n,
        "n_edges":      m,
        "edge_density": m / (n * (n - 1) / 2) if n > 1 else 0.0,
        "avg_degree":   2 * m / n if n > 0 else 0.0,
    }
    try:
        stats["avg_clustering"] = nx.average_clustering(G, weight="weight")
    except Exception:
        stats["avg_clustering"] = 0.0

    try:
        if nx.is_connected(G):
            lcc = G
        else:
            lcc = G.subgraph(max(nx.connected_components(G), key=len))
        stats["avg_path_length"] = nx.average_shortest_path_length(lcc, weight="weight")
        stats["diameter"]        = nx.diameter(lcc)
    except Exception:
        stats["avg_path_length"] = np.nan
        stats["diameter"]        = np.nan

    comps = list(nx.connected_components(G))
    stats["n_components"]           = len(comps)
    stats["largest_component_size"] = len(max(comps, key=len)) if comps else 0
    return stats


# ── Full analysis pipeline ────────────────────────────────────────────────────

def analyse_graph(
    edges: pd.DataFrame,
    output_dir: Optional[Path] = None,
    prefix: str = "",
) -> Dict[str, pd.DataFrame]:
    """
    Run the complete graph analysis pipeline on an edge list.

    Args:
        edges:      DataFrame with id1, id2, association
        output_dir: if provided, saves all CSVs here
        prefix:     filename prefix (e.g. '2016_SPRA_daytime_')

    Returns:
        dict with keys: nodes, edges, stats, community_info
        'nodes' has all per-bird metrics in a single flat DataFrame
    """
    G = build_graph(edges)
    node_list = sorted(G.nodes())

    metrics  = compute_node_metrics(G)
    metrics  = classify_positions(metrics)
    metrics  = add_connectors(metrics)
    comm_df, comm_info = detect_communities(G, node_list)

    nodes = metrics.merge(comm_df, on="bird_id", how="left")
    stats = pd.DataFrame([network_stats(G)])

    results = {
        "nodes":          nodes,
        "edges":          edges,
        "stats":          stats,
        "community_info": pd.DataFrame([comm_info]),
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        nodes.to_csv(output_dir / f"nodes_{prefix}.csv",          index=False)
        edges.to_csv(output_dir / f"edges_{prefix}.csv",          index=False)
        stats.to_csv(output_dir / f"stats_{prefix}.csv",          index=False)
        pd.DataFrame([comm_info]).to_csv(
            output_dir / f"community_info_{prefix}.csv", index=False
        )

    return results
