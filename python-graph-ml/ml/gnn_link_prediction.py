"""
GNN Link Prediction — Apple Silicon (MPS) native
=================================================
Predicts *which weaver dyads associate* using an attributed Graph Neural
Network, and asks whether that beats classical structural heuristics.

Why this task (and not sex classification)
------------------------------------------
Sex is not predictable from network position in this data (ROC-AUC ~= 0.5),
and shallow node2vec dims contribute ~zero feature importance. Link
prediction is a better-posed question for graph ML here because:

    1. There are *thousands* of dyads per context (vs ~100 nodes), so a GNN
       has enough supervision to actually learn.
    2. It maps onto the field's central question — what predicts who
       associates with whom (structure? identity? kinship?).

Method
------
    Encoder : per-node learnable embedding  ⊕  sex one-hot,  fed through
              2 layers of GraphSAGE message passing.  Crucially, message
              passing runs over TRAINING edges only — the val/test edges we
              score are never visible to the encoder, so there is no leakage.
    Decoder : dot product of the two endpoint embeddings → link logit.
    Split   : RandomLinkSplit (70/10/20 train/val/test) with negative
              sampling, undirected-aware.
    Eval    : ROC-AUC + Average Precision on held-out positive vs negative
              dyads, compared against an Adamic-Adar heuristic baseline
              computed on the same training graph.

Everything runs on the MPS (Apple GPU) backend when available, else CPU.

Usage
-----
    cd birds_new/
    python ml/gnn_link_prediction.py --year 2016 --plot SPRA --period daytime
    python ml/gnn_link_prediction.py                 # all contexts → CSV
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.transforms import RandomLinkSplit

from sklearn.metrics import roc_auc_score, average_precision_score

from config import YEARS, PLOTS, PERIODS, OUTPUT_ROOT, RANDOM_STATE
from ml.features import load_sex_labels

ML_OUT = Path(__file__).parent.parent / "workspace" / "ml"

# Contexts smaller than this many undirected edges can't support a
# meaningful 20% test split — skip them and report NaN.
MIN_EDGES = 40


# ── Device ─────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Data loading ────────────────────────────────────────────────────────────────

def load_context_graph(
    year: int, plot: str, period: str
) -> Tuple[Data, List[str]]:
    """
    Build a PyG Data object for one context.

    Node features = one-hot sex (M / F / unknown).  A learnable per-node
    embedding is added inside the model, so the only *input* features are
    intrinsic (sex), never edge-derived — this keeps link prediction honest.

    Returns:
        data       — PyG Data with x, edge_index (undirected, deduped)
        node_ids   — list of bird_id strings, index-aligned to data
    """
    label = f"{year}_{plot}_{period}"
    edge_file = OUTPUT_ROOT / label / f"edges_{label}.csv"
    if not edge_file.exists():
        raise FileNotFoundError(f"No edges for {label}: {edge_file}")

    edges = pd.read_csv(edge_file)
    edges = edges[edges["association"] > 0].copy()

    node_ids = sorted(set(edges["id1"]) | set(edges["id2"]))
    idx = {b: i for i, b in enumerate(node_ids)}
    n = len(node_ids)

    # Undirected edge_index (both directions), deduped
    src = edges["id1"].map(idx).to_numpy()
    dst = edges["id2"].map(idx).to_numpy()
    ei = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
    ei = torch.cat([ei, ei.flip(0)], dim=1)

    # Sex one-hot: columns [M, F, unknown]
    try:
        sex = load_sex_labels().set_index("bird_id")["sex"].to_dict()
    except Exception:
        sex = {}
    x = torch.zeros((n, 3), dtype=torch.float)
    for b, i in idx.items():
        s = sex.get(b)
        x[i, 0 if s == "M" else 1 if s == "F" else 2] = 1.0

    data = Data(x=x, edge_index=ei, num_nodes=n)
    return data, node_ids


# ── Model ────────────────────────────────────────────────────────────────────────

class GNNLinkPredictor(nn.Module):
    """
    GraphSAGE encoder (learnable node embedding ⊕ sex) + dot-product decoder.
    """

    def __init__(self, num_nodes: int, sex_dim: int = 3,
                 emb_dim: int = 32, hidden: int = 64, out: int = 32):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, emb_dim)
        self.sex_proj = nn.Linear(sex_dim, emb_dim)
        self.conv1 = SAGEConv(emb_dim, hidden)
        self.conv2 = SAGEConv(hidden, out)
        nn.init.xavier_uniform_(self.node_emb.weight)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor,
               node_idx: torch.Tensor) -> torch.Tensor:
        h = self.node_emb(node_idx) + self.sex_proj(x)
        h = F.relu(self.conv1(h, edge_index))
        h = self.conv2(h, edge_index)
        return h

    @staticmethod
    def decode(z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)


# ── Baseline ─────────────────────────────────────────────────────────────────────

def adamic_adar_auc(train_edge_index: torch.Tensor, num_nodes: int,
                    test_edge_label_index: torch.Tensor,
                    test_labels: torch.Tensor) -> float:
    """
    Adamic-Adar heuristic on the training graph, scored on the same held-out
    positive/negative dyads the GNN is tested on. Sets the 'no learning' bar.
    """
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    ei = train_edge_index.cpu().numpy()
    G.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))

    pairs = list(zip(*test_edge_label_index.cpu().numpy().tolist()))
    scores = np.zeros(len(pairs))
    for k, (u, v) in enumerate(pairs):
        if u == v:
            continue
        s = 0.0
        for w in nx.common_neighbors(G, u, v):
            deg = G.degree(w)
            if deg > 1:
                s += 1.0 / np.log(deg)
        scores[k] = s

    y = test_labels.cpu().numpy()
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


# ── Train / evaluate one context ────────────────────────────────────────────────

def run_context(year: int, plot: str, period: str,
                epochs: int = 300, lr: float = 0.01,
                patience: int = 40, verbose: bool = True,
                seed: int = RANDOM_STATE) -> Optional[Dict]:
    label = f"{year}_{plot}_{period}"
    torch.manual_seed(seed)

    data, node_ids = load_context_graph(year, plot, period)
    n_undirected = data.edge_index.shape[1] // 2
    if n_undirected < MIN_EDGES:
        if verbose:
            print(f"[skip] {label}: only {n_undirected} edges (< {MIN_EDGES})")
        return None

    device = get_device()

    splitter = RandomLinkSplit(
        num_val=0.1, num_test=0.2,
        is_undirected=True,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
    )
    train_data, val_data, test_data = splitter(data)

    node_idx = torch.arange(data.num_nodes, device=device)
    for d in (train_data, val_data, test_data):
        d.x = d.x.to(device)
        d.edge_index = d.edge_index.to(device)
        d.edge_label_index = d.edge_label_index.to(device)
        d.edge_label = d.edge_label.to(device)

    model = GNNLinkPredictor(data.num_nodes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    def eval_split(split) -> Tuple[float, float]:
        model.eval()
        with torch.no_grad():
            z = model.encode(train_data.x, train_data.edge_index, node_idx)
            logits = model.decode(z, split.edge_label_index)
            prob = torch.sigmoid(logits).cpu().numpy()
            y = split.edge_label.cpu().numpy()
        if len(np.unique(y)) < 2:
            return float("nan"), float("nan")
        return roc_auc_score(y, prob), average_precision_score(y, prob)

    best_val, best_test_auc, best_test_ap, best_epoch = -1.0, float("nan"), float("nan"), 0
    since_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index, node_idx)
        logits = model.decode(z, train_data.edge_label_index)
        loss = F.binary_cross_entropy_with_logits(logits, train_data.edge_label)
        loss.backward()
        opt.step()

        val_auc, _ = eval_split(val_data)
        if not np.isnan(val_auc) and val_auc > best_val:
            best_val = val_auc
            best_test_auc, best_test_ap = eval_split(test_data)
            best_epoch = epoch
            since_improve = 0
        else:
            since_improve += 1
        if since_improve >= patience:
            break

    aa_auc = adamic_adar_auc(
        train_data.edge_index.cpu(), data.num_nodes,
        test_data.edge_label_index, test_data.edge_label,
    )

    result = {
        "year": year, "plot": plot, "period": period,
        "n_nodes": data.num_nodes, "n_edges": n_undirected,
        "device": str(device),
        "gnn_test_auc": round(best_test_auc, 4),
        "gnn_test_ap": round(best_test_ap, 4),
        "adamic_adar_auc": round(aa_auc, 4),
        "gnn_minus_aa": round(best_test_auc - aa_auc, 4),
        "best_epoch": best_epoch,
    }
    if verbose:
        print(f"[{label}] {data.num_nodes} nodes, {n_undirected} edges | "
              f"GNN AUC={best_test_auc:.3f} AP={best_test_ap:.3f} | "
              f"AA AUC={aa_auc:.3f} | Δ={best_test_auc-aa_auc:+.3f} "
              f"(epoch {best_epoch}, {device})")
    return result


# ── Batch ────────────────────────────────────────────────────────────────────────

def run_all_contexts(years=None, plots=None, periods=None, **kw) -> pd.DataFrame:
    years = years or YEARS
    plots = plots or PLOTS
    periods = periods or PERIODS

    rows = []
    for year in years:
        for plot in plots:
            for period in periods:
                try:
                    r = run_context(year, plot, period, **kw)
                    if r is not None:
                        rows.append(r)
                except FileNotFoundError:
                    continue
                except Exception as e:
                    print(f"[error] {year}_{plot}_{period}: {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        ML_OUT.mkdir(parents=True, exist_ok=True)
        out = ML_OUT / "gnn_link_prediction.csv"
        df.to_csv(out, index=False)
        print(f"\nSaved → {out}  ({len(df)} contexts)")
        print(f"Mean GNN AUC={df.gnn_test_auc.mean():.3f}  "
              f"mean AA AUC={df.adamic_adar_auc.mean():.3f}  "
              f"mean Δ={df.gnn_minus_aa.mean():+.3f}")
    return df


# ── CLI ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GNN link prediction on weaver graphs")
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--plot", type=str, default=None)
    ap.add_argument("--period", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()

    if args.year and args.plot and args.period:
        run_context(args.year, args.plot, args.period, epochs=args.epochs)
    else:
        run_all_contexts(
            years=[args.year] if args.year else None,
            plots=[args.plot] if args.plot else None,
            periods=[args.period] if args.period else None,
            epochs=args.epochs,
        )
