"""
Does kinship predict association beyond topology?  (GNN ablation)
=================================================================
The topology-only link-prediction result (ml/gnn_link_prediction.py) showed
weaver associations are dominated by triadic closure — a GNN with sex +
node identity cannot beat an Adamic-Adar heuristic. The ceiling is the
*features*, not the model.

This script tests the obvious missing feature: **genetic relatedness**
(Wang's estimator, from Relateness_weaver.csv, ~80-90% dyad coverage).

Clean ablation
--------------
Two models, IDENTICAL architecture (GraphSAGE encoder + MLP link decoder),
same splits and seed. The only difference:

    topology-only : decoder sees [z_i ⊙ z_j , |z_i - z_j|]
    +relatedness  : decoder also sees the dyad's Wang relatedness scalar

ΔAUC = AUC(+relatedness) − AUC(topology-only) = the marginal predictive
value of kinship for who-associates-with-whom, above and beyond graph
structure. A positive, consistent Δ is direct evidence for kin-structured
sociality; ~0 means association is not kin-driven once closure is accounted
for (which would itself sharpen the earlier weak-Mantel result).

Usage
-----
    cd birds_new/
    python ml/gnn_relatedness_ablation.py --plot SPRA      # good coverage
    python ml/gnn_relatedness_ablation.py                  # all contexts
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.transforms import RandomLinkSplit

from sklearn.metrics import roc_auc_score, average_precision_score

from config import YEARS, PLOTS, PERIODS, OUTPUT_ROOT, RANDOM_STATE
from ml.features import load_relatedness
from ml.gnn_link_prediction import load_context_graph, get_device, MIN_EDGES

ML_OUT = Path(__file__).parent.parent / "workspace" / "ml"


# ── Relatedness matrix for a context ────────────────────────────────────────────

def build_relatedness_matrix(node_ids, rel_lookup) -> Tuple[torch.Tensor, float]:
    """
    n×n symmetric Wang-relatedness matrix aligned to node_ids.
    Uncovered dyads are imputed with 0 (mean Wang ≈ 0 = unrelated).
    Returns (matrix, coverage_fraction).
    """
    n = len(node_ids)
    R = torch.zeros((n, n), dtype=torch.float)
    covered = 0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            w = rel_lookup.get(frozenset((node_ids[i], node_ids[j])))
            if w is not None:
                R[i, j] = R[j, i] = float(w)
                covered += 1
    return R, (covered / total if total else 0.0)


# ── Model ────────────────────────────────────────────────────────────────────────

class GNNWithMLPDecoder(nn.Module):
    """GraphSAGE encoder + MLP decoder that optionally ingests a dyad scalar."""

    def __init__(self, num_nodes, sex_dim=3, emb_dim=32, hidden=64, out=32,
                 use_relatedness=False):
        super().__init__()
        self.use_relatedness = use_relatedness
        self.node_emb = nn.Embedding(num_nodes, emb_dim)
        self.sex_proj = nn.Linear(sex_dim, emb_dim)
        self.conv1 = SAGEConv(emb_dim, hidden)
        self.conv2 = SAGEConv(hidden, out)
        dec_in = 2 * out + (1 if use_relatedness else 0)
        self.dec = nn.Sequential(
            nn.Linear(dec_in, 32), nn.ReLU(), nn.Linear(32, 1),
        )
        nn.init.xavier_uniform_(self.node_emb.weight)

    def encode(self, x, edge_index, node_idx):
        h = self.node_emb(node_idx) + self.sex_proj(x)
        h = F.relu(self.conv1(h, edge_index))
        return self.conv2(h, edge_index)

    def decode(self, z, edge_label_index, rel_scalar=None):
        src, dst = edge_label_index
        feats = [z[src] * z[dst], (z[src] - z[dst]).abs()]
        if self.use_relatedness:
            feats.append(rel_scalar.unsqueeze(-1))
        return self.dec(torch.cat(feats, dim=-1)).squeeze(-1)


# ── Train one model ──────────────────────────────────────────────────────────────

def _train_eval(data, train_data, val_data, test_data, R, node_idx, device,
                use_relatedness, epochs, lr, patience, seed) -> Tuple[float, float]:
    torch.manual_seed(seed)
    model = GNNWithMLPDecoder(data.num_nodes, use_relatedness=use_relatedness).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    def rel_for(edge_label_index):
        if not use_relatedness:
            return None
        s, d = edge_label_index
        return R[s, d]

    def evaluate(split):
        model.eval()
        with torch.no_grad():
            z = model.encode(train_data.x, train_data.edge_index, node_idx)
            logits = model.decode(z, split.edge_label_index, rel_for(split.edge_label_index))
            prob = torch.sigmoid(logits).cpu().numpy()
            y = split.edge_label.cpu().numpy()
        if len(np.unique(y)) < 2:
            return float("nan"), float("nan")
        return roc_auc_score(y, prob), average_precision_score(y, prob)

    best_val, best_auc, best_ap, since = -1.0, float("nan"), float("nan"), 0
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index, node_idx)
        logits = model.decode(z, train_data.edge_label_index,
                              rel_for(train_data.edge_label_index))
        loss = F.binary_cross_entropy_with_logits(logits, train_data.edge_label)
        loss.backward()
        opt.step()

        val_auc, _ = evaluate(val_data)
        if not np.isnan(val_auc) and val_auc > best_val:
            best_val = val_auc
            best_auc, best_ap = evaluate(test_data)
            since = 0
        else:
            since += 1
        if since >= patience:
            break
    return best_auc, best_ap


# ── Context ablation ─────────────────────────────────────────────────────────────

def run_context(year, plot, period, rel_lookup, epochs=300, lr=0.01,
                patience=40, verbose=True, seed=RANDOM_STATE) -> Optional[Dict]:
    label = f"{year}_{plot}_{period}"
    data, node_ids = load_context_graph(year, plot, period)
    n_undirected = data.edge_index.shape[1] // 2
    if n_undirected < MIN_EDGES:
        if verbose:
            print(f"[skip] {label}: only {n_undirected} edges")
        return None

    device = get_device()
    R, coverage = build_relatedness_matrix(node_ids, rel_lookup)
    R = R.to(device)

    # One split shared by both models for a fair comparison
    torch.manual_seed(seed)
    splitter = RandomLinkSplit(num_val=0.1, num_test=0.2, is_undirected=True,
                               add_negative_train_samples=True, neg_sampling_ratio=1.0)
    train_data, val_data, test_data = splitter(data)
    node_idx = torch.arange(data.num_nodes, device=device)
    for d in (train_data, val_data, test_data):
        d.x = d.x.to(device); d.edge_index = d.edge_index.to(device)
        d.edge_label_index = d.edge_label_index.to(device)
        d.edge_label = d.edge_label.to(device)

    topo_auc, topo_ap = _train_eval(data, train_data, val_data, test_data, R,
                                    node_idx, device, False, epochs, lr, patience, seed)
    rel_auc, rel_ap = _train_eval(data, train_data, val_data, test_data, R,
                                  node_idx, device, True, epochs, lr, patience, seed)

    result = {
        "year": year, "plot": plot, "period": period,
        "n_nodes": data.num_nodes, "n_edges": n_undirected,
        "rel_coverage": round(coverage, 3),
        "topo_auc": round(topo_auc, 4), "rel_auc": round(rel_auc, 4),
        "delta_auc": round(rel_auc - topo_auc, 4),
        "topo_ap": round(topo_ap, 4), "rel_ap": round(rel_ap, 4),
    }
    if verbose:
        print(f"[{label}] n={data.num_nodes} cov={coverage:.2f} | "
              f"topo AUC={topo_auc:.3f} | +rel AUC={rel_auc:.3f} | "
              f"Δ={rel_auc-topo_auc:+.3f}")
    return result


def run_all(years=None, plots=None, periods=None, **kw) -> pd.DataFrame:
    years = years or YEARS
    plots = plots or PLOTS
    periods = periods or PERIODS

    rel = load_relatedness()
    rel_lookup = {frozenset((a, b)): w for a, b, w in
                  zip(rel.bird_a, rel.bird_b, rel.wang)}
    print(f"relatedness lookup: {len(rel_lookup)} dyads, {len(set(rel.bird_a)|set(rel.bird_b))} birds\n")

    rows = []
    for year in years:
        for plot in plots:
            for period in periods:
                try:
                    r = run_context(year, plot, period, rel_lookup, **kw)
                    if r:
                        rows.append(r)
                except FileNotFoundError:
                    continue
                except Exception as e:
                    print(f"[error] {year}_{plot}_{period}: {e}")

    df = pd.DataFrame(rows)
    if not df.empty:
        ML_OUT.mkdir(parents=True, exist_ok=True)
        out = ML_OUT / "gnn_relatedness_ablation.csv"
        df.to_csv(out, index=False)
        print(f"\nSaved → {out}  ({len(df)} contexts)")
        print(f"Mean topo AUC={df.topo_auc.mean():.3f}  "
              f"mean +rel AUC={df.rel_auc.mean():.3f}  "
              f"mean Δ={df.delta_auc.mean():+.4f}")
        pos = (df.delta_auc > 0).sum()
        print(f"Relatedness helped in {pos}/{len(df)} contexts")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GNN kinship ablation")
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--plot", type=str, default=None)
    ap.add_argument("--period", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    run_all(
        years=[args.year] if args.year else None,
        plots=[args.plot] if args.plot else None,
        periods=[args.period] if args.period else None,
        epochs=args.epochs,
    )
