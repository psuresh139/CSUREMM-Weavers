# Weaver Bird Social Networks

Graph-based analysis and machine learning on the social structure of colonial
weaver bird colonies, built from automated logger-detection data (2013–2017).
Networks are reconstructed from unstructured, second-resolution detection logs;
the analysis asks what shapes who-associates-with-whom, whether social role is
a stable individual trait, and whether it carries a fitness consequence.

Based on the study system of Yiru Cheng's PhD dissertation. This directory
(`birds_new/`) is the clean, reproducible rewrite of the original exploratory
R/Python work.

---

## Table of contents

- [The scientific story](#the-scientific-story)
- [Data](#data)
- [Pipeline](#pipeline)
- [Machine learning & analysis](#machine-learning--analysis)
- [Directory layout](#directory-layout)
- [Installation (Apple Silicon)](#installation-apple-silicon)
- [Usage](#usage)
- [Key findings](#key-findings)
- [Data availability](#data-availability)
- [Caveats & limitations](#caveats--limitations)

---

## The scientific story

Three questions, answered end-to-end and reproducibly:

1. **What drives association?** *Structure, not identity or kinship.* Predicting
   who associates with whom is dominated by triadic closure (friends-of-friends).
   A simple heuristic reaches ~0.95 AUC, and neither a GNN, nor sex, nor genetic
   relatedness improves on it — a set of well-controlled negative results.
2. **Is social role a stable trait?** *Yes for embeddedness, no for brokerage.*
   A bird's degree/strength percentile is predictable a year ahead (a model beats
   the naive persistence baseline), while betweenness (brokerage) is not — being
   a "connector" is circumstantial, not dispositional.
3. **Does role predict fitness?** *Modestly, and nonlinearly.* Locally-embedded
   birds (high clustering, low betweenness) breed somewhat better; centrality
   predicts breeding success at ROC-AUC ≈ 0.73–0.81 above a detection-effort
   baseline, but there is no simple "more central → more offspring" law.

See [Key findings](#key-findings) for numbers.

---

## Data

Automated loggers record each bird's presence at colony nest sites at
1-second resolution. A *colony* = birds nesting in the same tree (known from
field observation, not derived from the graph).

| Dimension | Values |
|-----------|--------|
| Plots     | MSTO, SPRA, LLOD |
| Years     | 2013–2017 |
| Periods   | daytime (06:30–18:00), nighttime (18:00–06:30) |

A full grid is 30 *contexts* (year × plot × period); 26 have enough data to
build a graph.

**This repository contains no study data** — neither the raw logger files nor
any per-individual derived data (association graphs, embeddings, individual
breeding records). Only code, methods, and aggregate, non-identifying result
tables are included. See [Data availability](#data-availability).

The pipeline expects the raw inputs to be present locally at the paths in
`config.py` (`../Birds/data/`):

- `observation_logs/combined_weaver_log.xlsx` — logger detections
- `observation_logs/weaver_sex.xlsx` — sex labels
- `observation_logs/daily_breeding.xlsx` — per-nest breeding outcomes
- `observation_logs/c_2012-2017_weaver_banding.xlsx` — banding / colony
- `behavior_social/Relateness_weaver.csv` — pairwise genetic relatedness

**Data-quality filter (important):** tag `K00000` is a hardware test tag, not a
real bird. It is excluded everywhere via `TEST_BIRD_IDS` in `config.py`.

---

## Pipeline

Raw logs become weighted social graphs in four stages (`pipeline/`):

```
Logger detections
    │  gmm_events.py        Gaussian-mixture event detection — cluster raw
    │                       detections into discrete presence "events"
    ▼
Co-presence events
    │  sri_construction.py  Simple Ratio Index (SRI) — association strength
    │                       for each dyad from shared events
    ▼
SRI-weighted edge list
    │  graph_construction.py  NetworkX graph + node/network metrics +
    │                         Louvain communities
    ▼
Per-context graph outputs (workspace/graphs/{context}/)
```

Each context directory holds:

| File | Contents |
|------|----------|
| `nodes_{ctx}.csv` | per-bird: degree, strength, betweenness, eigenvector, clustering, network_position, community |
| `edges_{ctx}.csv` | id1, id2, association (SRI weight) |
| `stats_{ctx}.csv` | network-level: density, diameter, avg clustering, … |
| `community_info_{ctx}.csv` | Louvain communities, modularity |
| `embeddings_{ctx}.csv` | node2vec embedding per bird (added by the embeddings step) |

**A biological note on graph structure:** all multi-component graphs are
nighttime contexts. Daytime birds mix freely (fully connected foraging network);
at night they roost in physically separate nest clusters that may never share a
detection, producing isolated components. This is signal, not artefact.

---

## Machine learning & analysis

All ML lives in `ml/`. The GNN modules run on the **Apple GPU (MPS)** when
available and fall back to CPU otherwise.

| Module | Question | Method |
|--------|----------|--------|
| `embeddings/node2vec_pipeline.py` | node embeddings | node2vec (gensim, ARM-native) |
| `ml/clustering.py` | do embeddings recover sex? | KMeans / HDBSCAN + ARI/NMI |
| `ml/classification.py` | is an attribute predictable from structure? | LogReg / RF / SVM, stratified CV |
| `ml/gnn_link_prediction.py` | can we predict who associates? | attributed GraphSAGE + Adamic-Adar baseline |
| `ml/gnn_relatedness_ablation.py` | does kinship help, beyond topology? | same GNN, with/without a relatedness feature |
| `ml/role_stability.py` | is social role a stable, predictable trait? | cross-year RF vs persistence baseline, grouped CV |
| `ml/breeding_labels.py` | bird → fledge/hatch outcome | logger↔breeding nest join |
| `ml/centrality_fitness.py` | does centrality predict breeding success? | RF vs detection-effort baseline |
| `analysis/day_night_comparison.py` | do day/night networks differ? | paired structural stats, Spearman |
| `analysis/sex_assortative.py` | do dyads associate assortatively by sex? | permutation tests |
| `ml/relatedness_sri.py` | does relatedness predict SRI? | OLS + Mantel test |
| `ml/rl/environment.py` | (scaffold) foraging/roosting RL | Gymnasium environment |

**Method note — leakage control.** GNN link prediction message-passes over
*training edges only* (via `RandomLinkSplit`); cross-year and fitness models use
`GroupKFold` grouped by `bird_id` so no individual appears in both train and
test; centrality→fitness models include a detection-effort baseline to guard
against the confound that heavily-logged birds look both more central and more
likely to be matched to a fledging nest.

---

## Directory layout

```
birds_new/
├── config.py                       all paths, parameters, K00000 filter
├── requirements.txt
├── README.md
├── pipeline/
│   ├── gmm_events.py               GMM event detection
│   ├── sri_construction.py         SRI association calculation
│   ├── graph_construction.py       NetworkX graph + metrics
│   └── run_pipeline.py             orchestrator (CLI)
├── embeddings/
│   └── node2vec_pipeline.py        ARM-native node2vec
├── ml/
│   ├── features.py                 feature assembly + label loaders
│   ├── classification.py           supervised attribute prediction
│   ├── clustering.py               unsupervised structure discovery
│   ├── gnn_link_prediction.py      GraphSAGE link prediction (MPS)
│   ├── gnn_relatedness_ablation.py kinship ablation (MPS)
│   ├── role_stability.py           cross-year role prediction
│   ├── breeding_labels.py          logger↔breeding join
│   ├── centrality_fitness.py       centrality → breeding success
│   ├── relatedness_sri.py          relatedness vs association
│   ├── day_night_duality.py        per-bird day/night role comparison
│   └── rl/environment.py           RL scaffold (Gymnasium)
├── analysis/
│   ├── day_night_comparison.py
│   └── sex_assortative.py
├── notebooks/                      01 pipeline → 04 ML walkthrough
├── results/                        aggregate result tables (tracked)
└── workspace/                      per-bird outputs, regenerated locally (gitignored)
```

`results/` holds the aggregate, per-context result tables the analyses produce
(model AUCs, correlations, test statistics) — no individual birds. Running the
pipeline locally regenerates the full per-bird graphs and embeddings into
`workspace/`, which is gitignored and never published.

---

## Installation (Apple Silicon)

Runs natively on Apple Silicon — **no special conda environment is needed.**
Modern PyTorch ships a working MPS (Apple GPU) backend, and PyTorch Geometric
≥ 2.5 vendors the sparse extensions, so the historically painful
`torch-sparse` / `torch-scatter` compilation step is gone. (The older
`pytorch::cpuonly` "for ARM64 compatibility" advice is obsolete — ignore it.)

```bash
cd birds_new
pip install -r requirements.txt
```

Verify the GPU backend:

```python
import torch
print(torch.backends.mps.is_available())   # True on Apple Silicon
```

The GNN code selects `mps` automatically when present, else CPU.

---

## Usage

```bash
cd birds_new

# 1. Build all graphs from the logger data
python pipeline/run_pipeline.py                    # all contexts
python pipeline/run_pipeline.py --year 2016 --plot SPRA --period daytime

# 2. Node embeddings
python embeddings/node2vec_pipeline.py             # all contexts

# 3. Analyses (each writes to workspace/ml/)
python ml/gnn_link_prediction.py                   # who associates? (MPS)
python ml/gnn_relatedness_ablation.py              # does kinship help?
python ml/role_stability.py --period daytime       # is role a stable trait?
python ml/breeding_labels.py                        # build fledge labels
python ml/centrality_fitness.py --years 2015 2016  # centrality → fitness
```

Most scripts accept `--year / --plot / --period` to run a single context.

---

## Key findings

**1 · Association is structural, not identity- or kin-driven.**

| Test | Result | Reading |
|------|--------|---------|
| Sex from network | ROC-AUC ≈ 0.50 | not encoded in structure |
| GNN link prediction | 0.935 vs 0.954 (Adamic-Adar) | GNN can't beat friends-of-friends |
| Kinship ablation | ΔAUC = +0.0004 (helped 11/24) | relatedness adds nothing over topology |

**2 · Social embeddedness is a stable trait; brokerage is not.**
Predicting next-year degree percentile (grouped CV, vs persistence baseline):

| | Daytime | Nighttime |
|--|--|--|
| RandomForest R² | **+0.17** | **+0.24** |
| Persistence baseline R² | −0.38 | −0.49 |
| Shuffled null R² | −1.08 | −1.09 |

Degree/strength persist across years (ρ ≈ 0.22–0.27, p < 1e-5); betweenness does
not (ρ ≈ −0.045, n.s.).

**3 · Centrality has a modest, nonlinear fitness signature.**
Predicting fledging (grouped CV by bird):

| Model | fledged AUC |
|-------|-------------|
| Detection-effort only | 0.61 |
| Centrality (linear) | 0.58 |
| Centrality (RandomForest) | **0.73** |
| Centrality + effort | **0.81** |
| Shuffled null | 0.53 |

Direction: higher clustering and *lower* betweenness weakly favour breeding —
locally-embedded birds over brokers.

The tables backing every number above are in [`results/`](results/).

---

## Data availability

The underlying weaver study data is **not distributed with this repository**.
This includes both the raw field data (logger detections, banding, breeding,
relatedness) and any per-individual derived data (association edge lists, node
metric tables, node2vec/GNN embeddings, individual breeding labels). These are
generated locally into the gitignored `workspace/` directory and never
committed.

What *is* included is the analysis code and the **aggregate result tables** in
`results/` — per-context model scores, correlations, and test statistics that
contain no individual-bird identifiers.

The study data belongs to the original field project (based on the study
system of Yiru Cheng's PhD dissertation) and is available only on request from
the data custodians. To reproduce the full results, place the raw files at the
paths in `config.py` and run the pipeline.

---

## Caveats & limitations

- **Breeding labels are a proxy.** Each bird is assigned its dominant
  breeding-season *detection* nest, not a certified individual nest; loggers are
  not on every nest.
- **Year heterogeneity.** Fledge rate varies sharply by year (2014 ≈ 0 %, 2017 ≈
  4 %, 2015–16 ≈ 38 %). Fitness analyses lean on the balanced 2015–16 years.
- **Nonlinear fitness signal.** The linear centrality model is near-chance while
  the RF is strong, so the fitness effect is interaction-driven; the RF's use of
  cross-context base-rate structure has not been fully ruled out.
- **Small contexts.** Some year/plot combinations have very few birds; per-context
  results from those are noisy and should be read at the pooled level.
