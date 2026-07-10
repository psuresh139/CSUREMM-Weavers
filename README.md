# CSUREMM Weavers

Analysis of the social structure of colonial weaver bird colonies, built from
automated logger-detection data (2013–2017). Two generations of work live here:

| Directory | What it is |
|-----------|------------|
| [`python-graph-ml/`](python-graph-ml/) | The clean, reproducible graph-ML pipeline and analyses. **Start here.** See its own [README](python-graph-ml/README.md). |
| [`r-analysis/`](r-analysis/) | Earlier exploratory R scripts (spatial/temporal dyad analysis, MCMCglmm, connector-bird and supercolony network analysis). Kept for reference; not maintained. |

## The short version

Reconstructing weaver social networks from second-resolution detection logs and
asking three questions:

1. **What drives association?** Structure (triadic closure), not identity or kinship.
2. **Is social role a stable trait?** Yes for embeddedness, no for brokerage —
   a bird's centrality is predictable a year ahead, its brokerage is not.
3. **Does role predict fitness?** Modestly and nonlinearly — locally-embedded
   birds breed somewhat better than brokers.

Full method and numbers are in [`python-graph-ml/README.md`](python-graph-ml/README.md).

## Data availability

**No study data is included in this repository** — neither raw field data
(logger detections, banding, breeding, relatedness) nor any per-individual
derived data. Only analysis code and aggregate, non-identifying result tables
are tracked. The underlying data belongs to the original field project (based on
the study system of Yiru Cheng's PhD dissertation) and is available only on
request from the data custodians. To reproduce results, place the raw files at
the paths in `python-graph-ml/config.py` and run the pipeline.
