"""
Central configuration for the weaver bird social network project.
All paths, parameters, and constants live here.
"""

from pathlib import Path

# ── Root paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT    = PROJECT_ROOT.parent / "Birds" / "data"
OUTPUT_ROOT  = PROJECT_ROOT / "workspace" / "graphs"

LOGGER_FILE       = DATA_ROOT / "observation_logs" / "combined_weaver_log.xlsx"
SEX_FILE          = DATA_ROOT / "observation_logs" / "weaver_sex.xlsx"
BREEDING_FILE     = DATA_ROOT / "observation_logs" / "daily_breeding.xlsx"
BANDING_FILE      = DATA_ROOT / "observation_logs" / "c_2012-2017_weaver_banding.xlsx"
SAMPLING_FILE     = DATA_ROOT / "observation_logs" / "sampling_effort" / "2013-2017_weaver_sampling_duration.xlsx"
RELATEDNESS_FILE  = DATA_ROOT / "behavior_social" / "Relateness_weaver.csv"

# ── Data quality filters ──────────────────────────────────────────────────────
TEST_BIRD_IDS = {"K00000"}   # hardware test tag — not a real bird, must be excluded

# ── Study scope ───────────────────────────────────────────────────────────────
YEARS  = [2013, 2014, 2015, 2016, 2017]
PLOTS  = ["MSTO", "SPRA", "LLOD"]
PERIODS = ["daytime", "nighttime"]

# ── Time windows (seconds since midnight) ────────────────────────────────────
DAYTIME_START   = 6 * 3600 + 30 * 60   # 06:30 → 23400 s
DAYTIME_END     = 18 * 3600             # 18:00 → 64800 s
NIGHTTIME_START = 18 * 3600             # 18:00 → 64800 s
NIGHTTIME_END   = 6 * 3600 + 30 * 60   # 06:30 → 23400 s  (wraps midnight)

# ── GMM event detection ───────────────────────────────────────────────────────
GMM_KMAX        = 8       # max components; optimal selected via BIC
GMM_REG_COVAR   = 1e-3
GMM_RANDOM_SEED = 0
GMM_MIN_GAP_S   = 600     # merge events within 10 min of each other
GMM_MIN_HITS    = 3       # min detections to form an event
GMM_MAX_EVENT_S = 2 * 3600  # drop events longer than 2 h (dissertation threshold)

# ── SRI construction ──────────────────────────────────────────────────────────
SRI_MIN_EVENTS_PER_BIRD = 3   # drop transient birds
SRI_MIN_DAYS_PER_BIRD   = 2
SRI_MIN_SHARED_DAYS     = 1   # min days co-present to keep a dyad
SRI_MIN_VALUE           = 0.0 # minimum SRI to keep edge

# ── Graph metrics ─────────────────────────────────────────────────────────────
CONNECTOR_TOP_PERCENTILE = 0.90  # top 10% betweenness = connector

# ── Node2Vec ──────────────────────────────────────────────────────────────────
N2V_DIMENSIONS  = 64    # 64 is plenty for graphs with 20–170 nodes
N2V_WALK_LENGTH = 20    # was 30
N2V_NUM_WALKS   = 30    # was 200 — biggest memory driver
N2V_WINDOW      = 10
N2V_P           = 1.0   # return parameter (1 = balanced BFS/DFS)
N2V_Q           = 1.0   # in-out parameter (1 = random walk)
N2V_WORKERS     = 1     # was 4 — avoids macOS multiprocessing spawn overhead

# ── ML ────────────────────────────────────────────────────────────────────────
RANDOM_STATE    = 42
CV_FOLDS        = 5
HDBSCAN_MIN_SAMPLES = 5
KMEANS_K_RANGE  = range(2, 9)

# ── RL environment ────────────────────────────────────────────────────────────
RL_MAX_STEPS    = 100
RL_REWARD_SCALE = 1.0
