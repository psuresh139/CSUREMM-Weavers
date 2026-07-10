"""
BirdColonyEnv — Foraging / Roosting RL Environment
====================================================
Models a weaver bird choosing a nest site (or roost partner) as a
reward-driven decision process.

The question being modelled
----------------------------
Can an agent learn *which nest to visit*, given its current network
state, in a way that maximises social association strength with
the birds already at that nest?

Framing
-------
  State   : current bird's network features (degree, strength, eigenvector,
             betweenness, clustering) + its node2vec embedding  → 1-D vector
  Action  : discrete — choose one of N available nest sites (indexed 0…N-1)
  Reward  : mean SRI weight of edges between this bird and the birds
             already at the chosen nest on this timestep
  Episode : T steps (each step = one nest visit decision)

This is a scaffold: the core Gym interface is implemented and the
reward function uses real SRI data.  Agent training (DQN, PPO, etc.)
is intentionally left to the experimenter.

Interface
---------
    env = BirdColonyEnv.from_context(year=2016, plot="SPRA", period="daytime")
    obs, info = env.reset()
    for _ in range(100):
        action = env.action_space.sample()   # random policy
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

Dependencies
------------
    pip install gymnasium numpy pandas
    (gymnasium is the maintained fork of gym)
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    raise ImportError("pip install gymnasium")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import OUTPUT_ROOT, RL_MAX_STEPS, RL_REWARD_SCALE


class BirdColonyEnv(gym.Env):
    """
    Gymnasium environment modelling nest-site / roost-partner selection.

    Parameters
    ----------
    edges_df     : SRI edge list (id1, id2, association)
    nodes_df     : per-bird graph metrics (output of graph_construction)
    embeddings_df: node2vec embeddings (optional; zeros used if None)
    max_steps    : episode length
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        edges_df: pd.DataFrame,
        nodes_df: pd.DataFrame,
        embeddings_df: Optional[pd.DataFrame] = None,
        max_steps: int = RL_MAX_STEPS,
        seed: Optional[int] = None,
    ):
        super().__init__()

        # ── Build internal data structures ──────────────────────────────────
        self._edges    = edges_df.copy()
        self._nodes    = nodes_df.copy().set_index("bird_id")
        self._birds    = sorted(self._nodes.index.tolist())
        self._n_birds  = len(self._birds)
        self._bird2idx = {b: i for i, b in enumerate(self._birds)}

        # Nest sites = unique nests mentioned in the edge context
        # We proxy nests as unique bird identities (each bird is tied to a nest)
        self._nests    = self._birds          # actions = choosing a bird's nest
        self._n_nests  = self._n_birds

        # SRI adjacency dict:  bird → {neighbour: weight}
        self._sri: Dict[str, Dict[str, float]] = {b: {} for b in self._birds}
        for _, row in edges_df.iterrows():
            a, b, w = str(row["id1"]), str(row["id2"]), float(row["association"])
            self._sri.setdefault(a, {})[b] = w
            self._sri.setdefault(b, {})[a] = w

        # ── Feature dimensions ────────────────────────────────────────────────
        graph_cols = ["degree", "strength", "betweenness", "eigenvector", "clustering"]
        self._graph_cols = [c for c in graph_cols if c in self._nodes.columns]

        if embeddings_df is not None:
            emb = embeddings_df.set_index("bird_id")
            self._emb_cols = [c for c in emb.columns if c.startswith("dim_")]
            self._emb = emb[self._emb_cols]
        else:
            self._emb_cols = []
            self._emb = None

        self._obs_dim = len(self._graph_cols) + len(self._emb_cols)

        # ── Gym spaces ────────────────────────────────────────────────────────
        self.action_space      = spaces.Discrete(self._n_nests)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

        # ── Episode state ─────────────────────────────────────────────────────
        self._max_steps   = max_steps
        self._current_bird: Optional[str] = None
        self._step_count  = 0
        self._rng         = np.random.default_rng(seed)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _obs(self, bird: str) -> np.ndarray:
        """Build observation vector for the current bird."""
        graph_feats = np.zeros(len(self._graph_cols), dtype=np.float32)
        if bird in self._nodes.index:
            row = self._nodes.loc[bird]
            graph_feats = row[self._graph_cols].fillna(0.0).values.astype(np.float32)

        emb_feats = np.zeros(len(self._emb_cols), dtype=np.float32)
        if self._emb is not None and bird in self._emb.index:
            emb_feats = self._emb.loc[bird].fillna(0.0).values.astype(np.float32)

        return np.concatenate([graph_feats, emb_feats])

    def _reward(self, bird: str, nest_bird: str) -> float:
        """
        Reward = SRI between current bird and the bird at the chosen nest.
        If the same bird is chosen, reward is 0 (can't associate with self).
        Scaled by RL_REWARD_SCALE.
        """
        if bird == nest_bird:
            return 0.0
        sri = self._sri.get(bird, {}).get(nest_bird, 0.0)
        return float(sri) * RL_REWARD_SCALE

    # ── Gym API ────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._current_bird = self._rng.choice(self._birds)
        self._step_count   = 0
        obs = self._obs(self._current_bird)
        return obs, {"bird": self._current_bird}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take one step: current bird visits the nest indexed by action.

        Returns:
            obs         — observation for the next timestep
            reward      — SRI-based reward
            terminated  — always False (no natural terminal state)
            truncated   — True when max_steps reached
            info        — {bird, chosen_nest, reward}
        """
        nest_bird = self._birds[action]
        reward    = self._reward(self._current_bird, nest_bird)

        # Transition: the bird at the chosen nest becomes the focal bird
        # (models following behaviour — bird goes to where its associate is)
        self._current_bird = nest_bird
        self._step_count  += 1

        obs       = self._obs(self._current_bird)
        truncated = self._step_count >= self._max_steps
        info      = {
            "bird":        self._current_bird,
            "chosen_nest": nest_bird,
            "reward":      reward,
        }
        return obs, reward, False, truncated, info

    def render(self):
        print(
            f"Step {self._step_count}/{self._max_steps} | "
            f"Bird: {self._current_bird} | "
            f"Degree: {self._nodes.loc[self._current_bird, 'degree'] if self._current_bird in self._nodes.index else 'N/A'}"
        )

    # ── Convenience constructors ───────────────────────────────────────────────

    @classmethod
    def from_context(
        cls,
        year: int,
        plot: str,
        period: str,
        max_steps: int = RL_MAX_STEPS,
        seed: Optional[int] = None,
    ) -> "BirdColonyEnv":
        """Load data from a pipeline output directory and construct the env."""
        label    = f"{year}_{plot}_{period}"
        ctx_dir  = OUTPUT_ROOT / label

        edge_path = ctx_dir / f"edges_{label}.csv"
        node_path = ctx_dir / f"nodes_{label}.csv"
        emb_path  = ctx_dir / f"embeddings_{label}.csv"

        if not edge_path.exists() or not node_path.exists():
            raise FileNotFoundError(
                f"Pipeline outputs not found in {ctx_dir}. "
                "Run pipeline/run_pipeline.py first."
            )

        edges = pd.read_csv(edge_path)
        nodes = pd.read_csv(node_path)
        embs  = pd.read_csv(emb_path) if emb_path.exists() else None

        return cls(edges, nodes, embs, max_steps=max_steps, seed=seed)

    @classmethod
    def from_dataframes(
        cls,
        edges: pd.DataFrame,
        nodes: pd.DataFrame,
        embeddings: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> "BirdColonyEnv":
        return cls(edges, nodes, embeddings, **kwargs)
