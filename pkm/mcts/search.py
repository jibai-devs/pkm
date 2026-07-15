"""Determinized IS-MCTS with PUCT selection, policy-network priors and
value-network leaf evaluation, running on the cabt search API.

For each of D determinizations we build a separate tree with search_begin and
advance it with search_step; root visit counts are summed across trees and the
most-visited action is played. Chance events re-randomize inside the engine on
every search_step, so an edge's child is a sample; we keep the first sampled
child per edge (a standard approximation).
"""

import math
import random

import numpy as np

from pkm.types.obs import Observation
from pkm.rl.encoder import encode_decision
from pkm.rl.numpy_policy import NumpyPolicy
from pkm.search import search_begin, search_end, search_step

from .determinize import sample_determinization

_MAX_FORCED_SKIP = 100


def _forced_picks(sel: dict) -> list[int] | None:
    """Return the forced selection if the decision offers no real choice."""
    n = len(sel["option"])
    if n == 1 and sel["minCount"] >= 1:
        return [0]
    if n == sel["minCount"] == sel["maxCount"]:
        return list(range(n))
    return None


class _Node:
    __slots__ = (
        "search_id",
        "obs",
        "player",
        "terminal_v0",
        "actions",
        "priors",
        "visits",
        "totals",
        "children",
        "expanded",
    )

    def __init__(self, state: dict):
        # skip forward through forced decisions
        for _ in range(_MAX_FORCED_SKIP):
            obs = state["observation"]
            if obs["current"]["result"] >= 0:
                break
            forced = _forced_picks(obs["select"])
            if forced is None:
                break
            state = search_step(state["searchId"], forced)

        self.search_id = state["searchId"]
        self.obs = state["observation"]
        result = self.obs["current"]["result"]
        self.player = self.obs["current"]["yourIndex"]
        self.terminal_v0: float | None = None
        if result >= 0:
            self.terminal_v0 = 1.0 if result == 0 else -1.0 if result == 1 else 0.0
        self.expanded = False
        self.actions: list[tuple[int, ...]] = []
        self.priors: np.ndarray | None = None
        self.visits: np.ndarray | None = None
        self.totals: np.ndarray | None = None
        self.children: dict[int, _Node] = {}


class MCTS:
    def __init__(
        self,
        policy: NumpyPolicy,
        n_determinizations: int = 2,
        n_simulations: int = 32,
        c_puct: float = 1.5,
        max_candidates: int = 6,
        dirichlet_eps: float = 0.0,
        dirichlet_alpha: float = 0.3,
        rng: random.Random | None = None,
    ):
        self.policy = policy
        self.n_determinizations = n_determinizations
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.max_candidates = max_candidates
        self.dirichlet_eps = dirichlet_eps
        self.dirichlet_alpha = dirichlet_alpha
        self.rng = rng or random.Random()
        self.np_rng = np.random.default_rng(self.rng.randrange(2**31))

    # --- expansion ---

    def _expand(self, node: _Node) -> float:
        """Enumerate actions + priors, return the value from node.player's view."""
        d = encode_decision(Observation.model_validate(node.obs))
        sel = node.obs["select"]
        n = len(sel["option"])
        value = self.policy.value(d)

        if sel["maxCount"] == 1:
            actions: list[tuple[int, ...]] = [(i,) for i in range(n)]
            priors = self.policy.priors(d)
            if sel["minCount"] == 0:
                # picking nothing is legal: give STOP a uniform-ish share
                actions.append(())
                priors = np.append(priors * (n / (n + 1.0)), 1.0 / (n + 1.0))
        else:
            # multi-pick: candidate sequences sampled from the policy
            cand: dict[tuple[int, ...], float] = {}
            greedy = tuple(self.policy.act_greedy(d))
            cand[greedy] = 1.0
            for _ in range(self.max_candidates - 1):
                picks, prob = self.policy.sample_picks(d, self.np_rng)
                key = tuple(picks)
                cand[key] = max(cand.get(key, 0.0), prob)
            actions = list(cand.keys())
            priors = np.array([cand[a] for a in actions], dtype=np.float64)
            priors /= priors.sum()

        node.actions = actions
        node.priors = np.asarray(priors, dtype=np.float64)
        node.visits = np.zeros(len(actions), dtype=np.int64)
        node.totals = np.zeros(len(actions), dtype=np.float64)
        node.expanded = True
        return value

    def _puct_index(self, node: _Node) -> int:
        sqrt_total = math.sqrt(node.visits.sum() + 1)
        q = np.where(node.visits > 0, node.totals / np.maximum(node.visits, 1), 0.0)
        u = self.c_puct * node.priors * sqrt_total / (1.0 + node.visits)
        return int(np.argmax(q + u))

    def _simulate(self, root: _Node) -> None:
        node = root
        path: list[tuple[_Node, int]] = []
        while True:
            if node.terminal_v0 is not None:
                v0 = node.terminal_v0
                break
            if not node.expanded:
                v = self._expand(node)
                v0 = v if node.player == 0 else -v
                break
            idx = self._puct_index(node)
            path.append((node, idx))
            child = node.children.get(idx)
            if child is None:
                child = _Node(search_step(node.search_id, list(node.actions[idx])))
                node.children[idx] = child
            node = child
        for parent, idx in path:
            parent.visits[idx] += 1
            parent.totals[idx] += v0 if parent.player == 0 else -v0

    # --- public API ---

    def choose(
        self,
        obs: dict,
        my_decklist: list[int],
        opp_decklist: list[int],
    ) -> tuple[list[int], dict[tuple[int, ...], int]]:
        """Run search for a real decision; returns (picks, aggregated visit counts)."""
        agg: dict[tuple[int, ...], int] = {}
        try:
            for _ in range(self.n_determinizations):
                det = sample_determinization(obs, my_decklist, opp_decklist, self.rng)
                root = _Node(search_begin(obs, **det))
                if root.terminal_v0 is not None:
                    continue
                self._expand(root)
                if self.dirichlet_eps > 0 and len(root.actions) > 1:
                    noise = self.np_rng.dirichlet(
                        [self.dirichlet_alpha] * len(root.actions)
                    )
                    root.priors = (
                        1 - self.dirichlet_eps
                    ) * root.priors + self.dirichlet_eps * noise
                for _ in range(self.n_simulations):
                    self._simulate(root)
                for a, n in zip(root.actions, root.visits):
                    agg[a] = agg.get(a, 0) + int(n)
        finally:
            search_end()

        if not agg:  # all determinizations degenerate; fall back to policy
            return self.policy.select(obs), {}
        best = max(agg.items(), key=lambda kv: kv[1])[0]
        return list(best), agg
