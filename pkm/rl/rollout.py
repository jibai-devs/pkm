"""Self-play game driver: runs battles via battle_start/battle_select and
collects encoded decisions for both players."""

import random
from dataclasses import dataclass

from kaggle_environments.envs.cabt.cg.game import (
    battle_finish,
    battle_select,
    battle_start,
)

from pkm.types.obs import Observation

from .encoder import EncodedDecision, encode_decision, prize_potential
from .model import PolicyValueNet

MAX_DECISIONS = 3000


class TorchPolicy:
    """Wraps a PolicyValueNet into a per-observation actor."""

    def __init__(
        self, model: PolicyValueNet, greedy: bool = False, temperature: float = 1.0
    ):
        self.model = model
        self.greedy = greedy
        self.temperature = temperature

    def act(self, obs: dict, collect: bool) -> tuple[list[int], EncodedDecision | None]:
        parsed = Observation.model_validate(obs)
        sel = parsed.select
        assert sel is not None
        n = len(sel.option)
        # forced decision: nothing to learn, don't run the network
        if n == 1 and sel.minCount >= 1:
            return [0], None
        if n == sel.minCount == sel.maxCount:
            return list(range(n)), None

        d = encode_decision(parsed)
        res = self.model.act(d, greedy=self.greedy, temperature=self.temperature)
        if not collect:
            return res.picks, None
        d.picks = res.picks
        d.stopped = res.stopped
        d.logprob = res.logprob
        d.value = res.value
        d.potential = prize_potential(parsed)
        return res.picks, d


class RandomPolicy:
    def act(self, obs: dict, collect: bool) -> tuple[list[int], None]:
        sel = obs["select"]
        return random.sample(range(len(sel["option"])), sel["maxCount"]), None


@dataclass
class GameResult:
    trajectories: tuple[list[EncodedDecision], list[EncodedDecision]]
    rewards: tuple[float, float]  # terminal reward per player index
    decisions: int
    turns: int


def play_game(
    policies,
    decks: tuple[list[int], list[int]],
    collect: tuple[bool, bool] = (True, True),
) -> GameResult:
    """Play one full game; returns per-player trajectories and rewards."""
    obs, start = battle_start(list(decks[0]), list(decks[1]))
    if obs is None:
        raise RuntimeError(f"battle_start failed: errorPlayer={start.errorPlayer}")

    trajectories: tuple[list[EncodedDecision], list[EncodedDecision]] = ([], [])
    count = 0
    try:
        while obs["current"]["result"] < 0 and count < MAX_DECISIONS:
            p = obs["current"]["yourIndex"]
            picks, record = policies[p].act(obs, collect=collect[p])
            if record is not None:
                trajectories[p].append(record)
            obs = battle_select(picks)
            count += 1
        result = obs["current"]["result"]
        turns = obs["current"]["turn"]
    finally:
        battle_finish()

    if result == 0:
        rewards = (1.0, -1.0)
    elif result == 1:
        rewards = (-1.0, 1.0)
    else:  # draw or decision cap hit
        rewards = (0.0, 0.0)
    return GameResult(
        trajectories=trajectories, rewards=rewards, decisions=count, turns=turns
    )
