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

from .encoder import (
    EncodedDecision,
    budew_first_turn_attack_bonus,
    dragapult_ex_attack_bonus,
    dreepy_energy_spread_penalty,
    encode_decision,
    energy_overattach_penalty,
    prize_potential,
    wrong_type_energy_penalty,
)
from .model import PolicyValueNet
from .ppo import compute_returns

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
        d.energy_penalty = energy_overattach_penalty(parsed, res.picks)
        d.budew_bonus = budew_first_turn_attack_bonus(parsed, res.picks)
        d.wrong_type_energy_penalty = wrong_type_energy_penalty(parsed, res.picks)
        d.dragapult_attack_bonus = dragapult_ex_attack_bonus(parsed, res.picks)
        d.dreepy_spread_penalty = dreepy_energy_spread_penalty(parsed, res.picks)
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


@dataclass
class GameSpec:
    """One game's pre-decided matchup, so rollout (sequential or parallel)
    and aggregation always agree on who played and what to count."""

    opponent_state: dict | None  # None = mirror self-play (both sides = current)
    side: int  # which side "current" plays if opponent_state is set; -1 if mirror
    collect: tuple[bool, bool]


def make_game_specs(
    games_per_iter: int, pool: list[dict], pool_prob: float, rng: random.Random
) -> list[GameSpec]:
    """Decide, up front, the full iteration's matchups — same logic regardless
    of whether rollout runs sequentially or across worker processes."""
    specs = []
    for _ in range(games_per_iter):
        if rng.random() < pool_prob and len(pool) > 1:
            opponent_state = rng.choice(pool[:-1])
            side = rng.randint(0, 1)
            collect = (side == 0, side == 1)
        else:
            opponent_state = None
            side = -1
            collect = (True, True)
        specs.append(GameSpec(opponent_state, side, collect))
    return specs


def play_one(
    current_model: PolicyValueNet,
    opponent_model: PolicyValueNet,
    deck: list[int],
    spec: GameSpec,
) -> GameResult:
    """Play one game per `spec`, reusing `opponent_model` as scratch space for
    the pooled-opponent case (avoids rebuilding a fresh module every game)."""
    if spec.opponent_state is None:
        cur = TorchPolicy(current_model)
        policies = (cur, cur)
    else:
        opponent_model.load_state_dict(spec.opponent_state)
        opponent_model.eval()
        opp = TorchPolicy(opponent_model)
        cur = TorchPolicy(current_model)
        policies = (cur, opp) if spec.side == 0 else (opp, cur)
    return play_game(policies, (deck, deck), collect=spec.collect)


def aggregate_result(
    spec: GameSpec,
    result: GameResult,
    data: list,
    gamma: float,
    lam: float,
    shaping_coef: float,
    energy_penalty_coef: float = 0.0,
    budew_bonus_coef: float = 0.0,
    wrong_type_penalty_coef: float = 0.0,
    dragapult_bonus_coef: float = 0.0,
    dreepy_spread_coef: float = 0.0,
) -> tuple[int, int, int]:
    """Extend `data` with this game's collected trajectories and return the
    (win, loss, draw) increment for `current` — same counting rule the
    sequential loop always used, factored out so the parallel path matches."""
    w = losses = d = 0
    for p in range(2):
        if not spec.collect[p]:
            continue
        compute_returns(
            result.trajectories[p],
            result.rewards[p],
            gamma=gamma,
            lam=lam,
            shaping_coef=shaping_coef,
            energy_penalty_coef=energy_penalty_coef,
            budew_bonus_coef=budew_bonus_coef,
            wrong_type_penalty_coef=wrong_type_penalty_coef,
            dragapult_bonus_coef=dragapult_bonus_coef,
            dreepy_spread_coef=dreepy_spread_coef,
        )
        data.extend(result.trajectories[p])
        if spec.side == -1 and p == 1:
            continue  # count mirror games once
        r = result.rewards[p if spec.side == -1 else spec.side]
        w, losses, d = w + (r > 0), losses + (r < 0), d + (r == 0)
    return w, losses, d
