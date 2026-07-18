"""Self-play game driver: runs battles via battle_start/battle_select and
collects encoded decisions for both players."""

import random
from dataclasses import dataclass

from pkm.engine import (
    battle_finish,
    battle_select,
    battle_start,
)

from pkm.heuristics.context import GameContext
from pkm.heuristics.deck_tracker import DeckTracker
from pkm.types.obs import Observation

from .encoder import (
    EncodedDecision,
    budew_active_second_potential,
    budew_first_turn_attack_bonus,
    budew_turn_bench_setup_bonus,
    drakloak_backup_ready_bonus,
    dragapult_backup_potential,
    dragapult_ex_attack_bonus,
    dreepy_energy_spread_penalty,
    dreepy_evolve_bonus,
    dreepy_line_active_charge_bonus,
    dreepy_line_bench_charge_bonus,
    dreepy_line_field_potential,
    encode_decision,
    energy_overattach_penalty,
    phantom_dive_attack_bonus,
    prize_potential,
    wasted_resources_attack_penalty,
    wrong_type_energy_penalty,
    xerosic_machinations_bonus,
)
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

    def act(
        self, obs: dict, collect: bool, ctx: GameContext | None = None
    ) -> tuple[list[int], EncodedDecision | None]:
        parsed = Observation.model_validate(obs)
        sel = parsed.select
        assert sel is not None
        # forced decision: nothing to learn, don't run the network
        forced = sel.forced_picks()
        if forced is not None:
            return forced, None

        d = encode_decision(parsed, ctx)
        res = self.model.act(d, greedy=self.greedy, temperature=self.temperature)
        if ctx is not None:
            # Task 8: carry the belief forward for the *next* decision's
            # GLOBAL feature read (see pkm/rl/features.py) -- one decision
            # stale by construction, never recomputed inside a pure
            # feature function.
            ctx.archetype_belief = res.belief
        if not collect:
            return res.picks, None
        d.picks = res.picks
        d.stopped = res.stopped
        d.logprob = res.logprob
        d.value = res.value
        d.potential = prize_potential(parsed)
        d.board_setup_potential = dragapult_backup_potential(parsed)
        d.budew_setup_potential = budew_active_second_potential(parsed)
        d.dreepy_line_field_potential = dreepy_line_field_potential(parsed)
        d.energy_penalty = energy_overattach_penalty(parsed, res.picks)
        d.budew_bonus = budew_first_turn_attack_bonus(parsed, res.picks)
        d.wrong_type_energy_penalty = wrong_type_energy_penalty(parsed, res.picks)
        d.dragapult_attack_bonus = dragapult_ex_attack_bonus(parsed, res.picks)
        d.phantom_dive_bonus = phantom_dive_attack_bonus(parsed, res.picks)
        d.dreepy_spread_penalty = dreepy_energy_spread_penalty(parsed, res.picks)
        d.xerosic_bonus = xerosic_machinations_bonus(parsed, res.picks)
        d.budew_bench_setup_bonus = budew_turn_bench_setup_bonus(parsed, res.picks)
        d.dreepy_evolve_bonus = dreepy_evolve_bonus(parsed, res.picks)
        d.dreepy_bench_charge_bonus = dreepy_line_bench_charge_bonus(parsed, res.picks)
        d.dreepy_active_charge_bonus = dreepy_line_active_charge_bonus(
            parsed, res.picks
        )
        d.wasted_resources_penalty = wasted_resources_attack_penalty(parsed, res.picks)
        d.drakloak_backup_ready_bonus = drakloak_backup_ready_bonus(parsed, res.picks)
        return res.picks, d


class RandomPolicy:
    def act(
        self, obs: dict, collect: bool, ctx: GameContext | None = None
    ) -> tuple[list[int], None]:
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

    # One GameContext per player, each owning its own DeckTracker over its
    # own deck -- never reused across games (see pkm/heuristics/context.py).
    contexts = (
        GameContext(list(decks[0]), DeckTracker(decks[0])),
        GameContext(list(decks[1]), DeckTracker(decks[1])),
    )

    trajectories: tuple[list[EncodedDecision], list[EncodedDecision]] = ([], [])
    count = 0
    try:
        while obs["current"]["result"] < 0 and count < MAX_DECISIONS:
            p = obs["current"]["yourIndex"]
            tracker = contexts[p].tracker
            tracker.observe(obs)
            if tracker.is_search_reveal(obs):
                tracker.record_search_reveal(obs)
            picks, record = policies[p].act(obs, collect=collect[p], ctx=contexts[p])
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
