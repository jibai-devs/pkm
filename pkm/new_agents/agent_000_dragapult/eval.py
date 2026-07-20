"""Evaluation harness — the real learning signal (README D7 / §9, infra-todo §11).

Self-play win-rate is ~50% by construction and tells us nothing. This plays the
agent under test (greedy) against a **fixed** opponent for many games and reports
win-rate from the agent's perspective. The learn-check gate: does win-rate vs a
random opponent climb above 50%?

Opponents start with `RandomAgent`; frozen checkpoints / scripted bots (a league)
come later. Runs single-process (no multiprocessing).
"""

from __future__ import annotations

import random
from typing import Any, Callable

import torch

from pkm.new_agents.agent_000_dragapult.cabt import (
    battle_finish,
    battle_select,
    battle_start,
)
from pkm.new_agents.agent_000_dragapult.agent import DragapultAgent, InferenceConfig
from pkm.new_agents.agent_000_dragapult.deck import DECK_60

AgentFn = Callable[[dict[str, Any]], list[int]]


class RandomAgent:
    """Uniform-random legal-option baseline (its own RNG for reproducibility)."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def __call__(self, obs: dict[str, Any]) -> list[int]:
        sel = obs.get("select")
        if sel is None or obs.get("current") is None:
            return list(DECK_60)
        n = len(sel["option"])
        if n == 0:
            return []
        k = max(sel["minCount"], min(sel["maxCount"], n))
        return self.rng.sample(range(n), k)


def play_match(agent_fn: AgentFn, opp_fn: AgentFn, agent_seat: int) -> int:
    """Play one game (both sides pilot our deck); return result from the agent's
    perspective: +1 win, -1 loss, 0 draw."""
    obs, _ = battle_start(list(DECK_60), list(DECK_60))
    it = 0
    while obs["current"]["result"] < 0 and it < 100000:
        if obs["select"] is None or obs["current"] is None:
            obs = battle_select(list(DECK_60))  # deck-selection phase
            it += 1
            continue
        who = obs["current"]["yourIndex"]
        obs = battle_select((agent_fn if who == agent_seat else opp_fn)(obs))
        it += 1
    result = obs["current"]["result"]
    battle_finish()
    if result == agent_seat:
        return 1
    if result in (0, 1):
        return -1
    return 0


def evaluate(
    agent_fn: AgentFn, opp_fn: AgentFn, n_games: int = 100
) -> dict[str, float]:
    """Win-rate of agent_fn vs opp_fn over n_games, alternating seats (removes
    first-player bias)."""
    wins = losses = draws = 0
    for g in range(n_games):
        res = play_match(agent_fn, opp_fn, agent_seat=g % 2)
        if res > 0:
            wins += 1
        elif res < 0:
            losses += 1
        else:
            draws += 1
    n = max(n_games, 1)
    return {
        "n": n_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / n,
        "loss_rate": losses / n,
        "draw_rate": draws / n,
    }


@torch.no_grad()
def winrate_vs_random(
    model: torch.nn.Module,
    n_games: int = 100,
    seed: int = 0,
    inference: InferenceConfig | None = None,
) -> dict[str, float]:
    """Convenience: greedy agent from `model` vs a RandomAgent baseline.

    Pass ``inference`` to evaluate the agent-under-test with MCTS search rather
    than the raw policy head.
    """
    agent = DragapultAgent(model=model, greedy=True, inference=inference)
    return evaluate(agent, RandomAgent(seed=seed), n_games=n_games)


@torch.no_grad()
def winrate_vs_agent(
    model: torch.nn.Module,
    opponent: AgentFn,
    n_games: int = 100,
    inference: InferenceConfig | None = None,
) -> dict[str, float]:
    """Win-rate of greedy `model` vs an arbitrary opponent agent callable."""
    agent = DragapultAgent(model=model, greedy=True, inference=inference)
    return evaluate(agent, opponent, n_games=n_games)


@torch.no_grad()
def winrate_vs_checkpoint(
    model: torch.nn.Module,
    opponent_path: str,
    n_games: int = 100,
    inference: InferenceConfig | None = None,
) -> dict[str, float]:
    """Win-rate of greedy `model` vs a greedy agent loaded from another checkpoint
    (a training ``ckpt_N.pt`` or a packed ``weights.pt``).

    This is the *discriminating* eval: unlike vs-random (which saturates near
    100% once the agent is any good), pitting two trained policies against each
    other ranks them on a signal that isn't pinned to the random-opponent ceiling.
    """
    opp = DragapultAgent.from_checkpoint(opponent_path, greedy=True)
    return winrate_vs_agent(model, opp, n_games=n_games, inference=inference)
