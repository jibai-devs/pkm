"""Fitness: how well does one genome do against the live opponent bundle?

**The central design problem.** "Beat Mega Abomasnow" is a binary target, and
a population that loses every game gets identical fitness (-1) for every
member -- selection has nothing to rank, and evolution stalls before it
starts. So fitness is deliberately *graded*: win/loss dominates, but a
prize-count margin makes "lost 6-1" measurably worse than "lost 6-5", and a
small shaping term rewards the deck's known-good habits. That gives the search
a slope to climb long before the first win appears.

The shaping term reuses the default agent's existing reward functions
(`pkm/rl/encoder.py`) *read-only* -- those already encode what a Dragapult
deck wants (charge the line, evolve, Phantom Dive, don't strand energy), and
there is no reason to invent a second opinion. Nothing here writes to `pkm`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from pkm.engine import battle_finish, battle_select, battle_start
from pkm.heuristics.context import GameContext
from pkm.heuristics.deck_tracker import DeckTracker
from pkm.rl.encoder import encode_decision
from pkm.rl.model import PolicyValueNet
from pkm.types.obs import Observation

from .config import DarwinConfig
from .shaping import shaping_score

MAX_DECISIONS = 3000


@dataclass
class GameOutcome:
    won: bool
    drew: bool
    prize_margin: int  # ours taken minus theirs taken; + is good
    shaping: float
    decisions: int


@dataclass
class Running:
    """A genome's fitness estimate accumulated over *all* its evaluations.

    The single biggest flaw in the first run: each genome was judged on one
    batch of games and never re-examined, so a genome that went 6/6 by luck
    was recorded as perfect and -- being an elite -- survived untouched
    forever. Nothing could beat it, because nothing else got that lucky, and
    the population stopped improving while chasing a fluke.

    Carrying the totals forward means an elite is re-evaluated every
    generation and its estimate regresses toward its true strength. Surviving
    ten generations now requires being good ten times, not lucky once.
    """

    games: int = 0
    wins: int = 0
    prize_sum: float = 0.0
    shaping_sum: float = 0.0

    def add(self, f: "Fitness") -> None:
        self.games += f.games
        self.wins += f.wins
        self.prize_sum += f.prize_margin * f.games
        self.shaping_sum += f.shaping * f.games

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.games, 1)

    def score(self, cfg) -> float:
        n = max(self.games, 1)
        return (
            cfg.w_win * self.win_rate
            + cfg.w_prize_margin * ((self.prize_sum / n) / 6.0)
            + cfg.w_shaping * (self.shaping_sum / n)
        )


@dataclass
class Fitness:
    """A genome's measured quality, plus the parts that produced it."""

    score: float = 0.0
    games: int = 0
    wins: int = 0
    draws: int = 0
    prize_margin: float = 0.0
    shaping: float = 0.0
    parts: dict[str, float] = field(default_factory=dict)

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.games, 1)


def _prizes_left(player: dict) -> int:
    return len(player.get("prize") or [])


def play_one_game(
    model: PolicyValueNet,
    opponent,
    our_deck: list[int],
    our_seat: int,
    cfg: DarwinConfig,
) -> GameOutcome:  # noqa: D401
    """One full battle: our network on `our_seat`, the bundle on the other."""
    opponent.new_game()
    decks = [None, None]
    decks[our_seat] = list(our_deck)
    decks[1 - our_seat] = list(opponent.deck)

    obs, start = battle_start(decks[0], decks[1])
    if obs is None:
        raise RuntimeError(f"battle_start failed: errorPlayer={start.errorPlayer}")

    ctx = GameContext(list(our_deck), DeckTracker(our_deck))
    shaping_total = 0.0
    scored = 0
    count = 0
    try:
        while obs["current"]["result"] < 0 and count < MAX_DECISIONS:
            seat = obs["current"]["yourIndex"]
            if seat == our_seat:
                picks, s = _our_move(model, obs, ctx, cfg, our_deck)
                if s is not None:
                    shaping_total += s
                    scored += 1
            else:
                picks = opponent.act(obs)
            obs = battle_select(picks)
            count += 1
        final = obs["current"]
        result = final.get("result", -1)
        ours_left = _prizes_left(final["players"][our_seat])
        theirs_left = _prizes_left(final["players"][1 - our_seat])
    finally:
        battle_finish()

    return GameOutcome(
        won=(result == our_seat),
        drew=(result < 0 or result > 1),
        # taken = 6 - left, so this is (6-ours_left) - (6-theirs_left)
        prize_margin=theirs_left - ours_left,
        shaping=shaping_total / max(scored, 1),
        decisions=count,
    )


def _our_move(model, obs: dict, ctx: GameContext, cfg: DarwinConfig, deck: list[int]):
    """Greedy pick from the network, plus this decision's shaping value."""
    ctx.tracker.observe(obs)
    if ctx.tracker.is_search_reveal(obs):
        ctx.tracker.record_search_reveal(obs)
    parsed = Observation.model_validate(obs)
    sel = parsed.select
    if sel is None:
        return list(deck), None  # deck-submission phase
    forced = sel.forced_picks()
    if forced is not None:
        return forced, None
    decision = encode_decision(parsed, ctx)
    with torch.no_grad():
        res = model.act(decision, greedy=True)
    return res.picks, shaping_score(parsed, res.picks, cfg.shaping_weights)


def evaluate(
    genome: np.ndarray,
    scratch: PolicyValueNet,
    opponent,
    our_deck: list[int],
    cfg: DarwinConfig,
    shapes: list,
) -> Fitness:
    """Play `cfg.games_per_genome` games and reduce them to one number.

    Sides alternate so a genome cannot win purely by exploiting the going-first
    advantage -- that would select for luck rather than play.
    """
    load_genome(scratch, genome, shapes)
    scratch.eval()
    f = Fitness()
    for g in range(cfg.games_per_genome):
        out = play_one_game(scratch, opponent, our_deck, our_seat=g % 2, cfg=cfg)
        f.games += 1
        f.wins += int(out.won)
        f.draws += int(out.drew and not out.won)
        f.prize_margin += out.prize_margin
        f.shaping += out.shaping
    n = max(f.games, 1)
    f.prize_margin /= n
    f.shaping /= n
    f.parts = {
        "win": cfg.w_win * f.win_rate,
        "prize_margin": cfg.w_prize_margin * (f.prize_margin / 6.0),
        "shaping": cfg.w_shaping * f.shaping,
    }
    f.score = sum(f.parts.values())
    return f


# --- genome <-> network -----------------------------------------------------


def genome_shapes(model: PolicyValueNet) -> list:
    """(name, shape, numel) for every trainable tensor, in a stable order."""
    return [(k, tuple(v.shape), v.numel()) for k, v in model.state_dict().items()]


def flatten(model: PolicyValueNet) -> np.ndarray:
    sd = model.state_dict()
    return np.concatenate(
        [
            sd[k].detach().cpu().numpy().ravel().astype(np.float32)
            for k, _, _ in genome_shapes(model)
        ]
    )


def load_genome(model: PolicyValueNet, genome: np.ndarray, shapes: list) -> None:
    """Write a flat genome back into the network, in place."""
    sd = model.state_dict()
    i = 0
    for name, shape, numel in shapes:
        chunk = genome[i : i + numel].reshape(shape)
        sd[name].copy_(torch.from_numpy(np.ascontiguousarray(chunk)))
        i += numel
    model.load_state_dict(sd)
