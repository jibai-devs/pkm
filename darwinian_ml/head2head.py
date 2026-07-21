"""Direct head-to-head between two of our own agents.

Both sides play the same deck, so this is a mirror match and a 50% result
means "indistinguishable". Sides alternate every game, and both agents are
rebuilt per game because each carries per-game memory (GameContext /
DeckTracker) that would otherwise leak across battles.

Run:
    python -m darwinian_ml.head2head --games 40
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import torch
import typer

from pkm.agents.dragapult_default_agent import make_dragapult_default_agent
from pkm.agents.first_turn_agent import make_first_turn_agent
from pkm.agents.random_agent import make_random_agent
from pkm.agents.singaporean_middleman import make_singaporean_middleman
from pkm.data import Deck
from pkm.engine import battle_finish, battle_select, battle_start

from .config import DarwinConfig
from .evaluate import _evolved_agent

MAX_DECISIONS = 3000


def play(make_a, make_b, deck: list[int], a_seat: int) -> tuple[int, int, int]:
    """One mirror game. Returns (result_code, a_prizes_taken, b_prizes_taken)."""
    a, b = make_a(), make_b()
    obs, start = battle_start(list(deck), list(deck))
    if obs is None:
        raise RuntimeError(f"battle_start failed: {start.errorPlayer}")
    n = 0
    try:
        while obs["current"]["result"] < 0 and n < MAX_DECISIONS:
            seat = obs["current"]["yourIndex"]
            picks = a(obs) if seat == a_seat else b(obs)
            obs = battle_select(picks)
            n += 1
        final = obs["current"]
        taken = [
            6 - len(final["players"][0].get("prize") or []),
            6 - len(final["players"][1].get("prize") or []),
        ]
        return final.get("result", -1), taken[a_seat], taken[1 - a_seat]
    finally:
        battle_finish()


def series(label_a, make_a, label_b, make_b, deck, games) -> None:
    a_w = b_w = draws = 0
    a_p = b_p = 0
    t0 = time.time()
    for g in range(games):
        a_seat = g % 2
        result, ap, bp = play(make_a, make_b, deck, a_seat)
        a_p += ap
        b_p += bp
        if result == a_seat:
            a_w += 1
        elif result == 1 - a_seat:
            b_w += 1
        else:
            draws += 1
    dt = time.time() - t0
    n = max(games, 1)
    print(f"\n{label_a}  vs  {label_b}   ({games} games, sides alternating, {dt:.0f}s)")
    print(f"  {label_a:<38}{a_w:>4}W  {a_w / n:>5.0%}   avg prizes {a_p / n:.2f}")
    print(f"  {label_b:<38}{b_w:>4}W  {b_w / n:>5.0%}   avg prizes {b_p / n:.2f}")
    if draws:
        print(f"  draws / decision-cap                  {draws:>4}")


app = typer.Typer(add_completion=False)


@app.command()
def main(
    games: int = typer.Option(40),
    checkpoint: str = typer.Option(
        "darwinian_ml/runs/default_dragapult_darwinian/best.pt"
    ),
    deck_path: str = typer.Option(DarwinConfig.deck_path),
    seed: int = typer.Option(0),
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    deck = Deck.from_csv(deck_path).card_ids
    if not Path(checkpoint).is_file():
        raise SystemExit(f"no evolved checkpoint at {checkpoint}")
    quiet = {"log_sink": lambda _m: None}

    darwin = lambda: _evolved_agent(checkpoint, deck)  # noqa: E731
    full = lambda: make_singaporean_middleman(deck, **quiet)  # noqa: E731

    def no_setup():
        # The ablation showed the setup agent costs win rate; include the
        # variant without it so the comparison isn't against a handicap.
        return make_singaporean_middleman(
            deck,
            agents={
                "dragapult_default": make_dragapult_default_agent(deck),
                "dragapult_setup": make_dragapult_default_agent(deck),
                "first_turn": make_first_turn_agent(deck),
                "random": make_random_agent(deck),
            },
            **quiet,
        )

    def middleman_with_darwinian():
        """The full stack, but the evolved policy fills the default slot.

        The cleanest comparison available: identical first-turn MCTS, identical
        setup agent, identical routing -- the *only* difference is which set of
        weights answers the turns the default agent would normally play. Any
        gap is attributable to the policy rather than the scaffolding.
        """
        from pkm.agents.dragapult_setup_agent import make_dragapult_setup_agent

        return make_singaporean_middleman(
            deck,
            agents={
                "dragapult_default": _evolved_agent(checkpoint, deck),
                "dragapult_setup": make_dragapult_setup_agent(deck, **quiet),
                "first_turn": make_first_turn_agent(deck),
                "random": make_random_agent(deck),
            },
            **quiet,
        )

    series("singaporean_middleman (full)", full, "darwinian", darwin, deck, games)
    series("middleman (setup->default)", no_setup, "darwinian", darwin, deck, games)
    series(
        "middleman (stock default)",
        full,
        "middleman (darwinian default)",
        middleman_with_darwinian,
        deck,
        games,
    )


if __name__ == "__main__":
    app()
