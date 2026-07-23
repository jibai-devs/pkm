"""Ablation: which part of the middleman stack is costing win rate?

The full middleman measured far *worse* than a bare evolved network, which
should not happen if its extra machinery helps. This isolates the pieces by
swapping individual sub-agents out of the registry, keeping everything else --
opponent, deck, game count, side alternation -- identical.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
import typer

from pkm.agents.dragapult_default_agent import make_dragapult_default_agent
from pkm.agents.first_turn_agent import make_first_turn_agent
from pkm.agents.random_agent import make_random_agent
from pkm.agents.singaporean_middleman import make_singaporean_middleman
from pkm.data import Deck

from .config import DarwinConfig
from .evaluate import _evolved_agent, run_side
from .opponent import BundleOpponent, extract_bundle

app = typer.Typer(add_completion=False)


@app.command()
def main(
    games: int = typer.Option(40),
    bundle: str = typer.Option(DarwinConfig.bundle),
    deck_path: str = typer.Option(DarwinConfig.deck_path),
    out_dir: str = typer.Option(DarwinConfig.out_dir),
    checkpoint: str = typer.Option(
        "darwinian_ml/runs/default_dragapult_darwinian/best.pt"
    ),
    seed: int = typer.Option(0),
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    deck = Deck.from_csv(deck_path).card_ids
    bundle_dir = extract_bundle(bundle, Path(out_dir) / "opponents")
    quiet = {"log_sink": lambda _m: None}

    def full():
        return make_singaporean_middleman(deck, **quiet)

    def no_setup():
        """Middleman with the setup agent replaced by the default policy."""
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

    def no_first_turn():
        """Middleman with the first-turn MCTS replaced by the default policy."""
        return make_singaporean_middleman(
            deck,
            agents={
                "dragapult_default": make_dragapult_default_agent(deck),
                "dragapult_setup": make_dragapult_default_agent(deck),
                "first_turn": make_dragapult_default_agent(deck),
                "random": make_random_agent(deck),
            },
            **quiet,
        )

    def darwinian_default():
        """The full stack with the evolved policy in the default slot.

        Setup agent and first-turn MCTS unchanged, so the only difference
        from `full` is which weights answer the default agent's turns. Setup
        routing is turn-based (each side's own second turn) -- the only
        routing rule there is now.
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

    def darwinian_no_setup():
        """Evolved policy in the default slot, setup agent removed."""
        return make_singaporean_middleman(
            deck,
            agents={
                "dragapult_default": _evolved_agent(checkpoint, deck),
                "dragapult_setup": _evolved_agent(checkpoint, deck),
                "first_turn": make_first_turn_agent(deck),
                "random": make_random_agent(deck),
            },
            **quiet,
        )

    def search_setup():
        """Setup turn played by SEARCH over the rubric, not a trained net.

        Same slot, same routing as `full` -- the only change is that the
        agent answering the setup turn maximises the rubric by simulating
        real move sequences instead of approximating it from weights.
        """
        from pkm.agents.dragapult_setup_search_agent import (
            make_dragapult_setup_search_agent,
        )

        return make_singaporean_middleman(
            deck,
            agents={
                "dragapult_default": make_dragapult_default_agent(deck),
                "dragapult_setup": make_dragapult_setup_search_agent(deck, **quiet),
                "first_turn": make_first_turn_agent(deck),
                "random": make_random_agent(deck),
            },
            **quiet,
        )

    def darwin_search_setup():
        """Evolved policy in the default slot + SEARCH setup + first-turn."""
        from pkm.agents.dragapult_setup_search_agent import (
            make_dragapult_setup_search_agent,
        )

        return make_singaporean_middleman(
            deck,
            agents={
                "dragapult_default": _evolved_agent(checkpoint, deck),
                "dragapult_setup": make_dragapult_setup_search_agent(deck, **quiet),
                "first_turn": make_first_turn_agent(deck),
                "random": make_random_agent(deck),
            },
            **quiet,
        )

    contenders = [
        (
            "raw dragapult_default (policy.npz)",
            lambda: make_dragapult_default_agent(deck),
        ),
        ("middleman: full (RL setup agent)", full),
        ("middleman: setup -> default (none)", no_setup),
        ("middleman: SEARCH setup agent", search_setup),
        ("middleman: setup+first_turn -> default", no_first_turn),
    ]
    if Path(checkpoint).is_file():
        contenders += [
            ("middleman: DARWINIAN + SEARCH setup", darwin_search_setup),
        ]
    else:
        print(f"! no evolved checkpoint at {checkpoint}; darwinian variants skipped")

    rows = []
    with BundleOpponent(bundle_dir) as opponent:
        print(f"opponent: {Path(bundle).name}\n")
        for name, factory in contenders:
            print(f"running {name} ...", flush=True)
            rows.append(run_side(name, factory, opponent, deck, games))

    print(
        f"\n{'variant':<42}{'games':>6}{'W':>5}{'L':>5}{'win rate':>10}{'avg prizes':>12}"
    )
    print("-" * 80)
    for r in rows:
        print(
            f"{r['name']:<42}{r['games']:>6}{r['wins']:>5}{r['losses']:>5}"
            f"{r['win_rate']:>9.0%}{r['avg_prizes_taken']:>12.2f}"
        )


if __name__ == "__main__":
    app()
