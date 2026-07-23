"""Focused re-measurement: how is the setup turn best decided?

Three configurations, identical in every other respect -- same deck, same
opponent, same first-turn MCTS, same late-game policy, sides alternating.
Only the agent answering each side's own second turn differs.

Kept separate from `ablate.py` because the search variant is now time-bound
(~1.7s/decision, ~25x its old cost), so running the full six-variant ablation
would spend most of its time on variants this question doesn't need.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path

import torch
import typer

from pkm.agents.dragapult_default_agent import make_dragapult_default_agent
from pkm.agents.dragapult_setup_agent import make_dragapult_setup_agent
from pkm.agents.dragapult_setup_search_agent import make_dragapult_setup_search_agent
from pkm.agents.first_turn_agent import make_first_turn_agent
from pkm.agents.random_agent import make_random_agent
from pkm.agents.singaporean_middleman import make_singaporean_middleman
from pkm.data import Deck

from .config import DarwinConfig
from .evaluate import run_side
from .opponent import BundleOpponent, extract_bundle

app = typer.Typer(add_completion=False)


@app.command()
def main(
    games: int = typer.Option(400, help="games per configuration"),
    bundle: str = typer.Option(DarwinConfig.bundle),
    deck_path: str = typer.Option(DarwinConfig.deck_path),
    out_dir: str = typer.Option(DarwinConfig.out_dir),
    search_seconds: float = typer.Option(1.7, help="search budget per decision"),
    seed: int = typer.Option(0),
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    deck = Deck.from_csv(deck_path).card_ids
    bundle_dir = extract_bundle(bundle, Path(out_dir) / "opponents")
    quiet = {"log_sink": lambda _m: None}

    def build(setup_factory):
        return lambda: make_singaporean_middleman(
            deck,
            agents={
                "dragapult_default": make_dragapult_default_agent(deck),
                "dragapult_setup": setup_factory(),
                "first_turn": make_first_turn_agent(deck),
                "random": make_random_agent(deck),
            },
            **quiet,
        )

    contenders = [
        (
            "SEARCH setup (time-bound)",
            build(
                lambda: make_dragapult_setup_search_agent(
                    deck, time_budget_s=search_seconds, **quiet
                )
            ),
        ),
        (
            "RL setup net (policy_setup.npz)",
            build(lambda: make_dragapult_setup_agent(deck, **quiet)),
        ),
        (
            "no setup (default net plays it)",
            build(lambda: make_dragapult_default_agent(deck)),
        ),
    ]

    rows = []
    with BundleOpponent(bundle_dir) as opponent:
        print(f"opponent: {Path(bundle).name}   games/variant: {games}\n")
        for name, factory in contenders:
            t0 = time.time()
            print(f"running {name} ...", flush=True)
            r = run_side(name, factory, opponent, deck, games)
            r["mins"] = (time.time() - t0) / 60
            rows.append(r)

    print(
        f"\n{'turn 3/4 (setup turn) played by':<34}{'games':>6}{'W':>5}{'L':>5}"
        f"{'win rate':>10}{'95% CI':>16}{'prizes':>8}{'mins':>7}"
    )
    print("-" * 92)
    for r in rows:
        p = r["win_rate"]
        se = math.sqrt(p * (1 - p) / max(r["games"], 1))
        ci = f"[{100 * (p - 1.96 * se):.1f},{100 * (p + 1.96 * se):.1f}]"
        print(
            f"{r['name']:<34}{r['games']:>6}{r['wins']:>5}{r['losses']:>5}"
            f"{p:>9.1%}{ci:>16}{r['avg_prizes_taken']:>8.2f}{r['mins']:>7.1f}"
        )

    def cmp(a, b):
        ra, rb = rows[a], rows[b]
        pa, pb = ra["win_rate"], rb["win_rate"]
        se = math.sqrt(pa * (1 - pa) / ra["games"] + pb * (1 - pb) / rb["games"])
        z = (pa - pb) / se if se else 0.0
        verdict = "SIGNIFICANT" if abs(z) >= 1.96 else "not distinguishable"
        print(
            f"  {ra['name']} - {rb['name']}: "
            f"{100 * (pa - pb):+.1f} pp  z={z:.2f}  {verdict}"
        )

    print("\nCOMPARISONS")
    cmp(0, 1)
    cmp(0, 2)
    cmp(1, 2)
    # what this sample size can actually resolve
    p = sum(r["win_rate"] for r in rows) / len(rows)
    d = 2.8 * math.sqrt(2 * p * (1 - p) / games)
    print(
        f"\n  at n={games} and ~{p:.0%} win rate, only differences above "
        f"~{100 * d:.1f} pp are detectable (80% power)"
    )


if __name__ == "__main__":
    app()
