"""Head-to-head evaluation against the opponent bundle.

Measures two things against the *same* opponent, over the *same* number of
games, with sides alternating so neither benefits from the going-first
advantage:

1. **default_dragapult_darwinian** -- the evolved network (`best.pt`).
2. **singaporean_middleman** -- the full deployed stack: the first-turn MCTS
   on the opening turn, the setup agent on each side's second turn, and the
   default policy thereafter.

Run:
    python -m darwinian_ml.evaluate --games 40
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import torch
import typer

from pkm.agents.singaporean_middleman import make_singaporean_middleman
from pkm.data import Deck
from pkm.engine import battle_finish, battle_select, battle_start
from pkm.heuristics.context import GameContext
from pkm.heuristics.deck_tracker import DeckTracker
from pkm.rl.encoder import encode_decision
from pkm.rl.model import PolicyValueNet
from pkm.types.obs import Observation

from .config import DarwinConfig
from .opponent import BundleOpponent, extract_bundle

MAX_DECISIONS = 3000


def _evolved_agent(checkpoint: str, deck: list[int]):
    """The evolved genome as a plain `agent(obs) -> picks` callable."""
    model = PolicyValueNet()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    ctx = GameContext(list(deck), DeckTracker(deck))

    def agent(obs: dict) -> list[int]:
        ctx.tracker.observe(obs)
        if ctx.tracker.is_search_reveal(obs):
            ctx.tracker.record_search_reveal(obs)
        if obs.get("select") is None:
            return list(deck)
        parsed = Observation.model_validate(obs)
        forced = parsed.select.forced_picks()
        if forced is not None:
            return forced
        with torch.no_grad():
            return model.act(encode_decision(parsed, ctx), greedy=True).picks

    return agent


def play_match(agent, opponent, our_deck: list[int], our_seat: int) -> tuple[int, int]:
    """One game. Returns (result_code, our_prizes_taken)."""
    opponent.new_game()
    decks = [None, None]
    decks[our_seat] = list(our_deck)
    decks[1 - our_seat] = list(opponent.deck)
    obs, start = battle_start(decks[0], decks[1])
    if obs is None:
        raise RuntimeError(f"battle_start failed: {start.errorPlayer}")
    n = 0
    try:
        while obs["current"]["result"] < 0 and n < MAX_DECISIONS:
            seat = obs["current"]["yourIndex"]
            picks = agent(obs) if seat == our_seat else opponent.act(obs)
            obs = battle_select(picks)
            n += 1
        final = obs["current"]
        taken = 6 - len(final["players"][our_seat].get("prize") or [])
        return final.get("result", -1), taken
    finally:
        battle_finish()


def run_side(name: str, make_agent, opponent, deck: list[int], games: int) -> dict:
    wins = draws = losses = 0
    prizes = 0
    t0 = time.time()
    for g in range(games):
        seat = g % 2
        # a fresh agent per game: both stacks carry per-game memory
        result, taken = play_match(make_agent(), opponent, deck, seat)
        prizes += taken
        if result == seat:
            wins += 1
        elif result < 0 or result > 1:
            draws += 1
        else:
            losses += 1
    return {
        "name": name,
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / max(games, 1),
        "avg_prizes_taken": prizes / max(games, 1),
        "seconds": time.time() - t0,
    }


app = typer.Typer(add_completion=False)


@app.command()
def main(
    games: int = typer.Option(40, help="games per contender (sides alternate)"),
    checkpoint: str = typer.Option(
        "darwinian_ml/runs/default_dragapult_darwinian/best.pt"
    ),
    bundle: str = typer.Option(DarwinConfig.bundle),
    deck_path: str = typer.Option(DarwinConfig.deck_path),
    out_dir: str = typer.Option(DarwinConfig.out_dir),
    seed: int = typer.Option(0),
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    deck = Deck.from_csv(deck_path).card_ids
    bundle_dir = extract_bundle(bundle, Path(out_dir) / "opponents")

    contenders = []
    if Path(checkpoint).is_file():
        contenders.append(
            ("default_dragapult_darwinian", lambda: _evolved_agent(checkpoint, deck))
        )
    else:
        print(f"! no evolved checkpoint at {checkpoint}; skipping that contender")
    contenders.append(
        (
            "singaporean_middleman",
            lambda: make_singaporean_middleman(deck, log_sink=lambda _m: None),
        )
    )

    rows = []
    with BundleOpponent(bundle_dir) as opponent:
        print(f"opponent: {Path(bundle).name} ({len(opponent.deck)} cards)\n")
        for name, factory in contenders:
            print(f"running {name} ...", flush=True)
            rows.append(run_side(name, factory, opponent, deck, games))

    print(
        f"\n{'contender':<32}{'games':>6}{'W':>5}{'D':>4}{'L':>5}{'win rate':>10}"
        f"{'avg prizes':>12}{'time':>8}"
    )
    print("-" * 84)
    for r in rows:
        print(
            f"{r['name']:<32}{r['games']:>6}{r['wins']:>5}{r['draws']:>4}{r['losses']:>5}"
            f"{r['win_rate']:>9.0%}{r['avg_prizes_taken']:>12.2f}{r['seconds']:>7.0f}s"
        )


if __name__ == "__main__":
    app()
