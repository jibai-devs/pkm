"""Play a match between two named agents; save an HTML replay + JSON log.

Usage:
    pkm play --p0 neural --p1 random
    pkm play --p0 mcts --p1 neural --html mcts_vs_neural.html
    pkm play --p0 neural --p1 random --games 20   # win-rate only
    pkm play --agent 01_psychic --p0 neural --p1 random

Agents: random | neural (greedy policy, needs pkm/policy.npz) | mcts | singaporean_middleman
The HTML file is self-contained — open it in a browser to watch the match.
The JSON log is kaggle-environments' full episode record (per-step
observations, actions, rewards); reload it with json.load for analysis.
"""

import typer
import json
from typing import Callable

from kaggle_environments import make

from pkm.agents import (
    make_neural_agent,
    make_random_agent,
    make_singaporean_middleman,
)
from pkm.agents.profile import AgentProfile
from pkm.data import Deck
from pkm.tui.session import HUMAN


def make_agent_by_name(
    name: str, deck: list[int], weights: str | None
) -> Callable[[dict], list[int]]:
    if name == HUMAN:
        raise ValueError(
            "human has no standalone agent: it needs a TUI session "
            "(handled by play_match)"
        )
    if name == "random":
        return make_random_agent(deck)
    if name == "neural":
        return make_neural_agent(deck, weights)
    if name == "mcts":
        from pkm.mcts.agent import make_mcts_agent

        return make_mcts_agent(deck, weights_path=weights)
    if name == "singaporean_middleman":
        return make_singaporean_middleman(deck, weights)
    raise ValueError(
        "unknown agent: "
        f"{name!r} (expected random|neural|mcts|singaporean_middleman|human)"
    )


def play_human_match(
    p0: str,
    p1: str,
    deck_path: str = "deck/02_dragapult.csv",
    weights: str | None = None,
    html_path: str | None = "result.html",
    replay_path: str | None = "replay.json",
) -> None:
    """Play one match with a human at the keyboard, in a Textual TUI."""
    from pkm.tui.app import BattleApp
    from pkm.tui.session import ThreadedEnvSession

    if p0 == HUMAN and p1 == HUMAN:
        raise ValueError("only one human player is supported")

    human_index = 0 if p0 == HUMAN else 1
    opponent = p1 if human_index == 0 else p0

    deck = Deck.from_csv(deck_path).card_ids
    session = ThreadedEnvSession(
        deck=deck,
        human_index=human_index,
        opponent=opponent,
        weights=weights,
        html_path=html_path,
        replay_path=replay_path,
    )
    BattleApp(session).run()


def play_match(
    p0: str,
    p1: str,
    deck_path: str = "deck/02_dragapult.csv",
    weights: str | None = None,
    deck0_path: str | None = None,
    deck1_path: str | None = None,
    weights0: str | None = None,
    weights1: str | None = None,
    html_path: str | None = "result.html",
    replay_path: str | None = "replay.json",
):
    """Run one rendered match; returns the finished kaggle environment.

    `deck0_path`/`deck1_path` and `weights0`/`weights1` let each side bring
    its own deck/policy (e.g. two different population-trained bots);
    each falls back to the shared `deck_path`/`weights` when not given, so
    existing single-deck callers are unaffected.

    Exception: with a human player the match runs inside the TUI, which owns the
    environment for the lifetime of the app, and this returns None.
    """
    if HUMAN in (p0, p1):
        return play_human_match(
            p0,
            p1,
            deck_path=deck_path,
            weights=weights,
            html_path=html_path,
            replay_path=replay_path,
        )
    deck0 = Deck.from_csv(deck0_path or deck_path).card_ids
    deck1 = Deck.from_csv(deck1_path or deck_path).card_ids
    agents = [
        make_agent_by_name(p0, deck0, weights0 if weights0 is not None else weights),
        make_agent_by_name(p1, deck1, weights1 if weights1 is not None else weights),
    ]
    env = make("cabt", configuration={"decks": [deck0, deck1]})
    env.run(agents)

    final = env.steps[-1]
    print(f"p0 ({p0}): {final[0].status}, reward {final[0].reward}")
    print(f"p1 ({p1}): {final[1].status}, reward {final[1].reward}")

    if html_path:
        with open(html_path, "w") as f:
            f.write(env.render(mode="html"))
        print(f"replay visualization: {html_path} (open in a browser)")
    if replay_path:
        data = env.toJSON()  # str or dict depending on kaggle-environments version
        with open(replay_path, "w") as f:
            if isinstance(data, str):
                f.write(data)
            else:
                json.dump(data, f)
        print(f"episode log: {replay_path}")
    return env


def win_rate(
    p0: str,
    p1: str,
    games: int,
    deck_path: str = "deck/02_dragapult.csv",
    weights: str | None = None,
    deck0_path: str | None = None,
    deck1_path: str | None = None,
    weights0: str | None = None,
    weights1: str | None = None,
) -> float:
    """Head-to-head win rate for p0's agent type, alternating sides.

    `deck0_path`/`deck1_path` and `weights0`/`weights1` let each side bring
    its own deck/policy, same fallback-to-shared-default convention as
    `play_match`."""
    if HUMAN in (p0, p1):
        raise ValueError("human play does not support --games > 1")
    deck0 = Deck.from_csv(deck0_path or deck_path).card_ids
    deck1 = Deck.from_csv(deck1_path or deck_path).card_ids
    w0 = weights0 if weights0 is not None else weights
    w1 = weights1 if weights1 is not None else weights
    score = 0.0
    for g in range(games):
        a = make_agent_by_name(p0, deck0, w0)
        b = make_agent_by_name(p1, deck1, w1)
        agents = [a, b] if g % 2 == 0 else [b, a]
        side = g % 2
        env = make("cabt", configuration={"decks": [deck0, deck1] if g % 2 == 0 else [deck1, deck0]})
        env.run(agents)
        r = env.steps[-1][side].reward or 0
        score += 1.0 if r > 0 else 0.5 if r == 0 else 0.0
        print(
            f"game {g + 1}/{games}: {'W' if r > 0 else 'L' if r < 0 else 'D'}",
            flush=True,
        )
    print(f"{p0} vs {p1}: {score}/{games} = {score / games:.1%}")
    return score / games


app = typer.Typer(help=__doc__)


def _resolve_agent_deck_weights(agent: str) -> tuple[str, str | None]:
    """AgentProfile -> (deck_path, npz_weights_path or None if unexported)."""
    profile = AgentProfile(agent)
    ckpt = profile.checkpoint_dir / "policy.npz"
    return str(profile.deck_path), str(ckpt) if ckpt.is_file() else None


@app.command()
def main(
    p0: str = typer.Option("neural", help="player 0 agent: random|neural|mcts|human"),
    p1: str = typer.Option("random", help="player 1 agent: random|neural|mcts|human"),
    agent: str | None = typer.Option(None, help="agent profile name (resolves deck + weights)"),
    p0_agent: str | None = typer.Option(
        None,
        "--p0-agent",
        help="agent profile name for player 0 only -- lets p0/p1 use different "
        "decks/weights (e.g. two different agents/pool_*/ bots). Overrides "
        "--agent/--deck/--weights for this side only; needs that profile's "
        "checkpoints/policy.npz exported first (`pkm export --agent <name> "
        "agents/<name>/checkpoints/policy.npz`)",
    ),
    p1_agent: str | None = typer.Option(
        None, "--p1-agent", help="same as --p0-agent, for player 1 only"
    ),
    deck: str = typer.Option("deck/02_dragapult.csv", help="path to deck CSV"),
    weights: str | None = typer.Option(None, help="path to policy .npz"),
    html: str = typer.Option("result.html", help="HTML replay output path"),
    replay: str = typer.Option("replay.json", help="JSON replay output path"),
    games: int = typer.Option(1, help=">1: win-rate mode, no replay"),
) -> None:
    if agent:
        profile = AgentProfile(agent)
        deck = str(profile.deck_path)
        if weights is None:
            ckpt = profile.checkpoint_dir / "policy.npz"
            if ckpt.is_file():
                weights = str(ckpt)

    deck0 = weights0 = deck1 = weights1 = None
    if p0_agent:
        deck0, weights0 = _resolve_agent_deck_weights(p0_agent)
    if p1_agent:
        deck1, weights1 = _resolve_agent_deck_weights(p1_agent)

    if games > 1:
        win_rate(
            p0,
            p1,
            games,
            deck_path=deck,
            weights=weights,
            deck0_path=deck0,
            deck1_path=deck1,
            weights0=weights0,
            weights1=weights1,
        )
    else:
        play_match(
            p0,
            p1,
            deck_path=deck,
            weights=weights,
            deck0_path=deck0,
            deck1_path=deck1,
            weights0=weights0,
            weights1=weights1,
            html_path=html,
            replay_path=replay,
        )


if __name__ == "__main__":
    app()
