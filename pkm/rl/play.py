"""Play a match between two named agents; save an HTML replay + JSON log.

Usage:
    python -m pkm.rl.play --p0 neural --p1 random
    python -m pkm.rl.play --p0 mcts --p1 neural --html mcts_vs_neural.html
    python -m pkm.rl.play --p0 neural --p1 random --games 20   # win-rate only
    python -m pkm.rl.play --agent 01_psychic --p0 neural --p1 random

Agents: random | neural (greedy policy, needs pkm/policy.npz) | mcts
The HTML file is self-contained — open it in a browser to watch the match.
The JSON log is kaggle-environments' full episode record (per-step
observations, actions, rewards); reload it with json.load for analysis.
"""

import typer
import json
from typing import Callable

from kaggle_environments import make

from pkm.agents import make_neural_agent, make_random_agent
from pkm.agents.profile import AgentProfile
from pkm.data import Deck


def make_agent_by_name(
    name: str, deck: list[int], weights: str | None
) -> Callable[[dict], list[int]]:
    if name == "random":
        return make_random_agent(deck)
    if name == "neural":
        return make_neural_agent(deck, weights)
    if name == "mcts":
        from pkm.mcts.agent import make_mcts_agent

        return make_mcts_agent(deck, weights_path=weights)
    raise ValueError(f"unknown agent: {name!r} (expected random|neural|mcts)")


def play_match(
    p0: str,
    p1: str,
    deck_path: str = "deck/02_dragapult.csv",
    weights: str | None = None,
    html_path: str | None = "result.html",
    replay_path: str | None = "replay.json",
):
    """Run one rendered match; returns the finished kaggle environment."""
    deck = Deck.from_csv(deck_path).card_ids
    agents = [
        make_agent_by_name(p0, deck, weights),
        make_agent_by_name(p1, deck, weights),
    ]
    env = make("cabt", configuration={"decks": [deck, deck]})
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
) -> float:
    """Head-to-head win rate for p0's agent type, alternating sides."""
    deck = Deck.from_csv(deck_path).card_ids
    score = 0.0
    for g in range(games):
        a = make_agent_by_name(p0, deck, weights)
        b = make_agent_by_name(p1, deck, weights)
        agents = [a, b] if g % 2 == 0 else [b, a]
        side = g % 2
        env = make("cabt", configuration={"decks": [deck, deck]})
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


@app.command()
def main(
    p0: str = typer.Option("neural", help="player 0 agent: random|neural|mcts"),
    p1: str = typer.Option("random", help="player 1 agent: random|neural|mcts"),
    agent: str | None = typer.Option(None, help="agent profile name (resolves deck + weights)"),
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
    if games > 1:
        win_rate(p0, p1, games, deck_path=deck, weights=weights)
    else:
        play_match(
            p0, p1, deck_path=deck, weights=weights, html_path=html, replay_path=replay
        )


if __name__ == "__main__":
    app()
