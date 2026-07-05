"""Play a match between two named agents; save an HTML replay + JSON log.

Usage:
    python -m pkm.rl.play --p0 neural --p1 random
    python -m pkm.rl.play --p0 mcts --p1 neural --html mcts_vs_neural.html
    python -m pkm.rl.play --p0 neural --p1 random --games 20   # win-rate only

Agents: random | neural (greedy policy, needs pkm/policy.npz) | mcts
The HTML file is self-contained — open it in a browser to watch the match.
The JSON log is kaggle-environments' full episode record (per-step
observations, actions, rewards); reload it with json.load for analysis.
"""

import argparse
import json
from typing import Callable

from kaggle_environments import make

from pkm.agents import make_neural_agent, make_random_agent
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
    deck_path: str = "deck.csv",
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
    deck_path: str = "deck.csv",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0", default="neural", help="random|neural|mcts")
    parser.add_argument("--p1", default="random", help="random|neural|mcts")
    parser.add_argument("--deck", default="deck.csv")
    parser.add_argument("--weights", default=None, help="path to policy .npz")
    parser.add_argument("--html", default="result.html")
    parser.add_argument("--replay", default="replay.json")
    parser.add_argument(
        "--games", type=int, default=1, help=">1: win-rate mode, no replay"
    )
    args = parser.parse_args()
    if args.games > 1:
        win_rate(
            args.p0, args.p1, args.games, deck_path=args.deck, weights=args.weights
        )
    else:
        play_match(
            args.p0,
            args.p1,
            deck_path=args.deck,
            weights=args.weights,
            html_path=args.html,
            replay_path=args.replay,
        )


if __name__ == "__main__":
    main()
