"""Generate paired replays: the middleman with and without the setup agent.

The 400-game ablation says the setup agent costs 5-9 points of win rate but
not *why*. These replays are the qualitative half: identical stack, identical
opponent, the only difference being whether the setup agent plays each side's
own second turn (engine turn 3 going first, turn 4 going second).

Written in the **kaggle-env** ``{steps: [...]}`` shape the React viewers
(`replay/05_vite_react_app`, `replay/07_vite_react_cards`) load -- the same
format `ian_tools/generate_vite_replays.py` produces, *not* the flat
``vis.json`` list `ian_tools/generate_replays.py` writes for the external ptcg
visualizer. The two are not interchangeable: the React viewer rejects the flat
one with "not a replay file -- missing steps[]".

Each file also carries ``subAgentLog`` -- the middleman's routing lines per
step -- so the viewer can show which sub-agent made each decision. That is the
point here: you can find the setup agent's turn directly instead of counting.

    python -m darwinian_ml.setup_replays --games 6
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import typer
from kaggle_environments import make

from pkm.agents.dragapult_default_agent import make_dragapult_default_agent
from pkm.agents.first_turn_agent import make_first_turn_agent
from pkm.agents.random_agent import make_random_agent
from pkm.agents.singaporean_middleman import make_singaporean_middleman
from pkm.data import Deck

from .config import DarwinConfig
from .evaluate import _evolved_agent
from .opponent import BundleOpponent, extract_bundle

# Both React viewers serve their own public/; write to each so either opens them.
VIEWER_DIRS = [
    Path("replay/07_vite_react_cards/public/ian"),
    Path("replay/05_vite_react_app/public/ian"),
]


class _Cap:
    """Per-call capture of the middleman's log_sink output."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def sink(self, msg: str) -> None:
        self.lines.append(msg)


def _wrap(agent, cap: _Cap, store: dict[int, list[str]]):
    """Store each real decision's routing lines, keyed by kaggle step index."""

    def wrapped(obs: dict) -> list[int]:
        cap.lines = []
        picks = agent(obs)
        step = obs.get("step")
        if step is not None and obs.get("select") is not None:
            store[int(step)] = list(cap.lines)
        return picks

    return wrapped


def _strip_heavy(data: dict) -> None:
    """Drop the huge per-obs search_begin_input blob the viewer never reads."""
    for step in data.get("steps", []):
        for entry in step:
            obs = entry.get("observation")
            if isinstance(obs, dict):
                obs.pop("search_begin_input", None)


def _play(make_agent, opponent, deck: list[int], our_seat: int) -> tuple[dict, dict]:
    cap = _Cap()
    store: dict[int, list[str]] = {}
    opponent.new_game()

    def opponent_agent(obs: dict) -> list[int]:
        return opponent.act(obs)

    agents = [None, None]
    agents[our_seat] = _wrap(make_agent(cap), cap, store)
    agents[1 - our_seat] = opponent_agent
    decks = [None, None]
    decks[our_seat] = list(deck)
    decks[1 - our_seat] = list(opponent.deck)

    env = make("cabt", configuration={"decks": decks})
    env.run(agents)
    raw = env.toJSON()
    data = json.loads(raw) if isinstance(raw, str) else raw
    _strip_heavy(data)
    data["subAgentLog"] = [store.get(i, []) for i in range(len(data["steps"]))]

    reward = env.steps[-1][our_seat].get("reward") or 0
    info = {
        "outcome": "WIN" if reward > 0 else "loss" if reward < 0 else "draw",
        "steps": len(data["steps"]),
        "seat": our_seat,
    }
    return data, info


app = typer.Typer(add_completion=False)


@app.command()
def main(
    games: int = typer.Option(6, help="games per configuration"),
    bundle: str = typer.Option(DarwinConfig.bundle),
    deck_path: str = typer.Option(DarwinConfig.deck_path),
    out_dir: str = typer.Option(DarwinConfig.out_dir),
    checkpoint: str = typer.Option(
        "darwinian_ml/runs/default_dragapult_darwinian/best.pt"
    ),
    variants: str = typer.Option(
        "all", help="all | setup | darwin -- which configurations to record"
    ),
    seed: int = typer.Option(0),
) -> None:
    random.seed(seed)
    deck = Deck.from_csv(deck_path).card_ids
    bundle_dir = extract_bundle(bundle, Path(out_dir) / "opponents")
    for d in VIEWER_DIRS:
        d.mkdir(parents=True, exist_ok=True)

    def with_setup(cap):
        return make_singaporean_middleman(deck, log_sink=cap.sink)

    def without_setup(cap):
        return make_singaporean_middleman(
            deck,
            agents={
                "dragapult_default": make_dragapult_default_agent(deck),
                "dragapult_setup": make_dragapult_default_agent(deck),
                "first_turn": make_first_turn_agent(deck),
                "random": make_random_agent(deck),
            },
            log_sink=cap.sink,
        )

    def search_setup(cap):
        """Setup turn played by SEARCH over the rubric instead of a net."""
        from pkm.agents.dragapult_setup_search_agent import (
            make_dragapult_setup_search_agent,
        )

        return make_singaporean_middleman(
            deck,
            agents={
                "dragapult_default": make_dragapult_default_agent(deck),
                "dragapult_setup": make_dragapult_setup_search_agent(
                    deck, log_sink=cap.sink
                ),
                "first_turn": make_first_turn_agent(deck),
                "random": make_random_agent(deck),
            },
            log_sink=cap.sink,
        )

    def darwin_scaffolded(cap):
        """Evolved policy in the default slot, full scaffolding around it."""
        from pkm.agents.dragapult_setup_agent import make_dragapult_setup_agent

        return make_singaporean_middleman(
            deck,
            agents={
                "dragapult_default": _evolved_agent(checkpoint, deck),
                "dragapult_setup": make_dragapult_setup_agent(deck, log_sink=cap.sink),
                "first_turn": make_first_turn_agent(deck),
                "random": make_random_agent(deck),
            },
            log_sink=cap.sink,
        )

    def darwin_no_setup(cap):
        """Evolved policy in the default slot, setup agent removed."""
        return make_singaporean_middleman(
            deck,
            agents={
                "dragapult_default": _evolved_agent(checkpoint, deck),
                "dragapult_setup": _evolved_agent(checkpoint, deck),
                "first_turn": make_first_turn_agent(deck),
                "random": make_random_agent(deck),
            },
            log_sink=cap.sink,
        )

    configs: list[tuple[str, object]] = []
    if variants in ("all", "setup"):
        configs += [
            ("setup_RL", with_setup),
            ("setup_SEARCH", search_setup),
            ("setup_NONE", without_setup),
        ]
    if variants in ("all", "darwin"):
        if not Path(checkpoint).is_file():
            raise SystemExit(f"no evolved checkpoint at {checkpoint}")
        configs += [
            ("darwin_scaffolded", darwin_scaffolded),
            ("darwin_no_setup", darwin_no_setup),
        ]
    if not configs:
        raise SystemExit(f"unknown --variants {variants!r} (all | setup | darwin)")

    # Prefix by opponent at *write* time. Renaming files afterwards leaves
    # index.json pointing at paths that no longer exist, and Vite answers a
    # missing file with index.html -- which surfaces as the JSON parser
    # choking on "<!doctype". Naming them correctly here keeps the manifest
    # true by construction.
    tag = Path(bundle).name.split("_")[0][:4].lower()
    if "abomasnow" in Path(bundle).name:
        tag = "abom"
    elif "alakazam" in Path(bundle).name:
        tag = "alak"

    manifest = []
    with BundleOpponent(bundle_dir) as opponent:
        print(f"opponent: {Path(bundle).name}\n")
        for label, factory in configs:
            for g in range(games):
                data, info = _play(factory, opponent, deck, our_seat=g % 2)
                fname = f"{tag}_{label}_{g:02d}.json"
                for d in VIEWER_DIRS:
                    with open(d / fname, "w") as fh:
                        json.dump(data, fh)
                manifest.append(
                    {
                        "file": fname,
                        "title": f"vs {tag} - {label} #{g} ({info['outcome']})",
                        "outcome": info["outcome"],
                        "steps": info["steps"],
                    }
                )
                print(
                    f"  {label:<18} game {g}  seat{info['seat']}  "
                    f"{info['outcome']:<4}  {info['steps']:4d} steps -> {fname}",
                    flush=True,
                )

    # Merge, don't clobber: the other opponent's replays are still on disk and
    # a fresh manifest listing only this run would orphan them in the picker.
    for d in VIEWER_DIRS:
        existing = []
        idx = d / "index.json"
        if idx.is_file():
            try:
                existing = [
                    e
                    for e in json.load(open(idx))
                    if (d / e["file"]).is_file()
                    and e["file"] not in {m["file"] for m in manifest}
                ]
            except Exception:
                existing = []
        with open(idx, "w") as fh:
            json.dump(existing + manifest, fh, indent=2)

    print(f"\nwrote {len(manifest)} replays to {VIEWER_DIRS[0]}/")
    print("Open, e.g.:")
    print("  http://localhost:5175/?replay=/ian/setup_ON_00.json")
    print("  http://localhost:5175/?replay=/ian/setup_OFF_00.json")
    print(
        "\nThe viewer's sub-agent log names the agent behind each decision, so "
        "the setup agent's turn is findable directly."
    )


if __name__ == "__main__":
    app()
