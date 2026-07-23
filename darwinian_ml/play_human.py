"""Play a *packed submission bundle* by hand, in the TUI.

`pkm play --p0 human` can only run mirror matches: it hands the same deck to
both seats. This runs a **cross-deck** game instead — you play `--deck`, the
bundle plays whatever deck it was packed with — by reusing the same subprocess
bridge the darwinian experiment fights (`opponent.BundleOpponent`).

Nothing decides the decks but the two agents themselves: the cabt env reads
each side's 60-card list from that side's first action, so you submitting
Dragapult and the bundle submitting Alakazam is all it takes.

    python -m darwinian_ml.play_human --bundle darwinian_ml/runs/opponents/alakazam_df

Lives here, not in `pkm/`, to keep the dependency one-way: `darwinian_ml`
imports `pkm`, never the reverse. `ThreadedEnvSession` takes the built
opponent as a plain callable and stays ignorant of bundles.
"""

from __future__ import annotations

from pathlib import Path

import typer

from pkm.data import Deck
from pkm.tui.app import BattleApp
from pkm.tui.session import ThreadedEnvSession

from .opponent import BundleOpponent, extract_bundle

app = typer.Typer(add_completion=False)


@app.command()
def main(
    bundle: str = typer.Option(
        "darwinian_ml/runs/opponents/alakazam_df",
        help="submission tarball, or an already-extracted bundle directory",
    ),
    deck_path: str = typer.Option("deck/03_pult_munki.csv", help="the deck YOU play"),
    seat: int = typer.Option(0, help="your seat: 0 goes first, 1 goes second"),
    html: str = typer.Option("result.html"),
    replay: str = typer.Option("replay.json"),
) -> None:
    if seat not in (0, 1):
        raise SystemExit("seat must be 0 or 1")
    src = Path(bundle)
    bundle_dir = src if (src / "main.py").is_file() else extract_bundle(src, src.parent)

    deck = Deck.from_csv(deck_path).card_ids
    with BundleOpponent(bundle_dir) as opponent:
        session = ThreadedEnvSession(
            deck=deck,
            human_index=seat,
            opponent=bundle_dir.name,
            html_path=html,
            replay_path=replay,
            # A bound method's __code__.co_argcount counts `self`, and kaggle
            # uses that count to decide how many args to pass. Wrap it.
            opponent_agent=lambda obs: opponent.act(obs),
        )
        BattleApp(session).run()


if __name__ == "__main__":
    app()
