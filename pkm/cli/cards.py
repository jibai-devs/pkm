"""Card data CLI.

Usage:
    python -m pkm.cli.cards dump cards.json
"""

import json
from dataclasses import asdict
from pathlib import Path

import typer
from rich.console import Console

from pkm.data.card_data import get_card_data, get_attack_data

app = typer.Typer(help="Card data commands")
console = Console()


@app.callback()
def _root() -> None:
    """Card data commands."""


@app.command()
def dump(
    out: Path = typer.Argument(help="Output JSON file path"),
) -> None:
    """Dump all card data to a pretty-printed JSON file."""
    cards = get_card_data()
    attacks = get_attack_data()

    card_list = [asdict(c) for c in cards.values()]
    attack_list = [asdict(a) for a in attacks.values()]

    data = {
        "cards": card_list,
        "attacks": attack_list,
    }

    out.write_text(json.dumps(data, indent=2) + "\n")
    console.print(
        f"Dumped [cyan]{len(card_list)}[/cyan] cards and "
        f"[cyan]{len(attack_list)}[/cyan] attacks to [green]{out}[/green]"
    )


if __name__ == "__main__":
    app()
