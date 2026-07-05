"""Deck management CLI.

Usage:
    python -m pkm.cli_deck list
    python -m pkm.cli_deck show 00_basic
    python -m pkm.cli_deck convert 00_basic --to json
"""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from pkm.data import Deck, list_decks, resolve_deck

app = typer.Typer(help=__doc__)
console = Console()


@app.command()
def list() -> None:
    """List available decks in the deck/ directory."""
    decks = list_decks()
    if not decks:
        console.print("[yellow]No decks found in deck/ directory[/yellow]")
        raise typer.Exit(1)

    table = Table(title="Available Decks")
    table.add_column("Name", style="cyan")
    table.add_column("Format", style="green")
    table.add_column("Cards", justify="right")

    for path in decks:
        fmt = path.suffix.lstrip(".")
        if path.suffix == ".csv":
            count = sum(1 for line in path.open() if line.strip())
        else:
            with open(path) as f:
                count = sum(e["count"] for e in json.load(f))
        table.add_row(path.stem, fmt, str(count))

    console.print(table)


@app.command()
def show(
    name: str = typer.Argument(help="Deck name (with or without extension)"),
) -> None:
    """Show the contents of a deck."""
    path = resolve_deck(name)
    if path.suffix == ".json":
        with open(path) as f:
            entries = json.load(f)
    else:
        deck = Deck.from_csv(path)
        deck.to_json("/tmp/_deck_show.json")
        with open("/tmp/_deck_show.json") as f:
            entries = json.load(f)

    table = Table(title=f"Deck: {path.name}")
    table.add_column("ID", justify="right", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Count", justify="right")

    total = 0
    for entry in entries:
        table.add_row(str(entry["id"]), entry["name"], str(entry["count"]))
        total += entry["count"]

    console.print(table)
    console.print(f"Total: {total} cards, {len(entries)} unique")


@app.command()
def convert(
    name: str = typer.Argument(help="Deck name (with or without extension)"),
    to: str = typer.Option("json", help="Output format: json or csv"),
    out: str | None = typer.Option(
        None, help="Output path (default: same name, new extension)"
    ),
) -> None:
    """Convert a deck between CSV and JSON formats."""
    path = resolve_deck(name)
    deck = Deck.from_csv(path) if path.suffix == ".csv" else Deck.from_json(path)

    if out:
        out_path = Path(out)
    else:
        out_path = path.with_suffix(".json" if to == "json" else ".csv")

    if to == "json":
        deck.to_json(out_path)
    else:
        deck.to_csv(out_path)

    console.print(f"Converted [cyan]{path.name}[/cyan] -> [green]{out_path}[/green]")


if __name__ == "__main__":
    app()
