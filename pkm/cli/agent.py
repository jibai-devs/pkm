"""Agent profile management CLI.

Usage:
    pkm agent list
"""

import csv
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from pkm.agents.profile import AgentProfile
from pkm.rl.reward_terms import DEFAULT_WEIGHTS, load_weights

app = typer.Typer(help=__doc__)
console = Console()


@app.callback()
def _root() -> None:
    """Agent profile commands."""


def _deck_card_count(path: Path) -> int | None:
    """Total card count for a deck file, or None if it doesn't exist."""
    if not path.is_file():
        return None
    if path.suffix == ".json":
        with open(path) as f:
            return sum(e["count"] for e in json.load(f))
    return sum(1 for line in path.open() if line.strip())


def _last_metrics_row(metrics_path: Path) -> dict | None:
    if not metrics_path.is_file():
        return None
    last = None
    with open(metrics_path, newline="") as f:
        for row in csv.DictReader(f):
            last = row
    return last


def _weight_overrides(reward_weights_path: Path) -> str:
    if not reward_weights_path.is_file():
        return "default"
    weights = load_weights(reward_weights_path)
    changed = sum(1 for name, val in weights.items() if val != DEFAULT_WEIGHTS[name])
    return f"{changed} changed" if changed else "default"


@app.command(name="list")
def list_() -> None:
    """List agent profiles under agents/, with deck, checkpoint, and reward-weight status."""
    names = AgentProfile.list_agents()
    if not names:
        console.print("[yellow]No agent profiles found in agents/[/yellow]")
        raise typer.Exit(1)

    table = Table(title="Agent Profiles")
    table.add_column("Name", style="cyan")
    table.add_column("Deck", style="green")
    table.add_column("PPO ckpt", justify="center")
    table.add_column("Exit ckpt", justify="center")
    table.add_column("Last iter", justify="right")
    table.add_column("Eval win rate", justify="right")
    table.add_column("Reward weights")

    for name in names:
        profile = AgentProfile(name)
        card_count = _deck_card_count(profile.deck_path)
        deck_label = (
            f"{profile.deck_path.name} ({card_count})"
            if card_count is not None
            else f"[red]{profile.deck_path.name} missing[/red]"
        )
        ppo_ckpt = "[green]yes[/green]" if profile.latest_checkpoint("ppo") else "-"
        exit_ckpt = "[green]yes[/green]" if profile.latest_checkpoint("exit") else "-"

        last_row = _last_metrics_row(profile.metrics_dir / "ppo_train.csv")
        last_iter = last_row["iter"] if last_row else "-"
        eval_wr = "-"
        if last_row:
            # eval_win_rate is only populated every --eval-every iterations;
            # scan backwards for the most recent row that has one.
            with open(profile.metrics_dir / "ppo_train.csv", newline="") as f:
                for row in reversed(list(csv.DictReader(f))):
                    if row.get("eval_win_rate"):
                        eval_wr = f"{float(row['eval_win_rate']):.1%}"
                        break

        table.add_row(
            name,
            deck_label,
            ppo_ckpt,
            exit_ckpt,
            last_iter,
            eval_wr,
            _weight_overrides(profile.reward_weights_path),
        )

    console.print(table)


if __name__ == "__main__":
    app()
