"""Typer + rich command line for training agent_000_dragapult.

This is the agent's own, self-contained training entry point — it does **not**
use the ``pkm/rl`` infrastructure. All artifacts (checkpoints, metric logs) are
written under a single data directory, defaulting to::

    <repo>/pkm_data/new_agents/agent_000_dragapult/

Commands::

    cli.py info                 # show config, engine backend, paths
    cli.py smoke                # tiny end-to-end sanity run
    cli.py train  [opts]        # PPO self-play training run
    cli.py resume [opts]        # continue from checkpoints/latest.pt
    cli.py eval   [opts]        # win-rate of a checkpoint vs random

Run ``cli.py <command> --help`` for the full flag list.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    # Type-only: gives editors/ruff the `Config` symbol for annotations without
    # importing it at runtime (that would pull in torch). Annotations are strings
    # under `from __future__ import annotations`, so they're never evaluated.
    from pkm.new_agents.agent_000_dragapult.config import Config

# NOTE: heavy imports (config/train/engine -> torch) are deferred into the command
# bodies below so that merely importing this module (e.g. when the main `pkm` CLI
# registers it as a subcommand) stays cheap and torch-free. Only `typer`/`rich`
# and stdlib load at import time.

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Self-contained PPO self-play trainer for agent_000_dragapult.",
)
console = Console()

# Default artifact root: <repo>/pkm_data/new_agents/agent_000_dragapult
#   cli.py -> agent_000_dragapult -> new_agents -> pkm -> <repo>
DATA_DIR = (
    Path(__file__).resolve().parents[3]
    / "pkm_data"
    / "new_agents"
    / "agent_000_dragapult"
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _paths(data_dir: Path) -> dict[str, Path]:
    return {
        "data": data_dir,
        "ckpt": data_dir / "checkpoints",
        "logs": data_dir / "logs",
        "csv": data_dir / "logs" / "train.csv",
    }


def _build_config(
    *,
    workers: int,
    lr: float,
    gamma: float,
    lam: float,
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
    epochs: int,
    minibatch_size: int,
    seed: int,
    ckpt_every: int,
) -> Config:
    from pkm.new_agents.agent_000_dragapult.config import Config, RunConfig, TrainConfig

    train = dataclasses.replace(
        TrainConfig(),
        num_workers=workers,
        lr=lr,
        gamma=gamma,
        gae_lambda=lam,
        clip_eps=clip_eps,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        epochs_per_update=epochs,
        minibatch_size=minibatch_size,
        seed=seed,
    )
    run = dataclasses.replace(RunConfig(), checkpoint_every_updates=ckpt_every)
    return Config(train=train, run=run)


def _config_table(cfg: Config) -> Table:
    t = Table(show_header=False, box=None, pad_edge=False)
    t.add_column(style="dim")
    t.add_column(style="bold")
    tc = cfg.train
    for k, v in [
        ("workers", tc.num_workers),
        ("lr", tc.lr),
        ("gamma", tc.gamma),
        ("gae_lambda", tc.gae_lambda),
        ("clip_eps", tc.clip_eps),
        ("entropy_coef", tc.entropy_coef),
        ("value_coef", tc.value_coef),
        ("epochs/update", tc.epochs_per_update),
        ("minibatch", tc.minibatch_size),
        ("seed", tc.seed),
        ("ckpt_every", cfg.run.checkpoint_every_updates),
        ("config_hash", cfg.hash()),
    ]:
        t.add_row(str(k), str(v))
    return t


def _make_on_update(target: int):
    def on_update(i: int, total: int, stats: dict) -> None:
        ev = stats.get("eval_win_rate", "")
        ev_s = f"[green]{ev:.1%}[/]" if isinstance(ev, (int, float)) else "[dim]-[/]"
        console.print(
            f"[bold cyan]{i:>4}[/]/[cyan]{total}[/]  "
            f"games=[bold]{stats.get('games', 0):>3}[/]  "
            f"steps=[bold]{stats.get('steps', 0):>5}[/]  "
            f"pol=[yellow]{stats.get('pol_loss', 0):+.4f}[/]  "
            f"val=[yellow]{stats.get('val_loss', 0):.4f}[/]  "
            f"ent=[magenta]{stats.get('entropy', 0):.3f}[/]  "
            f"p0/p1={stats.get('p0_win', 0):.0%}/{stats.get('p1_win', 0):.0%}  "
            f"eval={ev_s}"
        )

    return on_update


def _run_training(
    cfg: Config,
    updates: int,
    games: int,
    data_dir: Path,
    *,
    resume: bool,
    eval_every: int,
    eval_games: int,
    title: str,
) -> None:
    # Import here so `--help` / `info` don't pay the heavy engine + torch import.
    from pkm.new_agents.agent_000_dragapult.train import train

    p = _paths(data_dir)
    p["ckpt"].mkdir(parents=True, exist_ok=True)
    p["logs"].mkdir(parents=True, exist_ok=True)

    console.print(
        Panel.fit(_config_table(cfg), title=f"[bold]{title}[/]", border_style="cyan")
    )
    console.print(
        f"[dim]updates=[/]{updates}  [dim]games/update=[/]{games}  "
        f"[dim]resume=[/]{resume}  [dim]eval_every=[/]{eval_every}\n"
        f"[dim]checkpoints ->[/] {p['ckpt']}\n[dim]metrics ->[/] {p['csv']}\n"
    )

    ts = train(
        cfg,
        updates=updates,
        games_per_update=games,
        ckpt_dir=p["ckpt"],
        resume=resume,
        eval_every=eval_every,
        eval_games=eval_games,
        log_csv=p["csv"],
        on_update=_make_on_update(updates),
    )
    console.print(
        f"\n[bold green]done[/] at update [bold]{ts.update_idx}[/] · "
        f"latest -> {p['ckpt'] / 'latest.pt'}"
    )


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


@app.command()
def info(
    data_dir: Path = typer.Option(
        DATA_DIR,
        "--output-dir",
        "-o",
        help="Output/artifact root directory (checkpoints + logs).",
    ),
) -> None:
    """Print the default config, engine backend, and artifact paths."""
    from pkm.engine import ENGINE_BACKEND, ENGINE_LIB_PATH

    from pkm.new_agents.agent_000_dragapult.config import Config

    p = _paths(data_dir)
    console.print(
        Panel.fit(
            _config_table(Config()),
            title="[bold]default config[/]",
            border_style="cyan",
        )
    )
    tbl = Table(show_header=False, box=None)
    tbl.add_column(style="dim")
    tbl.add_column()
    tbl.add_row("engine backend", str(ENGINE_BACKEND))
    tbl.add_row("engine lib", str(ENGINE_LIB_PATH))
    for k in ("data", "ckpt", "logs", "csv"):
        exists = "[green]✓[/]" if p[k].exists() else "[dim]—[/]"
        tbl.add_row(k, f"{p[k]}  {exists}")
    console.print(tbl)


@app.command()
def smoke(
    data_dir: Path = typer.Option(
        DATA_DIR,
        "--output-dir",
        "-o",
        help="Output/artifact root directory (checkpoints + logs).",
    ),
    workers: int = typer.Option(
        1, help="Rollout workers (1 = verified single-process path)."
    ),
) -> None:
    """Tiny end-to-end sanity run: 2 updates x 2 games. Proves the pipeline works."""
    cfg = _build_config(
        workers=workers,
        lr=3e-4,
        gamma=0.997,
        lam=0.95,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        epochs=2,
        minibatch_size=64,
        seed=0,
        ckpt_every=2,
    )
    _run_training(
        cfg,
        updates=2,
        games=2,
        data_dir=data_dir,
        resume=False,
        eval_every=2,
        eval_games=10,
        title="smoke",
    )


@app.command()
def train(
    updates: int = typer.Option(200, help="Number of PPO updates to run."),
    games: int = typer.Option(16, help="Self-play games collected per update."),
    workers: int = typer.Option(
        1, help="Rollout workers (parallel.py path is unverified; 1 is safe)."
    ),
    lr: float = typer.Option(3e-4, help="Adam learning rate."),
    gamma: float = typer.Option(0.997, help="Discount factor."),
    lam: float = typer.Option(0.95, help="GAE lambda."),
    clip_eps: float = typer.Option(0.2, help="PPO clip epsilon."),
    entropy_coef: float = typer.Option(0.01, help="Entropy bonus coefficient."),
    value_coef: float = typer.Option(0.5, help="Value loss coefficient."),
    epochs: int = typer.Option(4, help="PPO epochs per update."),
    minibatch_size: int = typer.Option(64, help="Minibatch size (decisions)."),
    seed: int = typer.Option(0, help="RNG seed."),
    eval_every: int = typer.Option(
        10, help="Evaluate vs random every N updates (0 = never)."
    ),
    eval_games: int = typer.Option(100, help="Games per evaluation."),
    ckpt_every: int = typer.Option(50, help="Checkpoint snapshot every N updates."),
    data_dir: Path = typer.Option(
        DATA_DIR,
        "--output-dir",
        "-o",
        help="Output/artifact root directory (checkpoints + logs).",
    ),
    resume: bool = typer.Option(False, help="Resume from checkpoints/latest.pt."),
) -> None:
    """Run PPO self-play training."""
    cfg = _build_config(
        workers=workers,
        lr=lr,
        gamma=gamma,
        lam=lam,
        clip_eps=clip_eps,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        epochs=epochs,
        minibatch_size=minibatch_size,
        seed=seed,
        ckpt_every=ckpt_every,
    )
    _run_training(
        cfg,
        updates=updates,
        games=games,
        data_dir=data_dir,
        resume=resume,
        eval_every=eval_every,
        eval_games=eval_games,
        title="train" if not resume else "resume",
    )


@app.command()
def resume(
    updates: int = typer.Option(50, help="Additional PPO updates to run."),
    games: int = typer.Option(16, help="Self-play games collected per update."),
    workers: int = typer.Option(1, help="Rollout workers."),
    eval_every: int = typer.Option(
        10, help="Evaluate vs random every N updates (0 = never)."
    ),
    eval_games: int = typer.Option(100, help="Games per evaluation."),
    data_dir: Path = typer.Option(
        DATA_DIR,
        "--output-dir",
        "-o",
        help="Output/artifact root directory (checkpoints + logs).",
    ),
) -> None:
    """Resume training from checkpoints/latest.pt (config is restored from the checkpoint)."""
    from pkm.new_agents.agent_000_dragapult.train import TrainState

    p = _paths(data_dir)
    latest = p["ckpt"] / "latest.pt"
    if not latest.exists():
        console.print(f"[red]no checkpoint at[/] {latest}")
        raise typer.Exit(1)
    cfg = TrainState.load(latest).cfg
    cfg = dataclasses.replace(
        cfg, train=dataclasses.replace(cfg.train, num_workers=workers)
    )
    _run_training(
        cfg,
        updates=updates,
        games=games,
        data_dir=data_dir,
        resume=True,
        eval_every=eval_every,
        eval_games=eval_games,
        title="resume",
    )


@app.command()
def eval(
    games: int = typer.Option(100, help="Number of games vs the random baseline."),
    checkpoint: Optional[Path] = typer.Option(
        None, help="Checkpoint (defaults to checkpoints/latest.pt)."
    ),
    seed: int = typer.Option(0, help="Baseline RNG seed."),
    data_dir: Path = typer.Option(
        DATA_DIR,
        "--output-dir",
        "-o",
        help="Output/artifact root directory (checkpoints + logs).",
    ),
) -> None:
    """Report a checkpoint's win-rate vs the random baseline (alternating seats)."""
    from pkm.new_agents.agent_000_dragapult.eval import winrate_vs_random
    from pkm.new_agents.agent_000_dragapult.train import TrainState

    ckpt = checkpoint or (_paths(data_dir)["ckpt"] / "latest.pt")
    if not ckpt.exists():
        console.print(f"[red]no checkpoint at[/] {ckpt}")
        raise typer.Exit(1)
    console.print(f"[dim]evaluating[/] {ckpt} [dim]over[/] {games} [dim]games…[/]")
    model = TrainState.load(ckpt).model
    ev = winrate_vs_random(model, n_games=games, seed=seed)

    t = Table(title="vs random", title_style="bold")
    t.add_column("metric", style="dim")
    t.add_column("value", justify="right", style="bold")
    t.add_row("win rate", f"[green]{ev['win_rate']:.1%}[/]")
    t.add_row("W / L / D", f"{ev['wins']} / {ev['losses']} / {ev['draws']}")
    t.add_row("games", str(ev["n"]))
    console.print(t)


if __name__ == "__main__":
    app()
