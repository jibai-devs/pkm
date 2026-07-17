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
from datetime import datetime
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
        "runs": data_dir / "runs",  # TensorBoard event files
        "wandb": data_dir / "wandb",  # wandb (offline) run dirs
        "sweeps": data_dir / "sweeps",  # Optuna sqlite studies
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


def _build_observers(
    p: dict[str, Path],
    *,
    run_name: str,
    tb: bool,
    log_dir: Optional[Path],
    wandb_project: Optional[str],
    wandb_mode: str,
):
    """Assemble the metric sinks (observers) for a run from the enabled backends."""
    from pkm.new_agents.agent_000_dragapult.monitor import (
        ConsoleSink,
        CsvSink,
        TensorBoardSink,
        WandbSink,
    )

    observers = [ConsoleSink(console), CsvSink(p["csv"])]
    tb_dir = None
    if tb:
        tb_dir = Path(log_dir) if log_dir else p["runs"] / run_name
        observers.append(TensorBoardSink(tb_dir))
    if wandb_project:
        # wandb creates its own `wandb/` subdir inside `dir`, so point at the data
        # root -> runs land at <output>/wandb/ (not <output>/wandb/wandb/).
        observers.append(
            WandbSink(
                project=wandb_project, mode=wandb_mode, name=run_name, dir=p["data"]
            )
        )
    return observers, tb_dir


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
    tb: bool = True,
    log_dir: Optional[Path] = None,
    wandb_project: Optional[str] = None,
    wandb_mode: str = "offline",
    run_name: Optional[str] = None,
) -> None:
    # Import here so `--help` / `info` don't pay the heavy engine + torch import.
    from pkm.new_agents.agent_000_dragapult.train import train

    p = _paths(data_dir)
    p["ckpt"].mkdir(parents=True, exist_ok=True)
    p["logs"].mkdir(parents=True, exist_ok=True)
    run_name = run_name or f"{title}-{datetime.now():%Y%m%d-%H%M%S}"

    observers, tb_dir = _build_observers(
        p,
        run_name=run_name,
        tb=tb,
        log_dir=log_dir,
        wandb_project=wandb_project,
        wandb_mode=wandb_mode,
    )

    console.print(
        Panel.fit(_config_table(cfg), title=f"[bold]{title}[/]", border_style="cyan")
    )
    console.print(
        f"[dim]updates=[/]{updates}  [dim]games/update=[/]{games}  "
        f"[dim]resume=[/]{resume}  [dim]eval_every=[/]{eval_every}\n"
        f"[dim]checkpoints ->[/] {p['ckpt']}\n[dim]metrics ->[/] {p['csv']}"
    )
    if tb_dir is not None:
        console.print(
            f"[dim]tensorboard ->[/] {tb_dir}  [dim](tensorboard --logdir {p['runs']})[/]"
        )
    if wandb_project:
        console.print(
            f"[dim]wandb ->[/] project={wandb_project} mode={wandb_mode} dir={p['wandb']}"
        )
    console.print()

    ts = train(
        cfg,
        updates=updates,
        games_per_update=games,
        ckpt_dir=p["ckpt"],
        resume=resume,
        eval_every=eval_every,
        eval_games=eval_games,
        observers=observers,
        run_name=run_name,
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
    tb: bool = typer.Option(
        True, "--tb/--no-tb", help="Log to TensorBoard (<output>/runs/)."
    ),
    log_dir: Optional[Path] = typer.Option(
        None, help="TensorBoard log dir (default: <output>/runs/<run-name>)."
    ),
    wandb_project: Optional[str] = typer.Option(
        None, help="Enable Weights & Biases logging to this project (else off)."
    ),
    wandb_mode: str = typer.Option(
        "offline",
        help="wandb mode: offline (local, default), online (cloud), disabled.",
    ),
    run_name: Optional[str] = typer.Option(
        None, help="Run name (TB subdir + wandb run name)."
    ),
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
        tb=tb,
        log_dir=log_dir,
        wandb_project=wandb_project,
        wandb_mode=wandb_mode,
        run_name=run_name,
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
    tb: bool = typer.Option(
        True, "--tb/--no-tb", help="Log to TensorBoard (<output>/runs/)."
    ),
    wandb_project: Optional[str] = typer.Option(
        None, help="Enable Weights & Biases logging to this project (else off)."
    ),
    wandb_mode: str = typer.Option(
        "offline",
        help="wandb mode: offline (local, default), online (cloud), disabled.",
    ),
    run_name: Optional[str] = typer.Option(
        None, help="Run name (TB subdir + wandb run name)."
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
        tb=tb,
        wandb_project=wandb_project,
        wandb_mode=wandb_mode,
        run_name=run_name,
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


@app.command()
def sweep(
    trials: int = typer.Option(30, help="Number of Optuna trials."),
    updates: int = typer.Option(15, help="PPO updates per trial (keep short)."),
    games: int = typer.Option(32, help="Self-play games per update."),
    workers: int = typer.Option(8, help="Rollout workers per trial."),
    eval_games: int = typer.Option(100, help="Games per evaluation (the objective)."),
    study: str = typer.Option(
        "dragapult_ppo", help="Optuna study name (sqlite, resumable)."
    ),
    seed: int = typer.Option(0, help="Base RNG seed (offset per trial)."),
    data_dir: Path = typer.Option(
        DATA_DIR,
        "--output-dir",
        "-o",
        help="Output/artifact root directory (checkpoints + logs).",
    ),
) -> None:
    """Optuna hyperparameter sweep — maximize eval win-rate vs random.

    Each trial samples lr/entropy/clip/epochs/minibatch/gamma/lam, runs a short
    training, and returns its win-rate vs random. The study is SQLite-backed under
    <output>/sweeps/<study>.db (resumable; view with `optuna-dashboard`). Trials
    are pruned early via reported intermediate evals (MedianPruner).
    """
    import optuna

    from pkm.new_agents.agent_000_dragapult.eval import winrate_vs_random
    from pkm.new_agents.agent_000_dragapult.monitor import (
        MetricSink,
        StopTraining,
        TensorBoardSink,
    )
    from pkm.new_agents.agent_000_dragapult.train import train

    p = _paths(data_dir)
    p["sweeps"].mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{p['sweeps'] / (study + '.db')}"

    class PruningSink(MetricSink):
        """Report intermediate eval win-rate to Optuna and prune hopeless trials."""

        def __init__(self, trial: optuna.Trial):
            self.trial = trial

        def log(self, update: int, total: int, stats: dict) -> None:
            ev = stats.get("eval_win_rate")
            if isinstance(ev, (int, float)):
                self.trial.report(ev, update)
                if self.trial.should_prune():
                    raise StopTraining

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        entropy_coef = trial.suggest_float("entropy_coef", 1e-4, 5e-2, log=True)
        clip_eps = trial.suggest_float("clip_eps", 0.1, 0.3)
        epochs = trial.suggest_int("epochs", 2, 6)
        minibatch = trial.suggest_categorical("minibatch_size", [32, 64, 128])
        gamma = trial.suggest_float("gamma", 0.95, 0.999)
        lam = trial.suggest_float("lam", 0.9, 0.99)
        cfg = _build_config(
            workers=workers,
            lr=lr,
            gamma=gamma,
            lam=lam,
            clip_eps=clip_eps,
            entropy_coef=entropy_coef,
            value_coef=0.5,
            epochs=epochs,
            minibatch_size=minibatch,
            seed=seed + trial.number,
            ckpt_every=updates,
        )
        trial_dir = p["sweeps"] / study / f"trial_{trial.number}"
        observers = [
            TensorBoardSink(p["runs"] / f"sweep-{study}" / f"trial_{trial.number}"),
            PruningSink(trial),
        ]
        try:
            ts = train(
                cfg,
                updates=updates,
                games_per_update=games,
                ckpt_dir=trial_dir / "checkpoints",
                eval_every=max(1, updates // 3),
                eval_games=eval_games,
                observers=observers,
                run_name=f"{study}-trial{trial.number}",
            )
        except StopTraining as exc:  # pruned mid-run
            raise optuna.TrialPruned() from exc
        return winrate_vs_random(ts.model, n_games=eval_games, seed=seed)["win_rate"]

    console.print(
        f"[bold]sweep[/] study=[cyan]{study}[/] trials={trials} "
        f"updates/trial={updates} games={games} workers={workers}"
    )
    console.print(f"[dim]storage ->[/] {storage}\n")
    st = optuna.create_study(
        study_name=study,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )
    st.optimize(objective, n_trials=trials)

    try:
        best = st.best_trial
    except ValueError:
        console.print("[yellow]no completed trials (all pruned/failed).[/]")
        return
    t = Table(title="best trial", title_style="bold")
    t.add_column("param", style="dim")
    t.add_column("value", justify="right", style="bold")
    t.add_row("win_rate", f"[green]{best.value:.1%}[/]")
    for k, v in best.params.items():
        t.add_row(k, str(v))
    console.print(t)
    console.print(f"[dim]dashboard:[/] optuna-dashboard {storage}")


@app.command(name="import-csv")
def import_csv(
    csv: Optional[Path] = typer.Option(
        None, help="train.csv to import (default: <output>/logs/train.csv)."
    ),
    run_name: str = typer.Option(
        "csv-import", help="TensorBoard run name for the imported history."
    ),
    data_dir: Path = typer.Option(
        DATA_DIR,
        "--output-dir",
        "-o",
        help="Output/artifact root directory (checkpoints + logs).",
    ),
) -> None:
    """Backfill TensorBoard from an existing train.csv.

    Useful for runs started before TensorBoard logging existed: replays every CSV
    row through the same TensorBoard sink so the full history shows up under
    <output>/runs/<run-name>/ with the usual grouped scalars.
    """
    import csv as csvlib

    from pkm.new_agents.agent_000_dragapult.monitor import RunContext, TensorBoardSink

    p = _paths(data_dir)
    csv_path = Path(csv) if csv else p["csv"]
    if not csv_path.exists():
        console.print(f"[red]no CSV at[/] {csv_path}")
        raise typer.Exit(1)

    log_dir = p["runs"] / run_name
    sink = TensorBoardSink(log_dir)
    sink.start(
        RunContext(run_name=run_name, config={}, config_hash="", output_dir=p["data"])
    )
    n = 0
    with csv_path.open(newline="") as fh:
        for row in csvlib.DictReader(fh):
            if not row.get("update"):
                continue
            update = int(float(row["update"]))
            stats = {
                k: float(v)
                for k, v in row.items()
                if k != "update" and v not in (None, "")
            }
            sink.log(update, update, stats)
            n += 1
    sink.close()
    console.print(
        f"[green]imported[/] {n} rows from {csv_path}\n"
        f"[dim]->[/] {log_dir}\n"
        f"[dim]view:[/] tensorboard --logdir {p['runs']}"
    )


if __name__ == "__main__":
    app()
