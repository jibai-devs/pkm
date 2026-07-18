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
import re
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

# Every run belongs to a named *experiment* — its own subdirectory under the
# agent root (`<output>/experiments/<name>/`) holding that run's checkpoints,
# logs, TensorBoard events, sweeps and submission bundles. This lets the same
# agent keep many independent runs side by side. The name becomes a directory,
# so it must be filesystem-safe (validated by `_resolve_experiment`).
DEFAULT_EXPERIMENT = "000_default"
_SMOKE_EXPERIMENT = "000_smoke"  # keep sanity runs out of real experiments
_EXPERIMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_EXPERIMENT_HELP = (
    "Experiment name — artifacts nest under <output>/experiments/<name>/. "
    "Letters/digits/'.'/'_'/'-' only (no spaces or slashes). "
    f"Default: {DEFAULT_EXPERIMENT}."
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _resolve_experiment(data_dir: Path, experiment: str) -> Path:
    """Return the artifact dir for `experiment` under `data_dir`, validating the name.

    The name is used verbatim as a directory, so reject anything that is not a
    plain filesystem-safe token (in particular spaces and path separators).
    """
    name = experiment.strip()
    if not _EXPERIMENT_RE.fullmatch(name):
        console.print(
            f"[red]invalid experiment name[/] {experiment!r} — use only letters, "
            "digits, '.', '_' or '-' (no spaces or slashes)."
        )
        raise typer.Exit(2)
    return data_dir / "experiments" / name


def _confirm_overwrite(ckpt_dir: Path, experiment: str, force: bool) -> None:
    """Guard a fresh `train` against clobbering an experiment's existing checkpoint.

    No-op when the experiment has no `latest.pt`. Otherwise prompt (or, with
    `--force`, warn and proceed); declining aborts.
    """
    latest = ckpt_dir / "latest.pt"
    if not latest.exists():
        return

    idx = None
    try:  # read just the update counter; the blob is a TrainState dict
        import torch

        blob = torch.load(latest, map_location="cpu", weights_only=False)
        if isinstance(blob, dict):
            idx = blob.get("update_idx")
    except Exception:  # noqa: BLE001 — best-effort; a bad/old blob just omits the count
        idx = None
    at = f" (update {idx})" if idx is not None else ""

    if force:
        console.print(
            f"[yellow]--force:[/] overwriting existing checkpoint for experiment "
            f"[cyan]{experiment}[/]{at}."
        )
        return

    console.print(
        f"[yellow]![/] experiment [cyan]{experiment}[/] already has a checkpoint"
        f"{at} at\n    [dim]{latest}[/]"
    )
    if not typer.confirm("Overwrite and start a fresh run?", default=False):
        console.print(
            "[dim]aborted — use `resume` to continue this run, or pass "
            "--experiment <name> to start a separate one.[/]"
        )
        raise typer.Exit(1)


def _paths(data_dir: Path) -> dict[str, Path]:
    return {
        "data": data_dir,
        "ckpt": data_dir / "checkpoints",
        "logs": data_dir / "logs",
        "csv": data_dir / "logs" / "train.csv",
        "runs": data_dir / "runs",  # TensorBoard event files
        "wandb": data_dir / "wandb",  # wandb (offline) run dirs
        "sweeps": data_dir / "sweeps",  # Optuna sqlite studies
        "submissions": data_dir / "submissions",  # packed .tar.gz bundles
    }


# Local training defaults to the fast, direct-ctypes nix build (no
# `kaggle_environments` import). Override per-command with `--engine`.
_ENGINE_HELP = "Engine backend: kaggle | local | local-nix (default local-nix)."
_DEFAULT_ENGINE = "local-nix"


def _select_engine(engine: str) -> None:
    """Select the engine backend and load it now, printing an obvious banner.

    Call this as the first statement of a command body. Thanks to the loader's
    lazy loading nothing has loaded the engine yet, so :func:`set_backend` wins;
    we then force the load here so a missing local build fails fast with a build
    hint (rather than deep inside the first rollout).
    """
    from pkm.engine import loader

    try:
        loader.set_backend(engine)
        loader.get_lib()  # load now: fail fast with a build hint if missing
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        console.print(f"[red]engine '{engine}' unavailable:[/]\n{exc}")
        raise typer.Exit(1) from exc

    backend, path = loader.ENGINE_BACKEND, loader.ENGINE_LIB_PATH
    note = " [dim](slow: imports kaggle_environments)[/]" if backend == "kaggle" else ""
    console.print(f"⚙  [bold]engine:[/] [cyan]{backend}[/] → {path}{note}")


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
    experiment: str = typer.Option(
        DEFAULT_EXPERIMENT, "--experiment", "-e", help=_EXPERIMENT_HELP
    ),
    engine: str = typer.Option(_DEFAULT_ENGINE, help=_ENGINE_HELP),
) -> None:
    """Print the default config, engine backend, and artifact paths."""
    from pkm.engine import loader

    try:
        loader.set_backend(engine)  # report the selection; don't force a load
    except (ValueError, RuntimeError) as exc:
        console.print(f"[red]engine '{engine}' unavailable:[/]\n{exc}")
        raise typer.Exit(1) from exc
    ENGINE_BACKEND, ENGINE_LIB_PATH = loader.ENGINE_BACKEND, loader.ENGINE_LIB_PATH

    from pkm.new_agents.agent_000_dragapult.config import Config

    data_dir = _resolve_experiment(data_dir, experiment)
    console.print(f"[dim]experiment:[/] [cyan]{experiment}[/] → {data_dir}")
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
    experiment: str = typer.Option(
        _SMOKE_EXPERIMENT, "--experiment", "-e", help=_EXPERIMENT_HELP
    ),
    engine: str = typer.Option(_DEFAULT_ENGINE, help=_ENGINE_HELP),
) -> None:
    """Tiny end-to-end sanity run: 2 updates x 2 games. Proves the pipeline works."""
    _select_engine(engine)
    data_dir = _resolve_experiment(data_dir, experiment)
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
    experiment: str = typer.Option(
        DEFAULT_EXPERIMENT, "--experiment", "-e", help=_EXPERIMENT_HELP
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite an existing experiment checkpoint without prompting.",
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
    engine: str = typer.Option(_DEFAULT_ENGINE, help=_ENGINE_HELP),
) -> None:
    """Run PPO self-play training."""
    data_dir = _resolve_experiment(data_dir, experiment)
    if not resume:
        _confirm_overwrite(_paths(data_dir)["ckpt"], experiment, force)
    _select_engine(engine)
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
    experiment: str = typer.Option(
        DEFAULT_EXPERIMENT, "--experiment", "-e", help=_EXPERIMENT_HELP
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
    engine: str = typer.Option(_DEFAULT_ENGINE, help=_ENGINE_HELP),
) -> None:
    """Resume training from checkpoints/latest.pt (config is restored from the checkpoint)."""
    data_dir = _resolve_experiment(data_dir, experiment)
    _select_engine(engine)
    from pkm.new_agents.agent_000_dragapult.train import TrainState

    p = _paths(data_dir)
    latest = p["ckpt"] / "latest.pt"
    if not latest.exists():
        console.print(
            f"[red]no checkpoint at[/] {latest}\n"
            f"[dim](experiment [cyan]{experiment}[/] — start it with `train "
            f"--experiment {experiment}`)[/]"
        )
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
    experiment: str = typer.Option(
        DEFAULT_EXPERIMENT, "--experiment", "-e", help=_EXPERIMENT_HELP
    ),
    engine: str = typer.Option(_DEFAULT_ENGINE, help=_ENGINE_HELP),
) -> None:
    """Report a checkpoint's win-rate vs the random baseline (alternating seats)."""
    _select_engine(engine)
    from pkm.new_agents.agent_000_dragapult.eval import winrate_vs_random
    from pkm.new_agents.agent_000_dragapult.train import TrainState

    data_dir = _resolve_experiment(data_dir, experiment)
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
    experiment: str = typer.Option(
        DEFAULT_EXPERIMENT, "--experiment", "-e", help=_EXPERIMENT_HELP
    ),
    engine: str = typer.Option(_DEFAULT_ENGINE, help=_ENGINE_HELP),
) -> None:
    """Optuna hyperparameter sweep — maximize eval win-rate vs random.

    Each trial samples lr/entropy/clip/epochs/minibatch/gamma/lam, runs a short
    training, and returns its win-rate vs random. The study is SQLite-backed under
    <output>/sweeps/<study>.db (resumable; view with `optuna-dashboard`). Trials
    are pruned early via reported intermediate evals (MedianPruner).
    """
    _select_engine(engine)
    data_dir = _resolve_experiment(data_dir, experiment)
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
    experiment: str = typer.Option(
        DEFAULT_EXPERIMENT, "--experiment", "-e", help=_EXPERIMENT_HELP
    ),
) -> None:
    """Backfill TensorBoard from an existing train.csv.

    Useful for runs started before TensorBoard logging existed: replays every CSV
    row through the same TensorBoard sink so the full history shows up under
    <output>/runs/<run-name>/ with the usual grouped scalars.
    """
    import csv as csvlib

    from pkm.new_agents.agent_000_dragapult.monitor import RunContext, TensorBoardSink

    data_dir = _resolve_experiment(data_dir, experiment)
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


_MAX_BUNDLE_MIB = 197.7  # Kaggle submission size limit
_COMPETITION = "pokemon-tcg-ai-battle"


@app.command()
def pack(
    checkpoint: Optional[Path] = typer.Option(
        None, help="Checkpoint to pack (default: checkpoints/latest.pt)."
    ),
    data_dir: Path = typer.Option(
        DATA_DIR,
        "--output-dir",
        "-o",
        help="Output/artifact root directory (checkpoints + logs).",
    ),
    experiment: str = typer.Option(
        DEFAULT_EXPERIMENT, "--experiment", "-e", help=_EXPERIMENT_HELP
    ),
) -> None:
    """Pack the latest weights into a Kaggle submission bundle (.tar.gz).

    Extracts the model weights from the checkpoint into ``weights.pt``, adds the
    submission ``main.py`` entry point and the ``pkm/`` package, and writes a
    timestamped tarball under <output>/submissions/. Torch is NOT bundled (size
    limit); the bundle relies on torch existing in the cabt sandbox at inference.
    """
    import tarfile
    import tempfile
    from datetime import datetime as _dt

    import torch

    data_dir = _resolve_experiment(data_dir, experiment)
    p = _paths(data_dir)
    ckpt = checkpoint or (p["ckpt"] / "latest.pt")
    if not ckpt.exists():
        console.print(f"[red]no checkpoint at[/] {ckpt}")
        raise typer.Exit(1)
    repo_root = Path(__file__).resolve().parents[3]
    template = Path(__file__).with_name("submit_main.py")
    p["submissions"].mkdir(parents=True, exist_ok=True)

    # Extract just the model state_dict (checkpoints are TrainState blobs).
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    state_dict = blob["model"] if isinstance(blob, dict) and "model" in blob else blob

    def _no_pycache(info: tarfile.TarInfo):
        base = Path(info.name).name
        if "__pycache__" in info.name or base.endswith((".pyc", ".pyo")):
            return None
        return info

    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    out = p["submissions"] / f"submission_{ts}.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        weights_file = Path(tmp) / "weights.pt"
        torch.save(state_dict, weights_file)
        with tarfile.open(out, "w:gz") as tar:
            tar.add(template, arcname="main.py")
            tar.add(weights_file, arcname="weights.pt")
            tar.add(repo_root / "pkm", arcname="pkm", filter=_no_pycache)

    size_mib = out.stat().st_size / 1024 / 1024
    ok = size_mib <= _MAX_BUNDLE_MIB
    console.print(
        f"[green]packed[/] {ckpt.name} -> {out}\n"
        f"[dim]size:[/] {size_mib:.1f} MiB "
        + (
            f"[green](≤ {_MAX_BUNDLE_MIB} limit)[/]"
            if ok
            else f"[red](> {_MAX_BUNDLE_MIB} limit!)[/]"
        )
    )
    console.print(
        "[yellow]note:[/] inference uses torch; the bundle assumes the cabt "
        "sandbox provides it (no torch is bundled)."
    )
    console.print("[dim]submit with:[/] pkm new_agents 000_dragapult submit")


@app.command()
def submit(
    bundle: Optional[Path] = typer.Option(
        None, help="Bundle to submit (default: newest <output>/submissions/*.tar.gz)."
    ),
    message: str = typer.Option(
        "agent_000_dragapult", help="Kaggle submission message."
    ),
    competition: str = typer.Option(_COMPETITION, help="Kaggle competition slug."),
    data_dir: Path = typer.Option(
        DATA_DIR,
        "--output-dir",
        "-o",
        help="Output/artifact root directory (checkpoints + logs).",
    ),
    experiment: str = typer.Option(
        DEFAULT_EXPERIMENT, "--experiment", "-e", help=_EXPERIMENT_HELP
    ),
) -> None:
    """Upload a packed bundle to Kaggle (`kaggle competitions submit`).

    Requires the `kaggle` CLI and configured credentials (`~/.kaggle/kaggle.json`).
    Defaults to the newest bundle produced by `pack`.
    """
    import shutil
    import subprocess

    data_dir = _resolve_experiment(data_dir, experiment)
    p = _paths(data_dir)
    if bundle is None:
        bundles = sorted(p["submissions"].glob("submission_*.tar.gz"))
        if not bundles:
            console.print(
                f"[red]no bundles in[/] {p['submissions']} — run `pack` first."
            )
            raise typer.Exit(1)
        bundle = bundles[-1]
    if not bundle.exists():
        console.print(f"[red]no bundle at[/] {bundle}")
        raise typer.Exit(1)
    if shutil.which("kaggle") is None:
        console.print(
            "[red]`kaggle` CLI not found.[/] Install: uv add --group dev kaggle"
        )
        raise typer.Exit(1)

    cmd = [
        "kaggle",
        "competitions",
        "submit",
        "-c",
        competition,
        "-f",
        str(bundle),
        "-m",
        message,
    ]
    console.print(f"[dim]submitting[/] {bundle.name} [dim]to[/] {competition}")
    console.print(f"[dim]$[/] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    raise typer.Exit(result.returncode)


@app.command()
def status(
    competition: str = typer.Option(_COMPETITION, help="Kaggle competition slug."),
    watch: bool = typer.Option(
        False,
        "--watch",
        "-w",
        help="Keep polling until the latest submission finishes.",
    ),
    interval: int = typer.Option(5, help="Seconds between polls in --watch mode."),
    timeout: int = typer.Option(180, help="Give up watching after this many seconds."),
    limit: int = typer.Option(10, help="Max submissions to show."),
) -> None:
    """Show (or poll) your Kaggle submission status + score for this competition.

    Run after `submit` to see whether the agent ran or errored (e.g. a missing-torch
    import failure shows up as an errored submission) and what it scored. Requires
    the kaggle CLI + credentials.
    """
    import csv as csvlib
    import shutil
    import subprocess
    import time

    if shutil.which("kaggle") is None:
        console.print(
            "[red]`kaggle` CLI not found.[/] Install: uv add --group dev kaggle"
        )
        raise typer.Exit(1)

    def _poll_once() -> Optional[str]:
        r = subprocess.run(
            ["kaggle", "competitions", "submissions", "-c", competition, "--csv"],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            console.print(
                f"[red]kaggle error:[/] {r.stderr.strip() or r.stdout.strip()}"
            )
            return None
        rows = list(csvlib.DictReader(r.stdout.splitlines()))
        if not rows:
            console.print("[yellow]no submissions found for this competition.[/]")
            return None

        t = Table(title=f"submissions · {competition}", title_style="bold")
        t.add_column("date", style="dim")
        t.add_column("file / description")
        t.add_column("status")
        t.add_column("score", justify="right", style="bold")
        for row in rows[:limit]:
            st = (row.get("status") or "").lower()
            color = (
                "green"
                if "complete" in st
                else "red"
                if ("error" in st or "fail" in st)
                else "yellow"
            )
            label = st.replace("submissionstatus.", "") or "?"  # strip kaggle prefix
            name = row.get("fileName") or row.get("description") or ""
            score = row.get("publicScore") or "—"
            t.add_row(row.get("date", ""), name, f"[{color}]{label}[/]", str(score))
        console.print(t)
        return (rows[0].get("status") or "").lower()

    deadline = time.monotonic() + timeout
    while True:
        latest = _poll_once()
        if not watch or latest is None:
            break
        if "complete" in latest or "error" in latest or "fail" in latest:
            console.print("[bold]latest submission finished.[/]")
            break
        if time.monotonic() >= deadline:
            console.print(f"[yellow]gave up after {timeout}s (still pending).[/]")
            break
        console.print(f"[dim]pending… re-checking in {interval}s (Ctrl-C to stop)[/]")
        time.sleep(interval)


if __name__ == "__main__":
    app()
