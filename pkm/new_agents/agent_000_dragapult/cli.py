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
import sys
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


def _parse_reward_weights(pairs: list[str]) -> dict[str, float]:
    """Parse ``--reward-weight name=value`` pairs into a dict, validating names
    against the reward-term registry and values as floats. Exits with a helpful
    message on a bad name or value rather than raising deep in config build."""
    from pkm.rl.reward_terms import TERM_NAMES

    out: dict[str, float] = {}
    for pair in pairs:
        if "=" not in pair:
            console.print(f"[red]--reward-weight expects name=value, got:[/] {pair!r}")
            raise typer.Exit(1)
        name, _, raw = pair.partition("=")
        name = name.strip()
        if name not in TERM_NAMES:
            console.print(
                f"[red]unknown reward term[/] {name!r}; "
                f"choose from {sorted(TERM_NAMES)}"
            )
            raise typer.Exit(1)
        try:
            out[name] = float(raw)
        except ValueError:
            console.print(f"[red]reward weight for {name!r} is not a number:[/] {raw!r}")
            raise typer.Exit(1) from None
    return out


def _parse_aux_weights(pairs: list[str]) -> dict[str, float]:
    """Parse ``--aux-weight name=value`` pairs into a dict, validating names
    against the aux-task registry and values as floats."""
    from pkm.new_agents.agent_000_dragapult.aux_tasks import task_names

    names = set(task_names())
    out: dict[str, float] = {}
    for pair in pairs:
        if "=" not in pair:
            console.print(f"[red]--aux-weight expects name=value, got:[/] {pair!r}")
            raise typer.Exit(1)
        name, _, raw = pair.partition("=")
        name = name.strip()
        if name not in names:
            console.print(
                f"[red]unknown aux task[/] {name!r}; choose from {sorted(names)}"
            )
            raise typer.Exit(1)
        try:
            out[name] = float(raw)
        except ValueError:
            console.print(f"[red]aux weight for {name!r} is not a number:[/] {raw!r}")
            raise typer.Exit(1) from None
    return out


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
    shaping: str = "prize_potential",
    shaping_coef: float = 1.0,
    reward_weights: dict[str, float] | None = None,
    aux_weights: dict[str, float] | None = None,
    method: str = "ppo",
    mcts_simulations: int = 32,
    mcts_c_puct: float = 1.25,
    mcts_temperature: float = 1.0,
    determinization: str = "sample",
    model_preset: str = "small",
    model_overrides: dict[str, int | float | None] | None = None,
) -> Config:
    from pkm.new_agents.agent_000_dragapult.config import (
        Config,
        RunConfig,
        TrainConfig,
        build_model_config,
    )

    # Network architecture: a size preset (small=v1) with per-dim overrides.
    try:
        model = build_model_config(model_preset, model_overrides)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(2) from exc

    # Start from the default weights and overlay any CLI overrides, so unset
    # terms keep their documented defaults rather than dropping to 0.0.
    weights = dict(TrainConfig().reward_weights)
    if reward_weights:
        weights.update(reward_weights)

    # Same overlay for aux weights: start from the all-zero default so unset
    # tasks stay off, then apply CLI overrides.
    aux = dict(TrainConfig().aux_weights)
    if aux_weights:
        aux.update(aux_weights)

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
        shaping=shaping,
        shaping_coef=shaping_coef,
        reward_weights=weights,
        aux_weights=aux,
        method=method,
        mcts_simulations=mcts_simulations,
        mcts_c_puct=mcts_c_puct,
        mcts_temperature=mcts_temperature,
        determinization=determinization,
    )
    run = dataclasses.replace(RunConfig(), checkpoint_every_updates=ckpt_every)
    return Config(model=model, train=train, run=run)


def _config_table(cfg: Config) -> Table:
    t = Table(show_header=False, box=None, pad_edge=False)
    t.add_column(style="dim")
    t.add_column(style="bold")
    tc = cfg.train
    mc = cfg.model
    for k, v in [
        (
            "model",
            f"{mc.n_layers}L · d_state={mc.d_state} d_entity={mc.d_entity} "
            f"heads={mc.n_heads} d_opt={mc.d_opt} d_card={mc.d_card}",
        ),
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
        ("shaping", f"{tc.shaping} (coef={tc.shaping_coef})"),
        ("ckpt_every", cfg.run.checkpoint_every_updates),
        ("config_hash", cfg.hash()),
    ]:
        t.add_row(str(k), str(v))
    if tc.shaping == "heuristic":
        active = {k: v for k, v in sorted(tc.reward_weights.items()) if v}
        t.add_row("reward_weights", str(active) if active else "(all 0.0)")
    aux = {k: v for k, v in sorted(tc.aux_weights.items()) if v}
    if aux:
        t.add_row("aux_weights", str(aux))
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
    device: str = "cpu",
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
        device=device,
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
        1, help="Parallel self-play workers (1 = single-process; smoke stays simple)."
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
    updates: int = typer.Option(256, help="Number of PPO updates to run."),
    games: int = typer.Option(16, help="Self-play games collected per update."),
    workers: int = typer.Option(
        8, help="Parallel self-play workers (one engine/process; 1 = single-process)."
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
    shaping: str = typer.Option(
        "prize_potential",
        help="Reward shaping: 'prize_potential' (default), 'terminal' (sparse "
        "+/-1), or 'heuristic' (full deck-specific reward stack; weight it with "
        "--reward-weight).",
    ),
    shaping_coef: float = typer.Option(
        1.0, help="Scale on the shaping term (0.0 == terminal)."
    ),
    reward_weight: list[str] = typer.Option(
        [],
        "--reward-weight",
        help="Override a heuristic reward-term weight as name=value (repeatable), "
        "e.g. --reward-weight dragapult_bonus=0.3. Only used when "
        "--shaping heuristic.",
    ),
    aux_weight: list[str] = typer.Option(
        [],
        "--aux-weight",
        help="Enable/weight an auxiliary loss as name=value (repeatable), e.g. "
        "--aux-weight prize_margin=0.25. Weight > 0 turns the task on. Default: "
        "all off. Aux heads are training-only (stripped from the Kaggle bundle).",
    ),
    model: str = typer.Option(
        "small",
        "--model",
        help="Network size preset: small (v1, default), medium, large, xl. "
        "Override individual dims with the --d-*/--n-layers/--n-heads flags.",
    ),
    n_layers: Optional[int] = typer.Option(
        None, help="Override: entity-attention layers in the trunk (1 = v1 depth)."
    ),
    d_state: Optional[int] = typer.Option(
        None, help="Override: state (trunk output) embedding dim."
    ),
    d_entity: Optional[int] = typer.Option(
        None, help="Override: per-entity embedding dim (attention width)."
    ),
    n_heads: Optional[int] = typer.Option(
        None, help="Override: attention heads (must divide d_entity)."
    ),
    d_opt: Optional[int] = typer.Option(
        None, help="Override: option embedding / scorer width."
    ),
    d_card: Optional[int] = typer.Option(
        None, help="Override: card embedding dim."
    ),
    dropout: Optional[float] = typer.Option(
        None, help="Override: dropout in the extra transformer layers (regularization)."
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Learner device: cpu (default), cuda, or auto (cuda if available). "
        "Rollout workers + eval always run on CPU; only the PPO update uses the "
        "device. cuda needs a CUDA build of torch.",
    ),
    method: str = typer.Option("ppo", help="Training method: 'ppo' or 'exit'."),
    mcts_simulations: int = typer.Option(
        32, help="MCTS simulations per move (exit)."
    ),
    mcts_c_puct: float = typer.Option(
        1.25, help="PUCT exploration constant (exit)."
    ),
    mcts_temperature: float = typer.Option(
        1.0, help="Visit-count temperature (exit)."
    ),
    determinization: str = typer.Option(
        "sample", help="Hidden-state determinizer (exit)."
    ),
    eval_every: int = typer.Option(
        16, help="Evaluate vs random every N updates (0 = never)."
    ),
    eval_games: int = typer.Option(128, help="Games per evaluation."),
    ckpt_every: int = typer.Option(64, help="Checkpoint snapshot every N updates."),
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
    # Fail fast on a bad/unavailable device before building anything heavy.
    from pkm.new_agents.agent_000_dragapult.config import resolve_device

    try:
        device = resolve_device(device)
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(2) from exc
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
        shaping=shaping,
        shaping_coef=shaping_coef,
        reward_weights=_parse_reward_weights(reward_weight),
        aux_weights=_parse_aux_weights(aux_weight),
        method=method,
        mcts_simulations=mcts_simulations,
        mcts_c_puct=mcts_c_puct,
        mcts_temperature=mcts_temperature,
        determinization=determinization,
        model_preset=model,
        model_overrides={
            "n_layers": n_layers,
            "d_state": d_state,
            "d_entity": d_entity,
            "n_heads": n_heads,
            "d_opt": d_opt,
            "d_card": d_card,
            "dropout": dropout,
        },
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
        device=device,
    )


@app.command()
def resume(
    updates: int = typer.Option(50, help="Additional PPO updates to run."),
    games: int = typer.Option(16, help="Self-play games collected per update."),
    workers: int = typer.Option(
        8, help="Parallel rollout workers (one engine per process)."
    ),
    eval_every: int = typer.Option(
        16, help="Evaluate vs random every N updates (0 = never)."
    ),
    eval_games: int = typer.Option(128, help="Games per evaluation."),
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
    games: int = typer.Option(100, help="Number of games vs the opponent."),
    checkpoint: Optional[Path] = typer.Option(
        None, help="Checkpoint under test (defaults to checkpoints/latest.pt)."
    ),
    opponent: Optional[Path] = typer.Option(
        None,
        "--opponent",
        help="Opponent checkpoint for a head-to-head (a ckpt_N.pt or a packed "
        "weights.pt). Omit to play the random baseline. Head-to-head is the "
        "discriminating eval — it isn't pinned to the ~100%% random ceiling.",
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
    inference: str = typer.Option(
        "policy",
        "--inference",
        help="Decision mode for the agent under test: 'policy' or 'mcts'.",
    ),
    mcts_sims: int = typer.Option(
        0, "--mcts-sims", "-K", help="MCTS simulation budget K (0 = policy only)."
    ),
    mcts_c_puct: float = typer.Option(1.25, help="PUCT exploration constant for MCTS."),
    mcts_temperature: float = typer.Option(
        0.0, help="Root visit-count temperature for the MCTS pick (0 = most-visited)."
    ),
) -> None:
    """Report a checkpoint's win-rate vs the random baseline (alternating seats).

    Add ``--inference mcts -K <sims>`` to evaluate the checkpoint with
    inference-time PUCT search instead of the raw policy head — the direct way
    to measure whether search actually beats the plain policy for this net.
    """
    _select_engine(engine)
    from pkm.new_agents.agent_000_dragapult.agent import InferenceConfig
    from pkm.new_agents.agent_000_dragapult.eval import (
        winrate_vs_checkpoint,
        winrate_vs_random,
    )
    from pkm.new_agents.agent_000_dragapult.train import TrainState

    if inference not in ("policy", "mcts"):
        console.print(f"[red]--inference must be 'policy' or 'mcts', got[/] {inference!r}")
        raise typer.Exit(1)
    inf = InferenceConfig(
        type=inference,
        mcts_sims=mcts_sims,
        c_puct=mcts_c_puct,
        temperature=mcts_temperature,
    )

    data_dir = _resolve_experiment(data_dir, experiment)
    ckpt = checkpoint or (_paths(data_dir)["ckpt"] / "latest.pt")
    if not ckpt.exists():
        console.print(f"[red]no checkpoint at[/] {ckpt}")
        raise typer.Exit(1)
    model = TrainState.load(ckpt).model
    if inf.use_mcts:
        console.print(
            f"[dim]agent-under-test uses[/] MCTS "
            f"[dim](K={inf.mcts_sims}, c_puct={inf.c_puct}, temp={inf.temperature})[/]"
        )
    if opponent is not None:
        if not opponent.exists():
            console.print(f"[red]no opponent checkpoint at[/] {opponent}")
            raise typer.Exit(1)
        console.print(
            f"[dim]head-to-head[/] {ckpt.name} [dim]vs[/] {opponent.name} "
            f"[dim]over[/] {games} [dim]games…[/]"
        )
        ev = winrate_vs_checkpoint(model, str(opponent), n_games=games)
        title = f"vs {opponent.name}"
    else:
        console.print(f"[dim]evaluating[/] {ckpt} [dim]vs random over[/] {games} [dim]games…[/]")
        ev = winrate_vs_random(model, n_games=games, seed=seed)
        title = "vs random"

    t = Table(title=title, title_style="bold")
    t.add_column("metric", style="dim")
    t.add_column("value", justify="right", style="bold")
    t.add_row("win rate", f"[green]{ev['win_rate']:.1%}[/]")
    t.add_row("W / L / D", f"{ev['wins']} / {ev['losses']} / {ev['draws']}")
    t.add_row("games", str(ev["n"]))
    console.print(t)


SWEEP_OBJECTIVES = ("curve_auc", "final_winrate", "peak_winrate", "net_winrate")


def _score_objective(
    name: str, curve: list[float], final_ev: dict[str, float]
) -> float:
    """Reduce a trial's results to the single scalar Optuna maximizes.

    ``curve`` is the list of intermediate eval win-rates recorded during the
    trial; ``final_ev`` is a fresh end-of-trial :func:`evaluate` result.

    - ``curve_auc`` (default): mean of the eval learning curve (intermediate
      evals + the final one) — rewards fast *and* sustained learning and
      averages out per-eval noise. What a short trial can actually measure.
    - ``final_winrate``: the final eval win-rate only (legacy behaviour).
    - ``peak_winrate``: best eval reached — robust to an end-of-run collapse.
    - ``net_winrate``: final ``win_rate - loss_rate`` — credits *not losing*, a
      denser signal than raw wins once win-rate saturates near random's ceiling.
    """
    wr = final_ev["win_rate"]
    if name == "final_winrate":
        return wr
    if name == "net_winrate":
        return wr - final_ev["loss_rate"]
    points = [*curve, wr]  # learning curve incl. the final snapshot
    if name == "peak_winrate":
        return max(points)
    return sum(points) / len(points)  # curve_auc


@app.command()
def sweep(
    trials: int = typer.Option(30, help="Number of Optuna trials."),
    updates: int = typer.Option(15, help="PPO updates per trial (keep short)."),
    games: int = typer.Option(32, help="Self-play games per update."),
    workers: int = typer.Option(8, help="Rollout workers per trial."),
    eval_games: int = typer.Option(128, help="Games per evaluation (the objective)."),
    objective: str = typer.Option(
        "curve_auc",
        help=(
            "Trial score to maximize: curve_auc (default; mean of the eval "
            "learning curve), final_winrate, peak_winrate, or net_winrate "
            "(win_rate - loss_rate)."
        ),
    ),
    study: str = typer.Option(
        "dragapult_ppo", help="Optuna study name (sqlite, resumable)."
    ),
    seed: int = typer.Option(0, help="Base RNG seed (offset per trial)."),
    model: str = typer.Option(
        "small",
        "--model",
        help="Network size preset every trial trains at: small (v1, default), "
        "medium, large, xl. Architecture is fixed across the sweep; only the "
        "hyperparameters (and, with --tune-rewards, reward weights) are searched. "
        "Push beyond xl (an 'xxl') with the per-dim overrides below.",
    ),
    n_layers: Optional[int] = typer.Option(
        None, help="Override: trunk attention layers (every trial)."
    ),
    d_state: Optional[int] = typer.Option(
        None, help="Override: state (trunk output) dim (every trial)."
    ),
    d_entity: Optional[int] = typer.Option(
        None, help="Override: per-entity dim (every trial)."
    ),
    n_heads: Optional[int] = typer.Option(
        None, help="Override: attention heads, must divide d_entity (every trial)."
    ),
    d_opt: Optional[int] = typer.Option(
        None, help="Override: option/scorer width (every trial)."
    ),
    d_card: Optional[int] = typer.Option(
        None, help="Override: card embedding dim (every trial)."
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Learner device for every trial: cpu (default), cuda, or auto. "
        "Rollout + eval stay on CPU; only the PPO update uses the device.",
    ),
    tune_rewards: bool = typer.Option(
        False,
        "--tune-rewards",
        help="Also sweep the heuristic reward-term weights (sets shaping="
        "'heuristic' and samples every term in reward_terms.ALL_TERMS in "
        "[0, 1]). Off = tune only the PPO hyperparameters (shaping stays "
        "'prize_potential').",
    ),
    reset: bool = typer.Option(
        False,
        "--reset",
        help="Delete an objective-mismatched study and start it over (no prompt).",
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
    engine: str = typer.Option(_DEFAULT_ENGINE, help=_ENGINE_HELP),
) -> None:
    """Optuna hyperparameter sweep — maximize eval win-rate vs random.

    Each trial samples lr/entropy/clip/epochs/minibatch/gamma/lam (and, with
    --tune-rewards, every heuristic reward-term weight), runs a short training,
    and returns its win-rate vs random. The study is SQLite-backed under
    <output>/sweeps/<study>.db (resumable; view with `optuna-dashboard`). Trials
    are pruned early via reported intermediate evals (MedianPruner).
    """
    if objective not in SWEEP_OBJECTIVES:
        console.print(
            f"[red]unknown --objective[/] {objective!r} "
            f"[dim](choose from {', '.join(SWEEP_OBJECTIVES)})[/]"
        )
        raise typer.Exit(2)
    _select_engine(engine)
    from pkm.new_agents.agent_000_dragapult.config import resolve_device

    try:
        device = resolve_device(device)
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(2) from exc
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
        """Report intermediate eval win-rate to Optuna, record the learning curve
        for the objective, and prune hopeless trials."""

        def __init__(self, trial: optuna.Trial):
            self.trial = trial
            self.curve: list[float] = []

        def log(self, update: int, total: int, stats: dict) -> None:
            ev = stats.get("eval_win_rate")
            if isinstance(ev, (int, float)):
                self.curve.append(float(ev))
                self.trial.report(ev, update)
                if self.trial.should_prune():
                    raise StopTraining

    def _run_trial(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        entropy_coef = trial.suggest_float("entropy_coef", 1e-4, 5e-2, log=True)
        clip_eps = trial.suggest_float("clip_eps", 0.1, 0.3)
        epochs = trial.suggest_int("epochs", 2, 6)
        minibatch = trial.suggest_categorical("minibatch_size", [32, 64, 128])
        gamma = trial.suggest_float("gamma", 0.95, 0.999)
        lam = trial.suggest_float("lam", 0.9, 0.99)
        # Optionally sweep the heuristic reward stack too. Registry-driven: one
        # weight per term in reward_terms.ALL_TERMS, each in [0, 1] (the heuristic
        # functions already carry the sign, so weights stay non-negative).
        shaping = "prize_potential"
        reward_weights = None
        if tune_rewards:
            from pkm.rl.reward_terms import TERM_NAMES

            shaping = "heuristic"
            reward_weights = {
                name: trial.suggest_float(f"rw_{name}", 0.0, 1.0)
                for name in TERM_NAMES
            }
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
            shaping=shaping,
            reward_weights=reward_weights,
            model_preset=model,
            model_overrides={
                "n_layers": n_layers,
                "d_state": d_state,
                "d_entity": d_entity,
                "n_heads": n_heads,
                "d_opt": d_opt,
                "d_card": d_card,
            },
            ckpt_every=updates,
        )
        trial_dir = p["sweeps"] / study / f"trial_{trial.number}"
        pruning = PruningSink(trial)
        observers = [
            TensorBoardSink(p["runs"] / f"sweep-{study}" / f"trial_{trial.number}"),
            pruning,
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
                device=device,
            )
        except StopTraining as exc:  # pruned mid-run
            raise optuna.TrialPruned() from exc
        final_ev = winrate_vs_random(ts.model, n_games=eval_games, seed=seed)
        return _score_objective(objective, pruning.curve, final_ev)

    console.print(
        f"[bold]sweep[/] study=[cyan]{study}[/] trials={trials} "
        f"updates/trial={updates} games={games} workers={workers} "
        f"objective=[cyan]{objective}[/] model=[cyan]{model}[/] "
        f"rewards=[cyan]{'heuristic (tuned)' if tune_rewards else 'prize_potential'}[/]"
    )
    console.print(f"[dim]storage ->[/] {storage}\n")
    st = optuna.create_study(
        study_name=study,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
    )
    # A study's trials are only comparable if they were all scored by the same
    # objective. Tag the study with its objective; on a mismatched resume (or a
    # pre-tagging legacy study that already holds trials) offer to delete and start
    # over — otherwise the sampler would model a mixed target and best_trial would
    # compare unlike scores.
    prev_objective = st.user_attrs.get("objective")
    conflict = None
    if prev_objective is None and st.trials:
        conflict = (
            f"study [cyan]{study}[/] predates objective tagging and holds "
            f"{len(st.trials)} trials of unknown objective"
        )
    elif prev_objective is not None and prev_objective != objective:
        conflict = (
            f"study [cyan]{study}[/] was scored by [bold]{prev_objective}[/], "
            f"not [bold]{objective}[/]"
        )
    if conflict:
        console.print(
            f"[yellow]{conflict}[/] — scores under different objectives are not "
            f"comparable (the sampler would model a mixed target)."
        )
        do_reset = reset or (
            sys.stdin.isatty()
            and typer.confirm(
                f"Delete study '{study}' and start over with objective={objective}?",
                default=False,
            )
        )
        if not do_reset:
            console.print(
                "[dim]aborted — use a new --study name to keep the old results, "
                "or pass --reset to delete and start over.[/]"
            )
            raise typer.Exit(2)
        optuna.delete_study(study_name=study, storage=storage)
        st = optuna.create_study(
            study_name=study,
            storage=storage,
            direction="maximize",
            load_if_exists=False,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        )
        console.print(f"[green]reset[/] study [cyan]{study}[/].\n")
    st.set_user_attr("objective", objective)
    if st.trials:
        console.print(
            f"[dim]resuming[/] {len(st.trials)} [dim]existing trials "
            f"(objective={objective}).[/]\n"
        )
    st.optimize(_run_trial, n_trials=trials)

    try:
        best = st.best_trial
    except ValueError:
        console.print("[yellow]no completed trials (all pruned/failed).[/]")
        return
    t = Table(title="best trial", title_style="bold")
    t.add_column("param", style="dim")
    t.add_column("value", justify="right", style="bold")
    t.add_row(objective, f"[green]{best.value:.1%}[/]")
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
    inference: str = typer.Option(
        "policy",
        "--inference",
        help="Inference mode baked into the bundle: 'policy' (raw policy head, "
        "fast) or 'mcts' (PUCT search at decision time — stronger but slower). "
        "MCTS is also disabled whenever --mcts-sims is 0.",
    ),
    mcts_sims: int = typer.Option(
        0,
        "--mcts-sims",
        "-K",
        help="MCTS simulation budget K per decision. 0 turns MCTS off (so "
        "'--inference mcts --mcts-sims 0' still deploys as plain policy).",
    ),
    mcts_c_puct: float = typer.Option(1.25, help="PUCT exploration constant for MCTS."),
    mcts_temperature: float = typer.Option(
        0.0,
        help="Root visit-count temperature for the MCTS move pick (0 = pick the "
        "most-visited move, deterministic).",
    ),
    determinization: str = typer.Option(
        "sample", help="Hidden-info determinizer for MCTS (key into DETERMINIZERS)."
    ),
) -> None:
    """Pack the latest weights into a Kaggle submission bundle (.tar.gz).

    Extracts the model weights from the checkpoint into ``weights.pt``, adds the
    submission ``main.py`` entry point and the ``pkm/`` package, and writes a
    timestamped tarball under <output>/submissions/. Torch is NOT bundled (size
    limit); the bundle relies on torch existing in the cabt sandbox at inference.

    The bundle also records the inference config (policy vs MCTS + the K budget),
    so the same checkpoint can be packed twice — once plain, once with search —
    and each submission behaves accordingly with no code change at deploy time.
    """
    import tarfile
    import tempfile
    from datetime import datetime as _dt

    import torch

    from pkm.new_agents.agent_000_dragapult.agent import InferenceConfig

    if inference not in ("policy", "mcts"):
        console.print(f"[red]--inference must be 'policy' or 'mcts', got[/] {inference!r}")
        raise typer.Exit(1)
    inf = InferenceConfig(
        type=inference,
        mcts_sims=mcts_sims,
        c_puct=mcts_c_puct,
        temperature=mcts_temperature,
        determinization=determinization,
    )

    data_dir = _resolve_experiment(data_dir, experiment)
    p = _paths(data_dir)
    ckpt = checkpoint or (p["ckpt"] / "latest.pt")
    if not ckpt.exists():
        console.print(f"[red]no checkpoint at[/] {ckpt}")
        raise typer.Exit(1)
    repo_root = Path(__file__).resolve().parents[3]
    template = Path(__file__).with_name("submit_main.py")
    p["submissions"].mkdir(parents=True, exist_ok=True)

    # Extract the model state_dict + the architecture it was trained with
    # (checkpoints are TrainState blobs carrying the full config), so the bundle
    # rebuilds a non-default (e.g. large/deeper) net correctly at inference.
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    state_dict = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
    # Auxiliary heads are training-only: the inference model rebuilds from the
    # bare ModelConfig (no aux heads), so drop their weights here or a strict
    # load_state_dict would reject the unexpected keys. See aux_tasks.py.
    aux_keys = [k for k in state_dict if k.startswith("aux_heads.")]
    if aux_keys:
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("aux_heads.")}
        console.print(f"[dim]stripped {len(aux_keys)} aux-head tensors (training-only)[/]")
    model_config = (
        (blob.get("config") or {}).get("model") if isinstance(blob, dict) else None
    )

    def _no_pycache(info: tarfile.TarInfo):
        base = Path(info.name).name
        if "__pycache__" in info.name or base.endswith((".pyc", ".pyo")):
            return None
        return info

    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    out = p["submissions"] / f"submission_{ts}.tar.gz"
    with tempfile.TemporaryDirectory() as tmp:
        weights_file = Path(tmp) / "weights.pt"
        torch.save(
            {
                "state_dict": state_dict,
                "model_config": model_config,
                "inference": inf.to_dict(),
            },
            weights_file,
        )
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
    mode = (
        f"mcts (K={inf.mcts_sims}, c_puct={inf.c_puct}, temp={inf.temperature})"
        if inf.use_mcts
        else "policy (no search)"
    )
    console.print(f"[dim]inference:[/] {mode}")
    console.print(
        "[yellow]note:[/] inference uses torch; the bundle assumes the cabt "
        "sandbox provides it (no torch is bundled)."
    )
    if inf.use_mcts:
        console.print(
            "[yellow]note:[/] MCTS runs a forward search per decision — watch "
            "Kaggle's per-turn + cumulative 600s time budget; tune K accordingly."
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
