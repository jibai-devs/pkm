"""Observer-pattern metric sinks for training runs.

The training loop (:func:`pkm.new_agents.agent_000_dragapult.train.train`) is the
*subject*: after every update it notifies a list of :class:`MetricSink`
*observers* with that update's stats. Each sink decides what to do with them —
print to the console, append a CSV row, push scalars to TensorBoard, log to
Weights & Biases.

The heavy optional deps (``torch.utils.tensorboard``, ``wandb``) are imported
**lazily inside the sinks**, so importing this module — and the rollout workers
that import ``train.py`` — stays cheap and free of those imports.

Add a new backend by subclassing :class:`MetricSink` and overriding the hooks you
need; everything is wired through the same per-update ``stats`` dict.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

# Canonical scalar metrics we surface, as ``display_tag -> stats_key``. Sinks that
# group by namespace (TensorBoard, wandb) use the tag; flat sinks use the key.
SCALARS: dict[str, str] = {
    "loss/policy": "pol_loss",
    "loss/value": "val_loss",
    "policy/entropy": "entropy",
    "rollout/steps": "steps",
    "rollout/games": "games",
    "rollout/p0_win": "p0_win",
    "rollout/p1_win": "p1_win",
    "eval/win_rate": "eval_win_rate",
}

CSV_FIELDS = [
    "update",
    "games",
    "steps",
    "p0_win",
    "p1_win",
    "pol_loss",
    "val_loss",
    "entropy",
    "eval_win_rate",
]


class StopTraining(Exception):
    """Raised by a sink to intentionally halt the loop (e.g. Optuna pruning).

    :func:`notify` re-raises this instead of swallowing it, so a sink can abort a
    run on purpose while ordinary sink errors stay isolated.
    """


@dataclass
class RunContext:
    """Immutable description of a run, passed to each sink's :meth:`start`."""

    run_name: str
    config: dict
    config_hash: str
    output_dir: Path
    resume: bool = False


class MetricSink:
    """Base observer. Override only the hooks you need — all default to no-ops."""

    def start(self, ctx: RunContext) -> None:  # noqa: D401
        """Called once before the first update."""

    def log(self, update: int, total: int, stats: dict[str, Any]) -> None:
        """Called after every update with that update's metrics."""

    def close(self) -> None:
        """Called once when training ends (also on error, via the loop's finally)."""


def notify(observers: Sequence[MetricSink], method: str, *args: Any) -> None:
    """Invoke ``method`` on each observer, isolating failures.

    A broken sink (e.g. a wandb offline-dir permission issue) must never crash a
    training run — its error is reported to stderr and the others keep going.
    """
    for obs in observers:
        try:
            getattr(obs, method)(*args)
        except StopTraining:
            raise  # intentional halt (e.g. pruning) — propagate, don't swallow
        except Exception as exc:  # noqa: BLE001 — deliberately swallow per-sink errors
            print(
                f"[monitor] {type(obs).__name__}.{method} failed: {exc}",
                file=sys.stderr,
            )


# --------------------------------------------------------------------------- #
# Sinks
# --------------------------------------------------------------------------- #


class ConsoleSink(MetricSink):
    """Print a rich-formatted one-line summary per update."""

    def __init__(self, console: Any):
        self.console = console

    def log(self, update: int, total: int, stats: dict[str, Any]) -> None:
        ev = stats.get("eval_win_rate", "")
        ev_s = f"[green]{ev:.1%}[/]" if isinstance(ev, (int, float)) else "[dim]-[/]"
        self.console.print(
            f"[bold cyan]{update:>4}[/]/[cyan]{total}[/]  "
            f"games=[bold]{stats.get('games', 0):>3}[/]  "
            f"steps=[bold]{stats.get('steps', 0):>5}[/]  "
            f"pol=[yellow]{stats.get('pol_loss', 0):+.4f}[/]  "
            f"val=[yellow]{stats.get('val_loss', 0):.4f}[/]  "
            f"ent=[magenta]{stats.get('entropy', 0):.3f}[/]  "
            f"p0/p1={stats.get('p0_win', 0):.0%}/{stats.get('p1_win', 0):.0%}  "
            f"eval={ev_s}"
        )


class CsvSink(MetricSink):
    """Append one row of metrics per update to a CSV (header written once)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def start(self, ctx: RunContext) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="") as fh:
                csv.DictWriter(fh, fieldnames=CSV_FIELDS).writeheader()

    def log(self, update: int, total: int, stats: dict[str, Any]) -> None:
        row = {"update": update, **stats}
        with self.path.open("a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=CSV_FIELDS).writerow(
                {k: row.get(k, "") for k in CSV_FIELDS}
            )


class TensorBoardSink(MetricSink):
    """Log scalars to a TensorBoard event file (view: ``tensorboard --logdir``)."""

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.writer: Any = None

    def start(self, ctx: RunContext) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))

    def log(self, update: int, total: int, stats: dict[str, Any]) -> None:
        if self.writer is None:
            return
        for tag, key in SCALARS.items():
            v = stats.get(key)
            if isinstance(v, (int, float)):
                self.writer.add_scalar(tag, v, update)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None


class WandbSink(MetricSink):
    """Log to Weights & Biases. ``mode`` is 'offline' (local, default), 'online'
    (needs ``wandb login``), or 'disabled'."""

    def __init__(
        self,
        project: str,
        mode: str = "offline",
        name: str | None = None,
        dir: str | Path | None = None,
    ):
        self.project = project
        self.mode = mode
        self.name = name
        self.dir = dir
        self.run: Any = None

    def start(self, ctx: RunContext) -> None:
        import wandb

        if self.dir:
            Path(self.dir).mkdir(parents=True, exist_ok=True)
        self.run = wandb.init(
            project=self.project,
            name=self.name or ctx.run_name,
            mode=self.mode,
            dir=str(self.dir) if self.dir else None,
            config=ctx.config,
            resume="allow",
        )

    def log(self, update: int, total: int, stats: dict[str, Any]) -> None:
        if self.run is None:
            return
        import wandb

        payload = {
            tag: stats[key]
            for tag, key in SCALARS.items()
            if isinstance(stats.get(key), (int, float))
        }
        wandb.log(payload, step=update)

    def close(self) -> None:
        if self.run is not None:
            import wandb

            wandb.finish()
            self.run = None
