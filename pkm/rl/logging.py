"""Metric logging backends.

Any class with a `scalar(tag, value, step)` and `close()` method can be used
as a backend. Register multiple to log to TensorBoard + wandb simultaneously.

Usage:
    from pkm.rl.logging import MetricLog

    log = MetricLog("runs/ppo")
    log.add_wandb(project="pkm-ppo", config={"lr": 3e-4})
    log.scalar("loss/policy", 0.5, step=1)
    log.close()

Or pass backends directly to the training loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class Backend(Protocol):
    """Protocol for metric logging backends."""

    def scalar(self, tag: str, value: float, step: int) -> None: ...
    def close(self) -> None: ...


class TensorBoardBackend:
    """Logs to TensorBoard."""

    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter

        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(log_dir)

    def scalar(self, tag: str, value: float, step: int) -> None:
        self._writer.add_scalar(tag, value, step)

    def close(self) -> None:
        self._writer.close()


class WandbBackend:
    """Logs to Weights & Biases."""

    def __init__(
        self,
        project: str,
        config: dict[str, Any] | None = None,
        run_name: str | None = None,
        log_dir: str | None = None,
    ):
        import wandb

        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            config=config or {},
            name=run_name,
            dir=log_dir,
            reinit=True,
        )

    def scalar(self, tag: str, value: float, step: int) -> None:
        self._run.log({tag: value}, step=step)

    def close(self) -> None:
        self._run.finish()


class CsvBackend:
    """Logs to a flat CSV file (one row per step)."""

    def __init__(self, path: str):
        import csv as csv_module

        self._csv = csv_module
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self._path, "w", newline="")
        self._writer = None
        self._row: dict[str, Any] = {}
        self._step: int = -1

    def scalar(self, tag: str, value: float, step: int) -> None:
        if step != self._step:
            self._flush()
            self._step = step
            self._row = {"step": step}
        self._row[tag] = value

    def _flush(self) -> None:
        if not self._row:
            return
        if self._writer is None:
            self._writer = self._csv.DictWriter(
                self._f, fieldnames=list(self._row.keys())
            )
            self._writer.writeheader()
        self._writer.writerow(self._row)
        self._f.flush()

    def close(self) -> None:
        self._flush()
        self._f.close()


class MetricLog:
    """Composite logger that fans out to multiple backends."""

    def __init__(self, backends: list[Backend] | None = None):
        self._backends: list[Backend] = list(backends or [])

    def add(self, backend: Backend) -> None:
        """Register a logging backend."""
        self._backends.append(backend)

    def add_tensorboard(self, log_dir: str) -> None:
        """Add a TensorBoard backend."""
        self.add(TensorBoardBackend(log_dir))

    def add_wandb(
        self,
        project: str,
        config: dict[str, Any] | None = None,
        run_name: str | None = None,
        log_dir: str | None = None,
    ) -> None:
        """Add a wandb backend."""
        self.add(
            WandbBackend(
                project=project, config=config, run_name=run_name, log_dir=log_dir
            )
        )

    def add_csv(self, path: str) -> None:
        """Add a CSV backend."""
        self.add(CsvBackend(path))

    def scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar to all registered backends."""
        for b in self._backends:
            b.scalar(tag, value, step)

    def log_dict(self, data: dict[str, float], step: int) -> None:
        """Log multiple scalars from a dict."""
        for tag, value in data.items():
            self.scalar(tag, value, step)

    def close(self) -> None:
        for b in self._backends:
            b.close()
