"""PPO + self-play trainer for agent_000_dragapult (v1 baseline).

Pipeline: self-play rollout (one model pilots both seats) -> per-seat GAE with
terminal +/-1 reward -> clipped PPO update over the presented-option action
distribution (see :mod:`.policy`). Everything needed to resume a run is captured
in a `TrainState` checkpoint (weights, optimizer, RNG, update index, config).

This is the first learner (README D7). Hybrid inference-MCTS / expert iteration
are future work (README §9); they reuse this self-play + checkpoint plumbing and
the same `model.evaluate()` interface.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from pkm.new_agents.agent_000_dragapult.config import Config, _hash_dict, build_model
from pkm.new_agents.agent_000_dragapult.monitor import MetricSink, RunContext, notify

# Back-compat re-export: Step used to live here.
from pkm.new_agents.agent_000_dragapult.trainers.ppo import Step  # noqa: F401


# --------------------------------------------------------------------------- #
# Checkpoint / resume
# --------------------------------------------------------------------------- #


@dataclass
class TrainState:
    cfg: Config
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    update_idx: int = 0
    # LR scheduler (e.g. cosine); None for the constant-LR default. Its state is
    # persisted so a resumed run continues the schedule from the right point.
    scheduler: Any = None

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "config": self.cfg.to_dict(),
            "config_hash": self.cfg.hash(),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_idx": self.update_idx,
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "rng": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(blob, tmp)
        tmp.replace(path)  # atomic

    @classmethod
    def load(cls, path: str | Path) -> "TrainState":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        cfg = Config.from_dict(blob["config"])
        # Validate the STORED dict against its STORED hash, so additive schema
        # changes (new fields with defaults) never trip the guard for older
        # checkpoints. A tampered file still fails.
        if _hash_dict(blob["config"]) != blob["config_hash"]:
            raise ValueError("config hash mismatch on resume")
        model = build_model(cfg)
        model.load_state_dict(blob["model"])
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
        optimizer.load_state_dict(blob["optimizer"])
        # Rebuild the scheduler and restore its state (T_max/eta_min/last_epoch all
        # live in the saved state_dict, so a placeholder T_max is fine here).
        scheduler = None
        if cfg.train.lr_schedule == "cosine" and blob.get("scheduler"):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
            scheduler.load_state_dict(blob["scheduler"])
        torch.set_rng_state(blob["rng"]["torch"])
        np.random.set_state(blob["rng"]["numpy"])
        random.setstate(blob["rng"]["python"])
        return cls(
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            update_idx=blob["update_idx"],
            scheduler=scheduler,
        )


# --------------------------------------------------------------------------- #
# Train loop
# --------------------------------------------------------------------------- #


def train(
    cfg: Config,
    updates: int,
    games_per_update: int,
    ckpt_dir: str | Path,
    resume: bool = False,
    eval_every: int = 0,
    eval_games: int = 100,
    observers: Sequence[MetricSink] = (),
    run_name: str | None = None,
    device: str = "cpu",
) -> TrainState:
    """Run ``updates`` PPO self-play updates.

    ``observers`` are :class:`~.monitor.MetricSink` objects notified after every
    update with that update's ``stats`` (console, CSV, TensorBoard, wandb, …).
    Their ``start``/``log``/``close`` hooks are called with per-sink failures
    isolated (see :func:`~.monitor.notify`), so a broken sink never aborts a run.
    """
    ckpt_dir = Path(ckpt_dir)
    latest = ckpt_dir / "latest.pt"
    if resume and latest.exists():
        ts = TrainState.load(latest)
    else:
        model = build_model(cfg)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
        ts = TrainState(cfg=cfg, model=model, optimizer=opt)
        # Fresh run: build the LR scheduler spanning this run's planned updates
        # (cosine from lr down to lr_min). Resumed runs restore it in load().
        if cfg.train.lr_schedule == "cosine":
            ts.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=updates, eta_min=cfg.train.lr_min
            )

    # Device: the learner (model + optimizer + PPO update) runs on `dev`; rollout
    # workers always get CPU weights (parallel.py .cpu()'s the state_dict), since
    # self-play is CPU-bound engine work. Only the batched update benefits from GPU.
    from pkm.new_agents.agent_000_dragapult.config import resolve_device

    dev = resolve_device(device)
    ts.model.to(dev)
    # On resume the optimizer state loaded on CPU must follow the params to `dev`.
    for st in ts.optimizer.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                st[k] = v.to(dev)
    print(
        f"[device] learner on {dev}"
        + ("" if dev == "cpu" else " (rollout workers + eval stay on CPU)"),
        flush=True,
    )

    start_idx = ts.update_idx
    target = start_idx + updates
    ctx = RunContext(
        run_name=run_name or f"{cfg.run.name}-{cfg.hash()}",
        config=cfg.to_dict(),
        config_hash=cfg.hash(),
        output_dir=ckpt_dir.parent,
        resume=resume,
    )
    notify(observers, "start", ctx)

    from pkm.new_agents.agent_000_dragapult.trainers import get_trainer

    _trainer = get_trainer(cfg)

    pool = None
    if cfg.train.num_workers and cfg.train.num_workers > 1:
        import os

        from pkm.new_agents.agent_000_dragapult.parallel import ParallelRollout

        w = cfg.train.num_workers
        cores = os.cpu_count() or 0
        per = games_per_update // w
        rem = games_per_update % w
        spread = f"{per}" if rem == 0 else f"{per}-{per + 1}"
        idle = max(cores - w - 1, 0)  # -1 for the learner on the main process
        print(
            f"[parallel] {w} workers  |  {cores} cores "
            f"({idle} idle during rollout)  |  {games_per_update} games/update "
            f"= {spread} games/worker",
            flush=True,
        )
        if per <= 2:
            print(
                "[parallel]  note: <=2 games/worker → high straggler variance; "
                "raise --games or --workers for better utilization.",
                flush=True,
            )
        pool = ParallelRollout(
            cfg, cfg.train.num_workers, base_seed=cfg.train.seed, model=ts.model
        )
    try:
        for _ in range(updates):
            t0 = time.perf_counter()
            trainer = _trainer  # built once before the loop
            if pool is not None:
                steps, roll_stats = pool.collect(trainer, games_per_update)
            else:
                steps, roll_stats = trainer.collect(ts.model, games_per_update, cfg)
            t_rollout = time.perf_counter() - t0
            t1 = time.perf_counter()
            upd_stats = trainer.update(ts.model, ts.optimizer, steps, cfg)
            t_update = time.perf_counter() - t1
            if ts.scheduler is not None:
                ts.scheduler.step()
            ts.update_idx += 1

            t_total = time.perf_counter() - t0
            n_steps = roll_stats.get("steps", 0)
            time_stats = {
                "t_rollout": t_rollout,
                "t_update": t_update,
                "t_total": t_total,
                "sps": n_steps / t_total if t_total > 0 else 0.0,  # decisions/sec (1 step = 1 decision)
                "gps": games_per_update / t_total if t_total > 0 else 0.0,  # games/sec
                "eta_s": (target - ts.update_idx) * t_total,
                "lr": ts.optimizer.param_groups[0]["lr"],
            }
            # Parallel-efficiency diagnostics (only meaningful with a worker pool):
            #  - rollout_util: how balanced the barrier is. mean/max worker busy
            #    time; 1.0 = all workers finished together, 0.5 = the average
            #    worker sat idle for half of t_rollout waiting on the straggler.
            #  - core_util: fraction of the WHOLE cycle that workers spend doing
            #    real work vs. idling — folds in the serial PPO update, during
            #    which every worker is blocked. = worker_busy_sum / (W * t_total).
            #  - serial_frac: share of the cycle where all workers are idle
            #    (update + weight-broadcast overhead). Amdahl's serial fraction.
            if pool is not None and "worker_busy_max" in roll_stats:
                w = roll_stats.get("num_workers", 1)
                busy_max = roll_stats["worker_busy_max"]
                busy_sum = roll_stats["worker_busy_sum"]
                time_stats["rollout_util"] = (
                    roll_stats["worker_busy_mean"] / busy_max if busy_max > 0 else 0.0
                )
                time_stats["core_util"] = (
                    busy_sum / (w * t_total) if t_total > 0 else 0.0
                )
                time_stats["serial_frac"] = (
                    (t_total - t_rollout) / t_total if t_total > 0 else 0.0
                )
            stats = {**roll_stats, **upd_stats, **time_stats}
            if eval_every and ts.update_idx % eval_every == 0:
                from pkm.new_agents.agent_000_dragapult.eval import winrate_vs_random

                # Eval runs on CPU (batch-1 inference; DragapultAgent would also
                # move the model to CPU in place). On GPU, hand it a CPU clone so
                # the learner's device/optimizer are left untouched.
                eval_model = ts.model
                if dev != "cpu":
                    eval_model = build_model(ts.cfg)
                    eval_model.load_state_dict(
                        {k: v.detach().cpu() for k, v in ts.model.state_dict().items()}
                    )
                ev = winrate_vs_random(
                    eval_model, n_games=eval_games, deck_name=ts.cfg.run.deck
                )
                stats["eval_win_rate"] = ev["win_rate"]

            # notify() isolates ordinary sink errors but re-raises StopTraining
            # (e.g. an Optuna PruningSink deciding to abort this trial early).
            notify(observers, "log", ts.update_idx, target, stats)

            if ts.update_idx % cfg.run.checkpoint_every_updates == 0:
                ts.save(ckpt_dir / f"ckpt_{ts.update_idx}.pt")
            ts.save(latest)
    finally:
        notify(observers, "close")
        if pool is not None:
            pool.close()
    return ts
