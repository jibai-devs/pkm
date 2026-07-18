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
from typing import Sequence

import numpy as np
import torch

from pkm.new_agents.agent_000_dragapult.cabt import (
    battle_finish,
    battle_select,
    battle_start,
    to_observation,
)
from pkm.new_agents.agent_000_dragapult import deck, policy
from pkm.new_agents.agent_000_dragapult.config import Config, _hash_dict, build_model
from pkm.new_agents.agent_000_dragapult.features import Features, featurize
from pkm.new_agents.agent_000_dragapult.model import collate
from pkm.new_agents.agent_000_dragapult.monitor import MetricSink, RunContext, notify
from pkm.new_agents.agent_000_dragapult.shaping import assign_targets


@dataclass
class Step:
    """One recorded policy decision (fields after `seat` filled by GAE)."""

    features: Features
    action: list[int]
    logprob: float
    value: float
    seat: int
    reward: float = 0.0  # filled by the shaper (shaping.assign_targets)
    adv: float = 0.0
    ret: float = 0.0


# --------------------------------------------------------------------------- #
# Rollout (self-play)
# --------------------------------------------------------------------------- #


def play_game(
    model: torch.nn.Module, cfg: Config, gen: torch.Generator | None = None
) -> tuple[list[Step], int]:
    """Play one self-play game; return (recorded steps, result)."""
    steps: list[Step] = []
    obs, _ = battle_start(deck.DECK_60, deck.DECK_60)
    n_iter = 0
    while obs["current"]["result"] < 0 and n_iter < 100000:
        if obs["select"] is None or obs["current"] is None:
            obs = battle_select(list(deck.DECK_60))  # deck-selection phase
            n_iter += 1
            continue
        f = featurize(to_observation(obs))
        n = f.n_options
        if n == 0:
            obs = battle_select([])
            n_iter += 1
            continue
        b = collate([f])
        with torch.no_grad():
            logits, value = model(b)
        k = policy.select_count(f.min_count, f.max_count, n)
        valid = torch.zeros(logits.shape[1], dtype=torch.bool)
        valid[:n] = True
        picks, logprob = policy.sample_action(logits[0], valid, k, gen=gen)
        steps.append(
            Step(
                features=f,
                action=picks,
                logprob=logprob,
                value=float(value[0]),
                seat=obs["current"]["yourIndex"],
            )
        )
        obs = battle_select(picks)
        n_iter += 1
    result = obs["current"]["result"]
    battle_finish()
    assign_targets(steps, result, cfg)
    return steps, result


def collect_rollout(
    model: torch.nn.Module,
    n_games: int,
    cfg: Config,
    gen: torch.Generator | None = None,
) -> tuple[list[Step], dict[str, float]]:
    model.eval()
    steps: list[Step] = []
    results = []
    for _ in range(n_games):
        s, r = play_game(model, cfg, gen=gen)
        steps.extend(s)
        results.append(r)
    denom = max(n_games, 1)
    stats = {
        "games": n_games,
        "steps": len(steps),
        "p0_win": results.count(0) / denom,
        "p1_win": results.count(1) / denom,
    }
    return steps, stats


# --------------------------------------------------------------------------- #
# PPO update
# --------------------------------------------------------------------------- #


def _minibatch(steps: list[Step]) -> dict[str, torch.Tensor]:
    """Collate a list of steps into a training batch."""
    b = collate([s.features for s in steps])
    kmax = max(len(s.action) for s in steps)
    actions = torch.zeros(len(steps), kmax, dtype=torch.long)
    action_len = torch.tensor([len(s.action) for s in steps], dtype=torch.long)
    for i, s in enumerate(steps):
        actions[i, : len(s.action)] = torch.tensor(s.action, dtype=torch.long)
    b["actions"] = actions
    b["action_len"] = action_len
    b["old_logprob"] = torch.tensor([s.logprob for s in steps], dtype=torch.float32)
    b["adv"] = torch.tensor([s.adv for s in steps], dtype=torch.float32)
    b["ret"] = torch.tensor([s.ret for s in steps], dtype=torch.float32)
    return b


def ppo_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    steps: list[Step],
    cfg: Config,
) -> dict[str, float]:
    model.train()
    tc = cfg.train
    idx = np.arange(len(steps))
    adv_all = torch.tensor([s.adv for s in steps], dtype=torch.float32)
    adv_mean, adv_std = adv_all.mean().item(), adv_all.std().item() + 1e-8
    stats = {
        "pol_loss": 0.0,
        "val_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_frac": 0.0,
        "grad_norm": 0.0,
        "n": 0,
    }
    rng = np.random.default_rng(tc.seed)
    for _ in range(tc.epochs_per_update):
        rng.shuffle(idx)
        for start in range(0, len(idx), tc.minibatch_size):
            mb = [steps[i] for i in idx[start : start + tc.minibatch_size]]
            if not mb:
                continue
            b = _minibatch(mb)
            logits, value = model(b)
            new_lp = policy.batched_action_logprob(
                logits, b["option_mask"], b["actions"], b["action_len"]
            )
            ent = policy.batched_entropy(logits, b["option_mask"]).mean()
            adv = (b["adv"] - adv_mean) / adv_std
            logratio = new_lp - b["old_logprob"]
            ratio = logratio.exp()
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1 - tc.clip_eps, 1 + tc.clip_eps) * adv
            pol_loss = -torch.min(unclipped, clipped).mean()
            val_loss = (value - b["ret"]).pow(2).mean()
            loss = pol_loss + tc.value_coef * val_loss - tc.entropy_coef * ent
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), tc.max_grad_norm
            )
            optimizer.step()
            with torch.no_grad():
                # http://joschu.net/blog/kl-approx.html — low-variance, ≥0 KL est.
                approx_kl = ((ratio - 1) - logratio).mean().item()
                clip_frac = (ratio - 1).abs().gt(tc.clip_eps).float().mean().item()
            stats["pol_loss"] += pol_loss.item()
            stats["val_loss"] += val_loss.item()
            stats["entropy"] += ent.item()
            stats["approx_kl"] += approx_kl
            stats["clip_frac"] += clip_frac
            stats["grad_norm"] += float(grad_norm)
            stats["n"] += 1
    n = max(stats["n"], 1)
    # Explained variance of the value head over the whole batch (1.0 = perfect,
    # 0.0 = no better than predicting the mean, <0 = worse). Diagnoses whether
    # the critic is actually learning the return.
    with torch.no_grad():
        b_all = _minibatch(steps)
        _, v_all = model(b_all)
        ret_all = b_all["ret"]
        var_ret = ret_all.var().item()
        explained_var = 1.0 - (ret_all - v_all).var().item() / (var_ret + 1e-8)
    return {
        "pol_loss": stats["pol_loss"] / n,
        "val_loss": stats["val_loss"] / n,
        "entropy": stats["entropy"] / n,
        "approx_kl": stats["approx_kl"] / n,
        "clip_frac": stats["clip_frac"] / n,
        "grad_norm": stats["grad_norm"] / n,
        "explained_var": explained_var,
    }


# --------------------------------------------------------------------------- #
# Checkpoint / resume
# --------------------------------------------------------------------------- #


@dataclass
class TrainState:
    cfg: Config
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    update_idx: int = 0

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "config": self.cfg.to_dict(),
            "config_hash": self.cfg.hash(),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_idx": self.update_idx,
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
        torch.set_rng_state(blob["rng"]["torch"])
        np.random.set_state(blob["rng"]["numpy"])
        random.setstate(blob["rng"]["python"])
        return cls(
            cfg=cfg, model=model, optimizer=optimizer, update_idx=blob["update_idx"]
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
        pool = ParallelRollout(cfg, cfg.train.num_workers, base_seed=cfg.train.seed)
    try:
        for _ in range(updates):
            t0 = time.perf_counter()
            if pool is not None:
                steps, roll_stats = pool.collect(ts.model, games_per_update)
            else:
                steps, roll_stats = collect_rollout(ts.model, games_per_update, cfg)
            t_rollout = time.perf_counter() - t0
            t1 = time.perf_counter()
            upd_stats = ppo_update(ts.model, ts.optimizer, steps, cfg)
            t_update = time.perf_counter() - t1
            ts.update_idx += 1

            t_total = time.perf_counter() - t0
            n_steps = roll_stats.get("steps", 0)
            time_stats = {
                "t_rollout": t_rollout,
                "t_update": t_update,
                "t_total": t_total,
                "sps": n_steps / t_total if t_total > 0 else 0.0,
                "eta_s": (target - ts.update_idx) * t_total,
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

                ev = winrate_vs_random(ts.model, n_games=eval_games)
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
