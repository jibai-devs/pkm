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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pkm.cabt import battle_finish, battle_select, battle_start, to_observation
from pkm.agents.agent_000_dragapult import deck, policy
from pkm.agents.agent_000_dragapult.config import Config, build_model
from pkm.agents.agent_000_dragapult.features import Features, featurize
from pkm.agents.agent_000_dragapult.model import collate


@dataclass
class Step:
    """One recorded policy decision (fields after `seat` filled by GAE)."""

    features: Features
    action: list[int]
    logprob: float
    value: float
    seat: int
    adv: float = 0.0
    ret: float = 0.0


# --------------------------------------------------------------------------- #
# Rollout (self-play)
# --------------------------------------------------------------------------- #

def _seat_reward(result: int, seat: int) -> float:
    if result == seat:
        return 1.0
    if result in (0, 1):        # a decisive result for the other seat
        return -1.0
    return 0.0                  # draw / unknown


def _gae(steps: list[Step], gamma: float, lam: float, result: int) -> None:
    """Fill adv/ret per seat (each seat's decisions are its own trajectory)."""
    for seat in (0, 1):
        traj = [s for s in steps if s.seat == seat]
        adv = 0.0
        for t in reversed(range(len(traj))):
            last = t == len(traj) - 1
            next_v = 0.0 if last else traj[t + 1].value
            nonterm = 0.0 if last else 1.0
            r = _seat_reward(result, seat) if last else 0.0
            delta = r + gamma * next_v * nonterm - traj[t].value
            adv = delta + gamma * lam * nonterm * adv
            traj[t].adv = adv
            traj[t].ret = adv + traj[t].value


def play_game(
    model: torch.nn.Module, gamma: float, lam: float, gen: torch.Generator | None = None
) -> tuple[list[Step], int]:
    """Play one self-play game; return (recorded steps, result)."""
    steps: list[Step] = []
    obs, _ = battle_start(deck.DECK_60, deck.DECK_60)
    n_iter = 0
    while obs["current"]["result"] < 0 and n_iter < 100000:
        if obs["select"] is None or obs["current"] is None:
            obs = battle_select(list(deck.DECK_60))          # deck-selection phase
            n_iter += 1
            continue
        f = featurize(to_observation(obs))
        n = f.n_options
        if n == 0:
            obs = battle_select([]); n_iter += 1; continue
        b = collate([f])
        with torch.no_grad():
            logits, value = model(b)
        k = policy.select_count(f.min_count, f.max_count, n)
        valid = torch.zeros(logits.shape[1], dtype=torch.bool)
        valid[:n] = True
        picks, logprob = policy.sample_action(logits[0], valid, k, gen=gen)
        steps.append(Step(features=f, action=picks, logprob=logprob,
                          value=float(value[0]), seat=obs["current"]["yourIndex"]))
        obs = battle_select(picks)
        n_iter += 1
    result = obs["current"]["result"]
    battle_finish()
    _gae(steps, gamma, lam, result)
    return steps, result


def collect_rollout(
    model: torch.nn.Module, n_games: int, cfg: Config, gen: torch.Generator | None = None
) -> tuple[list[Step], dict[str, float]]:
    model.eval()
    steps: list[Step] = []
    results = []
    for _ in range(n_games):
        s, r = play_game(model, cfg.train.gamma, cfg.train.gae_lambda, gen=gen)
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
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, steps: list[Step], cfg: Config
) -> dict[str, float]:
    model.train()
    tc = cfg.train
    idx = np.arange(len(steps))
    adv_all = torch.tensor([s.adv for s in steps], dtype=torch.float32)
    adv_mean, adv_std = adv_all.mean().item(), adv_all.std().item() + 1e-8
    stats = {"pol_loss": 0.0, "val_loss": 0.0, "entropy": 0.0, "n": 0}
    rng = np.random.default_rng(tc.seed)
    for _ in range(tc.epochs_per_update):
        rng.shuffle(idx)
        for start in range(0, len(idx), tc.minibatch_size):
            mb = [steps[i] for i in idx[start : start + tc.minibatch_size]]
            if not mb:
                continue
            b = _minibatch(mb)
            logits, value = model(b)
            new_lp = policy.batched_action_logprob(logits, b["option_mask"], b["actions"], b["action_len"])
            ent = policy.batched_entropy(logits, b["option_mask"]).mean()
            adv = (b["adv"] - adv_mean) / adv_std
            ratio = (new_lp - b["old_logprob"]).exp()
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1 - tc.clip_eps, 1 + tc.clip_eps) * adv
            pol_loss = -torch.min(unclipped, clipped).mean()
            val_loss = (value - b["ret"]).pow(2).mean()
            loss = pol_loss + tc.value_coef * val_loss - tc.entropy_coef * ent
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.max_grad_norm)
            optimizer.step()
            stats["pol_loss"] += pol_loss.item(); stats["val_loss"] += val_loss.item()
            stats["entropy"] += ent.item(); stats["n"] += 1
    n = max(stats["n"], 1)
    return {"pol_loss": stats["pol_loss"] / n, "val_loss": stats["val_loss"] / n,
            "entropy": stats["entropy"] / n}


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
        tmp.replace(path)   # atomic

    @classmethod
    def load(cls, path: str | Path) -> "TrainState":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        cfg = Config.from_dict(blob["config"])
        if cfg.hash() != blob["config_hash"]:
            raise ValueError("config hash mismatch on resume")
        model = build_model(cfg)
        model.load_state_dict(blob["model"])
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
        optimizer.load_state_dict(blob["optimizer"])
        torch.set_rng_state(blob["rng"]["torch"])
        np.random.set_state(blob["rng"]["numpy"])
        random.setstate(blob["rng"]["python"])
        return cls(cfg=cfg, model=model, optimizer=optimizer, update_idx=blob["update_idx"])


# --------------------------------------------------------------------------- #
# Train loop
# --------------------------------------------------------------------------- #

def train(cfg: Config, updates: int, games_per_update: int, ckpt_dir: str | Path,
          resume: bool = False, eval_every: int = 0, eval_games: int = 100) -> TrainState:
    ckpt_dir = Path(ckpt_dir)
    latest = ckpt_dir / "latest.pt"
    if resume and latest.exists():
        ts = TrainState.load(latest)
        print(f"resumed at update {ts.update_idx}")
    else:
        model = build_model(cfg)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
        ts = TrainState(cfg=cfg, model=model, optimizer=opt)

    pool = None
    if cfg.train.num_workers and cfg.train.num_workers > 1:
        from pkm.agents.agent_000_dragapult.parallel import ParallelRollout
        pool = ParallelRollout(cfg, cfg.train.num_workers, base_seed=cfg.train.seed)
    try:
        for _ in range(updates):
            if pool is not None:
                steps, roll_stats = pool.collect(ts.model, games_per_update)
            else:
                steps, roll_stats = collect_rollout(ts.model, games_per_update, cfg)
            upd_stats = ppo_update(ts.model, ts.optimizer, steps, cfg)
            ts.update_idx += 1
            print(f"update {ts.update_idx}: {roll_stats} {upd_stats}")
            if eval_every and ts.update_idx % eval_every == 0:
                from pkm.agents.agent_000_dragapult.eval import winrate_vs_random
                ev = winrate_vs_random(ts.model, n_games=eval_games)
                print(f"  eval@{ts.update_idx} vs random: win_rate={ev['win_rate']:.2%} "
                      f"(W{ev['wins']}/L{ev['losses']}/D{ev['draws']})")
            if ts.update_idx % cfg.run.checkpoint_every_updates == 0:
                ts.save(ckpt_dir / f"ckpt_{ts.update_idx}.pt")
            ts.save(latest)
    finally:
        if pool is not None:
            pool.close()
    return ts
