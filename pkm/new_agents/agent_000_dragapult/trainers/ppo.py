"""PPO + self-play trainer (relocated from train.py, behavior unchanged)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from pkm.new_agents.agent_000_dragapult.cabt import (
    battle_finish,
    battle_select,
    battle_start,
    to_observation,
)
from pkm.new_agents.agent_000_dragapult import deck, policy
from pkm.new_agents.agent_000_dragapult.config import Config
from pkm.new_agents.agent_000_dragapult.features import Features, featurize
from pkm.new_agents.agent_000_dragapult.model import collate
from pkm.new_agents.agent_000_dragapult.shaping import assign_targets
from pkm.rl import encoder as H
from pkm.types.obs import Observation


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
    # Heuristic scalars, filled by `_fill_heuristics` during rollout and read by
    # the "heuristic" shaper (see shaping.py). Names match reward_terms.ALL_TERMS
    # attrs. All default to 0.0, so a step never touched by _fill_heuristics
    # (any shaping other than "heuristic") contributes nothing.
    potential: float = 0.0  # prize differential (POTENTIAL term "shaping")
    board_setup_potential: float = 0.0
    budew_setup_potential: float = 0.0
    dreepy_line_field_potential: float = 0.0
    energy_penalty: float = 0.0
    budew_bonus: float = 0.0
    wrong_type_energy_penalty: float = 0.0
    dragapult_attack_bonus: float = 0.0
    dreepy_spread_penalty: float = 0.0
    xerosic_bonus: float = 0.0
    budew_bench_setup_bonus: float = 0.0
    dreepy_evolve_bonus: float = 0.0
    dreepy_bench_charge_bonus: float = 0.0
    dreepy_active_charge_bonus: float = 0.0
    wasted_resources_penalty: float = 0.0
    phantom_dive_bonus: float = 0.0


def _fill_heuristics(step: "Step", parsed: Observation, picks: list[int]) -> None:
    """Compute every deck-specific heuristic on the live observation + picks and
    stash the scalars on `step`. Mirrors pkm/rl/rollout.py's collection block;
    the "heuristic" shaper combines them into rewards via the reward_terms
    registry. Only called when cfg.train.shaping == "heuristic"."""
    step.potential = H.prize_potential(parsed)
    step.board_setup_potential = H.dragapult_backup_potential(parsed)
    step.budew_setup_potential = H.budew_active_second_potential(parsed)
    step.dreepy_line_field_potential = H.dreepy_line_field_potential(parsed)
    step.energy_penalty = H.energy_overattach_penalty(parsed, picks)
    step.budew_bonus = H.budew_first_turn_attack_bonus(parsed, picks)
    step.wrong_type_energy_penalty = H.wrong_type_energy_penalty(parsed, picks)
    step.dragapult_attack_bonus = H.dragapult_ex_attack_bonus(parsed, picks)
    step.phantom_dive_bonus = H.phantom_dive_attack_bonus(parsed, picks)
    step.dreepy_spread_penalty = H.dreepy_energy_spread_penalty(parsed, picks)
    step.xerosic_bonus = H.xerosic_machinations_bonus(parsed, picks)
    step.budew_bench_setup_bonus = H.budew_turn_bench_setup_bonus(parsed, picks)
    step.dreepy_evolve_bonus = H.dreepy_evolve_bonus(parsed, picks)
    step.dreepy_bench_charge_bonus = H.dreepy_line_bench_charge_bonus(parsed, picks)
    step.dreepy_active_charge_bonus = H.dreepy_line_active_charge_bonus(parsed, picks)
    step.wasted_resources_penalty = H.wasted_resources_attack_penalty(parsed, picks)


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
        step = Step(
            features=f,
            action=picks,
            logprob=logprob,
            value=float(value[0]),
            seat=obs["current"]["yourIndex"],
        )
        if cfg.train.shaping == "heuristic":
            # The heuristics want a pkm.types.obs.Observation (not cabt's own
            # type), parsed straight from the wire dict as pkm/rl/rollout does.
            _fill_heuristics(step, Observation.model_validate(obs), picks)
        steps.append(step)
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


class PpoTrainer:
    """Classic PPO + self-play (the v1 baseline), behind the Trainer protocol."""

    def collect(self, model, n_games, cfg, gen=None):
        return collect_rollout(model, n_games, cfg, gen=gen)

    def update(self, model, opt, samples, cfg):
        return ppo_update(model, opt, samples, cfg)
