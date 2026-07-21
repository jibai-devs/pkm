"""PPO + self-play trainer (relocated from train.py, behavior unchanged)."""

from __future__ import annotations

from dataclasses import dataclass, field

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
from pkm.new_agents.agent_000_dragapult.shaping import HEURISTIC_SHAPERS, assign_targets
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
    # Auxiliary-task labels, name -> target, filled after each game by the active
    # aux tasks (see aux_tasks.py). Empty unless an aux task is on; the aux loss
    # only reads names it activated, so a missing key never matters.
    aux: dict = field(default_factory=dict)
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
    dev = next(model.parameters()).device  # cpu for rollout workers; gpu if single-process --device cuda
    deck_60 = deck.deck_60(cfg.run.deck)  # both self-play seats pilot the run's deck
    obs, _ = battle_start(deck_60, deck_60)
    n_iter = 0
    while obs["current"]["result"] < 0 and n_iter < 100000:
        if obs["select"] is None or obs["current"] is None:
            obs = battle_select(list(deck_60))  # deck-selection phase
            n_iter += 1
            continue
        f = featurize(to_observation(obs))
        n = f.n_options
        if n == 0:
            obs = battle_select([])
            n_iter += 1
            continue
        b = {k: v.to(dev) for k, v in collate([f]).items()}
        with torch.no_grad():
            state, ent = model.encode(b)
            value = model.value_from_state(state)
            if getattr(model, "policy_head", "marginal") == "autoreg":
                # Autoregressive multi-select: the head samples its own count
                # (may pick fewer than maxCount, incl. nothing when minCount==0).
                picks, logprob = policy.sample_action_autoreg(
                    model, state, ent, b, gen=gen
                )
            else:
                logits = model.policy_from_state(state, ent, b)
                k = policy.select_count(f.min_count, f.max_count, n)
                valid = torch.zeros(
                    logits.shape[1], dtype=torch.bool, device=logits.device
                )
                valid[:n] = True
                picks, logprob = policy.sample_action(logits[0], valid, k, gen=gen)
        step = Step(
            features=f,
            action=picks,
            logprob=logprob,
            value=float(value[0]),
            seat=obs["current"]["yourIndex"],
        )
        if cfg.train.shaping in HEURISTIC_SHAPERS:
            # The heuristics want a pkm.types.obs.Observation (not cabt's own
            # type), parsed straight from the wire dict as pkm/rl/rollout does.
            _fill_heuristics(step, Observation.model_validate(obs), picks)
        steps.append(step)
        obs = battle_select(picks)
        n_iter += 1
    result = obs["current"]["result"]
    terminal_obs = obs  # last observation carries the final prize piles
    battle_finish()
    assign_targets(steps, result, cfg)
    # Auxiliary-task labels: let each active task read the finished game and
    # stamp its per-step target. No-op when no aux task is active (the default).
    from pkm.new_agents.agent_000_dragapult.aux_tasks import AUX_TASKS, active_tasks

    for name in active_tasks(cfg.train.aux_weights):
        AUX_TASKS[name].assign(steps, terminal_obs)
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
    # Auxiliary labels: one tensor per task present on the steps, keyed
    # `aux__<name>`. Missing labels default to 0.0 (a step from a game where the
    # task wasn't active). Iterating the union of keys keeps this task-agnostic.
    aux_names = {name for s in steps for name in s.aux}
    for name in aux_names:
        b[f"aux__{name}"] = torch.tensor(
            [float(s.aux.get(name, 0.0)) for s in steps], dtype=torch.float32
        )
    return b


def ppo_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    steps: list[Step],
    cfg: Config,
) -> dict[str, float]:
    model.train()
    tc = cfg.train
    dev = next(model.parameters()).device  # learner device (cpu unless --device cuda)
    idx = np.arange(len(steps))
    adv_all = torch.tensor([s.adv for s in steps], dtype=torch.float32)
    adv_mean, adv_std = adv_all.mean().item(), adv_all.std().item() + 1e-8
    from pkm.new_agents.agent_000_dragapult.aux_tasks import AUX_TASKS, active_tasks

    aux_names = active_tasks(tc.aux_weights)
    stats = {
        "pol_loss": 0.0,
        "val_loss": 0.0,
        "aux_loss": 0.0,
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
            b = {k: v.to(dev) for k, v in b.items()}
            # Run the trunk once, then each head, so the aux heads share the same
            # forward as policy + value (that shared pass is the whole point).
            state, ent_emb = model.encode(b)
            value = model.value_from_state(state)
            if getattr(model, "policy_head", "marginal") == "autoreg":
                new_lp = policy.batched_action_logprob_autoreg(
                    model, state, ent_emb, b, b["actions"], b["action_len"]
                )
                ent = policy.batched_entropy_autoreg(model, state, ent_emb, b).mean()
            else:
                logits = model.policy_from_state(state, ent_emb, b)
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
            # Auxiliary losses: sum of weight * task.loss(pred, target) over the
            # active heads. Zero (and no extra compute) when none are active.
            aux_loss = value.new_zeros(())
            if aux_names:
                aux_preds = model.aux_from_state(state)
                for name in aux_names:
                    pred, target = aux_preds.get(name), b.get(f"aux__{name}")
                    if pred is not None and target is not None:
                        aux_loss = aux_loss + tc.aux_weights[name] * AUX_TASKS[
                            name
                        ].loss(pred, target)
            loss = (
                pol_loss
                + tc.value_coef * val_loss
                + aux_loss
                - tc.entropy_coef * ent
            )
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
            stats["aux_loss"] += float(aux_loss.detach())
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
        # Forward in minibatch-sized chunks, not one giant batch: a full-update
        # batch (~1e4 steps) is fine in CPU RAM but OOMs a GPU (each option/entity
        # tensor times thousands of steps). Concatenate the per-chunk values.
        v_parts, ret_parts = [], []
        for start in range(0, len(steps), tc.minibatch_size):
            mb = steps[start : start + tc.minibatch_size]
            b = {k: v.to(dev) for k, v in _minibatch(mb).items()}
            _, v = model(b)
            v_parts.append(v)
            ret_parts.append(b["ret"])
        v_all = torch.cat(v_parts)
        ret_all = torch.cat(ret_parts)
        var_ret = ret_all.var().item()
        explained_var = 1.0 - (ret_all - v_all).var().item() / (var_ret + 1e-8)
    return {
        "pol_loss": stats["pol_loss"] / n,
        "val_loss": stats["val_loss"] / n,
        "aux_loss": stats["aux_loss"] / n,
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
