"""MCTS expert-iteration trainer.

collect: self-play where each acting decision runs MCTS (guided by the net) to
produce an improved policy target π; the played move is sampled from π. At game
end the ±1 outcome becomes the value target for that seat's samples.
update: supervised — cross-entropy(policy, π) + value_coef · MSE(value, z).
See docs/specs §5.

**pi consistency (Task 7 <-> Task 8):** `mcts.search`'s `pi` is a per-option
MARGINAL INCLUSION frequency (not a distribution over k-combinations), because
a multi-count node's visit/value stats are attributed to every option in the
picked slate independently (see `mcts.py` module docstring). That is exactly
the same per-option interpretation the model's policy head uses (it scores
each option independently, see `model.py`'s `policy_from_state`). So both
consumers of `pi` here treat it identically:
  - sampling the played move draws `k` *distinct* options from `pi` without
    replacement (mirrors `policy.select_count`'s without-replacement
    convention, and how MCTS's own root visit counts were accumulated), and
  - the imitation loss cross-entropies the model's per-option masked
    log-softmax directly against `pi` as a per-option target.
No renormalization/reinterpretation is needed between the two uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from pkm.new_agents.agent_000_dragapult import deck, mcts, policy
from pkm.new_agents.agent_000_dragapult.cabt import (
    battle_finish, battle_select, battle_start, to_observation,
)
from pkm.new_agents.agent_000_dragapult.features import Features, featurize
from pkm.new_agents.agent_000_dragapult.model import collate
from pkm.new_agents.agent_000_dragapult.shaping import _seat_reward


@dataclass
class ExItSample:
    features: Features
    policy_target: np.ndarray  # over the node's options; sums to 1
    seat: int
    value_target: float = 0.0  # filled at game end
    # MCTS-refined root value at this state (acting seat's perspective), captured
    # during search. Only populated + used by the "tdlambda" value-target scheme
    # (the bootstrap for the TD(λ) blend); ignored under the default "mc" scheme.
    root_value: float = 0.0


def _play_game(model, cfg, gen) -> tuple[list[ExItSample], int]:
    samples: list[ExItSample] = []
    deck_60 = deck.deck_60(cfg.run.deck)  # both self-play seats pilot the run's deck
    obs, _ = battle_start(deck_60, deck_60)
    n_iter = 0
    while obs["current"]["result"] < 0 and n_iter < 100000:
        if obs["select"] is None or obs["current"] is None:
            obs = battle_select(list(deck_60))
            n_iter += 1
            continue
        o = to_observation(obs)
        n = len(o.select.option)
        if n == 0:
            obs = battle_select([])
            n_iter += 1
            continue
        seat = obs["current"]["yourIndex"]
        # W-world IS-MCTS (W=1 == plain single-world search). Ask for the refined
        # root value only when the TD(λ) scheme needs it as a bootstrap.
        w = cfg.train.mcts_worlds
        if cfg.train.exit_value_target == "tdlambda":
            pi, root_v = mcts.search_worlds(
                obs, seat, model, cfg, gen, n_worlds=w, return_value=True
            )
        else:
            pi = mcts.search_worlds(
                obs, seat, model, cfg, gen, n_worlds=w, return_value=False
            )
            root_v = 0.0
        samples.append(
            ExItSample(features=featurize(o), policy_target=pi, seat=seat, root_value=root_v)
        )
        # play a move sampled from π (respecting the multi-select count)
        k = policy.select_count(o.select.minCount, o.select.maxCount, n)
        k = max(1, min(k, n))
        idx = torch.multinomial(torch.from_numpy(pi), k, replacement=False, generator=gen)
        obs = battle_select(idx.tolist())
        n_iter += 1
    result = obs["current"]["result"]
    battle_finish()
    _assign_value_targets(samples, result, cfg)
    return samples, result


def _assign_value_targets(samples: list[ExItSample], result: int, cfg) -> None:
    """Fill each sample's value target.

    "mc" (default): the raw game outcome (±1/0) for the acting seat.
    "tdlambda": blend the outcome with the MCTS-refined root value backward along
    each seat's own trajectory (agent_001's scheme). Per seat, walking from the
    last decision to the first::

        value := outcome(seat)
        label := (value + root_value) / 2
        value := λ·value + (1-λ)·root_value

    so early-game targets lean on the search-refined value while late-game ones
    stay anchored to the true outcome — lower variance than pure Monte-Carlo.
    """
    if cfg.train.exit_value_target != "tdlambda":
        for s in samples:  # Monte-Carlo value target = game outcome
            s.value_target = _seat_reward(result, s.seat)
        return
    lam = cfg.train.exit_lambda
    for seat in (0, 1):
        value = _seat_reward(result, seat)
        for s in reversed([s for s in samples if s.seat == seat]):
            s.value_target = (value + s.root_value) * 0.5
            value = value * lam + s.root_value * (1.0 - lam)


class ExItTrainer:
    def collect(self, model, n_games, cfg, gen=None):
        model.eval()
        # gen=None rides the process-global torch RNG (mirrors PPO's
        # collect_rollout/sample_action): each worker seeds it distinctly via
        # torch.manual_seed(base_seed+rank) in parallel._worker and it
        # advances across updates, so determinization + π-sampling are NOT
        # identically reseeded every call. An explicit gen (e.g. from tests)
        # is passed through unchanged for determinism.
        samples: list[ExItSample] = []
        results = []
        for _ in range(n_games):
            s, r = _play_game(model, cfg, gen)
            samples.extend(s)
            results.append(r)
        denom = max(n_games, 1)
        stats = {
            "games": n_games, "steps": len(samples),
            "p0_win": results.count(0) / denom, "p1_win": results.count(1) / denom,
        }
        return samples, stats

    def update(self, model, opt, samples, cfg):
        model.train()
        tc = cfg.train
        idx = np.arange(len(samples))
        rng = np.random.default_rng(tc.seed)
        agg = {"policy_loss": 0.0, "value_loss": 0.0, "n": 0}
        for _ in range(tc.epochs_per_update):
            rng.shuffle(idx)
            for start in range(0, len(idx), tc.minibatch_size):
                mb = [samples[i] for i in idx[start:start + tc.minibatch_size]]
                if not mb:
                    continue
                b = collate([s.features for s in mb])
                logits, value = model(b)  # logits [B,L], value [B]
                logp = torch.log_softmax(logits.masked_fill(b["option_mask"] == 0, -1e9), dim=-1)
                L = logits.shape[1]
                tgt = torch.zeros(len(mb), L)
                for i, s in enumerate(mb):
                    tgt[i, : len(s.policy_target)] = torch.from_numpy(s.policy_target)
                policy_loss = -(tgt * logp).sum(dim=-1).mean()
                z = torch.tensor([s.value_target for s in mb], dtype=torch.float32)
                value_loss = F.mse_loss(value, z)
                loss = policy_loss + tc.value_coef * value_loss
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.max_grad_norm)
                opt.step()
                agg["policy_loss"] += policy_loss.item()
                agg["value_loss"] += value_loss.item()
                agg["n"] += 1
        n = max(agg["n"], 1)
        return {
            "policy_loss": agg["policy_loss"] / n,
            "value_loss": agg["value_loss"] / n,
            "pol_loss": agg["policy_loss"] / n,   # alias for the console/CSV sinks
            "val_loss": agg["value_loss"] / n,
            "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0,
            "grad_norm": 0.0, "explained_var": 0.0,
        }
