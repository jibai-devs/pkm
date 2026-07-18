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


def _play_game(model, cfg, gen) -> tuple[list[ExItSample], int]:
    samples: list[ExItSample] = []
    obs, _ = battle_start(deck.DECK_60, deck.DECK_60)
    n_iter = 0
    while obs["current"]["result"] < 0 and n_iter < 100000:
        if obs["select"] is None or obs["current"] is None:
            obs = battle_select(list(deck.DECK_60)); n_iter += 1; continue
        o = to_observation(obs)
        n = len(o.select.option)
        if n == 0:
            obs = battle_select([]); n_iter += 1; continue
        seat = obs["current"]["yourIndex"]
        pi = mcts.search(obs, seat, model, cfg, gen)  # [n]
        samples.append(ExItSample(features=featurize(o), policy_target=pi, seat=seat))
        # play a move sampled from π (respecting the multi-select count)
        k = policy.select_count(o.select.minCount, o.select.maxCount, n)
        k = max(1, min(k, n))
        idx = torch.multinomial(torch.from_numpy(pi), k, replacement=False, generator=gen)
        obs = battle_select(idx.tolist()); n_iter += 1
    result = obs["current"]["result"]
    battle_finish()
    for s in samples:  # Monte-Carlo value target = game outcome
        s.value_target = _seat_reward(result, s.seat)
    return samples, result


class ExItTrainer:
    def collect(self, model, n_games, cfg, gen=None):
        model.eval()
        gen = gen or torch.Generator().manual_seed(cfg.train.seed)
        samples: list[ExItSample] = []
        results = []
        for _ in range(n_games):
            s, r = _play_game(model, cfg, gen)
            samples.extend(s); results.append(r)
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
                opt.zero_grad(); loss.backward()
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
