"""Torch-free greedy inference: replays PolicyValueNet's forward pass in numpy.

Must stay in sync with pkm/rl/model.py. Used by the Kaggle submission agent so
the bundle doesn't need torch.
"""

import numpy as np

from pkm.types.obs import Observation

from .encoder import EncodedDecision, encode_decision

NEG_INF = -1e9


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _linear(w: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    return x @ w.T + b


class NumpyPolicy:
    def __init__(self, weights: dict[str, np.ndarray]):
        self.w = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}

    @classmethod
    def load(cls, path: str) -> "NumpyPolicy":
        with np.load(path) as z:
            return cls({k: z[k] for k in z.files})

    def _encode_state(self, d: EncodedDecision) -> np.ndarray:
        w = self.w
        card_emb = w["card_emb.weight"]
        board = card_emb[d.board_cards].reshape(-1)
        hand_e = card_emb[d.hand_cards]
        mask = (d.hand_cards > 0).astype(np.float32)[:, None]
        hand = (hand_e * mask).sum(0) / max(mask.sum(), 1.0)
        x = np.concatenate([board, hand, d.state_feats])
        h = _relu(_linear(w["state_fc1.weight"], w["state_fc1.bias"], x))
        return _relu(_linear(w["state_fc2.weight"], w["state_fc2.bias"], h))

    def _encode_options(self, d: EncodedDecision) -> np.ndarray:
        w = self.w
        x = np.concatenate(
            [
                w["card_emb.weight"][d.opt_card],
                w["card_emb.weight"][d.opt_card2],
                w["attack_emb.weight"][d.opt_attack],
                w["opt_type_emb.weight"][d.opt_type],
                d.opt_feats,
            ],
            axis=1,
        )
        return _relu(_linear(w["opt_fc.weight"], w["opt_fc.bias"], x))

    def _logits(
        self, h: np.ndarray, rows: np.ndarray, picked_sum: np.ndarray
    ) -> np.ndarray:
        w = self.w
        n = rows.shape[0]
        x = np.concatenate(
            [np.tile(h, (n, 1)), rows, np.tile(picked_sum, (n, 1))], axis=1
        )
        y = _relu(_linear(w["score_fc1.weight"], w["score_fc1.bias"], x))
        return _linear(w["score_fc2.weight"], w["score_fc2.bias"], y).reshape(-1)

    def value(self, d: EncodedDecision) -> float:
        w = self.w
        h = self._encode_state(d)
        y = _relu(_linear(w["value_fc1.weight"], w["value_fc1.bias"], h))
        return float(np.tanh(_linear(w["value_fc2.weight"], w["value_fc2.bias"], y))[0])

    def priors(self, d: EncodedDecision) -> np.ndarray:
        """First-pick probabilities over the option list (no STOP)."""
        h = self._encode_state(d)
        opts = self._encode_options(d)
        rows = np.concatenate([opts, self.w["stop_vec"][None, :]], axis=0)
        picked = np.zeros_like(self.w["stop_vec"])
        logits = self._logits(h, rows, picked)
        logits[-1] = NEG_INF  # exclude STOP from priors
        logits -= logits.max()
        p = np.exp(logits)
        return (p / p.sum())[:-1]

    def act_greedy(self, d: EncodedDecision) -> list[int]:
        h = self._encode_state(d)
        opts = self._encode_options(d)
        n = opts.shape[0]
        rows = np.concatenate([opts, self.w["stop_vec"][None, :]], axis=0)
        picked_sum = np.zeros_like(self.w["stop_vec"])
        available = np.ones(n + 1, dtype=bool)

        picks: list[int] = []
        while len(picks) < d.max_count:
            available[n] = len(picks) >= d.min_count
            logits = self._logits(h, rows, picked_sum)
            logits[~available] = NEG_INF
            idx = int(np.argmax(logits))
            if idx == n:
                break
            picks.append(idx)
            picked_sum = picked_sum + opts[idx]
            available[idx] = False
        return picks

    def sample_picks(
        self, d: EncodedDecision, rng: np.random.Generator, temperature: float = 1.0
    ) -> tuple[list[int], float]:
        """Sample a full pick sequence; returns (picks, joint probability)."""
        h = self._encode_state(d)
        opts = self._encode_options(d)
        n = opts.shape[0]
        rows = np.concatenate([opts, self.w["stop_vec"][None, :]], axis=0)
        picked_sum = np.zeros_like(self.w["stop_vec"])
        available = np.ones(n + 1, dtype=bool)

        picks: list[int] = []
        joint = 1.0
        while len(picks) < d.max_count:
            available[n] = len(picks) >= d.min_count
            logits = self._logits(h, rows, picked_sum) / max(temperature, 1e-6)
            logits[~available] = NEG_INF
            logits -= logits.max()
            p = np.exp(logits)
            p /= p.sum()
            idx = int(rng.choice(n + 1, p=p))
            joint *= float(p[idx])
            if idx == n:
                break
            picks.append(idx)
            picked_sum = picked_sum + opts[idx]
            available[idx] = False
        return picks, joint

    def select(self, obs: dict) -> list[int]:
        """Full agent decision for an observation with a select block."""
        parsed = Observation.model_validate(obs)
        sel = parsed.select
        assert sel is not None
        n = len(sel.option)
        if n == 1 and sel.minCount >= 1:
            return [0]
        if n == sel.minCount == sel.maxCount:
            return list(range(n))
        return self.act_greedy(encode_decision(parsed))
