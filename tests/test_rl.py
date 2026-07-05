"""Tests for the RL stack: encoding, act/evaluate consistency, numpy parity."""

import random

import numpy as np
import torch

from kaggle_environments.envs.cabt.cg.game import (
    battle_finish,
    battle_select,
    battle_start,
)

from pkm.data import Deck
from pkm.rl.encoder import (
    MAX_HAND,
    N_BOARD_SLOTS,
    STATE_FEATS,
    encode_decision,
)
from pkm.rl.model import PolicyValueNet
from pkm.rl.numpy_policy import NumpyPolicy
from pkm.rl.ppo import compute_returns
from pkm.rl.rollout import TorchPolicy, play_game


def _collect_decisions(n_decisions: int = 40) -> list:
    """Advance a random game and encode the first non-trivial decisions."""
    random.seed(0)
    deck = Deck.from_csv("deck.csv").card_ids
    obs, _ = battle_start(deck, deck)
    out = []
    try:
        while len(out) < n_decisions and obs["current"]["result"] < 0:
            sel = obs["select"]
            n = len(sel["option"])
            if not (n == 1 and sel["minCount"] >= 1):
                out.append(encode_decision(obs))
            obs = battle_select(random.sample(range(n), sel["maxCount"]))
    finally:
        battle_finish()
    return out


def test_encoder_shapes():
    for d in _collect_decisions(20):
        assert d.board_cards.shape == (N_BOARD_SLOTS,)
        assert d.hand_cards.shape == (MAX_HAND,)
        assert d.state_feats.shape == (STATE_FEATS,)
        n = len(d.opt_type)
        assert n >= 1
        assert d.opt_card.shape == (n,)
        assert d.opt_feats.shape == (n, 5)
        assert 0 <= d.min_count <= d.max_count <= n


def test_act_evaluate_consistency():
    """evaluate() must reproduce the log-probs computed during act()."""
    torch.manual_seed(0)
    random.seed(0)
    model = PolicyValueNet()
    model.eval()
    deck = Deck.from_csv("deck.csv").card_ids
    result = play_game((TorchPolicy(model), TorchPolicy(model)), (deck, deck))
    decisions = (result.trajectories[0] + result.trajectories[1])[:80]
    assert len(decisions) > 10
    logprobs, entropies, values = model.evaluate(decisions)
    old = torch.tensor([d.logprob for d in decisions])
    assert float((logprobs.detach() - old).abs().max()) < 1e-4
    assert bool((entropies >= 0).all())
    vals = torch.tensor([d.value for d in decisions])
    assert float((values.detach() - vals).abs().max()) < 1e-4


def test_compute_returns_terminal():
    torch.manual_seed(0)
    random.seed(0)
    model = PolicyValueNet()
    model.eval()
    deck = Deck.from_csv("deck.csv").card_ids
    result = play_game((TorchPolicy(model), TorchPolicy(model)), (deck, deck))
    traj = result.trajectories[0]
    compute_returns(traj, result.rewards[0])
    assert all(np.isfinite(d.advantage) and np.isfinite(d.ret) for d in traj)


def test_numpy_policy_matches_torch():
    torch.manual_seed(1)
    model = PolicyValueNet()
    model.eval()
    weights = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    np_pol = NumpyPolicy(weights)
    for d in _collect_decisions(20):
        t = model.act(d, greedy=True)
        assert abs(t.value - np_pol.value(d)) < 1e-4
        p = np_pol.priors(d)
        assert abs(p.sum() - 1.0) < 1e-5
        picks = np_pol.act_greedy(d)
        assert d.min_count <= len(picks) <= d.max_count
