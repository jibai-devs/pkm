"""Tests for the RL stack: encoding, act/evaluate consistency, numpy parity."""

import random

import numpy as np
import torch

from pkm.engine import (
    battle_finish,
    battle_select,
    battle_start,
)

from pkm.data import Deck
from pkm.types.obs import Observation
from pkm.rl.encoder import (
    MAX_HAND,
    N_BOARD_SLOTS,
    OPT_FEATS,
    STATE_FEATS,
    encode_decision,
)
from pkm.rl.model import PolicyValueNet
from pkm.rl.numpy_policy import NumpyPolicy
from pkm.rl.opponent_pool import load_pool_bots
from pkm.rl.ppo import compute_returns
from pkm.rl.rollout import GameSpec, TorchPolicy, make_game_specs, play_game, play_one


def _collect_decisions(n_decisions: int = 40) -> list:
    """Advance a random game and encode the first non-trivial decisions."""
    random.seed(0)
    deck = Deck.from_csv("deck/00_basic.csv").card_ids
    obs, _ = battle_start(deck, deck)
    out = []
    try:
        while len(out) < n_decisions and obs["current"]["result"] < 0:
            sel = obs["select"]
            n = len(sel["option"])
            if not (n == 1 and sel["minCount"] >= 1):
                out.append(encode_decision(Observation.model_validate(obs)))
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
        assert d.opt_feats.shape == (n, OPT_FEATS)
        assert 0 <= d.min_count <= d.max_count <= n


def test_act_evaluate_consistency():
    """evaluate() must reproduce the log-probs computed during act()."""
    torch.manual_seed(0)
    random.seed(0)
    model = PolicyValueNet()
    model.eval()
    deck = Deck.from_csv("deck/00_basic.csv").card_ids
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
    deck = Deck.from_csv("deck/00_basic.csv").card_ids
    result = play_game((TorchPolicy(model), TorchPolicy(model)), (deck, deck))
    traj = result.trajectories[0]
    compute_returns(traj, result.rewards[0])
    assert all(np.isfinite(d.advantage) and np.isfinite(d.ret) for d in traj)


def test_make_game_specs_no_archetype_pool_unchanged():
    """archetype_pool_prob=0 (default) must never touch opponent_deck --
    mechanical backward-compat guard for pre-3c callers."""
    rng = random.Random(0)
    pool = [{}] * 3
    specs = make_game_specs(50, pool, pool_prob=0.5, rng=rng)
    assert all(s.opponent_deck is None for s in specs)


def test_make_game_specs_cross_archetype_pool():
    rng = random.Random(0)
    deck_a = Deck.from_csv("deck/00_basic.csv").card_ids
    deck_b = Deck.from_csv("deck/01_psychic.csv").card_ids
    archetype_pool = [(deck_a, {"a": 1}), (deck_b, {"b": 2})]
    specs = make_game_specs(
        200,
        pool=[{}],
        pool_prob=0.0,
        rng=rng,
        archetype_pool=archetype_pool,
        archetype_pool_prob=1.0,
    )
    assert len(specs) == 200
    for s in specs:
        assert s.opponent_deck in (deck_a, deck_b)
        assert s.opponent_state in ({"a": 1}, {"b": 2})
        assert s.collect == (s.side == 0, s.side == 1)


def test_play_one_cross_archetype_deck():
    """play_one must send each side its own deck, not silently mirror the
    trainee's deck onto a cross-archetype opponent (Part 3c)."""
    torch.manual_seed(0)
    random.seed(0)
    current_model = PolicyValueNet()
    current_model.eval()
    opponent_model = PolicyValueNet()
    deck = Deck.from_csv("deck/00_basic.csv").card_ids
    opponent_deck = Deck.from_csv("deck/01_psychic.csv").card_ids
    spec = GameSpec(
        opponent_state=opponent_model.state_dict(),
        side=0,
        collect=(True, True),
        opponent_deck=opponent_deck,
    )
    result = play_one(current_model, opponent_model, deck, spec)
    assert result.decisions > 0
    assert result.rewards in {(1.0, -1.0), (-1.0, 1.0), (0.0, 0.0)}


def test_load_pool_bots_skips_untrained_profiles(tmp_path):
    """A pool-bot profile dir with no ppo_latest.pt yet must be skipped, not
    raise -- lets population sourcing proceed incrementally."""
    (tmp_path / "pool_untrained" / "checkpoints").mkdir(parents=True)
    bots = load_pool_bots(agents_dir=str(tmp_path))
    assert bots == []


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
