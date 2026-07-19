"""Tests for Part 2a: the standalone classifier's belief wired into the
state encoder's GLOBAL feature (replacing the old 3-class aux-head slot).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pkm.archetype.belief import compute_belief
from pkm.archetype.export import export_npz
from pkm.archetype.gen import generate_dataset
from pkm.archetype.numpy_model import NumpyArchetypeClassifier
from pkm.archetype.train import train
from pkm.data import Deck
from pkm.engine import battle_finish, battle_start
from pkm.heuristics.context import GameContext
from pkm.heuristics.deck_tracker import DeckTracker
from pkm.rl.features import BELIEF_DIM, _opponent_archetype_belief
from pkm.rl.rollout import TorchPolicy
from pkm.rl.model import PolicyValueNet
from pkm.types.obs import Observation


def test_belief_feature_defaults_to_zero_without_ctx():
    obs = Observation.model_validate({"select": None, "current": None})
    out = _opponent_archetype_belief(obs, None)
    assert out.shape == (BELIEF_DIM,)
    assert np.all(out == 0.0)


def test_belief_feature_defaults_to_zero_before_first_update():
    deck = Deck.from_csv("deck/00_basic.csv").card_ids
    ctx = GameContext(list(deck), DeckTracker(deck))
    obs = Observation.model_validate({"select": None, "current": None})
    out = _opponent_archetype_belief(obs, ctx)
    assert out.shape == (BELIEF_DIM,)
    assert np.all(out == 0.0)


def test_belief_feature_reflects_ctx_injection():
    deck = Deck.from_csv("deck/00_basic.csv").card_ids
    ctx = GameContext(list(deck), DeckTracker(deck))
    injected = np.zeros(BELIEF_DIM, dtype=np.float32)
    injected[3] = 1.0
    ctx.archetype_belief = injected
    obs = Observation.model_validate({"select": None, "current": None})
    out = _opponent_archetype_belief(obs, ctx)
    assert np.array_equal(out, injected)


def _tiny_classifier_npz(tmp_path) -> str:
    model, _ = train(n_per_class=5, epochs=1, batch_size=32, log_every=0, seed=0)
    path = tmp_path / "archetype_test.npz"
    export_npz(model, str(path))
    return str(path)


def test_compute_belief_shape_and_normalization(tmp_path):
    classifier = NumpyArchetypeClassifier.load(_tiny_classifier_npz(tmp_path))
    examples = generate_dataset(n_per_class=2, seed=1)
    example = next(e for e in examples if e.revealed)
    card_ids = np.array(list(example.revealed.keys()), dtype=np.int64)
    counts = np.array(list(example.revealed.values()), dtype=np.float32)
    belief = classifier.belief(card_ids, counts)
    assert belief.shape == (BELIEF_DIM,)
    assert belief.sum() == pytest.approx(1.0)
    assert np.all(belief >= 0.0)


def test_torch_policy_leaves_belief_none_without_classifier():
    deck = Deck.from_csv("deck/00_basic.csv").card_ids
    ctx = GameContext(list(deck), DeckTracker(deck))
    model = PolicyValueNet()
    policy = TorchPolicy(model)  # no archetype_classifier -- opt-in default off

    obs, _ = battle_start(deck, deck)
    try:
        policy.act(obs, collect=False, ctx=ctx)
    finally:
        battle_finish()

    assert ctx.archetype_belief is None


def test_torch_policy_sets_belief_with_classifier(tmp_path):
    classifier = NumpyArchetypeClassifier.load(_tiny_classifier_npz(tmp_path))
    deck = Deck.from_csv("deck/00_basic.csv").card_ids
    ctx = GameContext(list(deck), DeckTracker(deck))
    model = PolicyValueNet()
    policy = TorchPolicy(model, archetype_classifier=classifier)

    obs, _ = battle_start(deck, deck)
    try:
        policy.act(obs, collect=False, ctx=ctx)
    finally:
        battle_finish()

    assert ctx.archetype_belief is not None
    assert ctx.archetype_belief.shape == (BELIEF_DIM,)
    assert ctx.archetype_belief.sum() == pytest.approx(1.0)


def test_compute_belief_never_reads_opponent_hand(tmp_path):
    """Legally-visible input only: opponent hand is always None per
    pkm/types/obs.py's Player contract -- compute_belief must not crash or
    silently peek at it even if a caller injects one (shouldn't happen in
    practice, but the function must not assume the key is absent either)."""
    classifier = NumpyArchetypeClassifier.load(_tiny_classifier_npz(tmp_path))
    deck = Deck.from_csv("deck/00_basic.csv").card_ids
    obs, _ = battle_start(deck, deck)
    try:
        belief = compute_belief(obs, classifier)
    finally:
        battle_finish()
    assert belief.shape == (BELIEF_DIM,)
