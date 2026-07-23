"""Tests for Part 2b: classifier-biased opponent-decklist estimation in
pkm.mcts.determinize / pkm.mcts.agent."""

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from pkm.archetype.archetypes import get_archetypes
from pkm.archetype.export import export_npz
from pkm.archetype.numpy_model import NumpyArchetypeClassifier
from pkm.archetype.train import train
from pkm.data import Deck
from pkm.engine import battle_finish, battle_select, battle_start
from pkm.mcts.agent import make_mcts_agent
from pkm.mcts.determinize import _archetype_weighted_padding, infer_opponent_decklist
from pkm.rl.export import export_npz as export_policy_npz
from pkm.rl.model import PolicyValueNet


def _mid_game_obs(steps: int = 25):
    random.seed(4)
    deck = Deck.from_csv("deck/00_basic.csv").card_ids
    obs, _ = battle_start(deck, deck)
    for _ in range(steps):
        if obs["current"]["result"] >= 0:
            break
        sel = obs["select"]
        obs = battle_select(random.sample(range(len(sel["option"])), sel["maxCount"]))
    return obs, deck


def test_archetype_weighted_padding_favors_believed_archetype():
    archetypes = get_archetypes()
    target_idx = next(
        i for i, a in enumerate(archetypes) if any(s.card_id is not None for s in a.staples)
    )
    target_card_ids = {
        s.card_id for s in archetypes[target_idx].staples if s.card_id is not None
    }

    belief = np.zeros(len(archetypes) + 1, dtype=np.float32)
    belief[target_idx] = 1.0  # fully confident in this one archetype

    rng = random.Random(0)
    padding = _archetype_weighted_padding(belief, energy_fallback=3, need=200, rng=rng)

    hit_rate = sum(1 for c in padding if c in target_card_ids) / len(padding)
    assert hit_rate > 0.5, "padding should be dominated by the believed archetype's own staples"


def test_archetype_weighted_padding_falls_back_on_all_unknown_belief():
    archetypes = get_archetypes()
    belief = np.zeros(len(archetypes) + 1, dtype=np.float32)
    belief[-1] = 1.0  # all mass on "Unknown" -- no archetype staples to weight toward

    rng = random.Random(0)
    padding = _archetype_weighted_padding(belief, energy_fallback=3, need=50, rng=rng)
    assert padding == [3] * 50


def test_infer_opponent_decklist_with_classifier_stays_legal(tmp_path):
    model, _ = train(n_per_class=5, epochs=1, batch_size=32, log_every=0, seed=0)
    npz_path = tmp_path / "archetype_test.npz"
    export_npz(model, str(npz_path))
    classifier = NumpyArchetypeClassifier.load(str(npz_path))

    obs, _ = _mid_game_obs()
    try:
        est = infer_opponent_decklist(obs, classifier=classifier, rng=random.Random(1))
        assert len(est) == 60
        assert all(isinstance(c, int) and c > 0 for c in est)
    finally:
        battle_finish()


def test_make_mcts_agent_end_to_end_with_archetype_classifier(tmp_path):
    """Smoke test: the classifier-biased path drives real decisions through
    the real engine without crashing (Part 2 verification item 4)."""
    torch_model, _ = train(n_per_class=5, epochs=1, batch_size=32, log_every=0, seed=0)
    archetype_npz = tmp_path / "archetype_test.npz"
    export_npz(torch_model, str(archetype_npz))

    policy_model = PolicyValueNet()
    policy_npz = tmp_path / "policy_test.npz"
    export_policy_npz(policy_model, str(policy_npz))

    deck = Deck.from_csv("deck/00_basic.csv").card_ids
    agent = make_mcts_agent(
        deck,
        weights_path=str(policy_npz),
        n_determinizations=1,
        n_simulations=4,
        seed=0,
        archetype_weights_path=str(archetype_npz),
    )

    obs, _ = battle_start(deck, deck)
    try:
        for _ in range(5):
            if obs["current"]["result"] >= 0:
                break
            picks = agent(obs)
            obs = battle_select(picks)
    finally:
        battle_finish()
