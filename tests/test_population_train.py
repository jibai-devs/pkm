"""Tests for pkm/rl/population_train.py (Milestone 9,
docs/opponent-archetype-classifier-plan.md SS3b+3c)."""

import random

import torch

from pkm.agents.profile import AgentProfile
from pkm.data import Deck
from pkm.rl.features import archetype_index
from pkm.rl.model import PolicyValueNet
from pkm.rl.population_train import PopulationMember, make_pop_specs, run_pop_iteration


def _member(name: str, deck_path: str) -> PopulationMember:
    model = PolicyValueNet()
    model.eval()
    deck = Deck.from_csv(deck_path).card_ids
    return PopulationMember(
        name=name,
        deck=deck,
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=3e-4),
        weights={},
        archetype_label=archetype_index(deck_path),
        profile=AgentProfile(name),
    )


def test_population_matchmaking_coverage():
    """Every roster member besides the anchor (index 0) gets exactly
    games_per_pairing games per iteration; the anchor plays every one of
    them that many times too. No bot-vs-bot or self-play games (v1 scope)."""
    specs = make_pop_specs(games_per_pairing=3, num_members=4)
    counts = {i: 0 for i in range(4)}
    for s in specs:
        assert s.member_a_idx != s.member_b_idx
        assert 0 in (s.member_a_idx, s.member_b_idx)  # anchor always involved
        counts[s.member_a_idx] += 1
        counts[s.member_b_idx] += 1
    for bot_idx in (1, 2, 3):
        assert counts[bot_idx] == 3
    assert counts[0] == 9  # anchor: 3 bots * 3 games each


def test_population_trajectory_routing():
    """A mixed anchor-vs-bot game's trajectories land in the correct
    member's bucket, never cross-contaminate."""
    torch.manual_seed(0)
    random.seed(0)
    anchor = _member("00_basic", "deck/00_basic.csv")
    bot = _member("pool_test_bot", "deck/01_psychic.csv")
    assert (
        anchor.archetype_label != bot.archetype_label
    )  # precondition for a real check
    roster = [anchor, bot]

    specs = make_pop_specs(games_per_pairing=2, num_members=2)
    game_stats = run_pop_iteration(roster, specs, gamma=0.99, lam=0.95)

    assert len(anchor.buffer) > 0
    assert len(bot.buffer) > 0
    assert all(d.true_archetype == anchor.archetype_label for d in anchor.buffer)
    assert all(d.true_archetype == bot.archetype_label for d in bot.buffer)
    assert anchor.name in game_stats and bot.name in game_stats
    # anchor played every spec (2 games); each individual game's W/L/D sums to 1
    assert sum(game_stats[anchor.name]) == 2
    assert sum(game_stats[bot.name]) == 2


def _tiny_archetype_classifier(tmp_path):
    """Same helper as tests/test_rl.py -- a cheap real (not mocked)
    NumpyArchetypeClassifier, avoids depending on the gitignored
    pkm/archetype.npz export existing on disk."""
    from pkm.archetype.export import export_npz
    from pkm.archetype.numpy_model import NumpyArchetypeClassifier
    from pkm.archetype.train import train as train_archetype

    model, _ = train_archetype(
        n_per_class=5, epochs=1, batch_size=32, log_every=0, seed=0
    )
    path = tmp_path / "archetype_test.npz"
    export_npz(model, str(path))
    return NumpyArchetypeClassifier.load(str(path))


def test_run_pop_iteration_classifier_reaches_both_sides(tmp_path, monkeypatch):
    """Belief-classifier-routing plan Phase 2: unlike train.py's play_one
    (trainee only, never the frozen opponent), population training has no
    frozen side -- every roster member is simultaneously being trained, so
    archetype_classifier must reach BOTH TorchPolicy instances of every
    pairing. Sequential path (no executor)."""
    import pkm.rl.population_train as pop_mod

    classifier = _tiny_archetype_classifier(tmp_path)
    torch.manual_seed(0)
    random.seed(0)
    anchor = _member("00_basic", "deck/00_basic.csv")
    bot = _member("pool_test_bot", "deck/01_psychic.csv")
    roster = [anchor, bot]
    specs = make_pop_specs(games_per_pairing=2, num_members=2)

    seen = []
    real_init = pop_mod.TorchPolicy.__init__

    def spy_init(self, model, greedy=False, temperature=1.0, archetype_classifier=None):
        seen.append(archetype_classifier is classifier)
        real_init(self, model, greedy, temperature, archetype_classifier)

    monkeypatch.setattr(pop_mod.TorchPolicy, "__init__", spy_init)

    pop_mod.run_pop_iteration(
        roster, specs, gamma=0.99, lam=0.95, archetype_classifier=classifier
    )

    assert seen and all(seen)  # every TorchPolicy built (both sides) got it


def test_run_pop_iteration_no_classifier_by_default(tmp_path, monkeypatch):
    """The flip side: archetype_classifier=None (the default) must reach
    neither side, so existing/unflagged runs are unaffected."""
    import pkm.rl.population_train as pop_mod

    torch.manual_seed(0)
    random.seed(0)
    anchor = _member("00_basic", "deck/00_basic.csv")
    bot = _member("pool_test_bot", "deck/01_psychic.csv")
    roster = [anchor, bot]
    specs = make_pop_specs(games_per_pairing=2, num_members=2)

    seen = []
    real_init = pop_mod.TorchPolicy.__init__

    def spy_init(self, model, greedy=False, temperature=1.0, archetype_classifier=None):
        seen.append(archetype_classifier is None)
        real_init(self, model, greedy, temperature, archetype_classifier)

    monkeypatch.setattr(pop_mod.TorchPolicy, "__init__", spy_init)

    pop_mod.run_pop_iteration(roster, specs, gamma=0.99, lam=0.95)

    assert seen and all(seen)


def test_play_pop_chunk_classifier_reaches_both_sides(tmp_path, monkeypatch):
    """Parallel-path counterpart: _play_pop_chunk is what actually runs
    inside each worker process under --workers > 1. Tested by calling it
    directly (in-process) rather than through a real ProcessPoolExecutor --
    it's a plain function, and cross-process pickling of the classifier is
    already covered by train.py's --archetype-belief --workers 2 smoke
    test (see AGENTS.md)."""
    import pkm.rl.parallel_rollout as par_mod

    classifier = _tiny_archetype_classifier(tmp_path)
    torch.manual_seed(0)
    random.seed(0)
    deck_a = Deck.from_csv("deck/00_basic.csv").card_ids
    deck_b = Deck.from_csv("deck/01_psychic.csv").card_ids
    model_a = PolicyValueNet()
    model_b = PolicyValueNet()

    seen = []
    real_init = par_mod.TorchPolicy.__init__

    def spy_init(self, model, greedy=False, temperature=1.0, archetype_classifier=None):
        seen.append(archetype_classifier is classifier)
        real_init(self, model, greedy, temperature, archetype_classifier)

    monkeypatch.setattr(par_mod.TorchPolicy, "__init__", spy_init)

    games = [
        (0, deck_a, model_a.state_dict(), deck_b, model_b.state_dict(), (True, True))
    ]
    par_mod._play_pop_chunk(games, archetype_classifier=classifier)

    assert seen == [True, True]  # both TorchPolicy instances got it


def test_population_train_noop_on_solo_path():
    """Importing pkm.rl.population_train must not mutate or share state
    with train.py's existing single-deck/single-model flow."""
    import pkm.rl.population_train as pop_mod
    import pkm.rl.train as train_mod

    assert train_mod.CSV_FIELDS == [
        "iter",
        "games",
        "wins",
        "losses",
        "draws",
        "decisions",
        "samples",
        "pi_loss",
        "v_loss",
        "entropy",
        "clip_frac",
        "archetype_loss",
        "time_s",
        "eval_win_rate",
        "eval_games",
    ]
    assert train_mod.CSV_FIELDS is not pop_mod.CSV_FIELDS
