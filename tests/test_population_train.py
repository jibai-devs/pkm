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
