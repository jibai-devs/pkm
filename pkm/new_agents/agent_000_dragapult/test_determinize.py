"""Tests for single-sample (K=1) determinization.

Fake obs shapes below mirror the REAL raw observation dict (confirmed against
`tests/fixtures/observations.json` and `pkm.types.obs.Player`):
  - own `hand` is a list of card dicts (each with an "id"); opponent `hand` is
    the bare value `None` when hidden (NOT a list of `None` placeholders).
  - `active` / `bench` are lists of Pokemon dicts (with nested "id",
    "energyCards", "tools", "preEvolution") or `None` per slot.
  - `prize` is a list of 6 entries, each a card dict or `None` (prizes are
    face-down even to their own owner) -- or an empty list before prizes are
    dealt at all (the true coin-flip-root shape, see `_fresh_root()` in
    test_search_seam.py).
  - `deckCount` / `handCount` are plain ints.
"""

import collections

import torch

from pkm.new_agents.agent_000_dragapult import deck
from pkm.new_agents.agent_000_dragapult.cabt import all_card_data
from pkm.new_agents.agent_000_dragapult.determinize import DETERMINIZERS, sample_world


def _card(cid, serial=0, player_index=0):
    return {"id": cid, "serial": serial, "playerIndex": player_index}


def _pokemon(cid, serial=0, player_index=0, energy_cards=(), tools=(), pre_evo=()):
    return {
        "id": cid,
        "serial": serial,
        "playerIndex": player_index,
        "hp": 70,
        "maxHp": 70,
        "appearThisTurn": False,
        "energies": [],
        "energyCards": [_card(c, player_index=player_index) for c in energy_cards],
        "tools": [_card(c, player_index=player_index) for c in tools],
        "preEvolution": [_card(c, player_index=player_index) for c in pre_evo],
    }


def _fake_obs(seat=0):
    """A mid-game observation: hands drawn, boards populated, prizes dealt."""
    other = 1 - seat
    players = [None, None]
    players[seat] = {
        "deckCount": 47,
        "handCount": 5,
        "prize": [None] * 6,
        "active": [_pokemon(119, serial=1, player_index=seat)],
        "bench": [],
        "discard": [_card(2, serial=2, player_index=seat)],
        "hand": [
            _card(1079, serial=3, player_index=seat),
            _card(1086, serial=4, player_index=seat),
            _card(1121, serial=5, player_index=seat),
            _card(1152, serial=6, player_index=seat),
            _card(1182, serial=7, player_index=seat),
        ],
    }
    players[other] = {
        "deckCount": 46,
        "handCount": 4,
        "prize": [None] * 6,
        "active": [_pokemon(120, serial=8, player_index=other)],
        "bench": [],
        "discard": [_card(5, serial=9, player_index=other)],
        "hand": None,  # hidden from `seat`
    }
    return {"current": {"yourIndex": seat, "players": players}}


def _coin_flip_root_obs(seat=0):
    """Nothing drawn yet: matches the real `_fresh_root()` shape (prize=[])."""
    other = 1 - seat
    players = [None, None]
    for i in (seat, other):
        players[i] = {
            "deckCount": 60,
            "handCount": 0,
            "prize": [],
            "active": [],
            "bench": [],
            "discard": [],
            "hand": [] if i == seat else None,
        }
    return {"current": {"yourIndex": seat, "players": players}}


def test_sample_world_respects_opponent_counts():
    gen = torch.Generator().manual_seed(0)
    obs = _fake_obs(seat=0)
    w = sample_world(obs, seat=0, gen=gen)
    opp = obs["current"]["players"][1]
    assert len(w.opponent_hand) == opp["handCount"]
    assert len(w.opponent_deck) >= opp["deckCount"]
    assert len(w.opponent_prize) == len(opp["prize"])


def test_sampled_cards_come_from_known_deck_multiset():
    gen = torch.Generator().manual_seed(1)
    w = sample_world(_fake_obs(seat=0), seat=0, gen=gen)
    known = collections.Counter(deck.DECK_60)
    used = collections.Counter(w.opponent_hand + w.opponent_prize)
    # never assign more copies of a card than exist in one seat's DECK_60 copy
    for card_id, cnt in used.items():
        assert cnt <= known[card_id]


def test_your_deck_populated_and_within_multiset():
    """The Task-5 correction: your_deck must NOT be [] -- search_begin's own-deck
    validation (`len(your_deck) >= me["deckCount"]`) fires whenever
    `select.deck is None`, which is true at every in-game decision node.
    """
    gen = torch.Generator().manual_seed(2)
    obs = _fake_obs(seat=0)
    me = obs["current"]["players"][0]
    w = sample_world(obs, seat=0, gen=gen)
    assert len(w.your_deck) >= me["deckCount"]
    known = collections.Counter(deck.DECK_60)
    used = collections.Counter(w.your_deck + w.your_prize)
    for card_id, cnt in used.items():
        assert cnt <= known[card_id]


def test_coin_flip_root_your_deck_is_full_known_deck():
    """At the true root (deckCount==60, nothing drawn/dealt) your_deck is not a
    guess at all -- it is the fully-known remaining 60-card deck.
    """
    gen = torch.Generator().manual_seed(3)
    obs = _coin_flip_root_obs(seat=0)
    me = obs["current"]["players"][0]
    w = sample_world(obs, seat=0, gen=gen)
    assert len(w.your_deck) >= me["deckCount"] == 60
    assert collections.Counter(w.your_deck) == collections.Counter(deck.DECK_60)


def test_deterministic_given_generator_seed():
    obs = _fake_obs(seat=0)
    w1 = sample_world(obs, seat=0, gen=torch.Generator().manual_seed(42))
    w2 = sample_world(obs, seat=0, gen=torch.Generator().manual_seed(42))
    assert w1 == w2


def test_determinizers_registry():
    assert DETERMINIZERS["sample"] is sample_world


def test_face_down_opponent_active_picks_basic():
    """Task-6 regression: opponent's active placed face-down (real shape is
    `active == [None]`, momentarily during the setup coin-flip before both
    actives are revealed together) must not raise, and must predict a Basic
    Pokémon id for the hidden slot -- game rules require the placed Pokémon
    to be a Basic.
    """
    gen = torch.Generator().manual_seed(4)
    obs = _fake_obs(seat=0)
    obs["current"]["players"][1]["active"] = [None]

    w = sample_world(obs, seat=0, gen=gen)

    by_id = {cd.cardId: cd for cd in all_card_data()}
    assert len(w.opponent_active) == 1
    picked = w.opponent_active[0]
    assert by_id.get(picked) is not None
    assert by_id[picked].basic
