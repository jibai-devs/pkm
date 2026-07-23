"""Tests for pkm/rl/attack_damage_estimator.py (Phases 1 and 2 of
docs/superpowers/plans/2026-07-20-attack-damage-estimator.md).

Every case below uses a real attackId/text pulled from the live card
database (not synthetic Attack objects) -- the whole point is verifying the
regex patterns parse actual card text correctly, not a hypothetical.
"""

from pkm.data.card_data import get_attack_data
from pkm.heuristics.context import GameContext
from pkm.heuristics.deck_tracker import DeckTracker
from pkm.rl.attack_damage_estimator import (
    MAX_REASONABLE_DAMAGE,
    estimate_attack_damage,
    min_guaranteed_damage,
)
from pkm.types.obs import Observation

ATTACKS = get_attack_data()

# Real attackIds, verified against pkm.engine.all_attacks() during the
# investigation (see the plan doc's "Sizing it" section).
COMET_PUNCH = 6  # "Flip 4 coins. This attack does 30 damage for each heads."
BALL_ROLL_10 = 55  # "Flip a coin until you get tails. ...10 damage for each heads."
BALL_ROLL_20 = 1213  # same move, 20-damage variant
CURSED_DROP = 116  # "Put 4 damage counters on your opponent's Pokemon in any way you like."
POWER_SPLASH = 201  # "...40 damage for each Energy attached to this Pokemon."
CRESCENDO_WAVE = 586  # "...30 damage for each {W} Energy attached to this Pokemon..."
BACK_DRAFT = 355  # "...30 damage for each Basic Energy card in your opponent's discard pile."
RE_BREW = 15  # "Put 2 damage counters ... for each Basic {G} Energy card in your discard pile."
DO_THE_WAVE = 115  # "...20 damage for each of your Benched Pokemon."
IRRITATED_OUTBURST = 184  # "...60 damage for each Prize card your opponent has taken."
ROUND_20 = 707  # "...20 damage for each of your Pokemon in play that has the Round attack."
CRUEL_ARROW = 183  # "This attack does 100 damage to 1 of your opponent's Pokemon. (...)"
MIND_RULER = 123  # "...30 damage for each card in your opponent's hand."
GADGET_SHOW = 524  # "...30 damage for each Pokemon Tool attached to all of your Pokemon."
HAMMER_LANCHE = 1046  # Phase 2, deliberately not implemented here -- must stay 0.
JET_HEADBUTT = 153  # damage=70, text="" -- plain fallback case.

TYMPOLE = 500  # knows the Round attack (attackId 707/708/710/1214).


def _poke(card_id, hp=100, max_hp=100, energies=None, tools=None, serial=1, player_index=0):
    return {
        "id": card_id,
        "serial": serial,
        "playerIndex": player_index,
        "hp": hp,
        "maxHp": max_hp,
        "appearThisTurn": False,
        "energies": energies or [],
        "energyCards": [],
        "tools": tools or [],
        "preEvolution": [],
    }


def _card(card_id, serial, player_index):
    return {"id": card_id, "serial": serial, "playerIndex": player_index}


def _player(
    active,
    bench=None,
    discard=None,
    prize_taken=0,
    hand_count=0,
    player_index=0,
    deck_count=40,
):
    return {
        "active": [active] if active else [],
        "bench": bench or [],
        "benchMax": 5,
        "deckCount": deck_count,
        "discard": discard or [],
        "prize": [_card(0, i, player_index) for i in range(6 - prize_taken)],
        "handCount": hand_count,
        "hand": None,
        "poisoned": False,
        "burned": False,
        "asleep": False,
        "paralyzed": False,
        "confused": False,
    }


def _obs(
    attacker,
    opponent_active=None,
    my_bench=None,
    my_discard=None,
    opp_discard=None,
    my_prize_taken=0,
    opp_prize_taken=0,
    opp_hand_count=0,
    my_deck_count=40,
) -> Observation:
    return Observation.model_validate(
        {
            "select": None,
            "logs": [],
            "current": {
                "turn": 1,
                "turnActionCount": 1,
                "yourIndex": 0,
                "firstPlayer": -1,
                "supporterPlayed": False,
                "stadiumPlayed": False,
                "energyAttached": False,
                "retreated": False,
                "result": -1,
                "stadium": [],
                "looking": None,
                "players": [
                    _player(
                        attacker,
                        bench=my_bench,
                        discard=my_discard,
                        prize_taken=my_prize_taken,
                        player_index=0,
                        deck_count=my_deck_count,
                    ),
                    _player(
                        opponent_active,
                        discard=opp_discard,
                        prize_taken=opp_prize_taken,
                        hand_count=opp_hand_count,
                        player_index=1,
                    ),
                ],
            },
            "search_begin_input": None,
        }
    )


def _bare_obs() -> Observation:
    """No attacker/bench/discard/etc set -- for patterns whose bonus doesn't
    depend on board state (fixed constants, coin flips)."""
    return _obs(attacker=_poke(1))


def test_coin_flip_fixed_count():
    obs = _bare_obs()
    assert estimate_attack_damage(ATTACKS[COMET_PUNCH], obs) == 4 * 30 * 0.5  # 60.0


def test_coin_flip_until_tails():
    obs = _bare_obs()
    assert estimate_attack_damage(ATTACKS[BALL_ROLL_10], obs) == 10.0
    assert estimate_attack_damage(ATTACKS[BALL_ROLL_20], obs) == 20.0


def test_fixed_damage_counters():
    obs = _bare_obs()
    assert estimate_attack_damage(ATTACKS[CURSED_DROP], obs) == 4 * 10.0  # 40.0


def test_energy_attached_to_self_any_type():
    attacker = _poke(1, energies=[1, 3, 3])  # 3 energies, any type
    obs = _obs(attacker=attacker)
    assert estimate_attack_damage(ATTACKS[POWER_SPLASH], obs) == 40 * 3


def test_energy_attached_to_self_typed():
    attacker = _poke(1, energies=[3, 3, 1])  # 2 Water (type 3), 1 Grass
    obs = _obs(attacker=attacker)
    assert estimate_attack_damage(ATTACKS[CRESCENDO_WAVE], obs) == 30 * 2


def test_discard_pile_energy_opponent_untyped():
    # 2 real Basic Energy cards (any type) + 1 non-energy distractor.
    opp_discard = [_card(1, 1, 1), _card(2, 2, 1), _card(1071, 3, 1)]
    obs = _obs(attacker=_poke(1), opp_discard=opp_discard)
    assert estimate_attack_damage(ATTACKS[BACK_DRAFT], obs) == 30 * 2


def test_discard_pile_energy_own_typed_via_damage_counters():
    # Re-Brew: 2 counters (=20 dmg) per Basic {G} Energy in MY OWN discard.
    my_discard = [_card(1, 1, 0), _card(1, 2, 0), _card(3, 3, 0)]  # 2x G, 1x W
    obs = _obs(attacker=_poke(1), my_discard=my_discard)
    assert estimate_attack_damage(ATTACKS[RE_BREW], obs) == 20 * 2


def test_benched_pokemon_count():
    my_bench = [_poke(2, serial=10), _poke(2, serial=11)]
    obs = _obs(attacker=_poke(1), my_bench=my_bench)
    assert estimate_attack_damage(ATTACKS[DO_THE_WAVE], obs) == 20 * 2


def test_prize_count_opponent_taken():
    obs = _obs(attacker=_poke(1), opp_prize_taken=3)
    assert estimate_attack_damage(ATTACKS[IRRITATED_OUTBURST], obs) == 60 * 3


def test_teammate_has_named_attack():
    # Attacker itself + one benched copy of the same species both know Round.
    attacker = _poke(TYMPOLE)
    my_bench = [_poke(TYMPOLE, serial=2)]
    obs = _obs(attacker=attacker, my_bench=my_bench)
    assert estimate_attack_damage(ATTACKS[ROUND_20], obs) == 20 * 2


def test_fixed_damage_to_opponent_pokemon():
    # Pure constant -- doesn't depend on board state at all.
    obs = _bare_obs()
    assert estimate_attack_damage(ATTACKS[CRUEL_ARROW], obs) == 100.0


def test_opponent_hand_size():
    obs = _obs(attacker=_poke(1), opp_hand_count=4)
    assert estimate_attack_damage(ATTACKS[MIND_RULER], obs) == 30 * 4


def test_pokemon_tool_count():
    attacker = _poke(1, tools=[_card(50, 1, 0)])
    my_bench = [_poke(2, serial=2, tools=[_card(51, 2, 0), _card(52, 3, 0)])]
    obs = _obs(attacker=attacker, my_bench=my_bench)
    assert estimate_attack_damage(ATTACKS[GADGET_SHOW], obs) == 30 * 3


def test_clamped_to_max_reasonable_damage():
    # 25 (MAX_HAND) * 30 = 750 > MAX_REASONABLE_DAMAGE.
    obs = _obs(attacker=_poke(1), opp_hand_count=25)
    assert estimate_attack_damage(ATTACKS[MIND_RULER], obs) == MAX_REASONABLE_DAMAGE


def test_hammer_lanche_without_ctx_stays_zero():
    # No GameContext -> no deck-composition info -> can't estimate, falls
    # back to the pre-Phase-2 baseline (never worse than before).
    obs = _bare_obs()
    assert estimate_attack_damage(ATTACKS[HAMMER_LANCHE], obs) == 0.0
    assert estimate_attack_damage(ATTACKS[HAMMER_LANCHE], obs, ctx=None) == 0.0


def test_fallback_to_static_damage_when_no_pattern_matches():
    obs = _bare_obs()
    assert estimate_attack_damage(ATTACKS[JET_HEADBUTT], obs) == 70.0


# --- min_guaranteed_damage: excludes coin-flip expected values ---------------


def test_min_guaranteed_damage_excludes_coin_flips():
    # Comet Punch's expected value (60) is not a certainty -- a coin-flip
    # attack must never contribute to the "guaranteed" figure.
    obs = _bare_obs()
    assert min_guaranteed_damage(ATTACKS[COMET_PUNCH], obs) == 0.0
    assert min_guaranteed_damage(ATTACKS[BALL_ROLL_10], obs) == 0.0


def test_min_guaranteed_damage_includes_deterministic_patterns():
    # Fixed constants, board-state counts etc. ARE certain -- must still
    # count towards the guaranteed figure.
    obs = _bare_obs()
    assert min_guaranteed_damage(ATTACKS[CRUEL_ARROW], obs) == 100.0
    assert min_guaranteed_damage(ATTACKS[CURSED_DROP], obs) == 40.0

    my_bench = [_poke(2, serial=10), _poke(2, serial=11)]
    obs2 = _obs(attacker=_poke(1), my_bench=my_bench)
    assert min_guaranteed_damage(ATTACKS[DO_THE_WAVE], obs2) == 20 * 2


# --- Phase 2: Hammer-lanche's deck-mill family --------------------------------
#
# "Discard the top 6 cards of your deck, and this attack does 100 damage
# for each Basic {W} Energy card that you discarded in this way."
#
# Real card IDs: 1=Basic{G}Energy, 3=Basic{W}Energy (see the engine's
# all_cards() -- verified during the Phase 1 investigation, only 8 energy
# cards exist in this database, all "Basic {X} Energy", no specials).


def _ctx_with_deck(deck_list: list[int]) -> GameContext:
    """A GameContext whose tracker has observed nothing yet -- every card in
    `deck_list` reads as still-in-deck, exactly representing "cards not yet
    seen anywhere" (hand/board/discard/prize)."""
    return GameContext(my_deck=deck_list, tracker=DeckTracker(deck_list))


def test_deck_mill_expected_value():
    # 10-card tracked deck, 3 of them Basic {W} Energy: mill 6 -> expected
    # matches = 6 * 3/10 = 1.8 -> 1.8 * 100 = 180.
    deck_list = [3, 3, 3] + [1] * 7
    ctx = _ctx_with_deck(deck_list)
    obs = _obs(attacker=_poke(1), my_deck_count=10)
    assert estimate_attack_damage(ATTACKS[HAMMER_LANCHE], obs, ctx) == 180.0


def test_deck_mill_capped_by_real_remaining_deck_size():
    # Same 10-card tracked composition, but the real deck (deckCount) only
    # has 4 cards left -- can't mill more than that, regardless of what the
    # tracker's undifferentiated deck-or-prize bucket still contains.
    deck_list = [3, 3, 3] + [1] * 7
    ctx = _ctx_with_deck(deck_list)
    obs = _obs(attacker=_poke(1), my_deck_count=4)
    # n = min(6, 4) = 4; expected matches = 4 * 3/10 = 1.2 -> 120.0
    assert estimate_attack_damage(ATTACKS[HAMMER_LANCHE], obs, ctx) == 120.0


def test_deck_mill_zero_when_no_energy_in_tracked_deck():
    deck_list = [1] * 10  # no {W} energy at all
    ctx = _ctx_with_deck(deck_list)
    obs = _obs(attacker=_poke(1), my_deck_count=10)
    assert estimate_attack_damage(ATTACKS[HAMMER_LANCHE], obs, ctx) == 0.0


def test_deck_mill_zero_when_tracker_deck_empty():
    # Every card already observed elsewhere -- no ZeroDivisionError, no
    # damage claimed.
    ctx = _ctx_with_deck([])
    obs = _obs(attacker=_poke(1), my_deck_count=0)
    assert estimate_attack_damage(ATTACKS[HAMMER_LANCHE], obs, ctx) == 0.0


def test_deck_mill_excluded_from_min_guaranteed_damage():
    # Even with a fully informative ctx, the deck-mill estimate is an
    # expected value, not a certainty -- must never leak into
    # min_guaranteed_damage (lethal_this_turn's certainty claim).
    deck_list = [3, 3, 3] + [1] * 7
    ctx = _ctx_with_deck(deck_list)
    obs = _obs(attacker=_poke(1), my_deck_count=10)
    assert min_guaranteed_damage(ATTACKS[HAMMER_LANCHE], obs, ctx) == 0.0
