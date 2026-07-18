import copy
import json
from pathlib import Path

from pkm.heuristics.deck_tracker import CardLocation, DeckTracker

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "observations.json").read_text()
)
OBS = FIXTURE["observations"]

# "9:43" (yourIndex=0): hand of 5, active with 2 attached energies, one
# bench pokemon, two discards. Exercises every observe()-bound location
# except STADIUM/ATTACHED_TOOL/PRIZE, which this captured game never hits.
NINE_43_HAND = [1079, 112, 1182, 1227, 132]
NINE_43_ACTIVE = 119
NINE_43_ENERGY = [5, 5]  # two copies, serials 59 then 58 in that order
NINE_43_BENCH = 1071
NINE_43_DISCARD = [1152, 1198]
NINE_43_PADDING = list(range(9101, 9150))  # 49 fillers -> 60-card deck
NINE_43_DECK = (
    NINE_43_HAND
    + [NINE_43_ACTIVE]
    + NINE_43_ENERGY
    + [NINE_43_BENCH]
    + NINE_43_DISCARD
    + NINE_43_PADDING
)

# "1:7" (yourIndex=0): a genuine full-deck search reveal (select.deck length
# 45 == deckCount 45). hand(7) + active(1) + revealed deck(45) + 7 padding
# slots that must end up deduced as prizes.
ONE_7_HAND = [1198, 1227, 5, 2, 1182, 1079, 1097]
ONE_7_ACTIVE = 119
ONE_7_REVEALED = [c["id"] for c in OBS["1:7"]["select"]["deck"]]
ONE_7_PADDING = list(range(9001, 9008))  # 7 fillers -> must become prizes
ONE_7_DECK = ONE_7_HAND + [ONE_7_ACTIVE] + ONE_7_REVEALED + ONE_7_PADDING


def test_observe_binds_every_zone_by_serial():
    assert len(NINE_43_DECK) == 60
    tracker = DeckTracker(NINE_43_DECK)
    tracker.observe(OBS["9:43"])

    by_id = {c.card_id: c for c in tracker.cards.values() if c.serial is not None}

    assert by_id[1079].location == CardLocation.HAND
    assert by_id[1079].serial == 49
    assert by_id[112].location == CardLocation.HAND
    assert by_id[112].serial == 20

    assert by_id[119].location == CardLocation.ACTIVE
    assert by_id[119].serial == 4

    assert by_id[1071].location == CardLocation.BENCH
    assert by_id[1071].serial == 18

    assert by_id[1152].location == CardLocation.DISCARD
    assert by_id[1152].serial == 40
    assert by_id[1198].location == CardLocation.DISCARD
    assert by_id[1198].serial == 35

    energy_slots = [c for c in tracker.cards.values() if c.card_id == 5]
    assert {c.location for c in energy_slots} == {CardLocation.ATTACHED_ENERGY}
    assert {c.serial for c in energy_slots} == {58, 59}

    # Untouched padding slots are still assumed to be in the deck.
    assert len(tracker.by_location(CardLocation.DECK)) == len(NINE_43_PADDING)


def test_is_search_reveal_true_for_genuine_full_deck_reveal():
    tracker = DeckTracker(ONE_7_DECK)
    assert tracker.is_search_reveal(OBS["1:7"]) is True


def test_is_search_reveal_false_when_select_has_no_deck():
    # "9:43" is a real (non-search) decision; select.deck is null.
    assert OBS["9:43"]["select"]["deck"] is None
    tracker = DeckTracker(NINE_43_DECK)
    assert tracker.is_search_reveal(OBS["9:43"]) is False


def test_is_search_reveal_false_for_filtered_search():
    # Same shape as a genuine reveal ("1:7"), but truncated so the shown
    # count no longer matches deckCount -- a "look at the top N" search.
    filtered = copy.deepcopy(OBS["1:7"])
    filtered["select"]["deck"] = filtered["select"]["deck"][:10]
    tracker = DeckTracker(ONE_7_DECK)
    assert tracker.is_search_reveal(filtered) is False


def test_record_search_reveal_deduces_prizes_by_elimination():
    assert len(ONE_7_DECK) == 60
    tracker = DeckTracker(ONE_7_DECK)
    tracker.observe(OBS["1:7"])
    assert tracker.is_search_reveal(OBS["1:7"]) is True

    tracker.record_search_reveal(OBS["1:7"])

    assert tracker.prizes_known is True
    assert set(tracker.known_prizes()) == set(ONE_7_PADDING)
    assert len(tracker.known_prizes()) == len(ONE_7_PADDING)

    # Everything that was actually revealed is bound to DECK, not PRIZE.
    revealed_ids = {c.card_id for c in tracker.by_location(CardLocation.DECK)}
    assert revealed_ids.isdisjoint(set(ONE_7_PADDING))


def test_two_trackers_do_not_leak_state():
    # Card (id=1198, serial=35) appears in both source snapshots: as a
    # DISCARD in "9:43" and as a HAND card in "1:7". If any tracker state
    # were shared across instances (e.g. a class-level default), binding it
    # in one tracker would corrupt or short-circuit the binding in the
    # other.
    tracker_a = DeckTracker(NINE_43_DECK)
    tracker_b = DeckTracker(ONE_7_DECK)

    tracker_a.observe(OBS["9:43"])
    tracker_b.observe(OBS["1:7"])

    a_card = next(c for c in tracker_a.cards.values() if c.card_id == 1198)
    b_card = next(
        c for c in tracker_b.cards.values() if c.card_id == 1198 and c.serial == 35
    )

    assert a_card.serial == 35
    assert a_card.location == CardLocation.DISCARD
    assert b_card.location == CardLocation.HAND

    assert len(tracker_a.by_location(CardLocation.DISCARD)) == 2
    assert len(tracker_b.by_location(CardLocation.DISCARD)) == 0
    assert tracker_a.prizes_known is False
    assert tracker_b.prizes_known is False
