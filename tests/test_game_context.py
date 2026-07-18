from pkm.heuristics.context import GameContext
from pkm.heuristics.deck_tracker import CardLocation, DeckTracker

DECK = [1, 2, 3] * 20  # 60 cards


def test_constructs_cleanly():
    ctx = GameContext(my_deck=DECK, tracker=DeckTracker(DECK))
    assert ctx.my_deck == DECK
    assert isinstance(ctx.tracker, DeckTracker)
    assert ctx.opp_decklist is None


def test_opp_decklist_defaults_to_none_and_is_settable():
    ctx = GameContext(my_deck=DECK, tracker=DeckTracker(DECK), opp_decklist=[4, 5, 6])
    assert ctx.opp_decklist == [4, 5, 6]


def test_two_contexts_from_same_deck_have_independent_trackers():
    ctx_a = GameContext(my_deck=DECK, tracker=DeckTracker(DECK))
    ctx_b = GameContext(my_deck=DECK, tracker=DeckTracker(DECK))

    assert ctx_a.tracker is not ctx_b.tracker

    # Mutating one tracker must not be visible through the other -- guards
    # against a shared mutable default anywhere in the construction path.
    slot = next(iter(ctx_a.tracker.cards))
    ctx_a.tracker.cards[slot].location = CardLocation.HAND

    assert ctx_b.tracker.cards[slot].location == CardLocation.DECK
