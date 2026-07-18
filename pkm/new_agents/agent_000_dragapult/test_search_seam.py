import pytest

from pkm.new_agents.agent_000_dragapult import cabt, deck


pytestmark = pytest.mark.slow


def _fresh_root():
    obs, _ = cabt.battle_start(deck.DECK_60, deck.DECK_60)
    # advance through deck-selection until a real choice is presented
    n = 0
    while obs["select"] is None or obs["current"] is None:
        obs = cabt.battle_select(list(deck.DECK_60))
        n += 1
        if n > 50:
            break
    return obs


def test_search_begin_returns_state_with_options():
    obs = _fresh_root()
    if obs["current"]["result"] >= 0:
        cabt.battle_finish()
        pytest.skip("game ended during setup")
    try:
        # `_fresh_root` returns the very first select (IS_FIRST coin-flip choice),
        # before either player has drawn a hand: deckCount == 60 for both seats
        # and select.deck is None, so search_begin's own-deck validation
        # (`len(your_deck) >= me["deckCount"]`) requires a matching-length guess.
        # At this exact moment nothing has been drawn yet, so our own full 60-card
        # deck is not a guess at all -- it is the true, fully-known remaining deck.
        # single-sample determinization for the *opponent's* deck is Task 6; here
        # we just pass the known deck for both sides since nothing is hidden yet.
        st = cabt.search_begin(
            obs,
            your_deck=list(deck.DECK_60), your_prize=[],
            opponent_deck=list(deck.DECK_60),
            opponent_prize=[], opponent_hand=[], opponent_active=[],
        )
        assert isinstance(st.searchId, int)
        assert st.observation.select is not None
    finally:
        # Unconditional cleanup: the process-global battle/search (`Battle.battle_ptr`
        # / `_agent_ptr`) must be torn down even if an assertion above fails, so a
        # failing test doesn't leak state into the next test.
        try:
            cabt.search_end()
        finally:
            cabt.battle_finish()


def test_search_step_branching_semantics():
    """Pin down whether two steps from the same node persist as distinct nodes."""
    obs = _fresh_root()
    if obs["current"]["result"] >= 0:
        cabt.battle_finish()
        pytest.skip("game ended during setup")
    branched = False
    try:
        st = cabt.search_begin(
            obs, your_deck=list(deck.DECK_60), your_prize=[],
            opponent_deck=list(deck.DECK_60),
            opponent_prize=[], opponent_hand=[], opponent_active=[],
        )
        root_id = st.searchId
        n_opts = len(st.observation.select.option)
        child_a = cabt.search_step(root_id, [0])
        # Step the ROOT again with a different option; record whether it works and
        # whether the returned searchId differs from child_a.
        try:
            child_b = cabt.search_step(root_id, [min(1, n_opts - 1)])
            branched = child_b.searchId != child_a.searchId
        except Exception:
            branched = False
    finally:
        # Unconditional cleanup: must run even if search_begin/search_step raise
        # or an assertion below fails, so the process-global battle/search state
        # doesn't leak into the next test.
        try:
            cabt.search_end()
        finally:
            cabt.battle_finish()
    # Locked characterization (observed 2026-07-18): search_step(root_id, ...)
    # called twice with different selects DOES branch -- each call returns a
    # distinct, persistent searchId rather than mutating one cursor in place.
    # Task 7's MCTS must therefore track nodes by searchId (a real tree), not
    # assume a single mutable cursor per parent.
    print(f"SEARCH_STEP_BRANCHES={branched}")
    assert branched is True
