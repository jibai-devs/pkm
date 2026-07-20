"""Routes each decision to one of several sub-agents. Only agent kaggle sees."""

import sys
from typing import Callable

from pkm.data.card_data import get_card_by_id
from pkm.heuristics.deck_tracker import DeckTracker

from .first_turn_agent import make_first_turn_agent
from .dragapult_default_agent import make_dragapult_default_agent
from .random_agent import make_random_agent

AgentFn = Callable[[dict], list[int]]
SelectAgentFn = Callable[[dict, dict[str, AgentFn], dict], str]


def _in_textual_app() -> bool:
    """True if we're running inside a live Textual app (human TUI play)."""
    try:
        from textual.app import active_app
    except ImportError:
        return False  # textual isn't installed (e.g. the Kaggle sandbox)
    try:
        active_app.get()
    except LookupError:
        return False  # textual is installed but no app is running
    return True


def _card_name(card_id: int) -> str:
    card = get_card_by_id(card_id)
    return card.name if card else f"Card#{card_id}"


def _log(msg: str, log_sink: Callable[[str], None] | None) -> None:
    if log_sink is not None:
        # Caller (e.g. the TUI session) owns display; hand it the message
        # instead of guessing where console output would actually be seen.
        log_sink(msg)
        return
    if _in_textual_app():
        # Textual owns the whole terminal; a raw print (even to the real
        # stdout) gets overwritten or corrupts the display. textual.log is
        # only visible via a `textual console` running alongside `textual
        # run --dev`, but it's the one channel that's actually safe here.
        from textual import log as tlog

        tlog(msg)
        return
    # No live Textual app (bot-vs-bot / a plain script / the real Kaggle
    # submission). kaggle's env.run() wraps every agent call in
    # redirect_stdout, so a plain print() alone would vanish from a live
    # local terminal — write to the real stdout too so it's visible there.
    # But the plain print() still matters: it's what lands in *kaggle's own*
    # captured per-step stdout, which is what actually reflects in Kaggle's
    # submission logs (the same mechanism that surfaces a crashing agent's
    # traceback there). Emit both.
    print(msg, file=sys.__stdout__, flush=True)
    print(msg, flush=True)


def _log_prizes(tracker: DeckTracker, log_sink: Callable[[str], None] | None) -> None:
    if tracker.prizes_known:
        names = sorted(_card_name(cid) for cid in tracker.known_prizes())
        msg = f"prizes: {names}"
    else:
        msg = "prizes unknown"
    _log(msg, log_sink)


def _went_first_or_second(obs: dict) -> str:
    state = obs["current"]
    you = state["yourIndex"]
    first_player = state.get("firstPlayer", -1)
    if first_player == you:
        return "first"
    if first_player == 1 - you:
        return "second"
    return "unknown"  # not yet resolved (shouldn't happen once a real decision exists)


def _select_agent(obs: dict, agents: dict[str, AgentFn], state: dict) -> str:
    """Template: pick a registered agent name for the upcoming turn.

    `state["tracker"]` (a `DeckTracker`) is available here for routing logic
    that depends on card locations (deck/hand/discard/prize/board/attached).
    """
    cur = obs["current"]
    turn = cur["turn"]
    you = cur["yourIndex"]
    first_player = cur.get("firstPlayer", -1)
    # our own first turn: setup (turn 0), turn 1 going first (or before the
    # first-player coin resolves), turn 2 going second — the engine's turn
    # counter is shared across both players
    if turn == 0:
        return "first_turn"
    if turn == 1 and first_player != 1 - you:
        return "first_turn"
    if turn == 2 and first_player == 1 - you:
        return "first_turn"
    return "dragapult_default"


def make_singaporean_middleman(
    deck: list[int],
    weights_path: str | None = None,
    agents: dict[str, AgentFn] | None = None,
    select_agent: SelectAgentFn = _select_agent,
    log_sink: Callable[[str], None] | None = None,
) -> AgentFn:
    """Build the kaggle-facing agent that dispatches per turn."""
    registry: dict[str, AgentFn] = (
        agents
        if agents is not None
        else {
            "dragapult_default": make_dragapult_default_agent(deck, weights_path),
            "random": make_random_agent(deck),
            # pass our sink through so a first-turn search failure (and the
            # random fallback it triggers) shows up in the same log stream
            "first_turn": make_first_turn_agent(deck, log_sink=log_sink),
        }
    )

    state: dict = {
        "turn": None,
        "active": next(iter(registry)),
        "tracker": DeckTracker(deck),
        "announced_side": False,
        "planner": None,
    }

    def _build_planner():
        """The turn planner, or None unless PKM_TURN_PLAN_DIR is set.

        Booted here -- at agent construction, i.e. when kaggle imports the
        agent -- rather than on the first plan. The worker needs a ~3.5s
        engine load of its own, and starting it now lets that happen in the
        background while the opening decisions are played, so no decision
        ever waits on it. Diagnostic only, so failure just disables planning.
        """
        try:
            from .turn_planner import TurnPlanner, plan_dir

            if plan_dir() is None:
                return None
            planner = TurnPlanner(deck, weights_path=weights_path, log_sink=log_sink)
            planner.start()
            return planner
        except Exception as exc:
            _log(f"turn_planner: disabled ({type(exc).__name__}: {exc})", log_sink)
            return None

    state["planner"] = _build_planner()

    def agent(obs: dict) -> list[int]:
        tracker = state["tracker"]
        tracker.observe(obs)

        # A search card (e.g. an Item that searches the deck) was just
        # played: this obs exposes the whole deck, so hook it and deduce
        # which cards must be sitting in the prize pile.
        if tracker.is_search_reveal(obs):
            tracker.record_search_reveal(obs)

        if obs["select"] is None:
            state["announced_side"] = False  # new game starting
            return deck

        if not state["announced_side"]:
            # firstPlayer reads -1 ("unresolved") on the very first decision
            # of the game — that decision (SelectContext.IS_FIRST) is what
            # *determines* it, so it can't be reported yet. Keep checking
            # each subsequent decision until it's actually resolved.
            side = _went_first_or_second(obs)
            if side != "unknown":
                state["announced_side"] = True
                _log(f"went {side}", log_sink)

        _log_prizes(tracker, log_sink)

        turn = obs["current"]["turn"]
        first_decision_of_turn = turn != state["turn"]
        if first_decision_of_turn:
            state["turn"] = turn
            state["active"] = select_agent(obs, registry, state)

        planner = state["planner"]
        if planner is not None and first_decision_of_turn:
            # plan the whole turn up front, in the planner's own engine process
            try:
                planner.start_turn(obs)
            except Exception as exc:
                _log(f"turn_planner: start_turn failed ({exc})", log_sink)

        _log(f"decision made by: {state['active']}", log_sink)

        # obs is handed to the chosen sub-agent unmodified either way.
        picks = registry[state["active"]](obs)

        if planner is not None:
            # score what we actually did against what the plan expected
            try:
                planner.record_actual(obs, picks, state["active"])
            except Exception as exc:
                _log(f"turn_planner: record_actual failed ({exc})", log_sink)

        return picks

    return agent
