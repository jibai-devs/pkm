"""Routes each decision to one of several sub-agents. Only agent kaggle sees."""

import sys
from typing import Callable

from pkm.data.card_data import get_card_by_id
from pkm.heuristics.deck_tracker import DeckTracker

from .neural_agent import make_neural_agent
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


def _log_prizes(tracker: DeckTracker, log_sink: Callable[[str], None] | None) -> None:
    if tracker.prizes_known:
        names = sorted(_card_name(cid) for cid in tracker.known_prizes())
        msg = f"prizes: {names}"
    else:
        msg = "prizes unknown"
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
    # No live Textual app (bot-vs-bot / a plain script): kaggle's env.run()
    # wraps every agent call in redirect_stdout, so a plain print() would
    # vanish silently; write to the real stdout instead.
    print(msg, file=sys.__stdout__, flush=True)


def _select_agent(obs: dict, agents: dict[str, AgentFn], state: dict) -> str:
    """Template: pick a registered agent name for the upcoming turn.

    `state["tracker"]` (a `DeckTracker`) is available here for routing logic
    that depends on card locations (deck/hand/discard/prize/board/attached).
    """
    return "neural"


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
            "neural": make_neural_agent(deck, weights_path),
            "random": make_random_agent(deck),
        }
    )

    state: dict = {
        "turn": None,
        "active": next(iter(registry)),
        "tracker": DeckTracker(deck),
    }

    def agent(obs: dict) -> list[int]:
        tracker = state["tracker"]
        tracker.observe(obs)

        # A search card (e.g. an Item that searches the deck) was just
        # played: this obs exposes the whole deck, so hook it and deduce
        # which cards must be sitting in the prize pile.
        if tracker.is_search_reveal(obs):
            tracker.record_search_reveal(obs)

        if obs["select"] is None:
            return deck

        _log_prizes(tracker, log_sink)

        turn = obs["current"]["turn"]
        if turn != state["turn"]:
            state["turn"] = turn
            state["active"] = select_agent(obs, registry, state)

        # obs is handed to the chosen sub-agent unmodified either way.
        return registry[state["active"]](obs)

    return agent
