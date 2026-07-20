"""Index-free descriptions of decisions, so a *planned* move can be compared
to an *actual* one.

Option indices are only meaningful relative to one specific ``select.option``
list. The instant the simulated world diverges from the real game those lists
differ, and index equality becomes noise -- so plan/actual comparison keys on
what a pick *means* (its option type plus the card/attack it refers to)
instead of where it happened to sit in the list.

Deliberately dependency-free (no engine, no torch): it is imported by both the
parent process and the planner subprocess.
"""

from __future__ import annotations

from typing import Any

# AreaType members that name a zone a card can be chosen from.
AREA_DECK = 1
AREA_HAND = 2
AREA_TRASH = 3
AREA_ACTIVE = 4
AREA_BENCH = 5
AREA_PRIZE = 6

# Which player-state key each area reads from. A PLAY option carries *no*
# `area` at all (the hand is implied), so None maps to the hand too. Deck and
# prize entries are usually hidden, but a search effect reveals them -- and
# those are exactly the picks worth naming ("what did it fetch"), so they're
# mapped; when the zone isn't exposed the lookup just yields None.
_ZONE_KEY = {
    None: "hand",
    AREA_DECK: "deck",
    AREA_HAND: "hand",
    AREA_TRASH: "discard",
    AREA_ACTIVE: "active",
    AREA_BENCH: "bench",
    AREA_PRIZE: "prize",
}


def _card_in_zone(obs: dict, seat: int | None, area: int | None, index: int | None):
    """The card id an (area, index) pair points at, or None if not visible.

    Hidden zones (deck, prizes, the opponent's hand) simply resolve to None --
    the descriptor then carries no card, which is honest rather than a guess.
    """
    key = _ZONE_KEY.get(area)
    if key is None or index is None or seat is None:
        return None
    players = (obs.get("current") or {}).get("players") or []
    if not 0 <= seat < len(players):
        return None
    zone = players[seat].get(key) or []
    if not 0 <= index < len(zone):
        return None
    entry = zone[index]
    return entry.get("id") if isinstance(entry, dict) else None


def option_descriptor(obs: dict, opt: dict) -> dict[str, Any]:
    """A comparable, position-independent description of one option."""
    desc: dict[str, Any] = {"type": opt.get("type")}
    card_id = opt.get("cardId")
    if not card_id:
        seat = opt.get("playerIndex")
        if seat is None:
            seat = (obs.get("current") or {}).get("yourIndex")
        card_id = _card_in_zone(obs, seat, opt.get("area"), opt.get("index"))
    if card_id:
        desc["card"] = card_id
    if opt.get("attackId") is not None:
        desc["attack"] = opt["attackId"]
    # where it lands (bench slot vs active) distinguishes otherwise-identical plays
    if opt.get("inPlayArea") is not None:
        desc["to"] = [opt.get("inPlayArea"), opt.get("inPlayIndex")]
    return desc


def describe_picks(obs: dict, picks: list[int]) -> list[dict[str, Any]]:
    """Describe every option a pick-list selected, in a stable order."""
    sel = obs.get("select") or {}
    options = sel.get("option") or []
    out = [option_descriptor(obs, options[i]) for i in picks if 0 <= i < len(options)]
    return sorted(out, key=lambda d: repr(sorted(d.items())))


def picks_match(planned: list[dict], actual: list[dict]) -> bool:
    """True if two described pick-lists mean the same thing."""
    return planned == actual


def decision_context(obs: dict) -> dict[str, Any]:
    """The bits of a decision point worth recording alongside a pick."""
    state = obs.get("current") or {}
    sel = obs.get("select") or {}
    return {
        "turn": state.get("turn"),
        "seat": state.get("yourIndex"),
        "select_type": sel.get("type"),
        "select_context": sel.get("context"),
        "n_options": len(sel.get("option") or []),
    }
