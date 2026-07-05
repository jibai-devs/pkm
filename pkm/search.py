"""Bindings to the cabt C library search API (SearchBegin/SearchStep/...).

Signatures mirror the official competition ``cg/api.py``: SearchBegin takes an
agent handle from ``AgentStart()`` plus the observation's ``search_begin_input``
ASCII blob, and returns an ApiResult JSON string. Search IDs are int64.

Everything here works on plain dicts (same shape as agent observations) to
avoid dataclass conversion overhead in the MCTS hot loop.
"""

import ctypes
import json

from kaggle_environments.envs.cabt.cg.sim import lib

lib.AgentStart.restype = ctypes.c_void_p

lib.SearchBegin.restype = ctypes.c_char_p
lib.SearchBegin.argtypes = [
    ctypes.c_void_p,  # agent_ptr
    ctypes.c_char_p,  # search_begin_input
    ctypes.c_int,  # len(search_begin_input)
    ctypes.POINTER(ctypes.c_int),  # your_deck
    ctypes.POINTER(ctypes.c_int),  # your_prize
    ctypes.POINTER(ctypes.c_int),  # opponent_deck
    ctypes.POINTER(ctypes.c_int),  # opponent_prize
    ctypes.POINTER(ctypes.c_int),  # opponent_hand
    ctypes.POINTER(ctypes.c_int),  # opponent_active
    ctypes.c_int,  # manual_coin
]

lib.SearchStep.restype = ctypes.c_char_p
lib.SearchStep.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int64,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]

lib.SearchEnd.argtypes = [ctypes.c_void_p]
lib.SearchRelease.argtypes = [ctypes.c_void_p, ctypes.c_int64]

lib.AllCard.restype = ctypes.c_char_p
lib.AllAttack.restype = ctypes.c_char_p

_agent_ptr: int | None = None


def _get_agent_ptr() -> int:
    global _agent_ptr
    if _agent_ptr is None:
        _agent_ptr = lib.AgentStart()
    return _agent_ptr


def all_card_data() -> list[dict]:
    """Get all card metadata from the cabt engine."""
    return json.loads(lib.AllCard())


def all_attack() -> list[dict]:
    """Get all attack metadata from the cabt engine."""
    return json.loads(lib.AllAttack())


_SEARCH_BEGIN_ERRORS = {
    1: "Invalid Card ID.",
    2: "Active card must be the ID of a Pokémon card.",
    30: "agent_ptr broken.",
}

_SEARCH_STEP_ERRORS = {
    1: "There is no element with the specified search_id.",
    2: "Released item.",
    3: "Cannot be selected because the battle has ended.",
    4: "Must be select.minCount <= len(select) <= select.maxCount.",
    5: "Must be 0 <= select elements < len(select.option).",
    6: "Duplicate select elements.",
    30: "agent_ptr broken.",
}


def search_begin(
    observation: dict,
    your_deck: list[int],
    your_prize: list[int],
    opponent_deck: list[int],
    opponent_prize: list[int],
    opponent_hand: list[int],
    opponent_active: list[int],
    manual_coin: bool = False,
) -> dict:
    """Start a forward-simulation search from a real agent observation.

    Args:
        observation: The observation dict passed to the agent (must contain
            ``search_begin_input``).
        your_deck: Predicted card IDs of your deck. Ignored (pass []) when
            ``observation["select"]["deck"]`` is not None.
        your_prize: Predicted card IDs of your prize cards.
        opponent_deck: Predicted card IDs of the opponent's deck.
        opponent_prize: Predicted card IDs of the opponent's prizes.
        opponent_hand: Predicted card IDs of the opponent's hand.
        opponent_active: Predicted opponent active Pokémon card ID; only
            needed when the opponent's active is face-down.
        manual_coin: If True, coin results become selectable options.

    Returns:
        SearchState dict: {"observation": {...}, "searchId": int}
    """
    sbi = observation.get("search_begin_input")
    if sbi is None:
        raise ValueError("observation has no search_begin_input")

    state = observation["current"]
    your_index = state["yourIndex"]
    me = state["players"][your_index]
    opp = state["players"][1 - your_index]

    if observation["select"].get("deck") is not None:
        your_deck = []
    elif len(your_deck) < me["deckCount"]:
        raise ValueError("your_deck does not match your deck count")
    if len(your_prize) < len(me["prize"]):
        raise ValueError("your_prize does not match your prize count")
    if len(opponent_deck) < opp["deckCount"]:
        raise ValueError("opponent_deck does not match opponent deck count")
    if len(opponent_prize) < len(opp["prize"]):
        raise ValueError("opponent_prize does not match opponent prize count")
    if len(opponent_hand) < opp["handCount"]:
        raise ValueError("opponent_hand does not match opponent hand count")

    active = opp["active"]
    if len(active) > 0 and active[0] is None:
        if not opponent_active:
            raise ValueError("must predict the opponent's face-down active Pokémon")
    else:
        opponent_active = []

    raw = lib.SearchBegin(
        _get_agent_ptr(),
        sbi.encode("ascii"),
        len(sbi),
        (ctypes.c_int * len(your_deck))(*your_deck),
        (ctypes.c_int * len(your_prize))(*your_prize),
        (ctypes.c_int * len(opponent_deck))(*opponent_deck),
        (ctypes.c_int * len(opponent_prize))(*opponent_prize),
        (ctypes.c_int * len(opponent_hand))(*opponent_hand),
        (ctypes.c_int * len(opponent_active))(*opponent_active),
        1 if manual_coin else 0,
    )
    result = json.loads(raw)
    if result["error"] != 0:
        raise ValueError(
            _SEARCH_BEGIN_ERRORS.get(
                result["error"], f"SearchBegin error {result['error']}"
            )
        )
    return result["state"]


def search_step(search_id: int, select: list[int]) -> dict:
    """Advance the search by applying option indices at the given node.

    Returns:
        SearchState dict: {"observation": {...}, "searchId": int}
    """
    raw = lib.SearchStep(
        _get_agent_ptr(),
        search_id,
        (ctypes.c_int * len(select))(*select),
        len(select),
    )
    result = json.loads(raw)
    if result["error"] != 0:
        raise ValueError(
            _SEARCH_STEP_ERRORS.get(
                result["error"], f"SearchStep error {result['error']}"
            )
        )
    return result["state"]


def search_end() -> None:
    """End the current search; its memory is reused by the next search."""
    lib.SearchEnd(_get_agent_ptr())


def search_release(search_id: int) -> None:
    """Free a specific search node."""
    lib.SearchRelease(_get_agent_ptr(), search_id)
