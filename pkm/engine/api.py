"""The complete cabt engine API — one typed module the whole project calls.

This collates every engine entry point that used to be scattered across the
kaggle package (`cg/sim.py`, `cg/game.py`) and our own modules (`pkm/search.py`,
`pkm/data/card_data.py`). Import from :mod:`pkm.engine`, never from
`kaggle_environments.envs.cabt.cg.*`.

Return-type convention follows the codebase's design (documented in
`pkm/types/obs.py`): **dicts at the engine seam, pydantic inward of the ML
boundary.** So:

- ``battle_*`` return raw observation ``dict``s — 37 call sites read them as
  dicts and self-play rollouts must stay allocation-cheap. Call
  :func:`to_observation` (or ``Observation.model_validate``) at the point you
  need a typed model.
- ``search_*`` return :class:`~pkm.types.obs.SearchState`, a typed wrapper whose
  ``.observation`` validates lazily — the search API has a single consumer
  (`pkm/mcts`), so it can be typed without hot-loop cost.
- card / attack metadata come back as ``list[dict]`` (the engine primitive);
  `pkm/data/card_data.py` builds its dataclasses on top.
"""

from __future__ import annotations

import ctypes
import json

from pkm.types.obs import Observation, SearchState

from .loader import Battle, StartData, get_lib


def to_observation(obs: dict) -> Observation:
    """Validate a raw observation dict into a typed :class:`Observation`.

    The one canonical crossing of the "dict -> pydantic" boundary.
    """
    return Observation.model_validate(obs)


# --- battle ------------------------------------------------------------------


def _get_battle_data() -> dict:
    sd = get_lib().GetBattleData(Battle.battle_ptr)
    Battle.obs = json.loads(sd.json.decode())
    Battle.obs["search_begin_input"] = ctypes.string_at(sd.data, sd.count).decode(
        "ascii"
    )
    return Battle.obs


def battle_start(deck0: list[int], deck1: list[int]) -> tuple[dict | None, StartData]:
    """Start a battle from two 60-card decks. Returns (first obs, start data)."""
    if len(deck0) != 60 or len(deck1) != 60:
        raise ValueError("The deck must contain 60 cards.")
    cards = deck0 + deck1
    arg = (ctypes.c_int * len(cards))(*cards)
    start_data = get_lib().BattleStart(arg)
    Battle.battle_ptr = start_data.battlePtr
    if Battle.battle_ptr is None or Battle.battle_ptr == 0:
        return (None, start_data)
    return (_get_battle_data(), start_data)


def battle_select(select_list: list[int]) -> dict:
    """Apply option indices and return the next observation."""
    if not isinstance(select_list, list) or not all(
        isinstance(i, int) for i in select_list
    ):
        raise ValueError("select_list is not list[int]")
    arg = (ctypes.c_int * len(select_list))(*select_list)
    err = get_lib().Select(Battle.battle_ptr, arg, len(select_list))
    if err != 0:
        if err == 30:
            raise ValueError("battle_ptr broken.")
        raise IndexError()
    return _get_battle_data()


def battle_finish() -> None:
    """End the battle and free its memory."""
    get_lib().BattleFinish(Battle.battle_ptr)


def visualize_data() -> str:
    """Return the visualizer data blob for the current battle."""
    return get_lib().VisualizeData(Battle.battle_ptr).decode()


# --- card / attack metadata --------------------------------------------------


def all_cards() -> list[dict]:
    """All card metadata from the engine (raw)."""
    return json.loads(get_lib().AllCard())


def all_attacks() -> list[dict]:
    """All attack metadata from the engine (raw)."""
    return json.loads(get_lib().AllAttack())


# --- search (forward simulation) ---------------------------------------------

_agent_ptr: int | None = None


def _get_agent_ptr() -> int:
    global _agent_ptr
    if _agent_ptr is None:
        _agent_ptr = get_lib().AgentStart()
    return _agent_ptr


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
) -> SearchState:
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
        A :class:`SearchState` wrapping ``{"observation": ..., "searchId": ...}``.
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

    raw = get_lib().SearchBegin(
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
    return SearchState(result["state"])


def search_step(search_id: int, select: list[int]) -> SearchState:
    """Advance the search by applying option indices at the given node."""
    raw = get_lib().SearchStep(
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
    return SearchState(result["state"])


def search_end() -> None:
    """End the current search; its memory is reused by the next search."""
    get_lib().SearchEnd(_get_agent_ptr())


def search_release(search_id: int) -> None:
    """Free a specific search node."""
    get_lib().SearchRelease(_get_agent_ptr(), search_id)
