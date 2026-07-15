"""High-level battle wrappers over the loaded engine.

Copied from ``kaggle_environments.envs.cabt.cg.game`` and repointed at
:mod:`pkm.engine.loader`, so it works against whichever backend that module
selected. Behaviour is otherwise identical to the upstream wrapper.
"""

from __future__ import annotations

import ctypes
import json

from .loader import Battle, StartData, lib


def _get_battle_data() -> dict:
    sd = lib.GetBattleData(Battle.battle_ptr)
    Battle.obs = json.loads(sd.json.decode())
    Battle.obs["search_begin_input"] = ctypes.string_at(sd.data, sd.count).decode(
        "ascii"
    )
    return Battle.obs


def battle_start(deck0: list[int], deck1: list[int]) -> tuple[dict | None, StartData]:
    if len(deck0) != 60 or len(deck1) != 60:
        raise ValueError("The deck must contain 60 cards.")
    cards = deck0 + deck1
    arg = (ctypes.c_int * len(cards))(*cards)
    start_data = lib.BattleStart(arg)
    Battle.battle_ptr = start_data.battlePtr
    if Battle.battle_ptr is None or Battle.battle_ptr == 0:
        return (None, start_data)
    return (_get_battle_data(), start_data)


def battle_finish() -> None:
    lib.BattleFinish(Battle.battle_ptr)


def battle_select(select_list: list[int]) -> dict:
    if not isinstance(select_list, list) or not all(
        isinstance(i, int) for i in select_list
    ):
        raise ValueError("select_list is not list[int]")
    arg = (ctypes.c_int * len(select_list))(*select_list)
    err = lib.Select(Battle.battle_ptr, arg, len(select_list))
    if err != 0:
        if err == 30:
            raise ValueError("battle_ptr broken.")
        raise IndexError()
    return _get_battle_data()


def visualize_data() -> str:
    return lib.VisualizeData(Battle.battle_ptr).decode()
