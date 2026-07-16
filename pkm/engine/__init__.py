"""The single seam between pkm and the cabt game engine.

Import the engine from here instead of reaching into
``kaggle_environments.envs.cabt.cg.*`` directly. The backend (official Kaggle
build vs. our locally compiled ``engine/`` build) is chosen by
:mod:`pkm.engine.loader` via the ``PKM_ENGINE`` / ``PKM_ENGINE_LIB`` env vars.

The full typed API lives in :mod:`pkm.engine.api`; :mod:`pkm.engine.loader`
owns library loading, the ctypes ABI, and capability detection.
"""

from __future__ import annotations

from .api import (
    all_attacks,
    all_cards,
    battle_finish,
    battle_select,
    battle_start,
    search_begin,
    search_end,
    search_release,
    search_step,
    to_observation,
    visualize_data,
)
from .loader import (
    ENGINE_BACKEND,
    ENGINE_LIB_PATH,
    Battle,
    EngineCapabilities,
    SerialData,
    StartData,
    available_backends,
    capabilities,
    kaggle_available,
    lib,
    vendored_built,
)

__all__ = [
    "ENGINE_BACKEND",
    "ENGINE_LIB_PATH",
    "Battle",
    "EngineCapabilities",
    "SerialData",
    "StartData",
    "all_attacks",
    "all_cards",
    "available_backends",
    "battle_finish",
    "battle_select",
    "battle_start",
    "capabilities",
    "kaggle_available",
    "lib",
    "search_begin",
    "search_end",
    "search_release",
    "search_step",
    "to_observation",
    "vendored_built",
    "visualize_data",
]
