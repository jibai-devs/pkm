"""The single seam between pkm and the cabt game engine.

Import the engine from here instead of reaching into
``kaggle_environments.envs.cabt.cg.*`` directly. The backend (official Kaggle
build vs. our locally compiled ``engine/`` build) is chosen by
:mod:`pkm.engine.loader` via the ``PKM_ENGINE`` / ``PKM_ENGINE_LIB`` env vars.
"""

from __future__ import annotations

from .game import (
    battle_finish,
    battle_select,
    battle_start,
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
    "available_backends",
    "battle_finish",
    "battle_select",
    "battle_start",
    "capabilities",
    "kaggle_available",
    "lib",
    "vendored_built",
    "visualize_data",
]
