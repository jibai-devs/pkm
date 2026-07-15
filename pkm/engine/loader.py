"""Load the cabt game engine and expose its ctypes surface.

This is the single place that decides *which* build of the engine backs the
process. Both the official Kaggle `libcg.so` and our locally compiled
`engine/build/cg.so` export the identical C ABI, so switching backends is just
choosing a different shared-library path — everything above this module is
byte-for-byte the same code.

Backend selection (highest precedence first):
  1. ``PKM_ENGINE_LIB=/abs/path/to/cg.so``  — explicit override
  2. ``PKM_ENGINE=vendored``                — our compiled engine (engine/)
  3. default                                — the Kaggle-bundled engine

The default MUST stay "kaggle": that is the only engine available inside the
Kaggle submission sandbox, where ``engine/`` does not exist.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import platform
from dataclasses import dataclass, asdict
from pathlib import Path

# The 13 functions the cabt C ABI exports. Both builds must provide all of them.
CORE_SYMBOLS: tuple[str, ...] = (
    "GameInitialize",
    "BattleStart",
    "AgentStart",
    "BattleFinish",
    "GetBattleData",
    "Select",
    "VisualizeData",
    "AllCard",
    "AllAttack",
    "SearchBegin",
    "SearchStep",
    "SearchEnd",
    "SearchRelease",
)

# Optional symbols that a *patched* engine might export to allow deterministic,
# seed-injected battles (none exist in the stock engine — see docs/ENGINE.md).
# Detection is capability-based: if any of these appear, we can offer repro.
SEED_SYMBOLS: tuple[str, ...] = (
    "BattleStartSeed",
    "BattleStartWithSeed",
    "SetSeed",
    "GameSeed",
)


class StartData(ctypes.Structure):
    _fields_ = [
        ("battlePtr", ctypes.c_void_p),
        ("errorPlayer", ctypes.c_int),
        ("errorType", ctypes.c_int),
    ]


class SerialData(ctypes.Structure):
    _fields_ = [
        ("json", ctypes.c_char_p),
        ("data", ctypes.POINTER(ctypes.c_ubyte)),
        ("count", ctypes.c_int),
        ("selectPlayer", ctypes.c_int),
    ]


class Battle:
    """Process-global handle to the current battle (mirrors the Kaggle wrapper)."""

    battle_ptr = None
    obs = None
    raminingTime = [[], []]  # noqa: N815  (name kept identical to upstream)


def _kaggle_libname() -> str:
    system = platform.system()
    if system == "Windows":
        return "cg.dll"
    if system == "Darwin":
        return "libcg.dylib"
    if platform.machine() in ("arm64", "aarch64"):
        return "libcg-arm64.so"
    return "libcg.so"


def _kaggle_lib_path() -> Path:
    from kaggle_environments.envs.cabt.cg import sim

    return Path(sim.__file__).parent / _kaggle_libname()


def _vendored_lib_path() -> Path:
    # pkm/engine/loader.py -> repo root is parents[2]
    engine_dir = Path(__file__).resolve().parents[2] / "engine"
    # cmake (incremental) output first, then `nix build` result symlink.
    for candidate in (
        engine_dir / "build" / "cg.so",
        engine_dir / "result" / "lib" / "cg.so",
    ):
        if candidate.exists():
            return candidate
    return engine_dir / "build" / "cg.so"  # report the expected path in the error


def resolve_lib_path() -> tuple[str, Path]:
    """Return ``(backend_name, path)`` for the selected engine."""
    override = os.environ.get("PKM_ENGINE_LIB")
    if override:
        return "override", Path(override)
    backend = os.environ.get("PKM_ENGINE", "kaggle").lower()
    if backend == "vendored":
        return "vendored", _vendored_lib_path()
    if backend == "kaggle":
        return "kaggle", _kaggle_lib_path()
    raise ValueError(
        f"unknown PKM_ENGINE={backend!r} (expected 'kaggle' or 'vendored')"
    )


def _load() -> ctypes.CDLL:
    backend, path = resolve_lib_path()

    if backend == "kaggle":
        # Importing the upstream module already dlopen'd libcg.so, called
        # GameInitialize(), and set the base argtypes. dlopen caches by path, so
        # re-loading returns the same handle — calling GameInitialize() again
        # double-inits the engine's global state and aborts. Reuse it as-is.
        from kaggle_environments.envs.cabt.cg import sim

        return sim.lib

    if not path.exists():
        hint = " — build it with `just engine-build`" if backend == "vendored" else ""
        raise FileNotFoundError(
            f"engine backend '{backend}' library not found: {path}{hint}"
        )

    lib = ctypes.cdll.LoadLibrary(str(path))
    lib.GameInitialize()

    lib.BattleStart.restype = StartData
    lib.BattleStart.argtypes = [ctypes.POINTER(ctypes.c_int)]

    lib.BattleFinish.argtypes = [ctypes.c_void_p]

    lib.GetBattleData.restype = SerialData
    lib.GetBattleData.argtypes = [ctypes.c_void_p]

    lib.Select.restype = ctypes.c_int
    lib.Select.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]

    lib.VisualizeData.restype = ctypes.c_char_p
    lib.VisualizeData.argtypes = [ctypes.c_void_p]

    # AllCard / AllAttack / AgentStart / Search* argtypes are configured by the
    # consumers (pkm/search.py, pkm/data/card_data.py) on this same handle.
    return lib


ENGINE_BACKEND, ENGINE_LIB_PATH = resolve_lib_path()
lib = _load()


# --- capability detection ----------------------------------------------------


def _has_symbol(handle: ctypes.CDLL, name: str) -> bool:
    """True if ``name`` is a resolvable exported symbol of ``handle``."""
    try:
        getattr(handle, name)
    except AttributeError:
        return False
    return True


def kaggle_available() -> bool:
    """Is the official Kaggle-bundled engine importable in this environment?"""
    return importlib.util.find_spec("kaggle_environments.envs.cabt.cg.sim") is not None


def vendored_built() -> bool:
    """Has the vendored C++ engine been compiled to engine/build (or result)?"""
    return _vendored_lib_path().exists()


def available_backends() -> list[str]:
    """Backends that can actually be loaded right now, in precedence order."""
    out = []
    if kaggle_available():
        out.append("kaggle")
    if vendored_built():
        out.append("vendored")
    return out


@dataclass(frozen=True)
class EngineCapabilities:
    """A snapshot of what the currently loaded engine can do."""

    backend: str  # "kaggle" | "vendored" | "override"
    lib_path: str
    kaggle_available: bool
    vendored_built: bool
    present_symbols: tuple[str, ...]
    missing_symbols: tuple[str, ...]
    supports_seed_injection: bool  # a seed-setter symbol is exported
    deterministic: bool  # engine can reproduce a battle from a seed

    def as_dict(self) -> dict:
        return asdict(self)


def capabilities() -> EngineCapabilities:
    """Probe the loaded engine and report its capabilities.

    ``supports_seed_injection`` / ``deterministic`` are False for the stock
    engine: it seeds ``std::mt19937`` from ``std::random_device()`` with no seed
    entry point in the public ABI (see docs/ENGINE.md). They flip automatically
    if a future patched build exports one of :data:`SEED_SYMBOLS`.
    """
    present = tuple(s for s in CORE_SYMBOLS if _has_symbol(lib, s))
    missing = tuple(s for s in CORE_SYMBOLS if s not in present)
    seeded = any(_has_symbol(lib, s) for s in SEED_SYMBOLS)
    return EngineCapabilities(
        backend=ENGINE_BACKEND,
        lib_path=str(ENGINE_LIB_PATH),
        kaggle_available=kaggle_available(),
        vendored_built=vendored_built(),
        present_symbols=present,
        missing_symbols=missing,
        supports_seed_injection=seeded,
        deterministic=seeded,
    )
