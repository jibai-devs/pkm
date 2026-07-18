"""Load the cabt game engine and expose its ctypes surface.

This is the single place that decides *which* build of the engine backs the
process. Both the official Kaggle `libcg.so` and our locally compiled builds
export the identical C ABI, so switching backends is just choosing a different
shared-library path — everything above this module is byte-for-byte the same code.

Backend selection (highest precedence first):
  1. ``PKM_ENGINE_LIB=/abs/path/to/cg.so``  — explicit path override
  2. ``PKM_ENGINE=<backend>``               — one of:
       - ``kaggle``     the Kaggle-bundled engine (imports ``kaggle_environments``,
                        which is ~2.8s + registers every OpenSpiel env — slow)
       - ``local``      our cmake build   (``engine/build/cg.so``)
       - ``local-nix``  our nix build     (``engine/result/lib/cg.so``)
       - ``vendored``   deprecated alias: nix build if present, else cmake
  3. default                                — ``kaggle``

The default MUST stay "kaggle": that is the only engine available inside the
Kaggle submission sandbox, where ``engine/`` does not exist. The ``local*``
backends load their ``.so`` directly via ctypes and never import
``kaggle_environments`` — much faster startup for local training.
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


def _engine_dir() -> Path:
    # pkm/engine/loader.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2] / "engine"


def _cmake_lib_path() -> Path:
    """The cmake (incremental) build output — ``just engine-build``."""
    return _engine_dir() / "build" / "cg.so"


def _nix_lib_path() -> Path:
    """The ``nix build`` result symlink — ``just engine-build-nix``."""
    return _engine_dir() / "result" / "lib" / "cg.so"


def _vendored_lib_path() -> Path:
    """Deprecated ``vendored`` alias: nix build if present, else cmake."""
    for candidate in (_nix_lib_path(), _cmake_lib_path()):
        if candidate.exists():
            return candidate
    return _nix_lib_path()  # report the expected path in the error


# Build hint shown when a selected local backend's .so is missing.
_BUILD_HINTS: dict[str, str] = {
    "local": "just engine-build",
    "local-nix": "just engine-build-nix",
    "vendored": "just engine-build-nix  (or: just engine-build)",
}


def resolve_lib_path() -> tuple[str, Path]:
    """Return ``(backend_name, path)`` for the selected engine."""
    override = os.environ.get("PKM_ENGINE_LIB")
    if override:
        return "override", Path(override)
    backend = os.environ.get("PKM_ENGINE", "kaggle").lower()
    if backend in ("local-nix", "nix"):
        return "local-nix", _nix_lib_path()
    if backend in ("local", "cmake"):
        return "local", _cmake_lib_path()
    if backend == "vendored":  # deprecated alias
        return "vendored", _vendored_lib_path()
    if backend == "kaggle":
        return "kaggle", _kaggle_lib_path()
    raise ValueError(
        f"unknown PKM_ENGINE={backend!r} (expected 'kaggle', 'local', or 'local-nix')"
    )


def _configure_argtypes(handle: ctypes.CDLL) -> None:
    """Set restype/argtypes for all 13 exported functions on ``handle``.

    Single source of truth for the ABI signatures — previously these were
    scattered across sim.py (battle), pkm/search.py (search) and card_data.py
    (AllCard/AllAttack). Idempotent, so it is safe to run on the already-
    configured kaggle handle.
    """
    handle.BattleStart.restype = StartData
    handle.BattleStart.argtypes = [ctypes.POINTER(ctypes.c_int)]
    handle.BattleFinish.argtypes = [ctypes.c_void_p]
    handle.GetBattleData.restype = SerialData
    handle.GetBattleData.argtypes = [ctypes.c_void_p]
    handle.Select.restype = ctypes.c_int
    handle.Select.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    handle.VisualizeData.restype = ctypes.c_char_p
    handle.VisualizeData.argtypes = [ctypes.c_void_p]

    handle.AgentStart.restype = ctypes.c_void_p
    handle.SearchBegin.restype = ctypes.c_char_p
    handle.SearchBegin.argtypes = [
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
    handle.SearchStep.restype = ctypes.c_char_p
    handle.SearchStep.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    handle.SearchEnd.argtypes = [ctypes.c_void_p]
    handle.SearchRelease.argtypes = [ctypes.c_void_p, ctypes.c_int64]
    handle.AllCard.restype = ctypes.c_char_p
    handle.AllAttack.restype = ctypes.c_char_p


def _load() -> ctypes.CDLL:
    global _loaded_backend, _loaded_path
    backend, path = resolve_lib_path()
    _loaded_backend, _loaded_path = backend, path

    if backend == "kaggle":
        # Importing the upstream module already dlopen'd libcg.so and called
        # GameInitialize(). dlopen caches by path, so re-loading returns the same
        # handle — calling GameInitialize() again double-inits the engine's
        # global state and aborts. Reuse the handle; only (re)set argtypes.
        from kaggle_environments.envs.cabt.cg import sim

        handle = sim.lib
    else:
        if not path.exists():
            hint = _BUILD_HINTS.get(backend)
            msg = f"engine backend '{backend}' library not found: {path}"
            if hint:
                msg += f"\n  build it with:  {hint}"
            raise FileNotFoundError(msg)
        handle = ctypes.cdll.LoadLibrary(str(path))
        handle.GameInitialize()

    _configure_argtypes(handle)
    return handle


# --- lazy, late-bound loading ------------------------------------------------
# The engine loads on *first use*, not at import. This lets the backend be chosen
# at runtime (a CLI flag / :func:`set_backend`) instead of being frozen the moment
# ``pkm.engine`` is first imported, and lets commands that never touch the engine
# (``pkm deck list``, most of ``info``) skip loading entirely. Consumers should
# call :func:`get_lib`; the ``pkm.engine.lib`` / ``ENGINE_BACKEND`` /
# ``ENGINE_LIB_PATH`` names still resolve lazily via module ``__getattr__``.

_lib: ctypes.CDLL | None = None
_loaded_backend: str | None = None
_loaded_path: Path | None = None

_KNOWN_BACKENDS = frozenset(
    {"kaggle", "local", "local-nix", "nix", "cmake", "vendored"}
)


def get_lib() -> ctypes.CDLL:
    """Return the engine handle, loading + initializing it on first call (cached)."""
    global _lib
    if _lib is None:
        _lib = _load()
    return _lib


def set_backend(name: str) -> None:
    """Select the engine backend for this process. **Call before first use.**

    Points the lazy loader at ``name`` (via ``PKM_ENGINE``) so :func:`get_lib`
    picks it up. Raises if the engine is already loaded — it cannot be hot-swapped
    (a second ``GameInitialize`` aborts the process), so the backend is fixed once
    the first engine call has run.
    """
    if _lib is not None:
        raise RuntimeError(
            f"engine already loaded as {_loaded_backend!r}; set_backend() must run "
            "before the first engine call"
        )
    if name.lower() not in _KNOWN_BACKENDS:
        raise ValueError(
            f"unknown engine backend {name!r} "
            "(expected 'kaggle', 'local', or 'local-nix')"
        )
    os.environ["PKM_ENGINE"] = name


def _selected_backend() -> str:
    """The backend that *would* load now, by name, without importing anything."""
    if os.environ.get("PKM_ENGINE_LIB"):
        return "override"
    backend = os.environ.get("PKM_ENGINE", "kaggle").lower()
    return {"nix": "local-nix", "cmake": "local"}.get(backend, backend)


def __getattr__(name: str):  # PEP 562: lazy module attributes
    if name == "lib":
        return get_lib()
    if name == "ENGINE_BACKEND":
        return _loaded_backend if _lib is not None else _selected_backend()
    if name == "ENGINE_LIB_PATH":
        return _loaded_path if _lib is not None else resolve_lib_path()[1]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    if _cmake_lib_path().exists():
        out.append("local")
    if _nix_lib_path().exists():
        out.append("local-nix")
    return out


@dataclass(frozen=True)
class EngineCapabilities:
    """A snapshot of what the currently loaded engine can do."""

    backend: str  # "kaggle" | "local" | "local-nix" | "vendored" | "override"
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
    handle = get_lib()
    present = tuple(s for s in CORE_SYMBOLS if _has_symbol(handle, s))
    missing = tuple(s for s in CORE_SYMBOLS if s not in present)
    seeded = any(_has_symbol(handle, s) for s in SEED_SYMBOLS)
    return EngineCapabilities(
        backend=str(_loaded_backend),
        lib_path=str(_loaded_path),
        kaggle_available=kaggle_available(),
        vendored_built=vendored_built(),
        present_symbols=present,
        missing_symbols=missing,
        supports_seed_injection=seeded,
        deterministic=seeded,
    )
