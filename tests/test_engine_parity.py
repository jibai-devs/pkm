"""Parity between the official Kaggle engine and our vendored `engine/` build.

Both engines expose the identical C ABI. Full-game byte-parity is **not**
achievable and is not tested: `ApiBattleStart` (engine/src/api/Api.h) seeds
`std::mt19937` from `std::random_device()` with no seed injection through the
public `BattleStart(int*)` ABI, so every battle — even two runs of the *same*
official library — draws a different hand and diverges on the first `Select`.

What is deterministic, and what we assert here, is the **initial observation**
returned by `BattleStart` before any shuffle/draw. It is byte-identical across
independent seeds, so comparing it catches any real divergence in the card
tables, deck encoding, or observation serialization between the two builds. A
failure means the vendored engine has genuinely drifted from the official one.

`GameInitialize()` mutates per-library global state and must run exactly once
per handle, so the comparison runs in an isolated subprocess (`__main__` below)
that dlopens each engine once. Skips when the vendored build is absent.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DECK = REPO / "deck" / "02_dragapult.csv"


def _kaggle_lib() -> Path | None:
    """Locate libcg.so without importing (and thus initializing) the engine."""
    spec = importlib.util.find_spec("kaggle_environments.envs.cabt.cg.sim")
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).parent / "libcg.so"


def _vendored_lib() -> Path:
    return REPO / "engine" / "build" / "cg.so"


def test_initial_observation_identical() -> None:
    vendored = _vendored_lib()
    if not vendored.exists():
        pytest.skip("vendored engine not built (run `just engine-build`)")
    kaggle = _kaggle_lib()
    if kaggle is None or not kaggle.exists():
        pytest.skip("official Kaggle engine not available")

    proc = subprocess.run(
        [sys.executable, __file__, str(kaggle), str(vendored), str(DECK)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"engine parity failed:\n{proc.stdout}\n{proc.stderr}"


# --- subprocess entry point: dlopen each engine once, compare initial obs ----

if __name__ == "__main__":
    import ctypes
    import json

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

    def bind(path: str) -> ctypes.CDLL:
        lib = ctypes.cdll.LoadLibrary(path)
        lib.GameInitialize()
        lib.BattleStart.restype = StartData
        lib.BattleStart.argtypes = [ctypes.POINTER(ctypes.c_int)]
        lib.GetBattleData.restype = SerialData
        lib.GetBattleData.argtypes = [ctypes.c_void_p]
        lib.BattleFinish.argtypes = [ctypes.c_void_p]
        return lib

    def initial_obs(lib: ctypes.CDLL, cards: list[int]) -> tuple[int, dict]:
        arg = (ctypes.c_int * len(cards))(*cards)
        ptr = lib.BattleStart(arg).battlePtr
        return ptr, json.loads(lib.GetBattleData(ptr).json.decode())

    kaggle_path, vendored_path, deck_path = sys.argv[1:4]
    cards = [int(x) for x in Path(deck_path).read_text().split() if x.strip()]
    both = cards + cards

    k, v = bind(kaggle_path), bind(vendored_path)
    k_ptr, k_obs = initial_obs(k, both)
    v_ptr, v_obs = initial_obs(v, both)
    k.BattleFinish(k_ptr)
    v.BattleFinish(v_ptr)

    if k_obs != v_obs:
        differing = sorted(
            kk for kk in set(k_obs) | set(v_obs) if k_obs.get(kk) != v_obs.get(kk)
        )
        print(f"initial observation diverged; differing top-level keys: {differing}")
        sys.exit(1)
    print("initial-observation parity OK")
    sys.exit(0)
