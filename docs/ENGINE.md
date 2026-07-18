# The cabt engine: vendored source, backend switch, and the determinism compromise

This document covers the C++ game engine that powers every battle: where it
comes from, how we compile our own copy, how code selects which build to run,
what it can and cannot do, and the one hard limitation we had to accept.

- **Source of truth for the swap:** `pkm/engine/` — `loader.py` (backend switch,
  ctypes ABI, capabilities), `api.py` (the complete typed API), `__main__.py`
  (capability report). Import the engine from `pkm.engine`, never from
  `kaggle_environments.envs.cabt.cg.*` directly.
- **Vendored C++ source:** `engine/` (copied from the standalone `ptcg` repo,
  commit `0a56d34`, Competition-Use-Only license).

---

## 1. What the engine is

The "engine" is a single ~1.3 MB shared library exporting a **13-function C
ABI**. All real data crosses the boundary as JSON strings plus a small byte
blob — the ABI is essentially "pass some ints, get JSON back":

```
GameInitialize  BattleStart  AgentStart  BattleFinish  GetBattleData
Select  VisualizeData  AllCard  AllAttack
SearchBegin  SearchStep  SearchEnd  SearchRelease
```

Kaggle ships a prebuilt `libcg.so` (+ `cg.dll`, `libcg.dylib`,
`libcg-arm64.so`) inside the `kaggle_environments` package. We vendored the C++
**source** so we can compile a byte-compatible `cg.so` ourselves — for
instrumentation, speed, and eventually a seed-injection patch (see §6).

Our build exports the **exact same 13 symbols**, verified by the parity test and
by `just engine-info` (`13/13 present`).

---

## 1a. The Python API (`pkm/engine/api.py`)

The kaggle package only wrapped **6** of the 13 functions (`cg/sim.py` +
`cg/game.py`); the search API and card/attack data were bound separately, and
argtypes were configured in three different modules with `AllCard`/`AllAttack`
bound twice. `pkm/engine/api.py` collates **all 13** into one typed module, and
`loader._configure_argtypes` sets every ABI signature in one place. Import
everything from `pkm.engine`:

```python
from pkm.engine import (
    battle_start, battle_select, battle_finish, visualize_data,
    search_begin, search_step, search_end, search_release,
    all_cards, all_attacks, to_observation,
)
```

### Return-type convention: dicts at the seam, pydantic inward

`pkm/types/obs.py` documents the codebase's design — *"nothing inward of
`Observation.model_validate(raw)` sees a dict."* The API follows it, because
where each result is consumed dictates the right type:

| Function group | Returns | Why |
|---|---|---|
| `battle_*` | raw obs `dict` | ~37 call sites read these as dicts and self-play rollouts run millions of steps; validate with `to_observation()` only at the ML boundary |
| `search_*` | typed `SearchState` | one consumer (`pkm/mcts`); `.observation` validates **lazily and caches**, so traversal stays on `.raw_observation`/`.search_id` (cheap) and full validation is paid once per node, on encode |
| `all_cards` / `all_attacks` | `list[dict]` | engine primitive; `pkm/data/card_data.py` builds its dataclasses on top |

**Why not fully-typed returns everywhere?** Measured: full pydantic validation
per `search_step` costs **+20 µs on a ~28 µs step (+72%)** — MCTS/expert-iteration
would run ~1.7× slower. `SearchState`'s lazy-cached `.observation` gives the same
typed access at **~0%** hot-path cost, matching what the code already did by hand
(`Observation.model_validate` at `mcts/search.py`, `rollout.py`, `exit_train.py`,
`numpy_policy.py`, `tui/session.py`). Typing the `battle_*` returns would break
those 37 dict accesses and tax rollouts for no gain.

---

## 2. Building the engine

The engine is a header-heavy project whose single translation unit
(`src/Export.cpp`) includes everything, so the compile is memory-hungry — build
with `-j1` on low-RAM machines.

### With nix (recommended)

`engine/` carries its own flake with a pinned LLVM/clang C++20 devshell.

```bash
just engine-build         # cmake + ninja inside `nix develop` -> engine/build/cg.so
just engine-build-nix     # fully hermetic: `nix build` -> engine/result/lib/cg.so
```

`just engine-build` uses the flake's **libc++** (LLVM) standard library.
`engine/result/lib/cg.so` from `nix build` is a read-only store path; the loader
finds it as a fallback if `engine/build/cg.so` is absent.

> Nix flakes only see **git-tracked** files. Because `engine/` is nested in this
> repo, the engine sources must be `git add`-ed (staged is enough) before
> `nix build` / `nix develop` can see them.

### Without nix

Any system `cmake` + a C++20 compiler works. No ninja required (default
generator is fine):

```bash
just engine-build-cc      # cmake -S engine -B engine/build && cmake --build engine/build
# equivalently, by hand:
cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release
cmake --build engine/build            # add -j1 if the compile gets OOM-killed
```

A stock `g++` build links **libstdc++ (GNU)**, which happens to match the
standard library the official `libcg.so` links — so the non-nix build is
actually *closer* to the official engine's shuffle order than the nix libc++
build (see §5). It still is not seed-identical (§4).

### Fixed packaging bug

`engine/package.nix` originally whitelisted only `src/*.h` via `sourceByRegex`,
which dropped the `src/api`, `src/card`, … subdirectories after ptcg's
`refactor/restructure-src` reshuffle, breaking `nix build`. We changed the regex
to `"^src(/.*)?$"` so the whole tree is included.

---

## 3. Selecting a backend (the swap)

`pkm/engine/loader.py` decides which build backs the process. Because both
builds expose the identical ABI, "switching backends" is just choosing a
different `.so` path — everything above the loader is the same code.

Precedence (highest first):

| Selector | Effect |
|---|---|
| `PKM_ENGINE_LIB=/abs/path/cg.so` | load exactly that library |
| `PKM_ENGINE=local-nix` | our **nix** build `engine/result/lib/cg.so` |
| `PKM_ENGINE=local` | our **cmake** build `engine/build/cg.so` |
| `PKM_ENGINE=vendored` | deprecated alias: nix build if present, else cmake |
| *(default)* | the Kaggle-bundled `libcg.so` |

```bash
just play human neural                 # official engine (default)
PKM_ENGINE=local-nix just test         # whole suite against our nix build
PKM_ENGINE_LIB=/tmp/experiment/cg.so python -m pkm.engine
```

Only the `kaggle` backend imports `kaggle_environments` (≈2.8 s + registers every
OpenSpiel env); the `local*` backends `dlopen` their `.so` directly, so local
training starts fast and quiet.

**The default must stay `kaggle`.** The Kaggle submission sandbox has no
`engine/` directory, so the submission path must never require a local build.

### Lazy, late-bound loading

The engine loads on **first use**, not at import (`loader.get_lib()` caches the
handle; there is no module-level load). This means (a) the backend can be chosen
at runtime — `loader.set_backend("local-nix")` before the first engine call, or
the `--engine` flag on the agent CLIs — instead of being frozen the instant
`pkm.engine` is first imported; and (b) commands that never touch the engine
(`pkm deck list`, most of `info`) skip loading entirely. `set_backend()` raises
if the engine is already loaded — it cannot be hot-swapped (a second
`GameInitialize` aborts the process). `pkm.engine.lib` / `ENGINE_BACKEND` /
`ENGINE_LIB_PATH` still work; they resolve lazily via module `__getattr__`.

### What the switch covers (and what it doesn't)

`PKM_ENGINE` governs the **direct** engine paths that go through `pkm.engine`:
card data, the search API (`pkm/engine/api.py`), and the RL/MCTS rollouts
(`rollout.py`, `exit_train.py`, `test_mcts.py`, `test_rl.py`). `pkm/rl/play.py`
and the TUI
run matches through `kaggle_environments.make("cabt")`, which loads the
**bundled** engine internally — the switch does not reach those yet. Pointing
the `make()` harness at the vendored lib is a separate, larger change.

### The double-init gotcha

For the `kaggle` backend the loader **reuses** the already-initialized
`sim.lib` rather than re-`LoadLibrary`-ing the same path. `dlopen` caches by
path and returns the same handle, so a second `GameInitialize()` call
double-initializes the engine's global state and aborts the process with
`buffer full. capacity:7`. This is why the loader special-cases the kaggle
backend and why the parity test loads each engine exactly once in an isolated
subprocess.

---

## 4. Capability detection

The shim probes the loaded library and reports what it can do:

```bash
just engine-info                       # default backend
PKM_ENGINE=vendored just engine-info
```

```
cabt engine
  backend            : vendored
  lib path           : .../engine/build/cg.so
  available backends : kaggle, vendored
  core ABI symbols   : 13/13 present
  seed injection     : False
  deterministic      : False
```

From code:

```python
from pkm.engine import capabilities, available_backends, vendored_built

caps = capabilities()          # EngineCapabilities dataclass
caps.present_symbols           # the 13 core symbols found in the loaded lib
caps.supports_seed_injection   # True iff a seed-setter symbol is exported
available_backends()           # ["kaggle", "vendored"] — what can load right now
```

Detection is symbol-based (`ctypes` attribute resolution). `supports_seed_injection`
and `deterministic` are `False` for the stock engine but flip **automatically**
if a future patched build exports one of `SEED_SYMBOLS`
(`BattleStartSeed`, `SetSeed`, …). That is the hook for §6.

---

## 5. The determinism issue (root cause)

We wanted the vendored engine to be a byte-for-byte drop-in so that agents
trained locally see the exact same games as Kaggle. **It cannot be**, and the
reason is in the engine itself.

`engine/src/api/Api.h`, `ApiBattleStart`:

```cpp
std::random_device rd;
GameConfig config = {};
config.seed = rd();                       // hardware entropy
...
data->init(config);
std::seed_seq seq{rd(), rd(), rd(), rd()};
data->game.rng = std::mt19937(seq);       // reseeded again from random_device
```

The RNG is seeded from `std::random_device()`, and the public
`BattleStart(int* cards)` ABI takes **only the 60+60 card ids** — there is no
way to inject a seed. Consequences, all confirmed by measurement:

- **The official engine diverges from itself.** Two independent loads of the
  *same* `libcg.so`, same decks, produce different opening hands and disagree on
  the first `Select`.
- The **initial observation** (returned by `BattleStart` before any
  shuffle/draw) *is* identical across seeds — it contains no RNG-dependent
  state. The first `Select` triggers the shuffle and everything downstream
  diverges.

Two secondary, non-rules differences also perturb draw order (they would matter
only if the seed problem were fixed):

- **`std::shuffle` is implementation-defined.** libstdc++ and libc++ produce
  different sequences from the same `mt19937` state. Our `just engine-build`
  (nix, libc++) therefore shuffles differently from the official libstdc++ lib;
  `just engine-build-cc` (system g++, libstdc++) matches the official stdlib.
- These are draw-order differences, **not rules differences** — the observation
  schema and legal options are identical.

---

## 6. Our compromise

We accept that **full-game reproducibility is out of scope** with the stock
engine, and we test only what is actually deterministic:

- **`tests/test_engine_parity.py`** asserts the **initial observation** is
  byte-identical between the official and vendored builds. That is enough to
  catch real drift in the card tables, deck encoding, or observation
  serialization — the things a bad vendored build would break — without
  depending on RNG.
- For **training**, seed-identity is not required: self-play only needs the same
  *rules*, and different shuffles are fine (even desirable for variety). The
  vendored engine is fully usable for search, card data, and RL/MCTS rollouts
  today — the whole test suite passes on both backends.

### If we ever need seed-exact reproducibility

Patch the vendored C++ to accept an injected seed and expose it in the ABI —
e.g. a new `BattleStartSeed(int* cards, uint64_t seed)` export that skips the
`random_device` reseed. The capability layer already looks for exactly this
(`SEED_SYMBOLS`): the moment such a symbol exists, `capabilities().deterministic`
becomes `True` and `test_engine_parity.py` can be extended to walk full games in
lockstep. Until then, seed injection is deliberately **not** implemented.

---

## Quick reference

```bash
just engine-build        # build with nix (libc++)
just engine-build-nix    # build hermetically with nix
just engine-build-cc     # build without nix (system toolchain, libstdc++)
just engine-clean        # remove build outputs
just engine-info         # backend + capability report
just engine-parity       # initial-observation parity test
PKM_ENGINE=vendored just test    # run the suite against the vendored engine
```
