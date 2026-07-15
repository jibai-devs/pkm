# The cabt engine: vendored source, backend switch, and the determinism compromise

This document covers the C++ game engine that powers every battle: where it
comes from, how we compile our own copy, how code selects which build to run,
what it can and cannot do, and the one hard limitation we had to accept.

- **Source of truth for the swap:** `pkm/engine/` (`loader.py`, `game.py`,
  `__main__.py`). Import the engine from `pkm.engine`, never from
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
| `PKM_ENGINE=vendored` | our compiled `engine/build/cg.so` (or `engine/result/lib/cg.so`) |
| *(default)* | the Kaggle-bundled `libcg.so` |

```bash
just play human neural                 # official engine (default)
PKM_ENGINE=vendored just test          # whole suite against our build
PKM_ENGINE_LIB=/tmp/experiment/cg.so python -m pkm.engine
```

**The default must stay `kaggle`.** The Kaggle submission sandbox has no
`engine/` directory, so the submission path must never require the vendored
build.

### What the switch covers (and what it doesn't)

`PKM_ENGINE` governs the **direct** engine paths that go through `pkm.engine`:
card data, `pkm/search.py`, and the RL/MCTS rollouts (`rollout.py`,
`exit_train.py`, `test_mcts.py`, `test_rl.py`). `pkm/rl/play.py` and the TUI
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
