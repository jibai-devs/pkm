# Engine notes: nix vs kaggle backends, and seeding

Scratch reference for how this agent talks to the cabt engine, the difference
between the local (nix) build and Kaggle's, and where seeding does/doesn't stand.
Verified against the code + binaries on 2026-07-18.

## Backends (set via `--engine` / `PKM_ENGINE`)

The whole project loads the engine through one seam, `pkm/engine/`
(`loader.py` + `api.py`). Loading is **lazy** (`loader.get_lib()` on first use);
pick the backend with `set_backend()` / the CLI `--engine` flag before first use.

| `--engine` | library | notes |
|---|---|---|
| `kaggle` (default) | `.venv/.../cabt/cg/libcg.so` | imports `kaggle_environments` (~2.8s + all OpenSpiel envs) |
| `local-nix` | `engine/result/lib/cg.so` | our **nix** build; dlopen direct, fast |
| `local` | `engine/build/cg.so` | our **cmake** build |
| `vendored` | nix-first, else cmake | deprecated alias |

This agent's `just`/CLI training recipes default to **`local-nix`** (fast, quiet).
Submission always uses **kaggle** (sandbox has no `engine/`, sets no env).

## Are the nix build and Kaggle's the same?

**Same source, different binary.** Both are built from the `ptcg` C++ source
(commit `0a56d34`) and expose the identical 13-symbol ABI. `just engine-parity`
confirms the **initial observation is byte-identical** between them — so the
*rules/logic are the same code*. They differ only in:

- **C++ stdlib**: nix links **libc++**, Kaggle links **libstdc++**. `std::shuffle`
  is implementation-defined → the two deal cards in a different order *even for
  the same seed*.
- **RNG**: both seed `std::mt19937` from `std::random_device()` → each is
  nondeterministic and diverges from itself (and from the other) after the first
  `Select`.
- **Startup**: `local-nix` dlopens directly; `kaggle` drags in `kaggle_environments`.

Never a rules difference — only the RNG stream + shuffle implementation.

## Seeding: what we have vs. what we don't

**We did NOT add the ability to seed.** What exists in `loader.py` is seed-
*detection*, not seed-*injection*:

- `SEED_SYMBOLS = (BattleStartSeed, BattleStartWithSeed, SetSeed, GameSeed)` are
  **hypothetical** names a *future patched* build might export. `capabilities()`
  probes for them.
- Reality check (2026-07-18): `nm -D engine/result/lib/cg.so | grep seed` → nothing.
  `capabilities().supports_seed_injection` → **False**. Not exported, **not used**,
  can't be.

Nuance: the C++ *internally* has a `config.seed` and seeds its RNG from it
(`engine/src/api/Api.h:90-92`), so the engine is *capable* of deterministic play —
but the public `BattleStart(int*)` ABI always fills that seed from
`std::random_device()`, with no parameter to inject one. We never patched it.

`tests/test_engine_parity.py` only checks the deterministic *initial* observation
for exactly this reason.

## Do we need seeding?

Not to train or submit (nondeterministic self-play is standard; the submission
runs on Kaggle's unseeded engine anyway, so a seed only helps the *local* loop).

Where it would genuinely help:

1. **Variance-reduced evaluation (highest value).** Common random numbers — play
   two agents through the *same* seeded games — slashes win-rate variance. Directly
   addresses the sweep's saturated "100% vs random" being un-discriminating.
2. **Reproducible debugging** — replay the exact game that crashed/misplayed.
3. **Full-game parity** — byte-verify vendored == Kaggle across a whole game.

Cost: patch the vendored C++ to accept a seed and export e.g.
`BattleStartSeed(int* deck, uint32_t seed)`, rebuild. **Local-only** (can't patch
Kaggle's lib). For cross-engine byte-parity you'd also have to match the shuffle
(libc++ vs libstdc++), so cross-engine repro is a bigger ask than intra-vendored.

**Verdict:** not required. If pursued, the cheap high-value slice is: expose a seed,
use it *only* for paired local eval (common random numbers), leave training +
submission unchanged.
