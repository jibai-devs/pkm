# Pluggable Training Methods + MCTS Expert Iteration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add MCTS expert iteration (ExIt) as a second self-play training method behind a clean `Trainer` seam, keeping classic PPO working and bit-for-bit resumable.

**Architecture:** One `Trainer` protocol with one implementation per method (`PpoTrainer`, `ExItTrainer`) behind a `TRAINERS` registry keyed by `cfg.train.method`. `train.py` becomes a thin method-agnostic driver; the parallel worker pool calls `trainer.collect` generically. ExIt does PUCT search over the engine's `search_*` forward model, guided by `model.evaluate()`, with single-sample (K=1) determinization of the known `DECK_60`.

**Tech Stack:** Python 3, PyTorch, `torch.multiprocessing` (spawn), the vendored/Kaggle C engine via `pkm.engine` (reached through the agent's `cabt.py`), pytest.

## Global Constraints

- **Scope:** all changes confined to `pkm/new_agents/agent_000_dragapult/`. The engine is a dependency reached ONLY through `cabt.py`.
- **Back-compat:** the existing PPO checkpoint (experiment `000_default`, update 458) MUST still load and resume after the refactor.
- **Default method:** `cfg.train.method` defaults to `"ppo"`; omitting `--method` reproduces current behavior.
- **Determinism of workers:** rollout workers use the `spawn` start method (one engine per process); never `fork`.
- **Test command:** `uv run pytest <path> -q` (run from repo root `/home/df/projects/zeke/pkm_new`). Agent tests live flat in the agent dir as `test_*.py`, matching `test_shaping.py`.
- **Commit style:** conventional commits scoped `feat(agent_000_dragapult): …`; end message with the repo's `Co-Authored-By` trailer.
- **No secrets, no engine edits, no files outside the agent package.**

Throughout, `REPO=/home/df/projects/zeke/pkm_new` and `AG=pkm/new_agents/agent_000_dragapult`.

---

### Task 1: Config — `method` + MCTS knobs + resume hash-migration

**Files:**
- Modify: `pkm/new_agents/agent_000_dragapult/config.py`
- Modify: `pkm/new_agents/agent_000_dragapult/train.py:253-268` (`TrainState.load` hash guard)
- Test: `pkm/new_agents/agent_000_dragapult/test_config_migration.py` (create)

**Interfaces:**
- Consumes: existing `Config`, `TrainConfig`, `Config.hash()`, `TrainState`.
- Produces:
  - `TrainConfig.method: str = "ppo"`, `.mcts_simulations: int = 32`, `.mcts_c_puct: float = 1.25`, `.mcts_temperature: float = 1.0`, `.determinization: str = "sample"`.
  - `config._hash_dict(d: dict) -> str` — the canonical 12-char sha256 of a config dict.
  - `TrainState.load` validates `blob["config_hash"]` against `_hash_dict(blob["config"])` (the stored dict), not against a re-hash of the reconstructed config.

- [ ] **Step 1: Write the failing test**

Create `pkm/new_agents/agent_000_dragapult/test_config_migration.py`:

```python
import dataclasses
import torch

from pkm.new_agents.agent_000_dragapult.config import (
    Config,
    TrainConfig,
    _hash_dict,
    build_model,
)
from pkm.new_agents.agent_000_dragapult.train import TrainState


def test_new_fields_have_ppo_defaults():
    tc = TrainConfig()
    assert tc.method == "ppo"
    assert tc.mcts_simulations == 32
    assert tc.determinization == "sample"


def test_from_dict_backfills_missing_method():
    # An "old" config dict predating the method field.
    d = Config().to_dict()
    del d["train"]["method"]
    del d["train"]["mcts_simulations"]
    del d["train"]["mcts_c_puct"]
    del d["train"]["mcts_temperature"]
    del d["train"]["determinization"]
    cfg = Config.from_dict(d)  # must not raise; fills defaults
    assert cfg.train.method == "ppo"


def test_load_accepts_old_schema_checkpoint(tmp_path):
    # Save a current checkpoint, then rewrite its blob to look "old"
    # (no method/mcts fields, hash recomputed over the old dict).
    cfg = Config()
    model = build_model(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    ts = TrainState(cfg=cfg, model=model, optimizer=opt, update_idx=7)
    path = tmp_path / "latest.pt"
    ts.save(path)

    blob = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("method", "mcts_simulations", "mcts_c_puct",
              "mcts_temperature", "determinization"):
        blob["config"]["train"].pop(k, None)
    blob["config_hash"] = _hash_dict(blob["config"])
    torch.save(blob, path)

    loaded = TrainState.load(path)  # must not raise "config hash mismatch"
    assert loaded.update_idx == 7
    assert loaded.cfg.train.method == "ppo"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd $REPO && uv run pytest $AG/test_config_migration.py -q`
Expected: FAIL — `ImportError: cannot import name '_hash_dict'` and/or `TypeError` on unknown field.

- [ ] **Step 3: Add the fields and the hash helper in `config.py`**

In `TrainConfig` (after `shaping_coef`, config.py:65) add:

```python
    # --- training method selector (key into trainers.TRAINERS) ---
    method: str = "ppo"
    # MCTS expert-iteration knobs (inert unless method == "exit").
    mcts_simulations: int = 32
    mcts_c_puct: float = 1.25
    mcts_temperature: float = 1.0
    determinization: str = "sample"  # key into trainers.exit determinizers
```

Above `class Config` add the canonical dict-hasher and reuse it in `Config.hash`:

```python
def _hash_dict(d: dict[str, Any]) -> str:
    """Stable 12-char sha256 of a config dict (the one hashing definition)."""
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]
```

Change `Config.hash` (config.py:95-98) to:

```python
    def hash(self) -> str:
        """Stable short hash of the whole config (goes in checkpoints/run dirs)."""
        return _hash_dict(self.to_dict())
```

- [ ] **Step 4: Make the resume guard schema-additive-tolerant in `train.py`**

In `TrainState.load` (train.py:254-258) replace:

```python
        blob = torch.load(path, map_location="cpu", weights_only=False)
        cfg = Config.from_dict(blob["config"])
        if cfg.hash() != blob["config_hash"]:
            raise ValueError("config hash mismatch on resume")
```

with (import `_hash_dict` at the top of `train.py` alongside the existing config import):

```python
        blob = torch.load(path, map_location="cpu", weights_only=False)
        cfg = Config.from_dict(blob["config"])
        # Validate the STORED dict against its STORED hash, so additive schema
        # changes (new fields with defaults) never trip the guard for older
        # checkpoints. A tampered file still fails.
        if _hash_dict(blob["config"]) != blob["config_hash"]:
            raise ValueError("config hash mismatch on resume")
```

Update the import line in `train.py` (currently `from ...config import Config, build_model`) to also import `_hash_dict`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd $REPO && uv run pytest $AG/test_config_migration.py -q`
Expected: PASS (3 passed).

- [ ] **Step 6: Regression — existing tests still pass**

Run: `cd $REPO && uv run pytest $AG/test_shaping.py -q`
Expected: PASS (7 passed).

- [ ] **Step 7: Commit**

```bash
cd $REPO && git add $AG/config.py $AG/train.py $AG/test_config_migration.py
git commit -m "feat(agent_000_dragapult): config method selector + MCTS knobs + resume hash-migration

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `Trainer` protocol + registry

**Files:**
- Create: `pkm/new_agents/agent_000_dragapult/trainers/__init__.py`
- Test: `pkm/new_agents/agent_000_dragapult/test_trainers_registry.py` (create)

**Interfaces:**
- Produces:
  - `Trainer` (Protocol) with `collect(self, model, n_games, cfg, gen=None) -> tuple[list, dict]` and `update(self, model, opt, samples, cfg) -> dict`.
  - `TRAINERS: dict[str, Callable[[], Trainer]]` (populated lazily; PPO added in Task 3, ExIt in Task 8).
  - `get_trainer(cfg) -> Trainer` — instantiates `TRAINERS[cfg.train.method]`, raising a clear error on unknown method.

- [ ] **Step 1: Write the failing test**

Create `pkm/new_agents/agent_000_dragapult/test_trainers_registry.py`:

```python
import pytest

from pkm.new_agents.agent_000_dragapult.config import Config
from pkm.new_agents.agent_000_dragapult import trainers


def test_get_trainer_unknown_method_raises():
    cfg = Config()
    object.__setattr__(cfg.train, "method", "nope")  # frozen dataclass
    with pytest.raises(ValueError, match="unknown training method 'nope'"):
        trainers.get_trainer(cfg)


def test_registry_is_a_dict():
    assert isinstance(trainers.TRAINERS, dict)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd $REPO && uv run pytest $AG/test_trainers_registry.py -q`
Expected: FAIL — `ModuleNotFoundError: ...trainers`.

- [ ] **Step 3: Create the package + protocol**

Create `pkm/new_agents/agent_000_dragapult/trainers/__init__.py`:

```python
"""Pluggable training methods.

A ``Trainer`` owns the two method-specific halves of one PPO/ExIt update:
``collect`` (self-play → samples, runs in rollout workers) and ``update`` (the
learn step). Everything else — checkpoint/resume, observers, the parallel pool,
eval, timing/utilization diagnostics — lives in the method-agnostic driver
(:func:`..train.train`). Methods register in :data:`TRAINERS`, keyed by
``cfg.train.method`` (mirrors ``shaping.SHAPERS``/``ESTIMATORS``).
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

import torch


@runtime_checkable
class Trainer(Protocol):
    def collect(
        self, model: torch.nn.Module, n_games: int, cfg: Any,
        gen: "torch.Generator | None" = None,
    ) -> tuple[list, dict]:
        """Self-play → (samples, stats). Runs inside rollout workers."""
        ...

    def update(
        self, model: torch.nn.Module, opt: torch.optim.Optimizer,
        samples: list, cfg: Any,
    ) -> dict:
        """One learn step over ``samples`` → per-update stats."""
        ...


def _ppo_trainer() -> Trainer:
    from pkm.new_agents.agent_000_dragapult.trainers.ppo import PpoTrainer
    return PpoTrainer()


def _exit_trainer() -> Trainer:
    from pkm.new_agents.agent_000_dragapult.trainers.exit import ExItTrainer
    return ExItTrainer()


# Lazy factories so importing this package doesn't pull torch-heavy modules
# until a method is actually selected.
TRAINERS: dict[str, Callable[[], Trainer]] = {
    "ppo": _ppo_trainer,
    "exit": _exit_trainer,
}


def get_trainer(cfg: Any) -> Trainer:
    method = cfg.train.method
    try:
        factory = TRAINERS[method]
    except KeyError:
        raise ValueError(
            f"unknown training method {method!r}; choose from {sorted(TRAINERS)}"
        ) from None
    return factory()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd $REPO && uv run pytest $AG/test_trainers_registry.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd $REPO && git add $AG/trainers/__init__.py $AG/test_trainers_registry.py
git commit -m "feat(agent_000_dragapult): Trainer protocol + TRAINERS registry

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Relocate PPO into `trainers/ppo.py` (`PpoTrainer`)

**Files:**
- Create: `pkm/new_agents/agent_000_dragapult/trainers/ppo.py`
- Modify: `pkm/new_agents/agent_000_dragapult/train.py` (remove the relocated funcs; re-export `Step` for back-compat)
- Test: `pkm/new_agents/agent_000_dragapult/test_ppo_trainer.py` (create)

**Interfaces:**
- Consumes: `Trainer` (Task 2); existing `play_game`, `collect_rollout`, `ppo_update`, `Step`, `_minibatch` logic in `train.py`.
- Produces:
  - `trainers.ppo.Step` (the dataclass, moved verbatim from train.py:38-49).
  - `trainers.ppo.play_game`, `trainers.ppo.collect_rollout`, `trainers.ppo.ppo_update` (moved verbatim, bodies unchanged).
  - `trainers.ppo.PpoTrainer` with `collect` delegating to `collect_rollout` and `update` delegating to `ppo_update`.
  - `train.Step` remains importable (re-exported from `trainers.ppo`) so nothing downstream breaks.

- [ ] **Step 1: Write the failing test**

Create `pkm/new_agents/agent_000_dragapult/test_ppo_trainer.py`:

```python
import torch

from pkm.new_agents.agent_000_dragapult.config import Config, build_model
from pkm.new_agents.agent_000_dragapult.trainers.ppo import PpoTrainer, Step


def test_ppo_trainer_collect_then_update_smoke():
    cfg = Config()
    torch.manual_seed(0)
    model = build_model(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    trainer = PpoTrainer()

    samples, stats = trainer.collect(model, n_games=2, cfg=cfg)
    assert isinstance(samples, list) and len(samples) > 0
    assert all(isinstance(s, Step) for s in samples)
    assert stats["games"] == 2

    upd = trainer.update(model, opt, samples, cfg)
    for k in ("pol_loss", "val_loss", "entropy", "explained_var"):
        assert k in upd
        assert upd[k] == upd[k]  # not NaN
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd $REPO && uv run pytest $AG/test_ppo_trainer.py -q`
Expected: FAIL — `ModuleNotFoundError: ...trainers.ppo`.

- [ ] **Step 3: Create `trainers/ppo.py` by moving the PPO code**

Create `pkm/new_agents/agent_000_dragapult/trainers/ppo.py`. Move — verbatim, bodies unchanged — the following from `train.py` into it: the `Step` dataclass (train.py:38-49), `play_game` (57-96), `collect_rollout` (99-119), `_minibatch` (127-140), and `ppo_update` (143-219). Keep their imports (`deck`, `policy`, `featurize`, `collate`, `assign_targets`, `battle_*`, `to_observation`, `Config`, `numpy`, `torch`). Then add the trainer wrapper:

```python
class PpoTrainer:
    """Classic PPO + self-play (the v1 baseline), behind the Trainer protocol."""

    def collect(self, model, n_games, cfg, gen=None):
        return collect_rollout(model, n_games, cfg, gen=gen)

    def update(self, model, opt, samples, cfg):
        return ppo_update(model, opt, samples, cfg)
```

Module docstring:

```python
"""PPO + self-play trainer (relocated from train.py, behavior unchanged)."""
```

- [ ] **Step 4: Trim `train.py` and re-export `Step`**

Delete the moved definitions (`Step`, `play_game`, `collect_rollout`, `_minibatch`, `ppo_update`) from `train.py`. At the top of `train.py`, add:

```python
# Back-compat re-export: Step used to live here.
from pkm.new_agents.agent_000_dragapult.trainers.ppo import Step  # noqa: F401
```

(The `train()` driver is rewired in Task 4; it still references `collect_rollout`/`ppo_update` until then, so temporarily import them too:)

```python
from pkm.new_agents.agent_000_dragapult.trainers.ppo import (  # noqa: F401
    Step, collect_rollout, ppo_update,
)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd $REPO && uv run pytest $AG/test_ppo_trainer.py $AG/test_shaping.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd $REPO && git add $AG/trainers/ppo.py $AG/train.py $AG/test_ppo_trainer.py
git commit -m "feat(agent_000_dragapult): relocate PPO into trainers/ppo.py (PpoTrainer)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Slim `train.py` driver + generic parallel pool

**Files:**
- Modify: `pkm/new_agents/agent_000_dragapult/train.py` (the `train()` loop + pool wiring)
- Modify: `pkm/new_agents/agent_000_dragapult/parallel.py` (`_worker`, `ParallelRollout.collect`)
- Test: `pkm/new_agents/agent_000_dragapult/test_driver_parallel.py` (create)

**Interfaces:**
- Consumes: `trainers.get_trainer` (Task 2); `Trainer` (Task 2).
- Produces:
  - `train.train()` picks `trainer = get_trainer(cfg)` and calls `trainer.collect`/`trainer.update`.
  - `ParallelRollout.collect(self, trainer, total_games)` — now takes the trainer.
  - `_worker` builds its own trainer from `cfg.train.method` and calls `trainer.collect`.

- [ ] **Step 1: Write the failing test**

Create `pkm/new_agents/agent_000_dragapult/test_driver_parallel.py`:

```python
import torch

from pkm.new_agents.agent_000_dragapult.config import Config, build_model
from pkm.new_agents.agent_000_dragapult.parallel import ParallelRollout


def test_parallel_collect_takes_trainer_and_returns_samples():
    cfg = Config()
    torch.manual_seed(0)
    model = build_model(cfg)
    from pkm.new_agents.agent_000_dragapult.trainers.ppo import PpoTrainer

    pool = ParallelRollout(cfg, num_workers=2, base_seed=0)
    try:
        samples, stats = pool.collect(PpoTrainer(), total_games=4)
    finally:
        pool.close()
    assert stats["games"] == 4
    assert len(samples) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd $REPO && uv run pytest $AG/test_driver_parallel.py -q`
Expected: FAIL — `collect()` currently takes `(model, total_games)`, so `TypeError`.

- [ ] **Step 3: Make the worker + pool generic in `parallel.py`**

In `_worker` (parallel.py:25-51) replace the `collect_rollout` import + call:

```python
        from pkm.new_agents.agent_000_dragapult.config import Config, build_model
        from pkm.new_agents.agent_000_dragapult import trainers

        cfg = Config.from_dict(cfg_dict)
        model = build_model(cfg)
        trainer = trainers.get_trainer(cfg)
        while True:
            cmd = cmd_q.get()
            if cmd is None:
                break
            state_dict, n_games = cmd
            model.load_state_dict(state_dict)
            t0 = time.perf_counter()
            steps, stats = trainer.collect(model, n_games, cfg)
            stats["t_worker"] = time.perf_counter() - t0
            res_q.put((rank, steps, stats, None))
```

Change `ParallelRollout.collect` signature (parallel.py:79) to `def collect(self, trainer, total_games):` — the trainer is already known to each worker via `cfg.train.method`, so `trainer` here is only used by the single-process fallback callers; keep the parameter for interface symmetry and ignore it inside (workers self-select). Add a one-line comment saying so. The broadcast/gather body is unchanged.

- [ ] **Step 4: Rewire the `train()` driver in `train.py`**

Replace the rollout+update block (train.py:342-348, the `if pool is not None … ppo_update(...)`) with:

```python
            trainer = _trainer  # built once before the loop (see below)
            if pool is not None:
                steps, roll_stats = pool.collect(trainer, games_per_update)
            else:
                steps, roll_stats = trainer.collect(ts.model, games_per_update, cfg)
            t_rollout = time.perf_counter() - t0
            t1 = time.perf_counter()
            upd_stats = trainer.update(ts.model, ts.optimizer, steps, cfg)
            t_update = time.perf_counter() - t1
```

Before the loop (near where `pool` is created, train.py:314) add:

```python
    from pkm.new_agents.agent_000_dragapult.trainers import get_trainer
    _trainer = get_trainer(cfg)
```

Remove the temporary `collect_rollout, ppo_update` from the Task-3 back-compat import (keep only `Step`).

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd $REPO && uv run pytest $AG/test_driver_parallel.py $AG/test_ppo_trainer.py -q`
Expected: PASS.

- [ ] **Step 6: End-to-end PPO smoke through the driver (single + parallel)**

Run: `cd $REPO && uv run python -m pkm.new_agents.agent_000_dragapult.cli train -e _driver_smoke -o "$TMPDIR/exit_plan" --updates 2 --games 4 --workers 2 --eval-every 0 --force 2>&1 | tail -4`
Expected: two update lines print with the `util`/`core` columns; no error.
Cleanup: `rm -rf "$TMPDIR/exit_plan"`

- [ ] **Step 7: Commit**

```bash
cd $REPO && git add $AG/train.py $AG/parallel.py $AG/test_driver_parallel.py
git commit -m "feat(agent_000_dragapult): method-agnostic train() driver + generic parallel collect

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Surface the search seam in `cabt.py` + characterize the engine search API

**Files:**
- Modify: `pkm/new_agents/agent_000_dragapult/cabt.py` (re-export + typed wrappers)
- Test: `pkm/new_agents/agent_000_dragapult/test_search_seam.py` (create, marked `slow`)

**Interfaces:**
- Consumes: `pkm.engine.search_begin/search_step/search_end/search_release` (already exposed); `cabt.SearchState`, `cabt.to_observation`.
- Produces:
  - `cabt.search_begin(observation, your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active, manual_coin=False) -> SearchState`
  - `cabt.search_step(search_id: int, select: list[int]) -> SearchState`
  - `cabt.search_end() -> None`, `cabt.search_release(search_id: int) -> None`
  - A committed characterization of `search_step` semantics: **does stepping the same `search_id` twice with different selects yield two distinct persistent `searchId`s (branching tree) or mutate one cursor?** The MCTS in Task 7 consumes this finding.

- [ ] **Step 1: Write the characterization test**

Create `pkm/new_agents/agent_000_dragapult/test_search_seam.py`:

```python
import pytest

from pkm.new_agents.agent_000_dragapult import cabt, deck


pytestmark = pytest.mark.slow


def _fresh_root():
    obs, _ = cabt.battle_start(deck.DECK_60, deck.DECK_60)
    # advance through deck-selection until a real choice is presented
    n = 0
    while obs["select"] is None or obs["current"] is None:
        obs = cabt.battle_select(list(deck.DECK_60))
        n += 1
        if n > 50:
            break
    return obs


def test_search_begin_returns_state_with_options():
    obs = _fresh_root()
    if obs["current"]["result"] >= 0:
        cabt.battle_finish()
        pytest.skip("game ended during setup")
    o = cabt.to_observation(obs)
    seat = o.current.yourIndex
    # single-sample determinization is Task 6; here just pass the known deck
    st = cabt.search_begin(
        obs,
        your_deck=[], your_prize=[],
        opponent_deck=list(deck.DECK_60),
        opponent_prize=[], opponent_hand=[], opponent_active=[],
    )
    assert isinstance(st.searchId, int)
    assert st.observation.select is not None
    cabt.search_end()
    cabt.battle_finish()


def test_search_step_branching_semantics():
    """Pin down whether two steps from the same node persist as distinct nodes."""
    obs = _fresh_root()
    if obs["current"]["result"] >= 0:
        cabt.battle_finish()
        pytest.skip("game ended during setup")
    st = cabt.search_begin(
        obs, your_deck=[], your_prize=[],
        opponent_deck=list(deck.DECK_60),
        opponent_prize=[], opponent_hand=[], opponent_active=[],
    )
    root_id = st.searchId
    n_opts = len(st.observation.select.option)
    child_a = cabt.search_step(root_id, [0])
    # Step the ROOT again with a different option; record whether it works and
    # whether the returned searchId differs from child_a.
    branched = False
    try:
        child_b = cabt.search_step(root_id, [min(1, n_opts - 1)])
        branched = child_b.searchId != child_a.searchId
    except Exception:
        branched = False
    cabt.search_end()
    cabt.battle_finish()
    # Lock in the observed behavior so Task 7 can rely on it. Update the
    # asserted value to match reality on first run, then keep it as a guard.
    assert branched in (True, False)  # characterization: record actual below
    print(f"SEARCH_STEP_BRANCHES={branched}")
```

- [ ] **Step 2: Run it to characterize (expect import failure first)**

Run: `cd $REPO && uv run pytest $AG/test_search_seam.py -q -s`
Expected: FAIL — `AttributeError: module 'cabt' has no attribute 'search_begin'`.

- [ ] **Step 3: Add the re-exports/wrappers to `cabt.py`**

`pkm.engine.search_begin` returns a raw dict-backed state; wrap it into the typed `cabt.SearchState`. Near the existing `battle_*` re-export (cabt.py:26) add imports, and after the `SearchState` dataclass add typed wrappers:

```python
from pkm.engine import (  # noqa: F401
    search_begin as _engine_search_begin,
    search_step as _engine_search_step,
    search_end as _engine_search_end,
    search_release as _engine_search_release,
)


def _to_search_state(raw) -> SearchState:
    # pkm.engine returns an object/dict exposing .observation (raw obs dict)
    # and .searchId; normalize to the typed cabt.SearchState.
    obs = raw.observation if hasattr(raw, "observation") else raw["observation"]
    sid = raw.searchId if hasattr(raw, "searchId") else raw["searchId"]
    return SearchState(observation=to_observation(obs), searchId=int(sid))


def search_begin(observation, your_deck, your_prize, opponent_deck,
                 opponent_prize, opponent_hand, opponent_active,
                 manual_coin=False) -> SearchState:
    return _to_search_state(_engine_search_begin(
        observation, your_deck, your_prize, opponent_deck, opponent_prize,
        opponent_hand, opponent_active, manual_coin))


def search_step(search_id: int, select: list[int]) -> SearchState:
    return _to_search_state(_engine_search_step(search_id, select))


def search_end() -> None:
    _engine_search_end()


def search_release(search_id: int) -> None:
    _engine_search_release(search_id)
```

Verify the exact return shape of `pkm.engine.search_begin` first (`uv run python -c "import pkm.engine as e; help(e.search_begin)"`); adjust `_to_search_state` to the real attribute/dict access if needed. The engine's typed `SearchState` (engine/api.py) wraps `{"observation":…, "searchId":…}`.

- [ ] **Step 4: Run the characterization test and record the finding**

Run: `cd $REPO && uv run pytest $AG/test_search_seam.py -q -s`
Expected: PASS; capture the printed `SEARCH_STEP_BRANCHES=True|False`.
Then edit the final assertion to lock the observed value, e.g. `assert branched is True`, and add a one-line comment in `mcts.py`'s eventual header referencing it. **This finding decides Task 7's tree representation** (see Task 7 Step 0).

- [ ] **Step 5: Commit**

```bash
cd $REPO && git add $AG/cabt.py $AG/test_search_seam.py
git commit -m "feat(agent_000_dragapult): expose typed search_* seam in cabt + characterize search_step

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Single-sample determinization

**Files:**
- Create: `pkm/new_agents/agent_000_dragapult/determinize.py`
- Test: `pkm/new_agents/agent_000_dragapult/test_determinize.py` (create)

**Interfaces:**
- Consumes: `deck.DECK_60`; a raw observation dict (`obs["current"]["players"][i]` with `deckCount`, `handCount`, `prize`, `active`).
- Produces:
  - `determinize.sample_world(obs: dict, seat: int, gen) -> Predictions` where `Predictions` is a dataclass/namedtuple with fields matching `cabt.search_begin`'s hidden-state args: `your_deck, your_prize, opponent_deck, opponent_prize, opponent_hand, opponent_active` (each `list[int]`).
  - `determinize.DETERMINIZERS: dict[str, Callable]` keyed by `cfg.train.determinization` (`"sample"` registered).

- [ ] **Step 1: Write the failing test**

Create `pkm/new_agents/agent_000_dragapult/test_determinize.py`:

```python
import collections
import torch

from pkm.new_agents.agent_000_dragapult import deck
from pkm.new_agents.agent_000_dragapult.determinize import sample_world


def _fake_obs(seat=0):
    # Minimal shape sample_world reads: per-player counts + visible zones.
    other = 1 - seat
    players = [None, None]
    players[seat] = {
        "deckCount": 30, "handCount": 5, "prize": [None] * 6,
        "active": [1], "bench": [], "discard": [],
        "hand": [2, 3, 4, 5, 6],
    }
    players[other] = {
        "deckCount": 32, "handCount": 4, "prize": [None] * 6,
        "active": [7], "bench": [], "discard": [],
        "hand": [None] * 4,  # hidden from `seat`
    }
    return {"current": {"yourIndex": seat, "players": players}}


def test_sample_world_respects_opponent_counts():
    gen = torch.Generator().manual_seed(0)
    obs = _fake_obs(seat=0)
    w = sample_world(obs, seat=0, gen=gen)
    opp = obs["current"]["players"][1]
    assert len(w.opponent_hand) == opp["handCount"]
    assert len(w.opponent_deck) >= opp["deckCount"]
    assert len(w.opponent_prize) == len(opp["prize"])


def test_sampled_cards_come_from_known_deck_multiset():
    gen = torch.Generator().manual_seed(1)
    w = sample_world(_fake_obs(seat=0), seat=0, gen=gen)
    known = collections.Counter(deck.DECK_60)
    used = collections.Counter(w.opponent_hand + w.opponent_prize)
    # never assign more copies of a card than exist in the known deck
    for card_id, cnt in used.items():
        assert cnt <= known[card_id]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd $REPO && uv run pytest $AG/test_determinize.py -q`
Expected: FAIL — `ModuleNotFoundError: ...determinize`.

- [ ] **Step 3: Implement `determinize.py`**

Create `pkm/new_agents/agent_000_dragapult/determinize.py`:

```python
"""Single-sample (K=1) determinization of the opponent's hidden zones.

Both seats play the known DECK_60, so the hidden cards are a subset of that
multiset. From the acting seat's view we remove everything publicly visible,
then randomly deal the remainder into the opponent's deck/hand/prize by their
observed counts. This is honest IS-MCTS with K=1; full IS-MCTS averages over
K such worlds (logged upgrade). See docs/specs §5.3.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass

import torch

from pkm.new_agents.agent_000_dragapult import deck


@dataclass(frozen=True)
class Predictions:
    your_deck: list[int]
    your_prize: list[int]
    opponent_deck: list[int]
    opponent_prize: list[int]
    opponent_hand: list[int]
    opponent_active: list[int]


def _visible_card_ids(player: dict) -> list[int]:
    """Card IDs known to be NOT in the hidden pool (in play/discard/own hand)."""
    ids: list[int] = []
    for zone in ("active", "bench", "discard", "hand"):
        for c in player.get(zone, []) or []:
            if isinstance(c, int):
                ids.append(c)
            elif isinstance(c, dict) and isinstance(c.get("id"), int):
                ids.append(c["id"])
    return ids


def sample_world(obs: dict, seat: int, gen: torch.Generator) -> Predictions:
    state = obs["current"]
    me = state["players"][seat]
    opp = state["players"][1 - seat]

    known = collections.Counter(deck.DECK_60)
    # Remove everything the acting seat can see (its own zones + opponent's
    # face-up cards). Whatever remains could be in the opponent's hidden pool.
    for cid in _visible_card_ids(me):
        if known[cid] > 0:
            known[cid] -= 1
    for cid in _visible_card_ids(opp):  # opponent's own face-up cards
        if known[cid] > 0:
            known[cid] -= 1

    pool = [c for c, n in known.items() for _ in range(n)]
    order = torch.randperm(len(pool), generator=gen).tolist()
    pool = [pool[i] for i in order]

    n_hand = opp["handCount"]
    n_prize = len(opp["prize"])
    opp_hand = pool[:n_hand]
    opp_prize = pool[n_hand:n_hand + n_prize]
    opp_deck = pool[n_hand + n_prize:]

    return Predictions(
        your_deck=[], your_prize=[],
        opponent_deck=opp_deck,
        opponent_prize=opp_prize,
        opponent_hand=opp_hand,
        opponent_active=[],
    )


DETERMINIZERS = {"sample": sample_world}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd $REPO && uv run pytest $AG/test_determinize.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
cd $REPO && git add $AG/determinize.py $AG/test_determinize.py
git commit -m "feat(agent_000_dragapult): single-sample (K=1) determinization

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: `mcts.py` — PUCT search over the engine forward model

**Files:**
- Create: `pkm/new_agents/agent_000_dragapult/mcts.py`
- Test: `pkm/new_agents/agent_000_dragapult/test_mcts.py` (create, marked `slow`)

**Interfaces:**
- Consumes: `cabt.search_begin/search_step/search_end` + `SearchState` (Task 5); `determinize.sample_world` (Task 6); `featurize`, `collate`, `model.evaluate` (existing).
- Produces:
  - `mcts.search(root_obs: dict, seat: int, model, cfg, gen) -> np.ndarray` — returns the root visit-count policy `π` (length = number of root options, sums to 1, mass only on legal options).

- [ ] **Step 0: Consume the Task-5 finding**

Read the recorded `SEARCH_STEP_BRANCHES` value from Task 5.
- If **True** (engine persists distinct child nodes by `searchId`): store children by `searchId`; each simulation descends via `search_step` and revisits persisted nodes.
- If **False** (single mutable cursor): each simulation re-roots via `search_begin` and replays the selected action path with `search_step` from the root. Record the path (list of option indices) per node.

Write the implementation for the observed case; keep the other branch out (YAGNI). The test below is agnostic to which representation is used.

- [ ] **Step 1: Write the failing test**

Create `pkm/new_agents/agent_000_dragapult/test_mcts.py`:

```python
import numpy as np
import pytest
import torch

from pkm.new_agents.agent_000_dragapult import cabt, deck, mcts
from pkm.new_agents.agent_000_dragapult.config import Config, build_model

pytestmark = pytest.mark.slow


def _root_obs():
    obs, _ = cabt.battle_start(deck.DECK_60, deck.DECK_60)
    n = 0
    while obs["select"] is None or obs["current"] is None:
        obs = cabt.battle_select(list(deck.DECK_60))
        n += 1
        if n > 50:
            break
    return obs


def test_search_returns_valid_policy():
    cfg = Config()
    object.__setattr__(cfg.train, "mcts_simulations", 8)
    torch.manual_seed(0)
    model = build_model(cfg)
    obs = _root_obs()
    if obs["current"]["result"] >= 0:
        cabt.battle_finish()
        pytest.skip("game ended during setup")
    seat = obs["current"]["yourIndex"]
    gen = torch.Generator().manual_seed(0)
    pi = mcts.search(obs, seat, model, cfg, gen)
    cabt.battle_finish()

    n_opts = len(obs["select"]["option"])
    assert pi.shape == (n_opts,)
    assert np.isclose(pi.sum(), 1.0, atol=1e-5)
    assert (pi >= 0).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd $REPO && uv run pytest $AG/test_mcts.py -q`
Expected: FAIL — `ModuleNotFoundError: ...mcts`.

- [ ] **Step 3: Implement `mcts.py` (branching-tree case shown; adapt per Step 0)**

Create `pkm/new_agents/agent_000_dragapult/mcts.py`:

```python
"""PUCT MCTS over the engine's search_* forward model, guided by model.evaluate.

Node = one engine search node (searchId + observation). Value is backed up
negamax (sign flips when the child's acting seat differs). v1 uses single-sample
determinization (K=1) and lets the engine resolve chance inside search_step
(no explicit chance nodes) — see docs/specs §5.2, §5.3, §7.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from pkm.new_agents.agent_000_dragapult import cabt
from pkm.new_agents.agent_000_dragapult.determinize import DETERMINIZERS
from pkm.new_agents.agent_000_dragapult.features import featurize
from pkm.new_agents.agent_000_dragapult.model import collate


class _Node:
    __slots__ = ("search_id", "obs", "seat", "n_opts", "P", "N", "W", "children", "terminal_v")

    def __init__(self, state: cabt.SearchState):
        self.search_id = state.searchId
        self.obs = state.observation
        self.seat = self.obs.current.yourIndex if self.obs.current else 0
        self.n_opts = len(self.obs.select.option) if self.obs.select else 0
        self.P = np.zeros(self.n_opts, dtype=np.float32)
        self.N = np.zeros(self.n_opts, dtype=np.float32)
        self.W = np.zeros(self.n_opts, dtype=np.float32)
        self.children: dict[int, "_Node"] = {}
        self.terminal_v: float | None = None


def _evaluate(node: _Node, model, cfg) -> float:
    """Fill node.P from priors; return the node's value estimate in [-1, 1]."""
    if node.obs.current is not None and node.obs.current.result >= 0:
        # terminal: +1 if the node's seat won, else -1
        node.terminal_v = 1.0 if node.obs.current.result == node.seat else -1.0
        return node.terminal_v
    f = featurize(node.obs)
    b = collate([f])
    with torch.no_grad():
        priors, value = model.evaluate(b)  # (softmax over options, scalar)
    p = priors[0, : node.n_opts].cpu().numpy().astype(np.float32)
    s = p.sum()
    node.P = p / s if s > 0 else np.full(node.n_opts, 1.0 / max(node.n_opts, 1), np.float32)
    return float(value[0])


def _select(node: _Node, c_puct: float) -> int:
    sqrt_total = math.sqrt(max(node.N.sum(), 1.0))
    q = np.where(node.N > 0, node.W / np.maximum(node.N, 1), 0.0)
    u = c_puct * node.P * sqrt_total / (1.0 + node.N)
    return int(np.argmax(q + u))


def search(root_obs: dict, seat: int, model, cfg, gen: torch.Generator) -> np.ndarray:
    determinize = DETERMINIZERS[cfg.train.determinization]
    world = determinize(root_obs, seat, gen)
    root_state = cabt.search_begin(
        root_obs,
        your_deck=world.your_deck, your_prize=world.your_prize,
        opponent_deck=world.opponent_deck, opponent_prize=world.opponent_prize,
        opponent_hand=world.opponent_hand, opponent_active=world.opponent_active,
    )
    root = _Node(root_state)
    _evaluate(root, model, cfg)
    c_puct = cfg.train.mcts_c_puct

    try:
        for _ in range(cfg.train.mcts_simulations):
            path: list[tuple[_Node, int]] = []
            node = root
            # descend to a leaf
            while True:
                if node.n_opts == 0 or node.terminal_v is not None:
                    break
                a = _select(node, c_puct)
                path.append((node, a))
                if a in node.children:
                    node = node.children[a]
                    continue
                child = _Node(cabt.search_step(node.search_id, [a]))
                node.children[a] = child
                leaf_v = _evaluate(child, model, cfg)
                node = child
                break
            else:  # pragma: no cover
                leaf_v = 0.0
            if not path:
                break
            v = node.terminal_v if node.terminal_v is not None else leaf_v
            # negamax backup: flip sign when the edge's parent seat differs from leaf
            for parent, a in reversed(path):
                signed = v if parent.seat == node.seat else -v
                parent.N[a] += 1
                parent.W[a] += signed
    finally:
        cabt.search_end()

    if root.N.sum() == 0:  # no sims expanded (e.g. terminal root)
        pi = np.full(root.n_opts, 1.0 / max(root.n_opts, 1), np.float32)
        return pi
    tau = cfg.train.mcts_temperature
    counts = root.N ** (1.0 / tau) if tau > 0 else (root.N == root.N.max())
    return (counts / counts.sum()).astype(np.float32)
```

If Step 0 found `SEARCH_STEP_BRANCHES=False`, replace the descend loop with a
re-root-and-replay variant: keep `path` as option indices from root, and each
simulation calls `cabt.search_begin(...)` then `cabt.search_step` along the
recorded path before expanding. Everything else is unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd $REPO && uv run pytest $AG/test_mcts.py -q`
Expected: PASS (1 passed, or skipped if setup ends the game — re-run seeds until it exercises).

- [ ] **Step 5: Commit**

```bash
cd $REPO && git add $AG/mcts.py $AG/test_mcts.py
git commit -m "feat(agent_000_dragapult): PUCT MCTS over the engine search forward model

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: `trainers/exit.py` — `ExItTrainer` + `ExItSample` + imitation update

**Files:**
- Create: `pkm/new_agents/agent_000_dragapult/trainers/exit.py`
- Test: `pkm/new_agents/agent_000_dragapult/test_exit_trainer.py` (create, marked `slow`)

**Interfaces:**
- Consumes: `mcts.search` (Task 7); `Trainer` (Task 2); `featurize`, `collate`, `battle_start/select/finish`, `to_observation`, `policy.select_count`; `shaping._seat_reward` for the ±1 outcome.
- Produces:
  - `trainers.exit.ExItSample` dataclass: `features`, `policy_target: np.ndarray`, `value_target: float` (filled at game end), `seat: int`.
  - `trainers.exit.ExItTrainer` with `collect` (MCTS self-play → samples) and `update` (CE(policy) + MSE(value)).

- [ ] **Step 1: Write the failing test**

Create `pkm/new_agents/agent_000_dragapult/test_exit_trainer.py`:

```python
import pytest
import torch

from pkm.new_agents.agent_000_dragapult.config import Config, build_model
from pkm.new_agents.agent_000_dragapult.trainers.exit import ExItTrainer, ExItSample

pytestmark = pytest.mark.slow


def test_exit_collect_then_update_smoke():
    cfg = Config()
    object.__setattr__(cfg.train, "method", "exit")
    object.__setattr__(cfg.train, "mcts_simulations", 6)
    torch.manual_seed(0)
    model = build_model(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    trainer = ExItTrainer()

    samples, stats = trainer.collect(model, n_games=1, cfg=cfg)
    assert len(samples) > 0
    assert all(isinstance(s, ExItSample) for s in samples)
    # value targets filled to ±1 (zero-sum) or 0 (draw)
    assert all(s.value_target in (-1.0, 0.0, 1.0) for s in samples)

    upd = trainer.update(model, opt, samples, cfg)
    assert "policy_loss" in upd and upd["policy_loss"] == upd["policy_loss"]
    assert "value_loss" in upd and upd["value_loss"] == upd["value_loss"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd $REPO && uv run pytest $AG/test_exit_trainer.py -q`
Expected: FAIL — `ModuleNotFoundError: ...trainers.exit`.

- [ ] **Step 3: Implement `trainers/exit.py`**

Create `pkm/new_agents/agent_000_dragapult/trainers/exit.py`:

```python
"""MCTS expert-iteration trainer.

collect: self-play where each acting decision runs MCTS (guided by the net) to
produce an improved policy target π; the played move is sampled from π. At game
end the ±1 outcome becomes the value target for that seat's samples.
update: supervised — cross-entropy(policy, π) + value_coef · MSE(value, z).
See docs/specs §5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from pkm.new_agents.agent_000_dragapult import deck, mcts, policy
from pkm.new_agents.agent_000_dragapult.cabt import (
    battle_finish, battle_select, battle_start, to_observation,
)
from pkm.new_agents.agent_000_dragapult.features import Features, featurize
from pkm.new_agents.agent_000_dragapult.model import collate
from pkm.new_agents.agent_000_dragapult.shaping import _seat_reward


@dataclass
class ExItSample:
    features: Features
    policy_target: np.ndarray  # over the node's options; sums to 1
    seat: int
    value_target: float = 0.0  # filled at game end


def _play_game(model, cfg, gen) -> tuple[list[ExItSample], int]:
    samples: list[ExItSample] = []
    obs, _ = battle_start(deck.DECK_60, deck.DECK_60)
    n_iter = 0
    while obs["current"]["result"] < 0 and n_iter < 100000:
        if obs["select"] is None or obs["current"] is None:
            obs = battle_select(list(deck.DECK_60)); n_iter += 1; continue
        o = to_observation(obs)
        n = len(o.select.option)
        if n == 0:
            obs = battle_select([]); n_iter += 1; continue
        seat = obs["current"]["yourIndex"]
        pi = mcts.search(obs, seat, model, cfg, gen)  # [n]
        samples.append(ExItSample(features=featurize(o), policy_target=pi, seat=seat))
        # play a move sampled from π (respecting the multi-select count)
        k = policy.select_count(o.select.minCount, o.select.maxCount, n)
        idx = torch.multinomial(torch.from_numpy(pi), k, replacement=False, generator=gen)
        obs = battle_select(idx.tolist()); n_iter += 1
    result = obs["current"]["result"]
    battle_finish()
    for s in samples:  # Monte-Carlo value target = game outcome
        s.value_target = _seat_reward(result, s.seat)
    return samples, result


class ExItTrainer:
    def collect(self, model, n_games, cfg, gen=None):
        model.eval()
        gen = gen or torch.Generator().manual_seed(cfg.train.seed)
        samples: list[ExItSample] = []
        results = []
        for _ in range(n_games):
            s, r = _play_game(model, cfg, gen)
            samples.extend(s); results.append(r)
        denom = max(n_games, 1)
        stats = {
            "games": n_games, "steps": len(samples),
            "p0_win": results.count(0) / denom, "p1_win": results.count(1) / denom,
        }
        return samples, stats

    def update(self, model, opt, samples, cfg):
        model.train()
        tc = cfg.train
        idx = np.arange(len(samples))
        rng = np.random.default_rng(tc.seed)
        agg = {"policy_loss": 0.0, "value_loss": 0.0, "n": 0}
        for _ in range(tc.epochs_per_update):
            rng.shuffle(idx)
            for start in range(0, len(idx), tc.minibatch_size):
                mb = [samples[i] for i in idx[start:start + tc.minibatch_size]]
                if not mb:
                    continue
                b = collate([s.features for s in mb])
                logits, value = model(b)  # logits [B,L], value [B]
                logp = torch.log_softmax(logits.masked_fill(b["option_mask"] == 0, -1e9), dim=-1)
                L = logits.shape[1]
                tgt = torch.zeros(len(mb), L)
                for i, s in enumerate(mb):
                    tgt[i, : len(s.policy_target)] = torch.from_numpy(s.policy_target)
                policy_loss = -(tgt * logp).sum(dim=-1).mean()
                z = torch.tensor([s.value_target for s in mb], dtype=torch.float32)
                value_loss = F.mse_loss(value, z)
                loss = policy_loss + tc.value_coef * value_loss
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc.max_grad_norm)
                opt.step()
                agg["policy_loss"] += float(policy_loss)
                agg["value_loss"] += float(value_loss)
                agg["n"] += 1
        n = max(agg["n"], 1)
        return {
            "policy_loss": agg["policy_loss"] / n,
            "value_loss": agg["value_loss"] / n,
            "pol_loss": agg["policy_loss"] / n,   # alias for the console/CSV sinks
            "val_loss": agg["value_loss"] / n,
            "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0,
            "grad_norm": 0.0, "explained_var": 0.0,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd $REPO && uv run pytest $AG/test_exit_trainer.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
cd $REPO && git add $AG/trainers/exit.py $AG/test_exit_trainer.py
git commit -m "feat(agent_000_dragapult): ExItTrainer (MCTS self-play + imitation update)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: CLI `--method`/`--mcts-*` + `train-exit` recipe + end-to-end

**Files:**
- Modify: `pkm/new_agents/agent_000_dragapult/cli.py` (train command options → `TrainConfig`)
- Modify: `pkm/new_agents/agent_000_dragapult/justfile` (add `train-exit`)
- Test: `pkm/new_agents/agent_000_dragapult/test_cli_method.py` (create)

**Interfaces:**
- Consumes: everything above; the existing `train` command's `TrainConfig` construction (cli.py:~196-210).
- Produces: `train --method exit --mcts-simulations N …` flows into `TrainConfig`; `just train-exit train <exp> …` runs an ExIt run.

- [ ] **Step 1: Write the failing test**

Create `pkm/new_agents/agent_000_dragapult/test_cli_method.py`:

```python
from typer.testing import CliRunner

from pkm.new_agents.agent_000_dragapult.cli import app

runner = CliRunner()


def test_train_help_lists_method_and_mcts_options():
    res = runner.invoke(app, ["train", "--help"])
    assert res.exit_code == 0
    assert "--method" in res.output
    assert "--mcts-simulations" in res.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd $REPO && uv run pytest $AG/test_cli_method.py -q`
Expected: FAIL — `--method` not in help.

- [ ] **Step 3: Add the CLI options**

In the `train` command signature (cli.py:436-462) add options:

```python
    method: str = typer.Option("ppo", help="Training method: 'ppo' or 'exit'."),
    mcts_simulations: int = typer.Option(32, help="MCTS simulations per move (exit)."),
    mcts_c_puct: float = typer.Option(1.25, help="PUCT exploration constant (exit)."),
    mcts_temperature: float = typer.Option(1.0, help="Visit-count temperature (exit)."),
    determinization: str = typer.Option("sample", help="Hidden-state determinizer (exit)."),
```

Where the `train` command builds `TrainConfig` (cli.py:~196), add the new fields:

```python
        method=method,
        mcts_simulations=mcts_simulations,
        mcts_c_puct=mcts_c_puct,
        mcts_temperature=mcts_temperature,
        determinization=determinization,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd $REPO && uv run pytest $AG/test_cli_method.py -q`
Expected: PASS.

- [ ] **Step 5: Add the `train-exit` recipe to the justfile**

After the `train-opt-00` recipe in `pkm/new_agents/agent_000_dragapult/justfile` add:

```make
# MCTS expert-iteration self-play. Same mode contract as train-fast: `train`
# (fresh; injects --method exit + MCTS knobs) or `resume` (config restored from
# the checkpoint, so the flags must NOT be re-passed). e.g.
# `just train-exit train exit0 100 8` then `just train-exit resume exit0 50 8`.
train-exit mode="train" exp=exp updates="128" games="8" workers="8" engine="local-nix" force="":
    cd {{root}} && {{cli}} {{mode}} --experiment {{exp}} --updates {{updates}} --games {{games}} --workers {{workers}} {{ if mode == "train" { "--method exit --mcts-simulations 32" } else { "" } }} --engine {{engine}} {{ if mode == "train" { force } else { "" } }}
```

- [ ] **Step 6: End-to-end ExIt run (tiny)**

Run: `cd $REPO && uv run python -m pkm.new_agents.agent_000_dragapult.cli train -e _exit_e2e -o "$TMPDIR/exit_e2e" --method exit --mcts-simulations 6 --updates 2 --games 2 --workers 2 --eval-every 0 --force 2>&1 | tail -5`
Expected: two update lines print (policy/value losses finite); checkpoint written; no error.
Cleanup: `rm -rf "$TMPDIR/exit_e2e"`

- [ ] **Step 7: Verify PPO still resumes the real checkpoint (back-compat gate)**

Run: `cd $REPO && uv run python -m pkm.new_agents.agent_000_dragapult.cli info -e 000_default 2>&1 | tail -5`
Then a 1-update dry resume into a COPY (never mutate the real run):
```bash
cd $REPO && cp -r pkm_data/new_agents/agent_000_dragapult/experiments/000_default "$TMPDIR/000_default_copy"
uv run python -m pkm.new_agents.agent_000_dragapult.cli resume -e 000_default_copy \
  -o "$TMPDIR" --updates 1 --games 4 --workers 2 --eval-every 0 2>&1 | tail -5
rm -rf "$TMPDIR/000_default_copy" "$TMPDIR/experiments"
```
Expected: resumes from update 458 without "config hash mismatch"; prints update 459.

- [ ] **Step 8: Full agent test suite green**

Run: `cd $REPO && uv run pytest $AG/ -q`
Expected: PASS (fast tests; `slow`-marked search/MCTS/exit tests included).

- [ ] **Step 9: Commit**

```bash
cd $REPO && git add $AG/cli.py $AG/justfile $AG/test_cli_method.py
git commit -m "feat(agent_000_dragapult): CLI --method/--mcts-* + train-exit recipe

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- §3.1 Trainer protocol + registry → Task 2. §3.2 slim driver → Task 4. §3.3 generic parallel → Task 4. §3.4 config method + hash migration → Task 1.
- §4 PpoTrainer relocation + regression → Task 3.
- §5.1 cabt search seam → Task 5. §5.2 MCTS core → Task 7. §5.3 single-sample determinization → Task 6. §5.4 ExItTrainer + loss → Task 8. §5.5 ExItSample → Task 8.
- §6.1 config knobs → Task 1. §6.2 CLI + justfile → Task 9. §6.3 tests → PPO regression (Task 3), back-compat (Task 1 + Task 9 Step 7), MCTS unit (Task 7), ExIt smoke (Task 8), parallel (Task 4).
- §7 limitations → logged in code headers (determinize.py, mcts.py) + spec; no code owed.
- §8 file manifest → matches the Create/Modify lists across tasks.

**Placeholder scan:** no TBD/TODO; every code step shows complete code; the one genuine unknown (search_step branching) is resolved empirically in Task 5 and consumed in Task 7 Step 0, not hand-waved.

**Type consistency:** `Trainer.collect(model, n_games, cfg, gen=None) -> (list, dict)` and `.update(model, opt, samples, cfg) -> dict` are used identically in Tasks 3, 4, 8. `mcts.search(root_obs, seat, model, cfg, gen) -> np.ndarray` defined in Task 7, called with the same signature in Task 8. `determinize.sample_world(obs, seat, gen) -> Predictions` defined in Task 6, called in Task 7. `ExItSample(features, policy_target, seat, value_target=0.0)` consistent between Task 8 definition and test. `_hash_dict` defined in Task 1, used in Task 1's `train.py` guard.

**Known risk flagged for the executor:** Task 7 depends on the Task 5 characterization; if the engine mutates a single search cursor (`SEARCH_STEP_BRANCHES=False`), use the re-root-and-replay variant noted in Task 7 Step 3 (valid only where transitions are deterministic; chance is resolved inside `search_step`, so replay of a fixed determinization is consistent within one search).
