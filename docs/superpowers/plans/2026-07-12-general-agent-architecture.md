# General Agent Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace name-specific agent wiring with a profile-driven interface that can train, run, evaluate, and package multiple independent agents while preserving the working `02_dragapult` neural Kaggle submission.

**Architecture:** `AgentProfile` owns identity, deck, policy configuration, checkpoints, and output paths. A policy registry creates plain callable runtime agents; trainer implementations consume profiles and return checkpoints; a submission builder creates a minimal module-level `agent` entry point. Local games and evaluation receive two independently resolved profiles instead of one shared deck and weight path.

**Tech Stack:** Python 3.12, dataclasses, Typer, PyTorch, NumPy inference, kaggle-environments, pytest, YAML profile configuration.

---

## Guardrails

- Keep `02_dragapult` as the default profile throughout the migration.
- Keep the Kaggle contract as `agent(obs) -> list[int]` exposed at module scope.
- Do not put training, local battles, or CLI code in generated submission `main.py`.
- Do not change PPO math while introducing the interfaces; refactor callers first.
- Every phase must leave `python -m pytest tests/ -q` passing.
- Build a real `02_dragapult` submission before changing the multi-agent trainer.

## File Map

### New files

- `pkm/agents/spec.py` — immutable profile configuration and resolved paths.
- `pkm/agents/registry.py` — policy and trainer registration tables.
- `pkm/agents/factory.py` — profile-to-callable runtime construction.
- `pkm/submission.py` — profile submission packaging and generated entry point.
- `tests/test_agent_profile.py` — profile loading, paths, and validation.
- `tests/test_agent_factory.py` — policy factory and callable behavior.
- `tests/test_submission.py` — generated bundle and module-level `agent` contract.
- `tests/test_multi_agent_play.py` — independent decks and policy instances.
- `agents/02_dragapult/profile.yaml` — explicit default profile configuration.
- `agents/02_dragapult/deck.csv` — the profile-owned 60-card deck data.

### Existing files to modify

- `pkm/agents/profile.py` — retain compatibility while delegating to `AgentSpec`.
- `pkm/agents/neural_agent.py` — expose strict weight resolution for configured profiles.
- `pkm/agents/__init__.py` — export the public profile/factory API.
- `pkm/rl/train.py` — expose a trainer function that accepts a resolved profile.
- `pkm/rl/exit_train.py` — expose the same trainer contract for expert iteration.
- `pkm/rl/play.py` — resolve two profiles independently and support profile names.
- `pkm/rl/rollout.py` — accept separate policies and decks without profile knowledge.
- `pkm/cli/__init__.py` — add profile train/build-submit commands.
- `submit.sh` — become a compatibility wrapper around the submission builder.
- `justfile` — route train/build/upload/log recipes through profile paths.
- `AGENTS.md` — document the profile lifecycle and multi-agent commands.

## Task 1: Define Profile Configuration And Paths

**Files:**
- Create: `agents/02_dragapult/profile.yaml`
- Create: `pkm/agents/spec.py`
- Modify: `pkm/agents/profile.py`
- Modify: `pkm/agents/__init__.py`
- Create: `tests/test_agent_profile.py`

- [ ] **Step 1: Write failing profile tests.**

Test that the default profile resolves the existing deck, policy, checkpoint,
metrics, runs, and submission paths without depending on the current working
directory:

```python
def test_dragapult_profile_resolves_paths():
    profile = AgentProfile.load("02_dragapult")

    assert profile.name == "02_dragapult"
    assert profile.deck_path == Path("deck/02_dragapult.csv")
    assert profile.policy == "neural"
    assert profile.trainer == "ppo"
    assert profile.checkpoint_path.name == "ppo_latest.pt"
```

- [ ] **Step 2: Run the profile test and verify it fails.**

Run: `python -m pytest tests/test_agent_profile.py -q`

Expected: FAIL because `AgentProfile.load`, profile configuration loading, and
the resolved path properties do not exist.

- [ ] **Step 3: Add the explicit profile configuration.**

Create `agents/02_dragapult/deck.csv` by copying the current Dragapult deck
data, then create `agents/02_dragapult/profile.yaml`:

```yaml
name: 02_dragapult
deck: agents/02_dragapult/deck.csv
policy: neural
trainer: ppo
checkpoint: agents/02_dragapult/checkpoints/ppo_latest.pt
strategy: null
```

- [ ] **Step 4: Implement `AgentSpec` and profile loading.**

Define a frozen dataclass with `name`, `deck_path`, `policy`, `trainer`,
`strategy`, and `checkpoint_path`. Load YAML relative to the repository root,
reject unknown required fields, and expose `metrics_dir`, `runs_dir`, and
`submissions_dir`. `load_deck()` must validate exactly 60 card IDs and return a
new list for each caller.
Keep `AgentProfile(name)` as a compatibility constructor that calls
`AgentProfile.load(name)` internally.

- [ ] **Step 5: Run the profile tests.**

Run: `python -m pytest tests/test_agent_profile.py -q`

Expected: PASS, including tests for a missing profile and a missing deck with a
clear `FileNotFoundError`.

- [ ] **Step 6: Commit the profile contract.**

```bash
git add agents/02_dragapult/profile.yaml pkm/agents/spec.py pkm/agents/profile.py pkm/agents/__init__.py tests/test_agent_profile.py
git commit -m "feat: add declarative agent profiles"
```

## Task 2: Create The Policy Factory

**Files:**
- Create: `pkm/agents/registry.py`
- Create: `pkm/agents/factory.py`
- Modify: `pkm/agents/neural_agent.py`
- Modify: `pkm/agents/__init__.py`
- Create: `tests/test_agent_factory.py`

- [ ] **Step 1: Write failing factory tests.**

Test that a profile creates a plain callable, returns its deck during deck
submission, and delegates action selection to the configured policy:

```python
def test_profile_make_agent_returns_kaggle_callable(monkeypatch):
    profile = AgentProfile.load("02_dragapult")
    expected = list(range(60))
    monkeypatch.setattr(profile, "load_deck", lambda: expected)
    monkeypatch.setattr("pkm.agents.factory.make_neural_agent", lambda deck, weights: lambda obs: deck)

    policy = profile.make_agent()

    assert callable(policy)
    assert policy({"select": None}) == expected
```

- [ ] **Step 2: Run the test and verify it fails.**

Run: `python -m pytest tests/test_agent_factory.py -q`

Expected: FAIL because no registry-backed profile factory exists.

- [ ] **Step 3: Implement the registries.**

Use explicit typed maps:

```python
POLICY_FACTORIES = {
    "random": make_random_agent,
    "neural": make_neural_agent,
    "mcts": make_mcts_agent,
}
```

Add a separate strategy registry for optional deck-specific behavior. It starts
empty because the first profile uses plain neural inference:

```python
STRATEGY_FACTORIES: dict[str, StrategyFactory] = {}
```

`strategy: null` must be valid for a plain neural policy. Unknown policy or
strategy names must raise during profile loading.

Resolve weights from the profile checkpoint/export path. For a configured
neural profile, raise a clear error when the required exported policy is absent;
do not silently fall back to random during submission construction.

- [ ] **Step 4: Implement `AgentProfile.make_agent()`.**

Load the validated profile-owned deck, select the configured policy and
optional strategy factories, and return only the callable. The factory must
not return a class instance to Kaggle; a closure is acceptable and should
preserve the existing `obs["select"] is None` deck behavior. Factories receive
the resolved profile/spec so special logic can inspect deck metadata without
hardcoding an agent name.

- [ ] **Step 5: Run all agent tests and the existing suite.**

Run: `python -m pytest tests/test_agent_profile.py tests/test_agent_factory.py tests/test_main.py -q`

Expected: all targeted tests pass and the existing submission tests remain
green.

- [ ] **Step 6: Commit the policy factory.**

```bash
git add pkm/agents/registry.py pkm/agents/factory.py pkm/agents/neural_agent.py pkm/agents/__init__.py tests/test_agent_factory.py
git commit -m "feat: add profile policy factory"
```

## Task 3: Add Profile Training Facades

**Files:**
- Modify: `pkm/agents/profile.py`
- Modify: `pkm/rl/train.py`
- Modify: `pkm/rl/exit_train.py`
- Modify: `pkm/cli/__init__.py`
- Create: `tests/test_profile_training.py`

- [ ] **Step 1: Write failing facade tests.**

Verify that the profile passes its deck, checkpoint, metrics, and run paths to
the configured trainer without importing the CLI from the trainer:

```python
def test_profile_train_delegates_to_registered_trainer(monkeypatch):
    profile = AgentProfile.load("02_dragapult")
    calls = {}

    def fake_train(**kwargs):
        calls.update(kwargs)
        return TrainingResult(checkpoint=Path("agents/02_dragapult/checkpoints/ppo_latest.pt"))

    monkeypatch.setitem(TRAINERS, "ppo", fake_train)
    result = profile.train(iterations=1, games=1)

    assert result.checkpoint.name == "ppo_latest.pt"
    assert calls["deck_path"] == profile.deck_path
    assert calls["checkpoint_dir"] == profile.checkpoint_dir
```

- [ ] **Step 2: Run the test and verify it fails.**

Run: `python -m pytest tests/test_profile_training.py -q`

Expected: FAIL because `TrainingResult`, `TRAINERS`, and `AgentProfile.train`
do not exist.

- [ ] **Step 3: Define the trainer result and registration contract.**

Create a small result dataclass containing `checkpoint`, optional `metrics`,
and `iterations`. Register wrappers around the existing `train()` and
expert-iteration functions. The wrappers must accept explicit paths and return
the result rather than making the CLI responsible for path construction.

- [ ] **Step 4: Add `AgentProfile.train()` and `train_exit()`.**

`train()` selects `profile.trainer`; `train_exit()` selects the expert-iteration
trainer. Both call `profile.ensure_dirs()` first and use the profile's deck and
output paths by default. Explicit keyword arguments may override iteration,
games, learning rate, and seed, but may not silently change the profile deck.

- [ ] **Step 5: Refactor the Typer commands to call the facade.**

The CLI command should resolve the profile once and call:

```python
AgentProfile.load(agent).train(iterations=iterations, games=games, lr=lr)
```

Keep the existing command-line flags working during migration.

- [ ] **Step 6: Run training facade tests and a one-iteration smoke test.**

Run: `python -m pytest tests/test_profile_training.py -q`

Run: `python -m pkm.rl.train --agent 02_dragapult --iterations 1 --games 1 --eval-every 1 --eval-games 1`

Expected: the test passes and the smoke test writes a profile checkpoint and
metrics row under `agents/02_dragapult/`.

- [ ] **Step 7: Commit the training facade.**

```bash
git add pkm/agents/profile.py pkm/rl/train.py pkm/rl/exit_train.py pkm/cli/__init__.py tests/test_profile_training.py
git commit -m "feat: expose profile training methods"
```

## Task 4: Build Profile-Driven Submissions

**Files:**
- Create: `pkm/submission.py`
- Modify: `submit.sh`
- Modify: `justfile`
- Create: `tests/test_submission.py`

- [ ] **Step 1: Write failing submission tests.**

Build a temporary profile bundle and assert that it contains `main.py`,
`deck.csv`, and inference code, and that importing the generated `main.py`
exposes a callable `agent` without executing a local battle:

```python
def test_build_submit_creates_kaggle_entrypoint(tmp_path):
    bundle = AgentProfile.load("02_dragapult").build_submit(output_dir=tmp_path)

    assert (bundle / "main.py").is_file()
    assert (bundle / "deck.csv").read_text().count("\n") == 60
    namespace = runpy.run_path(bundle / "main.py")
    assert callable(namespace["agent"])
```

- [ ] **Step 2: Run the test and verify it fails.**

Run: `python -m pytest tests/test_submission.py -q`

Expected: FAIL because `AgentProfile.build_submit` and the submission builder
do not exist.

- [ ] **Step 3: Implement the submission builder.**

The builder must:

1. export the selected profile checkpoint to `policy.npz`;
2. write the profile deck as flat `deck.csv`;
3. generate `main.py` using the selected policy registry key;
4. copy only required inference modules into a temporary staging directory;
5. create a timestamped tarball under `submissions/`;
6. remove the staging directory even when packaging succeeds.

The generated entry point should resolve `ROOT = Path(__file__).resolve().parent`
and assign `agent = make_<policy>_agent(DECK, weights_path=...)` at module scope.

- [ ] **Step 4: Make `submit.sh` a compatibility wrapper.**

Change it to call the profile builder for `02_dragapult` and reject unsupported
profile names. Keep the existing output naming convention and `submissions/`
directory.

- [ ] **Step 5: Add the profile build command.**

Add a Typer command and Just recipe equivalent to:

```bash
pkm build-submit --agent 02_dragapult
```

Keep `just build_submit 02_dragapult` as a compatibility alias.

- [ ] **Step 6: Verify the real bundle.**

Run: `python -m pytest tests/test_submission.py tests/test_main.py -q`

Run: `just build_submit 02_dragapult`

Run: `tar -tzf submissions/submission_02_dragapult_*.tar.gz`

Expected: the archive contains one `main.py`, one 60-card `deck.csv`, the
neural inference modules, and `pkm/policy.npz`; it must not contain training
checkpoints or a local-battle entry point.

- [ ] **Step 7: Commit the submission builder.**

```bash
git add pkm/submission.py submit.sh justfile tests/test_submission.py
git commit -m "feat: build submissions from agent profiles"
```

## Task 5: Refactor Local Play For Two Independent Profiles

**Files:**
- Modify: `pkm/rl/play.py`
- Modify: `pkm/rl/rollout.py`
- Create: `tests/test_multi_agent_play.py`

- [ ] **Step 1: Write failing independent-player tests.**

Use two fake callables and two different 60-card decks. Assert that the game
runner passes the two policies and the two decks independently to the
environment configuration:

```python
def test_play_match_uses_independent_agents_and_decks(monkeypatch):
    captured = {}

    def fake_make(config):
        captured["decks"] = config["decks"]
        return FakeEnvironment()

    monkeypatch.setattr("pkm.rl.play.make", fake_make)
    play_match(agent0=agent_a, agent1=agent_b, deck0=deck_a, deck1=deck_b)

    assert captured["decks"] == [deck_a, deck_b]
```

- [ ] **Step 2: Run the test and verify it fails.**

Run: `python -m pytest tests/test_multi_agent_play.py -q`

Expected: FAIL because `play_match` currently resolves one deck and constructs
both players from one agent type.

- [ ] **Step 3: Change the internal play API.**

Use explicit pair arguments internally:

```python
play_match(
    agents=(agent0, agent1),
    decks=(deck0, deck1),
    html_path=None,
    replay_path=None,
)
```

Keep a CLI adapter that resolves `--p0-agent` and `--p1-agent` profile names.
Do not make the game engine or rollout code import `AgentProfile`.

- [ ] **Step 4: Add profile-aware CLI options.**

Support:

```bash
pkm play --p0-agent 02_dragapult --p1-agent 01_psychic --games 30
```

Retain `--p0 neural --p1 random` as a low-level compatibility mode for tests
and quick experiments.

- [ ] **Step 5: Run the multi-agent and existing play tests.**

Run: `python -m pytest tests/test_multi_agent_play.py tests/ -q`

Expected: all tests pass and the existing replay command still writes a valid
replay.

- [ ] **Step 6: Commit the independent-player runner.**

```bash
git add pkm/rl/play.py pkm/rl/rollout.py tests/test_multi_agent_play.py
git commit -m "feat: support independent agents in local play"
```

## Task 6: Generalize The PPO Opponent Pool

**Files:**
- Modify: `pkm/rl/train.py`
- Modify: `pkm/rl/rollout.py`
- Create: `pkm/rl/opponents.py`
- Create: `tests/test_opponents.py`

- [ ] **Step 1: Write failing opponent-pool tests.**

Test that a pool can contain current policies, checkpoint policies, random
policies, and policies from another profile, and that sampling is deterministic
under a supplied seed:

```python
def test_opponent_pool_samples_registered_sources():
    pool = OpponentPool([current, old_checkpoint, random_policy], seed=7)
    assert pool.sample() in {current, old_checkpoint, random_policy}
```

- [ ] **Step 2: Run the test and verify it fails.**

Run: `python -m pytest tests/test_opponents.py -q`

Expected: FAIL because the generalized opponent source abstraction does not
exist.

- [ ] **Step 3: Implement `OpponentPool`.**

Represent each source with a name and zero-argument factory so a fresh policy
can be constructed for every game. Include helpers:

```python
OpponentPool.from_profile(profile)
OpponentPool.add_profile(profile)
OpponentPool.add_random()
OpponentPool.add_checkpoint(path, deck)
```

Keep checkpoint state dictionaries out of the pool API; checkpoint loading
belongs in the policy factory.

- [ ] **Step 4: Replace the local PPO pool.**

Use `OpponentPool` in `train()` while preserving `pool_size` and `pool_prob`.
The default pool must preserve current behavior: current policy plus past
learner checkpoints. Additional profiles are optional CLI inputs, not silently
included by default.

- [ ] **Step 5: Add cross-profile training options.**

Support a repeatable option such as:

```bash
pkm train --agent 02_dragapult --opponent 01_psychic --opponent random
```

Resolve each opponent profile independently and record opponent names in the
metrics CSV.

- [ ] **Step 6: Run a small cross-play training smoke test.**

Run: `python -m pkm.rl.train --agent 02_dragapult --opponent random --iterations 1 --games 2 --eval-every 1 --eval-games 1`

Expected: training completes, the learner checkpoint is written under the
Dragapult profile, and the metrics row identifies the opponent source.

- [ ] **Step 7: Commit the generalized opponent pool.**

```bash
git add pkm/rl/opponents.py pkm/rl/train.py pkm/rl/rollout.py tests/test_opponents.py
git commit -m "feat: support configurable training opponents"
```

## Task 7: Documentation And End-To-End Regression

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/ideas/general-agent-architecture.md`
- Modify: `README.md` if present, otherwise `docs/GUIDE.md`
- Create: `tests/test_agent_lifecycle.py`

- [ ] **Step 1: Write the lifecycle regression test.**

The test should exercise the public sequence with mocked training and a
temporary output directory:

```python
profile = AgentProfile.load("02_dragapult")
policy = profile.make_agent()
result = profile.train(iterations=1, games=1)
bundle = profile.build_submit(output_dir=tmp_path)
assert callable(policy)
assert result.checkpoint.is_file()
assert (bundle / "main.py").is_file()
```

- [ ] **Step 2: Run the complete verification suite.**

Run: `python -m pytest tests/ -q`

Run: `ruff check pkm/ tests/`

Run: `ruff format --check pkm/ tests/`

Run: `just build_submit 02_dragapult`

Run: `just upload submissions/<new_bundle>.tar.gz`

Run: `just poll`

Then retrieve the episode and logs:

```bash
kaggle competitions episodes <submission_id> --format json
just logs <episode_id> 0
just logs <episode_id> 1
```

Read both files and confirm there are no `Traceback`, `Exception`, or non-empty
`stderr` entries.

- [ ] **Step 3: Update documentation.**

Document the public lifecycle:

```bash
pkm train --agent 02_dragapult
pkm build-submit --agent 02_dragapult
pkm play --p0-agent 02_dragapult --p1-agent 01_psychic --games 30
```

Document profile layout, generated submission contents, cross-play metrics, and
the rule that generated bundles/logs live under `submissions/`.

- [ ] **Step 4: Commit the completed lifecycle documentation.**

```bash
git add AGENTS.md docs/ideas/general-agent-architecture.md docs/GUIDE.md tests/test_agent_lifecycle.py
git commit -m "docs: document agent lifecycle"
```

## Completion Criteria

- `AgentProfile("02_dragapult").make_agent()` returns the working neural
  callable.
- `AgentProfile("02_dragapult").train(...)` writes profile-scoped checkpoints
  and metrics.
- `AgentProfile("02_dragapult").build_submit()` creates a Kaggle bundle with a
  module-level `agent` and no local battle execution.
- Local play can run two profiles with different decks and policies.
- PPO can train against a configurable opponent pool without changing its
  existing default self-play behavior.
- Full tests and lint pass, and a real Kaggle submission completes without
  runtime errors in either downloaded agent log.
