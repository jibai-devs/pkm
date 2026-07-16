# Agent architecture: code map, composition, and refactor notes

**Status:** Design discussion / notes. Nothing here is built yet unless a file
path says "already done."

This captures a discussion about (1) mapping what the current RL/MCTS code
actually does, (2) whether it's in the right place, and (3) how to decompose "an
agent" into composable, reusable pieces (heuristics + models + search + training)
without duplicating code.

Companion doc: [`general-agent-architecture.md`](./general-agent-architecture.md)
already covers agent **identity & packaging** (profiles, registries, decks,
opponent pools, Kaggle entry points). This doc covers the part that one does
*not*: **composition of deciders** and the **code-level refactor**.

---

## 0. Terminology: are "delegation / injection / pipeline" ML terms?

No. They are **software-engineering / design-pattern** terms, used here as
lenses on the decision code. They are **not** machine-learning or
reinforcement-learning terms. The RL-native names for the same concepts are
different — know both so you can search the right literature:

| Lens used here (software term) | What it means | RL / ML-native name(s) to search |
|---|---|---|
| **Injection** (dependency injection) | one component *calls* another as a sub-oracle | AlphaZero-style *search guided by a learned policy/value*; actor–critic (value "injected" as a baseline) |
| **Pipeline** (transform chain) | pass a shared bundle through ordered transforms | action masking; feature/decision pre- & post-processing; "shielding" in safe RL |
| **Delegation** (routing / fallback) | choose *which* sub-agent handles this state | **hierarchical RL / options framework**; **mixture-of-experts / gating**; ensembles |
| **Distillation** (not discussed as a lens but present) | train a cheap student on an expert's outputs | **policy distillation / DAgger** — this repo's `exit_train.py` already is this |

Takeaway: the *compositional* ideas are software patterns; the *learning*
ideas (options, MoE, DAgger, actor-critic) are the RL versions. They line up
one-to-one, which is why the design below is buildable.

---

## 1. Code map — where everything lives today

| Concern | File | What it actually is |
|---|---|---|
| **Neural network** (policy + value, one module) | `pkm/rl/model.py` → `PolicyValueNet` | Shared trunk `encode_state → h (128-d)`, then two heads |
| **Policy function** (torch) | `model.py` → `encode_options` + `option_logits` + `act` / `evaluate` | Scores a variable-length option list → softmax logits over (options + STOP) |
| **Value function** (torch) | `model.py` → `value(h)` | `tanh` scalar in [-1,+1], V(s) |
| **Same net, torch-free** | `pkm/rl/numpy_policy.py` → `NumpyPolicy` | Hand-replays the forward pass in numpy for the Kaggle bundle. Exposes `value`, `priors`, `act_greedy`, `sample_picks`, `select` |
| **obs → tensors** | `pkm/rl/encoder.py` → `encode_decision` / `EncodedDecision` | The one shared feature encoder; both torch and numpy consume `EncodedDecision` |
| **MCTS** | `pkm/mcts/search.py` → `MCTS`, `_Node` | PUCT tree on the engine search API; uses `NumpyPolicy` for **priors** (policy) and **leaf eval** (value) |
| **Hidden-info sampling** | `pkm/mcts/determinize.py` | Samples opponent deck/hand/prizes for IS-MCTS |
| **Self-play driver** | `pkm/rl/rollout.py` → `play_game`, `TorchPolicy`, `RandomPolicy` | Runs a battle, collects `EncodedDecision`s |
| **PPO math** | `pkm/rl/ppo.py` | GAE + potential shaping (`compute_returns`), clip update (`ppo_update`) |
| **Phase 1 orchestration** | `pkm/rl/train.py` | PPO self-play + opponent pool + eval-vs-random |
| **Phase 2 orchestration** | `pkm/rl/exit_train.py` | Expert iteration (DAgger): MCTS visits → policy target, outcome → value target |
| **Serving adapters** (`obs → picks`) | `pkm/agents/neural_agent.py`, `pkm/mcts/agent.py`, `pkm/agents/random_agent.py` | Wrap a policy/MCTS/random into the Kaggle `agent(obs)` callable |
| **Agent base factory** | `pkm/agents/base.py` → `make_agent(deck, strategy_fn)` | Handles the `select is None → return deck` boilerplate |
| **Agent identity/resources** | `pkm/agents/profile.py` → `AgentProfile` | Name → deck path + checkpoint/metrics/runs dirs |
| **Match runner + eval** | `pkm/rl/play.py` | `make_agent_by_name`, `play_match`, `win_rate` (runs through `kaggle_environments.make`) |
| **Weights torch → npz** | `pkm/rl/export.py` | Checkpoint → `policy.npz` |

### The two functions you asked about specifically

- **Value function** = one implementation (`value(h)`), reused in **three** roles:
  PPO baseline (`ppo.py`), GAE bootstrap (`ppo.py`), MCTS leaf eval
  (`search.py`). Clean reuse.
- **Policy function** = used in **two** roles: acting (rollout/serving) and MCTS
  priors (`search.py`). Clean reuse.

The universal contract that ties everything together:

```
agent(obs: dict) -> list[int]     # indices into obs["select"]["option"]
```

The observation already carries the legal action set. That uniform signature is
what makes any composition possible at all.

---

## 2. What's messy (the smells)

The concepts are sound; the problems are all **missing interfaces**:

1. **The forward pass exists twice, synced by hand.** `PolicyValueNet` (torch)
   and `NumpyPolicy` (numpy) are the same math in two languages; `numpy_policy.py`
   literally says *"Must stay in sync with model.py."* Change a layer → must edit
   both files or inference silently diverges. Biggest fragility.

2. **No `Policy` interface — five different "act" shapes:**
   ```
   PolicyValueNet.act(d: EncodedDecision)      -> ActResult
   TorchPolicy.act(obs: dict, collect)         -> (picks, decision)
   NumpyPolicy.select(obs) / .act_greedy(d)    -> picks
   RandomPolicy.act(obs, collect)              -> (picks, None)
   agent(obs: dict)                            -> picks     # kaggle callable
   ```
   MCTS depends on a specific subset of `NumpyPolicy` (`value`, `priors`,
   `act_greedy`, `sample_picks`) — an implicit contract written nowhere.

3. **Forced-decision handling is copy-pasted 4×.** The "if one legal option / if
   minCount==maxCount==n, just take it" logic lives in `search.py:_forced_picks`,
   `rollout.py` (`TorchPolicy.act`), `numpy_policy.py` (`select`), and is
   re-called in `exit_train.py` and `mcts/agent.py`. It is already a duplicated
   heuristic.

4. **The two training loops duplicate the rollout driver.** `train.py:play_game`
   and `exit_train.py:play_exit_game` both reimplement
   `battle_start / while result<0 / battle_select / battle_finish`.

---

## 3. Refactor proposals (ranked by payoff / effort)

1. **One `forced_picks(select) -> list[int] | None`** helper; delete the other 3
   copies. (Quick, pure win.)
2. **Named `Agent = Callable[[dict], list[int]]`** + one `make_agent` seam so the
   serving adapters stop each reimplementing `select is None → return deck`.
   (`base.py` already half-does this.)
3. **`Policy` Protocol** (`value` / `priors` / `act_greedy` / `sample_picks` over
   `EncodedDecision`). Make both `NumpyPolicy` and a thin `PolicyValueNet` adapter
   satisfy it. Then MCTS depends on the *protocol*, not `NumpyPolicy` concretely
   — so you can run MCTS on the torch net during training (no export round-trip).
4. **Kill the hand-maintained numpy mirror risk**: at minimum add a CI parity
   test asserting `NumpyPolicy` and `PolicyValueNet` agree on random inputs;
   ideally generate one from the other's layer list. (Highest value, most work.)
5. **One shared `collect_trajectory(policies, decks, record_fn)`** to de-dup the
   two training loops (see §6).

Ranking: **1 → 2 → 3 → 5 → 4**. Items 1–2 are quick; 3 is the satisfying one; 4
removes the silent-drift risk.

---

## 4. Decomposing "an agent" into layers

Five independently swappable layers:

```
┌─ Resources ──────────────────────────────────────────────┐
│  declared needs: deck, weights, card_data, search_api,    │  manifest + DI
│  opponent_model. Resolved by the profile/builder.         │  (AgentProfile
└───────────────────────────────────────────────────────────┘   half-does this)
┌─ Model(s) ───────────────────────────────────────────────┐
│  pure functions of features. Policy: decision→scores,     │  Policy protocol
│  Value: decision→float. Random & heuristic are trivial    │  (injectable oracle)
│  Policies. Neural is Policy+Value. NOT search.            │
└───────────────────────────────────────────────────────────┘
┌─ Deciders ───────────────────────────────────────────────┐
│  transform (obs, candidates, scores, mask) → refined      │  Stage protocol
│  bundle. rule-filter, neural-scorer, mcts-rerank,         │  (pipeline unit;
│  argmax/sample. heuristic|neural stacking lives here.     │   heuristic|neural)
└───────────────────────────────────────────────────────────┘
┌─ Agent ──────────────────────────────────────────────────┐
│  binds a decider pipeline + resolved resources →          │  Agent =
│  the obs→picks callable kaggle wants.                     │  Callable[[dict],list[int]]
└───────────────────────────────────────────────────────────┘
┌─ Trainer ────────────────────────────────────────────────┐
│  consumes trajectories + a *trainable Model*, emits new   │  separate procedure,
│  weights. NOT a method on the agent.                      │  targets the Model
└───────────────────────────────────────────────────────────┘
```

---

## 5. Composition — the key distinction

**You cannot naively pipe `picks → picks`.** Once agent A returns picks, the
decision is *consumed*; B has nothing to act on. What you compose is the
**intermediate representation** — candidates + scores + mask — before it
collapses to a pick. There are **three genuinely different composition modes**;
conflating them is the classic over-engineering trap:

| Mode | What flows | Example here | Interface |
|---|---|---|---|
| **Pipeline** (transform) | a `(candidates, scores, mask)` bundle | `neural scores → heuristic masks bad ones → argmax` | `Stage: Decision -> Decision` |
| **Injection** (sub-oracle) | one component *calls* another | `MCTS(prior=net, value=net)` — **already done** (`search.py`) | `Policy` |
| **Delegation** (routing/fallback) | choose *which* decider handles this obs | `forced-move rule → else neural`; setup-heuristic → midgame-neural | `FirstMatch([...])` |

Critical: `heuristic | neural` (pipeline of transforms) and `MCTS(neural)`
(dependency injection) are **not the same kind of composition**. Use **two small
interfaces, not one god-interface.**

### Interface sketches (in this repo's idiom)

```python
# Injectable oracle — MCTS and scorers consume THIS.
# NumpyPolicy already nearly implements it; PolicyValueNet can via an adapter.
class Policy(Protocol):
    def value(self, d: EncodedDecision) -> float: ...
    def priors(self, d: EncodedDecision) -> np.ndarray: ...
    def act_greedy(self, d: EncodedDecision) -> list[int]: ...

# Pipeline unit — heuristic|neural lives here.
@dataclass
class Decision:
    obs: Observation
    candidates: list[tuple[int, ...]]     # legal action sequences
    scores: np.ndarray | None             # utility/prior per candidate
    mask: np.ndarray                      # still-allowed
    cache: dict                           # EncodedDecision etc., computed once

class Stage(Protocol):
    def apply(self, d: Decision) -> Decision: ...

# An agent is a stage list + a collapse:
agent = Pipeline([
    LegalCandidates(),           # seed from obs["select"]
    ForcedMove(),                # the _forced_picks heuristic — short-circuits
    NeuralScorer(policy),        # fill scores from priors
    ForbidActiveEnergyDiscard(), # a hand-written rule masks candidates
    Argmax(),                    # collapse to picks
]).as_agent(deck)
```

`neural | heuristic` vs `heuristic | neural` is just **stage order**. Fallback is
a `FirstMatch` stage.

### Where MCTS sits (important)

MCTS does **not** become a `Stage`. It internally simulates via the engine search
API + determinization — it is not a pure re-scorer over the same candidate list.
MCTS stays **injection**: `MCTS(policy: Policy)`, exactly as today. Forcing MCTS
into the pipeline is the trap. Two interfaces, each stays honest.

---

## 6. Training: do NOT bolt it onto the agent

Training is a **procedure that produces/updates a Model**, not a property of an
agent. Most agents have nothing to train (random, pure heuristic, and MCTS itself
— MCTS trains its *injected net* via ExIt, not "itself"). A `.train()` on every
agent yields empty methods everywhere.

The real duplication is elsewhere: `train.py:play_game` and
`exit_train.py:play_exit_game` reimplement the same engine loop. Fix with **one
shared collector**, parameterized by "what to record per decision":

```python
def collect_trajectory(policies, decks, record_fn) -> Trajectory: ...
# PPO's record_fn stores logprob/value/potential;
# ExIt's record_fn stores the MCTS visit target.
```

Principle: **Model is the shared artifact. Acting consumes it, training updates
it** — different lifecycles over the same Model + one collector.

---

## 7. Prior art — this has been done; steal these

- **Behavior trees / utility AI** (game AI): selectors = fallback; scorer
  ensembles = weighted `Stage`s. The `ForcedMove` + rule filters *are* a small
  behavior tree.
- **DAgger / policy distillation**: expert → student — **`exit_train.py` already
  is this** (MCTS teacher → net student).
- **AlphaZero**: search-over-net by injection — **already done** in `search.py`.
- **Options framework / hierarchical RL**: high-level policy picks a sub-policy
  that runs until termination — the principled "setup-heuristic then
  midgame-neural."
- **Mixture-of-experts / gating**: a router weights sub-policies — the principled
  "ensemble two neural decks."

Two of the important compositions (injection + distillation) are **already
built**. What's missing is the cheap **pipeline/rule layer** — and there is not a
single heuristic written yet to justify a general framework.

---

## 8. Recommendation — build the seam, not the cathedral

Three deciders exist (neural, MCTS-over-neural, random) and exactly one real
composition (MCTS injects neural), already working. A full composition framework
with **zero heuristics written** is premature. Minimal path that earns each
abstraction:

1. **Extract the two protocols** — `Policy` (oracle) and
   `Agent = Callable[[dict], list[int]]`. Make `NumpyPolicy` + a `PolicyValueNet`
   adapter both satisfy `Policy`. Bonus: MCTS can then run on the torch net during
   training with no export round-trip.
2. **Turn `_forced_picks` into the first real `Stage`** (already a duplicated
   heuristic — perfect pilot). Build a tiny `Pipeline` that runs
   `ForcedMove → NeuralScorer → Argmax` and reproduces today's neural agent
   exactly. If that's clean, the pipeline abstraction has earned its place.
3. **One shared `collect_trajectory`** to de-dup the two training loops.

Only after a 2nd/3rd real heuristic exists do you build `FirstMatch`, routing,
ensembles, and the YAML `strategy` hook that `general-agent-architecture.md`
already sketches.

---

## 9. Future questions / open decisions

- **Which candidate representation?** For multi-pick decisions, MCTS already
  enumerates *sequences* (`tuple[int, ...]`); the `Decision.candidates` bundle
  should match. Confirm sequences (not per-option indices) are the right unit.
- **Does `Stage` operate on `EncodedDecision` or raw obs?** Cheap rules want raw
  obs; scorers want the encoded tensors. Probably `Decision.cache` holds the
  lazily-encoded `EncodedDecision` so a stage computes it at most once.
- **Numpy/torch parity**: unify the two forward passes, or just CI-test parity?
  (Decide before adding layers.)
- **Should the opponent pool hold `Policy` objects or `Agent` callables?**
  `general-agent-architecture.md` wants a general pool (self, past checkpoints,
  random, other profiles, MCTS) — needs a uniform handle.
- **Where do deck-specific heuristics live** relative to the profile's
  `strategy:` field? Registry of `Stage` factories keyed by strategy name?
- **Is a full behavior-tree/MoE layer ever worth it**, or do 1–2 rule `Stage`s +
  injection cover everything this game needs?

---

## 10. Reusable prompt (to ask another LLM the same thing)

> I have an RL agent for a Pokémon-TCG-style card-battle engine. Every agent is a
> function `agent(obs: dict) -> list[int]` returning indices into a legal-option
> list carried by the observation. I have: a `PolicyValueNet` (shared trunk →
> policy head over variable-length options + value head in [-1,1]); a hand-written
> numpy copy of the same forward pass for torch-free deployment; a feature encoder
> (obs → arrays); IS-MCTS with determinization that uses the net as prior + leaf
> value via the engine's search API; a self-play rollout driver; PPO
> (GAE + potential-based shaping); and expert-iteration (MCTS visits → policy
> target, outcome → value target). Serving adapters wrap each into the
> `agent(obs)` callable.
>
> 1. Map which code does the neural net, the policy function, the value function,
>    the MCTS, the training — and judge whether each is in the right place.
> 2. I want to decompose "an agent" into composable, reusable pieces: hardcoded
>    rules/heuristics, a model (policy+value, or dumb random, or DFS/BFS), and
>    search — where the agent declares what resources it needs (deck, weights,
>    card data, search API) and where to get them. What layers and interfaces?
> 3. Should training be a method on the agent, or separate? How do I avoid
>    duplicating the rollout/training loop across PPO and expert iteration?
> 4. I want to *stack* deciders — pipe one into another, `heuristic | neural`,
>    `neural | heuristic`, `MCTS(neural)`. What's the right interface for each?
>    Warn me where a single "pipe" abstraction would be wrong. Is
>    delegation/injection/pipeline software terminology or RL terminology, and
>    what are the RL-native names (options, mixture-of-experts, DAgger, etc.)?
> 5. Name the prior art I'm reinventing (behavior trees, utility AI, hierarchical
>    RL, distillation, AlphaZero) and tell me the minimal seam to build first
>    instead of an over-engineered framework.
