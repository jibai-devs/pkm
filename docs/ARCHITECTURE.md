# Architecture Reference (Phase 1 PPO + heuristics integration)

Reflects `feature/heuristics-integration` as of commit `73356e5` — Tasks 1–8
of `docs/superpowers/plans/2026-07-16-heuristics-integration-architecture.md`,
merged with the reward-shaping heuristics per
`docs/superpowers/plans/2026-07-18-merge-architecture-with-heuristics.md`.
Task 9 (hard-rule forced picks) was deliberately not built — see that plan's
"Session Status" section.

This is a from-scratch walkthrough of what actually runs, file by file. If it
disagrees with the code, trust the code — this doc rots, `git blame` doesn't.

---

## 1. The one-sentence version

A per-decision `Observation` is turned into (a) a declarative registry of
float features plus raw card-ID arrays, fed through a pointer-style
policy/value network with three heads (policy, value, opponent-archetype),
and (b) a set of hand-tuned reward-shaping terms that don't touch the
network's inputs at all but change what PPO trains it to chase. A per-game
`GameContext` threads through both, giving the network memory of what's been
seen without ever letting that memory leak between games or into MCTS's
imagined branches.

---

## 2. One decision, start to finish

```
engine obs (dict)
   │
   ▼
GameContext.tracker.observe(obs)              pkm/heuristics/deck_tracker.py
   (+ .record_search_reveal if a full-deck
   search reveal just happened)
   │
   ▼
Observation.model_validate(obs)               pkm/types/obs.py (pydantic)
   │
   ▼
encode_decision(obs, ctx) ─────────────────────────────────────────────┐
   │                                                                    │
   ├─ encode_state  → board_cards, hand_cards,                         │
   │                  state_feats (registry, §4 below),                │
   │                  deck_card_ids/counts (deck ledger, §4)            │
   │                                                                    │
   └─ encode_options → opt_type/opt_card/opt_card2/opt_attack,         │
                        opt_feats (registry)                            │
   │                                                                    pkm/rl/encoder.py
   ▼
PolicyValueNet.act(EncodedDecision)           pkm/rl/model.py
   │
   ├─ h = encode_state(...)                    "the trunk"
   ├─ picks, logprob = policy head (sequential, masked, STOP-terminated)
   ├─ value = value head
   └─ belief = archetype_belief(h)  (@torch.no_grad softmax)
   │
   ▼
ctx.archetype_belief = belief                 fed back as a GLOBAL feature
   │                                          for the *next* decision only
   ▼
picks returned to the engine (battle_select)
   │
   ▼ (if collecting a training trajectory)
EncodedDecision.<reward fields> populated      pkm/rl/rollout.py:TorchPolicy.act
   (prize_potential, dragapult_backup_potential, budew_bonus, ... — §6)
```

Two independent consumers read the same `GameContext`/tracker fact:
**learned** (fed through the `FeatureSpec` registry into the trunk) and, if a
fact is certain enough, **hard** (bypasses the network via `forced_picks` —
not built yet, see §9). Nothing here duplicates the tracker query logic
between the two.

---

## 3. `GameContext` + `DeckTracker` — the per-game memory

`pkm/heuristics/context.py`:

```python
@dataclass
class GameContext:
    my_deck: list[int]
    tracker: DeckTracker
    opp_decklist: list[int] | None = None
    archetype_belief: np.ndarray | None = None
```

- **`my_deck`** — the literal 60 card IDs in your own deck. Just a reference
  list; not learned.
- **`tracker`** — a `DeckTracker` (`pkm/heuristics/deck_tracker.py`). Binds
  each of your own 60 deck-list slots to an engine-assigned `serial` the
  first time it's observed in a public zone (hand/board/discard/prize/
  stadium/attached energy or tool). Critically, `record_search_reveal`
  deduces prize-pile contents by elimination: when a search effect reveals
  your *entire* remaining deck, whatever's still unbound has nowhere left to
  be but the prizes.
- **`opp_decklist`** — **only consumed by MCTS** (`pkm/mcts/determinize.py`),
  not by the trunk/registry pipeline above. It's the literal 60-card list
  MCTS samples hidden information from: passed directly when training
  self-play knows the true opponent deck, or produced by
  `infer_opponent_decklist(obs)` — a crude "count what's already visible,
  pad with basics/energy" heuristic — when it isn't known. Do not confuse
  this with `archetype_belief` below; they don't feed each other today (see
  `docs/superpowers/plans/2026-07-16-heuristics-integration-architecture.md`
  → "Explicitly deferred" for the future idea of wiring them together).
- **`archetype_belief`** — the trunk's own archetype-head output from the
  *previous* decision, re-read as a `GLOBAL` feature (§4, §5). One-decision
  stale by construction.

One `GameContext` per player per game, constructed right after
`battle_start`, discarded at the end. **Never reused across games** — a
leaked reference silently contaminates one game's prize knowledge into the
next (`tests/test_deck_tracker.py`, `tests/test_game_context.py` assert this).

Construction sites (5 total, all following the same pattern):
`pkm/rl/rollout.py:play_game`, `pkm/rl/exit_train.py:play_exit_game`,
`pkm/agents/neural_agent.py:make_neural_agent`, `pkm/mcts/agent.py:make_mcts_agent`,
`pkm/tui/session.py`.

---

## 4. `FeatureSpec` registry — declarative float features

`pkm/rl/features.py`. Every scalar the network sees (besides raw card/attack
IDs, which the network embeds itself) is one `FeatureSpec`:

```python
class Scope(Enum):
    GLOBAL = auto()       # one block for the whole state
    PER_SLOT = auto()     # one block per board slot (N_POKEMON_SLOTS of them)
    PER_OPTION = auto()   # one block per legal option this decision
    PER_DECK_CARD = auto()  # unused by any spec today — deck ledger (§4.1) bypasses this

@dataclass
class FeatureSpec:
    name: str
    width: int
    scope: Scope
    fn: Callable[[Observation, GameContext | None], np.ndarray]
    deterministic: bool   # True = pure function of obs alone, ignores ctx
```

Registration order is the single source of truth for tensor layout *and*
total width — `STATE_FEATS`/`OPT_FEATS` are computed once at import time as
`sum(s.width for s in ...)`, not hand-maintained integers.

**`GLOBAL_FEATURES`** (10 specs, summing to `10+8+4+1+1+4+2+NUM_SELECT_TYPES+4+ARCHETYPE_OUT`):
status conditions (me+opp), zone counts (hand/deck/prize/discard, both
sides), bench counts, turn, turn-action-count, turn flags
(energyAttached/supporterPlayed/stadiumPlayed/retreated), first-player flags,
select-type one-hot, select counts (min/max/energy-cost/damage-counter), and
`opponent_archetype_belief` (§5 — the one non-deterministic GLOBAL spec,
reads `ctx.archetype_belief`).

**`PER_SLOT_FEATURES`** (6 specs, one row per board slot): present, hp,
max_hp, energy_count, appeared_this_turn, and Tier-1's `retreat_viable`.

**`PER_OPTION_FEATURES`** (7 specs, one row per legal option): option_number,
option_count, attack_damage, attack_cost, option_is_mine, and Tier-1's
`lethal_this_turn` + `type_effectiveness`.

`FeatureConfig(disabled: frozenset[str])` zero-masks a named spec's output
slice without changing total width — this is the ablation mechanism the
Task 6/7/8 "retrain and measure" comparisons are meant to use (not yet run —
see §10).

### 4.1 Tier-1 deterministic features (`pkm/rl/deterministic_features.py`)

Pure functions of `obs` alone (ignore `ctx`), reusing `pkm/data/card_data.py`:

- **`lethal_this_turn`** (PER_OPTION, attack options only) — 1.0 if
  `opp_active.hp - attack.damage <= 0`. Ignores type effectiveness
  deliberately; the network combines the two signals itself.
- **`type_effectiveness`** (PER_OPTION) — weakness → 1.0, neutral → 0.5,
  resisted → 0.25, non-attack → 0.0 (distinguishable from neutral).
- **`retreat_viable`** (PER_SLOT, bench only) — 1.0 at a bench slot whose
  card's `retreat_cost` is covered by the active's current energy count.

One documented correction to `plan.md` §4's wording: `PokemonRef.hp` is
already *current* remaining HP in this engine (verified against
`tests/fixtures/observations.json`), not undamaged max minus a separate
damage-counter tally — so `lethal_this_turn` subtracts directly, no separate
damage-counters term.

### 4.2 Deck ledger — not a `FeatureSpec` (Task 7)

`deck_ledger(ctx)` returns raw `(card_id, count)` arrays — unique still-unseen
card IDs from `ctx.tracker.by_location(CardLocation.DECK)` and their counts —
**not** a float feature slice. The network pools these through its own
`card_emb` table (§5), the same pattern as `board_cards`/`hand_cards`, rather
than a fixed-width slot-indexed vector. This is `plan.md` §5's one deliberate
deviation: no 60-wide cap, generalizes across decklists for free.

### 4.3 Checkpoint stamping

`feature_stamp()` fingerprints the registry as an ordered
`(scope, name, width)` tuple. `write_stamp_sidecar(path)` writes it next to a
saved `.pt` as `<path>.stamp.json`; `check_stamp_sidecar(path)` raises
`FeatureStampMismatch` on load if the *current* registry doesn't match —
called automatically from `AgentProfile.latest_checkpoint()`
(`pkm/agents/profile.py`). A missing sidecar (pre-stamping checkpoint) is not
an error — can't verify it, so it's let through. The `.npz` export path
(`pkm/rl/export.py`) embeds the same stamp for the numpy/Kaggle side.

**Any tensor-width change here is an accepted, checkpoint-breaking cost**
(training volume is small enough that retraining from scratch is cheap) —
this stamping is what makes that safe instead of silently corrupting.

---

## 5. `PolicyValueNet` — the trunk + three heads

`pkm/rl/model.py`. Embedding tables: `card_emb` (`NUM_CARDS=1268 × 32`),
`attack_emb` (`NUM_ATTACKS=1557 × 16`), `opt_type_emb` (`17 × 8`).

### 5.1 `encode_state` → `h` (128-dim)

Board split into **my slots** / **opponent's slots** (Task 7 — previously one
flat positional concatenation across all `N_BOARD_SLOTS`), each mean-pooled
through `card_emb` separately, plus:

```
h = ReLU(state_fc2(ReLU(state_fc1(
      concat[my_board_pool, opp_board_pool, stadium_emb, hand_pool,
             deck_ledger_pool, state_feats]
))))

deck_ledger_pool = Σ_c unseen_count[c] · card_emb[c]   (§4.2, a weighted
                                                          sum, not a mean —
                                                          magnitude legitimately
                                                          grows with how much
                                                          of the deck is
                                                          still unseen)
```

Five `EMB_CARD`-wide (32) blocks total, independent of bench size:
`STATE_IN = 5*32 + STATE_FEATS`.

### 5.2 `encode_options` → per-option encodings

`OPT_IN = 2*EMB_CARD + EMB_ATTACK + EMB_OPT_TYPE + OPT_FEATS`, one `opt_fc`
linear+ReLU per option, producing `OPT_ENC=64`-wide rows.

### 5.3 Three heads off `h`

```
                     ┌── policy head (score each option + STOP vs h,
                     │    sequential multi-pick, masked)
h (128) ─────────────┼── value head    → tanh([-1,1]) V(s) estimate
                     └── archetype head → logits, width ARCHETYPE_OUT
```

- **Policy head** (`option_logits`): each option row + `h` + a running
  `picked_sum` of already-chosen options is scored by a 2-layer MLP; a
  learned `stop_vec` row is appended and becomes selectable once `min_count`
  picks are made. Multi-select decisions decompose into sequential
  pick→mask→rescore steps; joint log-prob is the sum of per-pick log-probs.
- **Value head**: `V(s)`, used as the PPO baseline and MCTS leaf eval.
- **Archetype head** (Task 8): `archetype_logits(h)` for training (cross-
  entropy against ground truth); `archetype_belief(h)` for re-injection —
  wrapped in `@torch.no_grad()` (not a bare `.detach()`) so no autograd graph
  is ever built for that call path, making it structurally impossible for a
  caller to accidentally backprop through it.

  `ARCHETYPE_CLASSES = ["00_basic", "01_psychic", "02_dragapult"]`,
  `ARCHETYPE_OUT = 4` (+1 reserved "Other" slot for any future unrecognized
  deck, chosen deliberately since growing this list is checkpoint-breaking).

### 5.4 The re-injection loop (Task 8's core design decision)

```
decision t:   h_t = encode_state(...)
              belief_t = archetype_belief(h_t)      [@no_grad]
              ctx.archetype_belief = belief_t
decision t+1: opponent_archetype_belief feature reads ctx.archetype_belief
              → becomes part of state_feats → part of h_{t+1}
```

Chosen: **ctx-mediated, one-decision-stale**, not same-decision two-pass
re-encoding (that would require encoding state twice per decision). The
detach is the non-negotiable line — if gradient reached the archetype head
through this path, cross-entropy-vs-truth would degrade into "whatever helps
win rate," an unfalsifiable, uninterpretable signal.
`tests/test_archetype_head.py` asserts `archetype_fc1.weight.grad` stays
`None`/zero when only `pi_loss`/`v_loss` are backpropped.

---

## 6. Reward shaping — what PPO trains the network to *chase*

This is a **completely separate axis** from everything above: it never
touches `EncodedDecision`'s inputs to the network, only the reward signal
computed from it. `pkm/rl/reward_terms.py`:

```python
POTENTIAL_TERMS: list[tuple[str, str]] = [
    ("shaping", "potential"),                    # prize differential
    ("board_setup", "board_setup_potential"),    # Dragapult ex + charged backup Drakloak
    ("budew_setup", "budew_setup_potential"),     # Budew active, going 2nd, turn<=2
    ("dreepy_field", "dreepy_line_field_potential"),  # 0→3 ramp, -1 at 4+
]
DIRECT_TERMS: list[tuple[str, str]] = [
    # ~13 action-conditioned bonuses/penalties: energy over-attach penalty,
    # Budew turn-1 attack bonus, wrong-type-energy penalty (Dreepy line),
    # Dragapult ex attack bonus, Phantom Dive bonus, energy-spread penalty,
    # Xerosic Machinations bonus/penalty, evolve bonus, bench/active charge
    # bonuses, wasted-resources-on-attack penalty, Drakloak backup-ready bonus
]
DEFAULT_WEIGHTS = {"shaping": 0.2, <everything else>: 0.0}
```

Every deck-specific term (Dreepy/Drakloak/Dragapult ex/Budew/Xerosic) defaults
to **off** — `DEFAULT_WEIGHTS` reproduces the pre-merge hardcoded
`shaping_coef=0.2` behavior exactly, so no existing agent's training changes
unless it opts in via its own `agents/<name>/reward_weights.json`.

`pkm/rl/rollout.py:TorchPolicy.act` computes and stashes every term's raw
value onto `EncodedDecision` each decision (`d.board_setup_potential = ...`,
etc. — see §2's flow diagram). `pkm/rl/ppo.py:compute_returns` then loops
`POTENTIAL_TERMS`/`DIRECT_TERMS`, weighted by whatever `weights` dict it's
given:

- **Potential-based terms**: `r_t += coef * (γ·φ(s_{t+1}) − φ(s_t))` — leaves
  the optimal policy unchanged, rewards *reaching* a state rather than
  paying out every step it holds.
- **Direct terms**: `r_t += coef * value_at_t` — added straight into that
  step's reward, since they're conditioned on the specific action taken.
- **`win_reward`**: optionally scales a *win's* terminal reward only (not
  losses/draws) — a knob for "make winning matter more" without also making
  losses hurt more.

`compute_returns`'s own default is `weights=None` → empty dict → **all**
terms off, including `shaping`. It's `train()`'s job
(`effective_weights = {**DEFAULT_WEIGHTS, **(weights or {})}`) to apply the
real defaults — so calling `compute_returns` directly in a test with no
`weights` arg is deliberately zero-shaping, not the training default.

---

## 7. Training loop (`pkm/rl/train.py`) + PPO update (`pkm/rl/ppo.py`)

```
for each iteration:
    for each of games_per_iter:
        opponent = current model, or sampled from a checkpoint pool
                   (pool_prob chance, prevents self-play cycling)
                   — or, if --archetype-pool is set, a Part 3b pool bot on
                   its own deck (archetype_pool_prob chance, rolled first;
                   see below and §10)
        play_game(...) → per-player trajectories of EncodedDecisions
        compute_returns(trajectory, terminal_reward, weights=effective_weights)
            → fills .advantage / .ret via GAE(λ) on top of the shaped rewards
        dec.true_archetype = archetype_index(deck_path)  for every decision
                              (still the trainee's own fixed deck label,
                              regardless of opponent — see §10)
    ppo_update(model, optimizer, all_collected_decisions)
        policy_loss (PPO clip) + value_coef·value_loss − entropy_coef·entropy
            + archetype_coef · CE(archetype_logits, true_archetype)
        (archetype_coef defaults 0.1; masked out entirely for unlabeled
         decisions — none exist today since every run trains one deck)
    pool.append(current weights); trim to pool_size
    every eval_every iters: eval vs random, checkpoint + write_stamp_sidecar
```

**Cross-archetype opponent sampling (Part 3c, `docs/opponent-archetype-classifier-plan.md`
Part 3, 2026-07-19).** `GameSpec` (`pkm/rl/rollout.py`) gained an optional
`opponent_deck` field (`None` = mirror `deck`, the only value every pre-3c
`GameSpec` ever had). `make_game_specs` rolls `archetype_pool_prob` *before*
the existing `pool_prob` check: when it hits, the opponent is a
`pkm/rl/opponent_pool.py:load_pool_bots()` entry — a trained Part 3b pool
bot's weights *and its own deck* — instead of a past checkpoint of the
trainee's deck. `play_one` sends each side its own deck accordingly. Opt-in
via `--archetype-pool [--archetype-pool-prob 0.2]`, off by default, so
`pool_prob`'s pre-existing meaning (fraction of games vs a past checkpoint of
*this same* deck) is unchanged when unset.

**Belief-in-encoder actually wired into training, same date.** §4's
`opponent_archetype_belief` GLOBAL feature and `TorchPolicy(model,
archetype_classifier=...)` (§2 above) existed since Parts 1-2 shipped, but
were only ever exercised directly in `tests/test_archetype_integration.py`
— `train.py` never built a classifier or passed one to any `TorchPolicy`,
so every real training run to date saw an all-zero belief regardless of the
dim-4→26 resize. `play_one` now takes an optional `archetype_classifier`,
threaded from `train()` through both the sequential and `parallel_rollout.py`
paths, and attached only to the trainee's `TorchPolicy` — never a frozen
opponent's, mirror or Part 3c pool bot alike. Opt-in via `--archetype-belief
[--archetype-weights pkm/archetype.npz]`, off by default.

CLI entrypoints (both must stay in sync — see §8's note on why there are two):

```
pkm train --agent 02_dragapult --iterations 50 --games 8 \
           --weights agents/02_dragapult/reward_weights.json   # optional, JSON {term: weight}
```

`--weights <path>` (replacing the old `--shaping <float>`) defaults to the
resolved agent's own `reward_weights.json` when `--agent` is given and
`--weights` is omitted; falls back to `DEFAULT_WEIGHTS` if neither exists.

### numpy/torch parity

`pkm/rl/numpy_policy.py` is a hand-written mirror of `model.py`'s forward
pass for torch-free Kaggle inference (submission has a 197.7 MiB cap — no
torch at eval time). `tests/test_numpy_torch_parity.py` builds a random-init
`PolicyValueNet`, exports it, and asserts `value()`/`priors()` agree within
float32 tolerance on the same batch. **Any task touching `model.py`'s forward
pass must update `numpy_policy.py` in the same change and re-run this test —
it's the standing CI gate** (there is no other automated check that the two
stay in sync).

---

## 8. Two CLI entrypoints — a trap if you only edit one

`pkm` (the actual installed command, per `pyproject.toml`'s
`[project.scripts] pkm = "pkm.cli:app"`) is `pkm/cli/__init__.py`. Its `train`
command calls `pkm.rl.train.main(...)` **as a plain function**, by keyword.
`pkm/rl/train.py` *also* defines its own typer `app`/`main` (reachable via
`python -m pkm.rl.train`, not the installed `pkm` command). **Both signatures
must match** — this bit during the reward-shaping merge: the first pass
updated `pkm/rl/train.py`'s signature (`shaping` → `weights`) but missed that
`pkm/cli/__init__.py`'s `train` command still called it with the old
`shaping=` keyword, which would have thrown at the first real `pkm train`
invocation. Fixed in commit `73356e5`; flagging here so the next signature
change checks both files.

**Hit again, 2026-07-19, Part 3c:** adding `--archetype-pool`/
`--archetype-pool-prob` to `pkm/rl/train.py`'s CLI only made them work via
`python -m pkm.rl.train` — the real `pkm train --help` didn't show them at
all (no error, just silently absent) until the same two options were added
to `pkm/cli/__init__.py`'s shim and threaded through its call to
`pkm.rl.train.main(...)`. Two occurrences now; treat "add the flag in both
files" as a hard rule for any future `pkm train`/`pkm exit-train` CLI change,
not a one-off fix.

---

## 9. MCTS boundary (`pkm/mcts/search.py`, `pkm/mcts/agent.py`)

```
make_mcts_agent(deck, opp_decklist=None)
    │
    ▼
ctx = GameContext(deck, DeckTracker(deck), opp_decklist)   # real per-game context
    │  (updated every real decision: ctx.tracker.observe(obs), reveal-check)
    ▼
MCTS.choose(obs, ctx)  ← reads GameContext ONCE here, read-only,
    │                     to bias sample_determinization (which hidden
    │                     cards to imagine for the opponent)
    ▼
_Node / _expand / _simulate  ← NO GameContext parameter, by construction.
    Every node past the root lives on a hypothetical determinized branch
    (search_step on an imagined action sequence), not real game history.
    Feeding an imagined observation into the real tracker would silently
    corrupt the AI's actual beliefs about the live game.
```

Enforced by explicit comments at `_Node.__init__` and `MCTS._expand` in
`pkm/mcts/search.py` — "do not fix this by threading ctx through by habit."
`deck_ledger(ctx=None)` and `_opponent_archetype_belief(ctx=None)` both
degrade gracefully (empty ledger, zero/uninformative belief) for exactly
this case.

---

## 10. Known gaps (as of this doc)

- **No retrain-and-measure yet** for Tasks 6/7/8's win-rate/accuracy lift
  claims — the architecture is unit-tested but nobody has run an on-vs-
  ablated (`FeatureConfig`) comparison with real training. `pkm/policy.npz`
  locally, if present, is a throwaway random-init smoke-test export, not a
  trained checkpoint.
- **Task 9 (hard-rule `forced_picks` extension) not built** — every
  candidate condition considered was either already covered by existing
  structural checks or turned out to be a policy preference in disguise
  ("always take the kill" isn't a certainty). See
  `docs/superpowers/plans/2026-07-16-heuristics-integration-architecture.md`
  for the full reasoning and reopening options.
- **Every decision in a run still gets one fixed archetype label** —
  `archetype_index(deck_path)` labels every collected decision with the
  trainee's own deck, regardless of who the opponent was; the archetype head
  (§4.2) is real machinery but degenerately single-class *for this label*
  until multi-deck **self** play exists (not planned — the trainee always
  trains one deck per run by design). What Part 3c (above) *does* add is
  opponent diversity — `--archetype-pool` lets the opponent be one of 25 real
  archetypes instead of always a past checkpoint of the same deck — but
  that's who you play against, not a change to whose label gets stamped.
  Opt-in, not yet exercised for a real training run (`AGENTS.md` → "What's
  Next").
- **`opp_decklist` and `archetype_belief` don't talk to each other** — MCTS's
  determinization and the trunk's learned belief are two independent "what
  might the opponent have" mechanisms today. Explicitly deferred: building a
  general opponent-side deck ledger from the guessed archetype (would need
  Task 8's classifier validated first).

---

## 11. File map

| File | Role |
|---|---|
| `pkm/heuristics/context.py` | `GameContext` |
| `pkm/heuristics/deck_tracker.py` | `DeckTracker`, `CardLocation`, `CardState` |
| `pkm/rl/features.py` | `FeatureSpec`/`Scope`/`FeatureConfig` registry, deck ledger, checkpoint stamping |
| `pkm/rl/deterministic_features.py` | Tier-1 features: `lethal_this_turn`, `type_effectiveness`, `retreat_viable` |
| `pkm/rl/encoder.py` | `encode_state`/`encode_options`/`encode_decision`, raw card-ID arrays, `EncodedDecision`, reward-shaping term functions |
| `pkm/rl/reward_terms.py` | `POTENTIAL_TERMS`/`DIRECT_TERMS` registry, `DEFAULT_WEIGHTS`, `load_weights` |
| `pkm/rl/model.py` | `PolicyValueNet` — trunk, policy/value/archetype heads |
| `pkm/rl/numpy_policy.py` | Torch-free mirror of `model.py`'s forward pass (Kaggle inference) |
| `pkm/rl/ppo.py` | `compute_returns` (GAE + shaping), `ppo_update` |
| `pkm/rl/rollout.py` | `play_game`, `TorchPolicy` (per-decision `GameContext` wiring + reward-term population), `GameSpec`/`make_game_specs`/`play_one` (opponent matchmaking incl. Part 3c cross-archetype sampling) |
| `pkm/rl/opponent_pool.py` | `load_pool_bots()` — loads trained `agents/pool_*/` checkpoints for Part 3c |
| `pkm/rl/parallel_rollout.py` | `ProcessPoolExecutor` self-play (`_play_chunk`/`collect_parallel`); threads `archetype_classifier` to workers alongside model state/deck |
| `pkm/rl/exit_train.py` | Phase 2 expert-iteration game loop (separate from `rollout.py`, no reward shaping — trains off MCTS visit-count targets) |
| `pkm/rl/train.py` | Phase 1 PPO training loop + its own CLI (`python -m pkm.rl.train`) |
| `pkm/cli/__init__.py` | The real `pkm` CLI (`pkm train`, `pkm play`, `pkm export`, ...) |
| `pkm/rl/sweep.py` | Optuna hyperparameter sweep |
| `pkm/agents/profile.py` | `AgentProfile` — per-agent directories, checkpoint stamp checking |
| `pkm/agents/neural_agent.py`, `pkm/mcts/agent.py` | Per-match agent factories, each constructing their own `GameContext` |
| `pkm/mcts/search.py`, `pkm/mcts/determinize.py` | IS-MCTS + determinization; the `GameContext` read-once boundary |
| `pkm/tui/session.py` | Human-play `GameContext` construction site |

Related design docs: `plan.md` (repo root — the original Tier-1/registry/
checkpoint-stamping/auxiliary-head design, still the source of truth for the
*why* behind §4/§5/§8 above),
`docs/superpowers/plans/2026-07-16-heuristics-integration-architecture.md`
(Tasks 1–8 implementation plan + session status),
`docs/superpowers/plans/2026-07-18-merge-architecture-with-heuristics.md`
(the reward-shaping merge, §6/§7/§8 above),
`docs/opponent-archetype-classifier-plan.md` (the archetype belief head, its
Kaggle-checkpoint-invalidation breakage, and Part 3's pool decks/pool
bots/cross-archetype sampling referenced in §7/§10 above).
