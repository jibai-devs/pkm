# 00 — Auxiliary losses

## What this is

An **auxiliary loss** hangs an extra prediction head off the *shared trunk*
(`state`, the `[B, d_state]` encoder summary in `model.py`) and trains it to
predict something extra. The point is not the prediction itself — it's the
**gradient**: the head forces the trunk to encode strategically-relevant
structure it might otherwise ignore, which (usually) makes the *policy and value*
that ride the same trunk better.

Cheap because it piggybacks on the existing forward pass: one extra `Linear` per
head, one extra term in the loss. Nothing at inference time uses it.

## Why we want it here

agent_000 is stuck at a Kaggle **600 eval ceiling** in an *imperfect-information
mirror match*. That specific failure mode points at auxiliary tasks that make the
trunk **build a belief about hidden state** (Tier B below) — the thing that lets
a policy exploit an opponent instead of playing a fixed mirror line, and the
thing inference-time MCTS determinization is guessing at. So a good aux head here
is synergistic with the MCTS we already ship.

## The menu of possible auxiliary targets

Ordered from cheapest-to-label to most-valuable-for-this-game.

### Tier A — denser versions of the value signal (terminal labels only, zero rollout plumbing)

| Name | Target | Label source |
|---|---|---|
| `prize_margin` | final prize margin, seat-signed (−6…+6) | terminal obs `prize` |
| `prizes_taken` | prizes you took this game (0…6) | terminal obs |
| `turns_left` | decisions remaining until game end | `len(steps)` from that step |
| `next_reward` | the immediate shaped reward | `shaping.py` scalars |

These need **no** opponent observation — labels are computed from the terminal
result and the step index. Best first experiment: `prize_margin`.

### Tier B — hidden-information / belief prediction (needs the opponent seat's obs, free in self-play)

| Name | Target | Label source |
|---|---|---|
| `opp_hand_size` | opponent's hand-card count | opponent seat obs |
| `opp_has_gust` | does opp hold a Boss/gust effect? (binary) | opponent seat obs |
| `opp_board_energy` | energy on opponent's board | opponent seat obs |
| `own_prizes` | which of *your* cards are stuck in prizes | `DeckTracker` deduction |
| `opp_lethal_next` | can opponent take lethal next turn? (binary) | sim roll / heuristic |

**This is the strong category** for our ceiling. Cost: plumb the opponent seat's
observation onto each `Step` (the same plumbing a centralized critic needs).

### Tier C — action-quality / representation (rarely the best, listed for completeness)

| Name | Target | Label source |
|---|---|---|
| `adv_sign` | was this a good pick? (sign of GAE advantage) | own `adv` |
| `reconstruct` | autoencode the option features | the inputs themselves |

## Recommendation

1. Ship `prize_margin` (Tier A) first — one-hour change, terminal label only.
   Watch `explained_var` and head-to-head win-rate move (or not).
2. Graduate to `opp_has_gust` / `opp_hand_size` (Tier B) once the opponent-obs
   plumbing exists.

## Design: modular, configurable, off by default (NOT bolted on forever)

Follow the codebase's existing idiom (`SHAPERS`/`ESTIMATORS` registries +
`reward_weights` dict defaulting to all-zero, stamped into the checkpoint config
hash). Do **not** hard-wire a single head.

- **Registry** `aux_tasks.py`: `AUX_TASKS: dict[str, AuxTask]`, where an
  `AuxTask` bundles `(label_fn, head_factory, loss_fn)`. Adding a new auxiliary =
  register one entry; nothing else changes.
- **Config** `TrainConfig.aux_weights: dict[str, float]` defaulting to every
  registered task at `0.0` (mirrors `reward_weights`). A task contributes to the
  loss **iff** its weight > 0. Serialized into the checkpoint config + folded
  into the config hash, so a run's aux setup is reproducible and a
  registry/weights mismatch is rejected on load.
- **Model**: build a head per task with weight > 0, keyed by name in an
  `nn.ModuleDict`. Because the set of active heads is config-derived, checkpoints
  stay compatible (a run with no aux has an empty `ModuleDict`).

Turn one on: `aux_weights={"prize_margin": 0.25}`. Turn all off (v1 behaviour,
bit-for-bit): `aux_weights={}` / all-zero — the default.

### Why registry, not "bolt it on forever"

- A permanently-on head **contaminates the config hash and checkpoint shape**
  even when you'd rather ablate it — you could never cleanly measure whether it
  helped.
- Different auxiliaries suit different decks/experiments; the ceiling work is
  explicitly about *measuring* levers. A registry + weight dict gives free
  ablation (set weight 0) and free A/B (two configs), exactly like the reward
  terms already do.
- Cost of the registry over a hard-coded head is ~30 lines, once.

## Inference: always ignored

Auxiliary heads are **training-only**. `evaluate()` / `value()` /
`policy_from_state()` never call them, and the numpy export (`policy.npz`, the
Kaggle bundle) simply doesn't export the aux `ModuleDict`. So:

- No inference cost, no submission-size cost.
- No risk to the torch↔numpy parity gate — the exported graph is unchanged.
- The heads exist only inside the `.pt` checkpoint, consumed only by the trainer.

## Sketch of the wiring (files touched)

- `aux_tasks.py` — new registry (label fns + head factories + loss fns).
- `model.py` — `nn.ModuleDict` of active heads built from config; a
  `aux_from_state(state) -> dict[name, pred]` accessor. Untouched inference path.
- `trainers/ppo.py` — record per-task labels on `Step` during `play_game`;
  collate them in `_minibatch`; add `sum(w * loss_fn(pred, label))` to the PPO
  loss; log each `aux/<name>` loss.
- `config.py` — `TrainConfig.aux_weights` field (default all-zero), in the hash.

See `docs/plans/` for the eventual implementation plan if/when we build it.
