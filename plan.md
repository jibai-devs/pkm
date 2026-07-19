# Heuristic & Soft-Signal Feature Architecture
 
> Status: Locked-in architecture reference. This is a design decision meant to
> stand for the rest of the competition — prefer extending it (new registry
> entries, new heads) over revisiting the shapes described here. Supersedes/extends
> docs/ideas/heuristic-features-plan.md (read that first for deck context and
> motivation; this doc is the concrete "how").
 
Goal: Give the policy/value network exact game facts (deterministic features)
and learned beliefs about hidden information (probabilistic features), without
capping the network's skill ceiling at what a hand-written heuristic could do, and
without introducing silent correctness bugs from statefulness where it isn't
needed.
 
Target deck for validation: deck/02_dragapult.csv (Dragapult ex / Dusknoir).
 
---
 
## 1. Two axes that fully determine where a new feature goes
 
Every new heuristic must be classified along two independent axes before writing
any code. Getting a feature's home right on both axes is the whole point of this
document — everything below is a consequence of these two axes.
 
### Axis A — What is the value *about*? (scope)
 
| Scope | Meaning | Tensor it joins | Example |
|---|---|---|---|
| GLOBAL | one fact about the whole game | state_feats | turn number, prize counts |
| PER_SLOT | one fact about one Pokémon on the board | joins that slot's row (board is 19 slots: my active + 8 bench, opp active + 8 bench, stadium) | HP, energy count |
| PER_OPTION | one fact about one *legal move being considered right now* | opt_feats | "does this attack KO the target" |
| PER_DECK_CARD | one fact about one unique card in my own decklist | new tensor, fixed-width (§4) | "copies of this card not yet seen" |
 
This choice is forced by what the number is about, not a style preference — see
docs/ideas/heuristic-features-plan.md §"Architecture for extensibility" for the
original reasoning.
 
### Axis B — How is the value computed? (provenance)
 
| Provenance | Computed by | Validated by | Can be wrong? |
|---|---|---|---|
| Deterministic | a pure function of the current observation (+ static card data) | a unit test with one known right answer | No — if it's ever wrong, that's a bug, not noise |
| Probabilistic | a trained sub-model (an auxiliary head, §6) | held-out accuracy / calibration, plus an ablation A/B on win rate | Yes, and that's expected — it's an estimate over hidden information |
 
Axis B does not change which scope a feature belongs to. It changes how you
test it, how confident to treat its output, and (for probabilistic features only)
that it needs a training procedure before it can be trusted.
 
A third property, orthogonal to both axes, matters for implementation but not for
scope/provenance classification:
 
### Axis C — Does it need memory across turns? (statefulness)
 
Default answer is no — nearly everything here is a pure function of the
*current* observation snapshot, because the game already re-shows accumulated
history every snapshot (discard piles only grow, board/hand reflect current
truth). Treat every feature as stateless unless you can name the specific reason
it can't be (§7 documents the one confirmed exception: prize deduction via
full-deck reveals).
 
---
 
## 2. The registry pattern
 
Replace the current hand-built, manually-indexed feature lists in
pkm/rl/encoder.py (encode_state/encode_options appending floats to a list in
a fixed order, with widths tracked as separate constants) with a declarative
registry:
 
`python
class Scope(Enum):
    GLOBAL = auto()
    PER_SLOT = auto()
    PER_OPTION = auto()
    PER_DECK_CARD = auto()
 
@dataclass
class FeatureSpec:
    name: str
    width: int
    scope: Scope
    fn: Callable[..., np.ndarray]   # pure fn of (obs, static card data, ...) -> width-shaped array
    deterministic: bool             # Axis B, for tooling/tests, not behavior
 
Each scope is assembled by its own loop over the specs tagged with it, in a fixed
registration order (order matters for tensor layout — see `FEATURE_ORDER` below).
Concretely:
 
- `encode_state` becomes: for each `GLOBAL` spec in order, call `fn(obs)`,
  concatenate. Same for `PER_SLOT` specs, once per occupied board slot.
- `encode_options` becomes: for each `PER_OPTION` spec in order, call
  `fn(obs, option)` for every option, stack.
- A new `encode_deck_ledger` assembles `PER_DECK_CARD` specs against the fixed
  60-wide card-slot mapping (§4).
 
**One global constant enumerates registration order and is the single source of
truth for widths** — no more hand-maintained `STATE_FEATS`/`OPT_FEATS` integers
computed separately from what the functions actually emit. `STATE_IN`/`OPT_IN` in
`pkm/rl/model.py` are derived from summing registered widths, not hard-coded.
 
**Ablation config:** a `FeatureConfig` (dict of `name -> bool`) threaded through
`encode_state`/`encode_options`/`encode_deck_ledger`; a disabled feature is masked
to zero rather than omitted, so tensor shape never depends on the config
(critical: shape must be config-independent, or checkpoints trained under
different configs become incomparable/unloadable).
 
**Checkpoint/feature-set versioning:** every checkpoint is stamped with the
ordered list of registered feature names + widths it was trained with (a hash or
literal list, saved alongside model weights). Loading a checkpoint against a
registry whose stamp doesn't match **must fail loudly** — never silently
misalign the input vector. This is the mechanism that makes "retrain from
scratch is acceptable" (below) safe rather than merely assumed.
 
---
 
## 3. Breaking-change acceptance
 
Any width change to any tensor (`STATE_FEATS`, `OPT_FEATS`, the new
`PER_DECK_CARD` tensor) changes the first linear layer's input shape in
`pkm/rl/model.py` (`STATE_IN`/`OPT_IN`) — **existing checkpoints will not load**.
This is accepted, permanently, as the cost of this architecture: Phase 1 is short
(~200 iterations) and Phase 2 has barely started, so retraining from scratch after
an architecture change is cheap relative to engineering a weight-preserving
migration. Every phase below (§8) ends in a full retrain, not an incremental
patch.
 
---
 
## 4. Deterministic — Tier 1 (per-option / per-slot, no new plumbing)
 
These reuse data already in `pkm/data/card_data.py` (`weakness`, `resistance`,
`retreat_cost`, `energy_type`, per-attack `damage`/`energies`) and the current
observation. Land as one batch — same risk tier, no new plumbing required.
 
| Feature | Scope | Definition |
|---|---|---|
| `lethal_this_turn` | `PER_OPTION` (attack options only) | `1.0` if `target.hp - damage_counters - attack.damage <= 0`, else `0.0` |
| `type_effectiveness` | `PER_OPTION` (attack options only) | lookup: attacker `energy_type` vs. target `weakness`/`resistance` → a small fixed set of multiplier buckets |
| `retreat_viable` | `PER_SLOT` (bench) | bench Pokémon's `retreat_cost` vs. active's attached energy count |
 
Each gets a fixture-based unit test in `tests/fixtures/observations.json` with a
hand-computed expected value — no statistical validation needed, these are exact.
 
---
 
## 5. Deterministic — Tier 2: your own deck ledger
 
**What it answers:** for every unique card in *your* decklist, "how many copies
have I not seen yet (still in my draw pile or hidden in my prizes)?"
 
**Computation (stateless, recomputed every decision):**
 
unseen[card_id] = decklist_count[card_id] - count_visible_now[card_id]
count_visible_now[card_id] = count of card_id across:
    my hand, my discard, my board (active + bench, including attached tools/energies), my stadium (if mine)

 
This is a pure function of the current observation plus the static decklist known
at deck-load time. **No history tracking required**: if I take a prize, the
physical card enters my hand, and hand is already fully visible in the current
snapshot — `count_visible_now` picks it up automatically next decision. There is
no separate "prize reveal" event to detect for this tier.
 
**Width decision — locked in:** fixed-width, **60 slots**, zero-padded for unused
slots, indexed by a per-deck `card_id -> slot` mapping built once at deck-load
time and cached for the game's duration. 60 is a safe upper bound (a 60-card deck
has at most 60 unique card types) and — critically — this makes the tensor shape
identical across *any* deck, so "multi-deck training" (`AGENTS.md` → "What's Next"
#5) never forces another architecture change. This resolves the "Open decision:
card-count feature width" from `docs/ideas/heuristic-features-plan.md` in favor of
option (b).
 
**Scope:** `PER_DECK_CARD`, new tensor, feeds forward as its own input block next
to `state_feats` (conceptually a `GLOBAL`-like block, just wide and indexed by
card rather than a handful of hand-picked scalars).
 
**Plumbing needed:** `encode_decision` must receive the acting player's decklist
(already carried by `TorchPolicy`/`play_game` at construction time per
`pkm/rl/rollout.py`, just not yet passed into the encoder) plus a per-deck
`card_id -> slot` mapping built once when the deck loads.
 
**Testing:** fixture-based, exact — same rigor as Tier 1.
 
---
 
## 6. Deterministic — Tier 3: opponent sighting tally
 
**What it answers:** "how many of this specific card have I actually seen the
opponent play or discard so far?" — **not** a full ledger, because you don't know
their decklist size or composition.
 
**Computation (stateless):** a tally, per card ID seen in the opponent's discard
pile or board, of how many copies are currently visible there. No "remaining"
count is derivable (unknown total), so this is deliberately a smaller, capped
signal (e.g. normalize by a small constant like 4, since most cards cap at 3-4
copies) rather than forced into the same 60-wide-ledger shape as Tier 2.
 
**Priority:** lower than Tier 2. This tier's main value is as future input to the
opponent-archetype classifier (§6.1) rather than as a standalone signal — build it
alongside that work, not before.
 
### 6.1 Non-goal for this tier (explicitly deferred)
 
A full opponent-side ledger (assuming their decklist from a guessed archetype,
then running Tier-2-style math against the assumption) is a legitimate future
extension but requires the archetype classifier (§9) to exist first, and layers a
probabilistic assumption on top of deterministic math. **Deferred; do not build
until §9 lands and is validated.**
 
---
 
## 7. Deterministic-but-stateful — Tier 4: prize deduction via full-deck reveals
 
This is the one confirmed exception to "everything is stateless." When a
search/tutor effect reveals your **entire remaining deck** (not a filtered
subset), you can deduce your prize contents by elimination:
 
prize_multiset = full_decklist - count_visible_now (hand+discard+bnot revealed_remaining_deck
`
 
This is **not** recoverable by recomputing from the current snapshot at a later
turn, because the observation's `deck`/`looking` fields (`pkm/types/obs.py`) are
transient — tied to that one decision, not re-shown afterward. Once resolved, the
information disappears from future snapshots unless captured at the moment it's
revealed.
 
### 7.1 Correctness requirements (non-negotiable — get any of these wrong and the
feature poisons the rest of thOnly trust reveals verified as whole-deck reveals.st reveals verified as whole-deck reveals.** A filtered search (e.g.
   "search for a Water not") that only shows matching cards is **not**    sufficient — treating it as a full reveal would wrongly mark non-matching
   cards as "confirmed in deck" or "confirmed in prizes" when they were simply
   never shown. Each search-capable card in the deck's pool must be individually
   checked against its actual effect text / engine behavior before being added to
   the trusted-trigger list. Default to not capturing unless verified.
2. Shuffling scrambles order, not composition. Deduced prize contents remain
   valid across a subsequent shuffle of the same known card pool.
3. **Any effect that shuffles a prize back into the deck, or swaps a prize,
   invalidates previously-locked-in prize knowledge.** Before trusting this
   tier in production, audit the card pool for such effects. If any exist (or
   cannot be ruled out with confidence), the tracker must listen for them and
   invalidate the relevant locked-in facts — silently stale-but-confident data is
   worse than admitting "unknown."
 
### 7.2 Lifecycle
 
- One tracker instance per game, created fresh alongside the decklist already
  threaded into play_game/rollout construction (pkm/rl/rollout.py) — this
  rides the existing per-game plumbing rather than adding a new one.
- Must never persist across game boundaries. Self-play runs many games
  back-to-back; a leaked tracker reference would silently contaminate one game's
  prize knowledge into an unrelated game. Guard this explicitly (construct
  per-episode, never reuse/cache the object).
- Feeds the same PER_DECK_CARD tensor as Tier 2, by resolving specific
  previously-"unseen" slots to a confirmed-prized state once deduced (still a
  count, not a new tensor — Tier 4 sharpens Tier 2's numbers, it doesn't add a
  parallel structure).
 
### 7.3 Testing
 
Not fixture-snapshot testable (it's sequence-dependent). Requires scripted
multi-turn test sequences: draw → search/reveal → shuffle → take a prize →
assert the tracker's belief. Also test the never-leaks-across-games invariant
directly (construct two trackers for two distinct games, assert no shared state).
 
### 7.4 Priority
 
Build after Tiers 1-2 are landed and measured, and only once the correctness
audit (§7.1.3) is complete. This is the highest-risk deterministic tier precisely
because it's stateful — errors compound silently across a whole game instead of
self-correcting each turn the way stateless tiers do.
 
---
 
## 8. Probabilistic features — multi-head model architecture
 
### 8.1 Shape
 
Extend the existing shared-trunk design in pkm/rl/model.py (h already feeds a
policy head and a value head) with auxiliary heads, one per probabilistic
estimate, off the same trunk:
 
board/hand/state_feats/deck_ledger ──► shared trunk ──► h
                                          │
                ┌─────────────────────────┼─────────────────────────┐
                ▼                         ▼                         ▼
          policy head              value head              auxiliary head(s)
       (existing, unchanged)   (existing, unchanged)   e.g. opponent-archetype classifier
                                                                  │
                                                        softmax, .detach()
                                                                  │
                                                                  ▼
                                                  re-injected as a GLOBAL feature,
                                                  same registry as deterministic features
 
### 8.2 Non-negotiable design rules
 
1. Auxiliary heads train only against their own supervised loss (cross-entropy
   against the true label — available for free in self-play, since both decks are
   known to the training process). They must never receive gradient from the    PPO policy/value loss. Enforce this with .detach() on the head's output
   before it's concatenated back in as a feature. Without this, the auxiliary head
   can drift into predicting "whatever helps win rate" instead of the true hidden
   variable, destroying its calibratability and the reason it's testable/trusted
   at all.
2. Combined loss is policy_loss + value_loss + Σ λᵢ · aux_loss_i, added at
   the same point pkm/rl/train.py already combines policy_loss/value_loss.
   Each auxiliary head gets its own λᵢ weight, tuned independently.
3. **A new auxiliary head must be pretrained/validated standalone before its
   output is wired into the re-injection path.** Train it (or at least confirm
   its standalone accuracy) with the rest of the network frozen or ignored first.
   Do not combine "new untrained head" with "feed its noisy output back into the
   policy" in the same experiment — this is the same confound-isolation argument
   used throughout this doc: change one unproven thing at a time.
4. Each auxiliary head's output width is fixed at design time (include an
   explicit "Other/Unknown" class for archetype-style classifiers to absorb
   future unseen categories without a width change) — but note that even with
   this mitigation, growing the head's class count is still a breaking change per
   §3, since the output re-joins state_feats. Choose the initial class count
   deliberately; do not plan on painless resizing later.
 
### 8.3 First concrete instance: opponent-archetype belief
 
- Input: the shared trunk's h (which already has access to everything
  visible about the opponent's board/discard) plus, once built, the Tier 3
  sighting tally (§6).
- Output: a probability vector over a fixed, deliberately chosen list of
  tracked archetypes + one "Other" bucket.
- Label source: self-play games know both decks exactly — train the head
  with real cross-entropy against ground truth.
- Re-injection: detached softmax output, GLOBAL scope, same registry as
  everything else.
 
### 8.4 Non-goal for this section
 
Do not build a second, fully standalone classifier network outside the trunk —
it forfeits the trunk's learned board representation and creates two models to
keep in sync. The auxiliary-head-on-shared-trunk shape (§8.1) is the only
sanctioned pattern for probabilistic features in this architecture.
 
> **2026-07-19 deliberate exception:** the opponent-archetype classifier
> described in `docs/opponent-archetype-classifier-plan.md` is being built as
> a standalone network in its own package (`pkm/archetype/`), not as a trunk
> aux head, overriding this rule on purpose. Reasoning: it recognizes 25 real
> competitive meta-archetypes from external data (`staples.json`), a
> different and much larger problem than the trunk's original 3-class
> own-deck bookkeeping head (`pkm/rl/features.py:ARCHETYPE_CLASSES`); keeping
> it separate means its class list/architecture can change without forcing a
> full retrain of policy/value. The old 3-class aux head's re-injection path
> is slated for retirement in favor of this classifier's belief output once
> Part 2 (same doc) lands — see that doc for the full plan.
 
---
 
## 9. Build order (do not parallelize across tiers — each step's lift must be
measured in isolation before starting the next)
 
1. Registry refactor (FeatureSpec, scopes, FeatureConfig ablation,
   checkpoint stamping) — mechanical, behavior-preserving. No new heuristics yet.
2. Tier 1 deterministic features (lethal, type effectiveness, retreat
   viability) as one batch. Retrain from scratch. Measure lift via pkm eval
   with ablation flags.
3. Tier 2 own-deck ledger. Resolve the card_id -> slot mapping, thread the
   decklist into encode_decision. Retrain, measure in isolation from Tier 1's
   already-confirmed lift.
4. Tier 3 opponent sighting tally — small, cheap, can land alongside the start
   of the auxiliary-head work below since it's an input to it.
5. Auxiliary-head infrastructure + opponent-archetype classifier (§8) — build
   standalone, validate accuracy, only then wire in the detached re-injection
   path. Retrain, measure in isolation.
6. Tier 4 prize deduction via full-deck reveals — only after the correctness
   audit (§7.1.3) is complete. Highest engineering risk; build last.
7. Sequencing/combo heuristics (ability-ping + Boss's Orders + attack ordering)
   and Risky Ruins risk timing remain out of scope for this document — per
   docs/ideas/heuristic-features-plan.md, these require lookahead and belong in
   MCTS priors (Phase 2 expert iteration, pkm/mcts/), not as input features here.
 
---
 
## 10. Summary table
 
| Tier | Scope | Provenance | Stateful? | Priority | Risk |
|---|---|---|---|---|---|
| 1: lethal/type/retreat | PER_OPTION/PER_SLOT | Deterministic | No | Highest | Low |
| 2: own-deck ledger | PER_DECK_CARD | Deterministic | No | High | Low |
| 3: opponent sighting tally | PER_DECK_CARD (capped) | Deterministic | No | Medium | Low |
| 4: prize deduction | PER_DECK_CARD (resolves Tier 2 slots) | Deterministic | Yes | Medium-low | High — silent, non-self-correcting corruption if wrong |
| Opponent archetype belief | GLOBAL (re-injected) | Probabilistic | No (recomputed from h each decision) | Medium | Medium — calibration risk, mitigated by .detach() |
 
Every row lands through the same registry (§2); the difference between rows is
entirely captured by scope + provenance + statefulness, never by a special case
in the model architecture itself.
 