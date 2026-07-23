# Attack-Damage Estimator — Scoping Plan

**Status: Phase 1 and Phase 2 DONE (2026-07-20), but coverage is lower than
this file originally claimed.** This file exists so the investigation and
design aren't lost if work stops before implementation is complete.

**Correction (2026-07-20, later same day):** the "Phase 1 result" and "What
actually got built" text below says coverage "turned out broader than
originally scoped." That was measured wrong — it checked whether each
pattern produced a nonzero *value* under an artificially empty test board,
which hides misses on patterns that need real board state (discard
contents, bench occupants, etc.) to fire at all. A direct text-match check
(does any pattern's regex match the attack's text at all, regardless of
board state) gives the real number: **only 86 of 195 real text-computed-damage
attacks (44%) are matched.** See `AGENTS.md` What's Next #12 for the full
breakdown of what's missing and why a growing regex list may not be the
right long-term shape for this at all — worth reading before extending this
module further.

**Phase 1 result:** New module `pkm/rl/attack_damage_estimator.py`
(`estimate_attack_damage` + `min_guaranteed_damage`), wired into
`pkm/rl/features.py:_attack_damage` and
`pkm/rl/deterministic_features.py:lethal_this_turn`. Implements 14 regex
patterns (more than originally scoped below -- see "What actually got built"
at the end of this file), verified with 0 exceptions across all 1556 real
attacks in the card database, plus 18 targeted unit tests using real
attackIds/text and 2 integration tests through the actual `lethal_this_turn`
feature function.

**Phase 2 result:** Added a 15th pattern -- the deck-mill family
("discard the top N cards of your deck, D damage per matching card
discarded that way"), which is what Hammer-lanche actually needed. Searched
the full card database for this text shape: Hammer-lanche (attackId 1046)
is currently the only real instance, though the pattern is written
generally. Uses `ctx.tracker.by_location(CardLocation.DECK)`
(GameContext/DeckTracker, already built for prize deduction -- no new
tracking capability needed) plus a hypergeometric expected-value
calculation: `min(mill_count, real_deck_count) * (matching / total_tracked)`.
Verified unbiased even before any full-deck reveal has happened in a game
(see `_remaining_deck_energy_fraction`'s docstring for the exchangeability
argument -- the undifferentiated deck-or-prize bucket has the same
matching-card density as the true deck alone, in expectation). Marked
non-deterministic (like the coin-flip patterns), so it's excluded from
`min_guaranteed_damage`/`lethal_this_turn` -- Hammer-lanche's damage stays
an estimate, never a claimed certainty. `ctx` is optional throughout
(defaults to `None`), so every pre-Phase-2 call site keeps working
unchanged; without a `ctx`, deck-mill patterns just contribute 0, same as
the pre-Phase-2 baseline. Verified with 0 exceptions across all 1556
attacks with a real `ctx` supplied, plus 5 new targeted tests (basic
expected value, capped-by-real-deck-size, zero-matching, empty-tracker,
exclusion-from-certainty).

**Combined test suite: 175 passing.** **Not yet retrained** -- Phase 3
(below) is still required before this changes any deployed checkpoint's
actual behavior; right now it only changes what the *feature* reports,
which an already-trained network hasn't learned to interpret differently
yet.

**Origin:** started from the user asking "why is the anchor deck doing so
poorly against `pool_400_mega_abomasnow_ex`" (Milestone 9 snapshot eval said
5%). Investigating that turned up two separate bugs — see
`AGENTS.md` → "Abomasnow Matchup Investigation (2026-07-20)" for the first one
(an eval-methodology mismatch: `eval-vs-pool` never wires in the archetype
belief classifier, so its numbers don't match what `pkm play`/Kaggle actually
run). **This plan is about the second one**, which is real and generalizes
well beyond Abomasnow.

---

## The story, in one paragraph

Decoded 20 real replays between `03_pult_munki` and `pool_400_mega_abomasnow_ex`
(scripts and raw replays exist locally — see "Artifacts" below). Across those
20 games the anchor lost 73 Pokémon total to the opponent's 31 — not just its
main attacker, its *entire* fragile support line (Budew, Dreepy, Drakloak,
Munkidori, Meowth ex, Moltres, Fezandipiti ex) kept dying. The single most-used
move in every game, and the most common finishing blow, was Mega Abomasnow ex's
**Hammer-lanche**. Its card data says `damage: 0`. Its actual effect, from the
card text: *"Discard the top 6 cards of your deck, and this attack does 100
damage for each Basic {W} Energy card that you discarded in this way."* That's
0–600 real damage, and the network's threat-assessment features never see any
of it, because two different places in the codebase read the static `damage`
field and stop there:

- `pkm/rl/deterministic_features.py:lethal_this_turn` (~line 40):
  `opp_active.hp - atk.damage <= 0`. For Hammer-lanche `atk.damage` is always
  `0`, so this attack can **never** be flagged as lethal, no matter how much
  damage it's about to do.
- `pkm/rl/features.py:_attack_damage` (~line 364): same static field, fed to
  the network directly as "how hard does this attack hit" for every option.
  Hammer-lanche always looks like a free 0-damage move.

This isn't an Abomasnow-specific quirk. It's a hole in how attack threat is
represented at all, and Abomasnow's deck just happens to lean on that hole
harder than anything else currently in the pool.

## Sizing it

Queried `all_attacks()` (1556 total attacks in the card database):

- 381 declare `damage: 0`. Of those, **195 actually deal damage via card
  text** (the other 186 are genuinely 0-damage utility moves — search, heal,
  status — correctly zero, not bugs).
- A second, smaller-but-related bug: attacks with `damage > 0` can *still*
  hide bonus damage in text. Dragapult ex's own **Phantom Dive** is
  `damage: 200` (captured correctly) but its text — "put 6 damage counters on
  your opponent's Benched Pokémon" — adds 60 more that no feature sees either.

Breaking the 195 down by mechanism (regex over attack text; counts are how
many of the 195 match each pattern):

| Mechanism | Count | Data needed |
|---|---:|---|
| Coin flip (`Comet Punch`, `Double Hit`, ...) | 45 | none — closed-form expected value |
| Fixed damage-counter placement (`Cursed Drop`: "4 counters") | 31 | none — literal constant in text |
| Energy attached to the attacker itself | 14 | `obs` — exact, see below |
| "each of your Pokémon with trait X" (`Round`) | 8 | own board + card data — exact |
| Energy of a specific type, other phrasings | 6 | `obs` — exact |
| Energy in **discard pile** (`Back Draft`, `Re-Brew`) | 5 | `obs` — exact, see below |
| Benched Pokémon count | 4 | `obs` — exact |
| Prizes taken | 3 | `obs` — exact |
| Unclassified (regex missed the phrasing) | 88 | mixed — mostly fixed bench-snipe constants ("60 damage to 1 of your opponent's Benched Pokémon") plus Hammer-lanche's own deck-mill family |

Two things made most of this cheaper than expected once checked against
`obs_data_structure/OBSERVATION_SCHEMA.md`:

- **`discard` is a public zone with full card identity** (`array<CardRef |
  null>`), for both players. "Energy in your discard pile" needs zero new
  tracking — just count matching cards in `player.discard`.
- **Per-Pokémon attached energy already has a full type breakdown**:
  `energies: array<int>` is one entry per attached energy unit, not just a
  total count. "40 damage per {W} Energy attached to this Pokémon" is a
  direct read, no new state.

So almost everything except two categories is *exactly* computable from the
observation alone, no `GameContext`/`DeckTracker` involvement needed:

1. **Coin flips** — inherently random even in principle; use closed-form
   expected value (`n_coins * damage_per_head * 0.5`, or the geometric-series
   form for "flip until tails" attacks like `Ball Roll`).
2. **Deck-mill effects (Hammer-lanche's own shape)** — "discard the top N of
   your deck, damage per matching card" depends on the *unrevealed order* of
   the attacker's own remaining deck. Not exactly computable, but the
   attacker's own remaining deck *composition* is knowable (60 cards minus
   everything already seen), so a **hypergeometric expected value** — P(k
   matching cards in the next N draws) × per-card damage — is a well-founded
   estimate. This is the category that specifically explains Hammer-lanche,
   and it's the hardest one on the list.

## Design

New module: `pkm/rl/attack_damage_estimator.py`, same shape as
`deterministic_features.py`/`reward_terms.py` — a registry, not a monolith.

```
estimate_attack_damage(attack, obs, ctx) -> float
```

Pattern-matched against attack text **once, at attack-table load time** (build
a `attackId -> estimator_fn` dict up front, the same way `get_attack_data()`
builds its lookup), falling back to the existing `atk.damage` when nothing
matches. That fallback is what makes this a strict refinement rather than a
rewrite — an attack that's already correctly represented can't get worse.

Two call sites change to use it instead of the raw field:
- `pkm/rl/features.py:_attack_damage`
- `pkm/rl/deterministic_features.py:lethal_this_turn` — this one needs more
  thought than a drop-in swap, since a mill-based estimate is a *probability*
  of lethal, not a certainty the way a fixed-damage attack's lethality is.
  Might need to become "probability this KOs," or gain a companion
  "damage is uncertain" flag feature, rather than staying a hard boolean for
  this one attack shape.

## Phasing

**Phase 1 — exact, no new tracking. DONE (2026-07-20).** Coin flips
(closed-form), fixed damage-counter constants, energy-attached-to-self, bench
count, prizes taken, discard-pile counts, plus a text-regex pass over the 88
"unclassified" bucket. All closed-form, no per-player deck tracking. Real
coverage turned out **broader** than this bullet originally guessed: while
implementing the "unclassified" pass, the single biggest win was a general
`"This attack does N damage to ..."` pattern (any target phrasing -- 1 of/2
of/each of your opponent's (Benched) Pokémon) that covers a large slice of
the 88 as pure fixed constants (`Cruel Arrow` 100, `Explosion Y` 280, `Sonic
Ripper` 220, etc. -- previously all reported as 0). Also added: opponent
hand-*size*-based damage (hand contents are hidden, but `handCount` is always
public), and Pokémon-Tool-count-based damage. Final pattern count: 14 (vs.
~8 categories originally scoped), split 11 deterministic / 3 expected-value
(coin flips) -- see `min_guaranteed_damage` below, added specifically so
`lethal_this_turn` doesn't treat a coin flip's expected damage as a
guarantee. Explicitly **not** chased in Phase 1 (fragile or out of scope):
deck-archetype keyword-in-name patterns ("Ancient", "Team Rocket's", "Erika's"
Pokémon -- no generic trait-tagging field exists in `CardData` for these),
opponent-hand-*contents*-based damage (the one genuinely hidden zone),
Pokémon-{V}-flag-based counting (not modeled in `CardData`, only `.ex` is).

**Phase 2 — deck-mill expected value. DONE (2026-07-20).** The Hammer-lanche
family specifically. Used (a) `GameContext`/`DeckTracker`'s existing
remaining-deck tracking (`by_location(CardLocation.DECK)`) -- extended, not
rebuilt; (b) a hypergeometric-expectation calculation. See "Phase 2 result"
at the top of this file for the details.

**Phase 3 — retrain, not optional.** Changing what an existing feature
*reports* for a checkpoint that's already trained is the same class of issue
as the belief-mismatch finding in the Abomasnow investigation: no dimension
change, but a real distribution shift for a network that's only ever seen
these attacks report ~0. Skipping a retrain here would mean deploying a
feature the current checkpoint was never trained to interpret. This folds into
the already-outstanding `AGENTS.md` "What's Next" item #1 (retrain-and-measure
ablations never actually run) rather than being a silent value swap on the
live checkpoint.

## Explicit non-goals (for now)

- Not attempting a general natural-language parser for arbitrary attack text.
  The regex-per-mechanism approach is deliberately narrow and additive; new
  mechanisms get their own pattern when they matter, not a general parser
  upfront.
- Not touching attacks that depend on genuinely hidden information (the
  opponent's hand — the one zone that really is hidden). None of the 195
  fell into this bucket in the initial pass, but it's worth a final audit
  before calling Phase 1 done.
- Not fixing the `eval-vs-pool` belief-classifier mismatch here — that's a
  separate, already-documented issue (`AGENTS.md`, What's Next #10).

## Artifacts from the investigation (local only, not committed)

- `scripts/run_matchup_replays.py`, `scripts/analyze_matchup_replays.py` —
  committed, used to gather and decode the 20 replays.
- `runs/03_pult_munki_vs_pool_400_mega_abomasnow_ex/` — the 20 raw replays +
  `summary.json` (gitignored via `runs/` in `.gitignore`; regenerate with
  `python scripts/run_matchup_replays.py 03_pult_munki
  pool_400_mega_abomasnow_ex --games 20` if this directory is gone by the time
  someone picks this back up).

## Where to resume

Phase 1 and Phase 2 are done and committed. The next concrete action is
Phase 3: the actual retrain-and-measure pass. Bundle both phases' fixes into
**one** retraining run, not two separate ones -- retraining twice would
waste a run and muddy which fix caused which win-rate change. This folds
into the already-outstanding `AGENTS.md` "What's Next" item #1
(retrain-and-measure ablations never actually run). Concretely: retrain
`03_pult_munki` (or whichever checkpoint this targets) with the new feature
values in place, then re-run `pkm eval-vs-pool` (belief-on, per the
belief-classifier-routing plan's Phase 1 fix) against the pool, especially
`pool_400_mega_abomasnow_ex`, and compare to the pre-fix baseline (45% from
the original investigation, 35% from the corrected `eval-vs-pool` number) to
see whether either fix actually moved the needle. This is a real decision
point (which checkpoint, how many iterations, whether to fold in the
belief-classifier-routing plan's Phase 3 redo question too) -- don't start a
training run without confirming scope first.

## What actually got built (for reference)

**Phase 1:** 14 regex patterns in `pkm/rl/attack_damage_estimator.py`, 11
deterministic + 3 expected-value-only (coin flips): fixed N-coin flips,
single-coin flips, flip-until-tails, fixed damage-counter placement ("in any
way you like"), energy attached to the attacker (typed and untyped),
Basic-Energy count in either player's discard pile (both the "X damage for
each" and "put N counters for each" phrasings), own benched-Pokémon count,
prizes taken (either side), teammate-has-a-named-attack (the `Round`
mechanic), the general fixed "this attack does N damage to ..." constant
(the single biggest coverage win, not originally sized in the table above),
opponent hand *size* (not contents), and Pokémon-Tool count on the caster's
own side.

**Phase 2:** a 15th pattern, the deck-mill family (Hammer-lanche's own
shape), non-deterministic like the coin flips. `estimate_attack_damage` and
`min_guaranteed_damage` both gained an optional `ctx: GameContext | None =
None` parameter (threaded through from `_attack_damage`/`lethal_this_turn`,
which already received `ctx` but previously ignored it) -- every handler
function's signature was updated to accept `ctx` uniformly, even the ones
that don't use it yet, so future patterns needing `ctx` don't require
another signature-wide change.
