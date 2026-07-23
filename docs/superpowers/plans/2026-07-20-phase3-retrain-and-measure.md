# Phase 3: Retrain-and-Measure Pass — Scoping Plan

**Status: DONE (2026-07-20) — ran, and it didn't work.** Two retrain
attempts (300 then 1500 more iterations, `--archetype-pool-prob` raised from
0.4 to 1.0 between them so every game samples one of the 26 decks equally
rather than mostly self-mirroring), both bundling the attack-damage-estimator
and belief-classifier-routing fixes as planned. Result: Abomasnow-specific
win rate (n=100 samples) went 28.0% → 22.0% → 29.0% — back to baseline, no
real gain. Overall pool average stayed flat (60-62%) throughout. A post-retrain
replay re-analysis (same methodology as the original investigation) found
the underlying casualty pattern **unchanged** — KOs scored against
`03_pult_munki` actually ticked up slightly (76 vs. the original 73),
Hammer-lanche still dominant in nearly every game, same fragile support line
still dying repeatedly. Full numbers in `AGENTS.md` → "Abomasnow Matchup
Investigation." **Follow-up direction:** `AGENTS.md` What's Next #13 — the
fix changed what the `attack_damage` feature reports, not what the AI
actually does; the real lever is likely a denser, more direct training
signal for card/attack danger (an auxiliary head, like the archetype-belief
one), not more iterations of the current setup.

**Origin:** the closing "Phase 3" of both
`docs/superpowers/plans/2026-07-20-attack-damage-estimator.md` (Phases 1+2
done) and `docs/superpowers/plans/2026-07-20-belief-classifier-routing.md`
(Phases 1+2 done). Both plans explicitly say not to run separate retrains —
bundle both fixes into one pass so any win-rate change stays attributable.
This file is that bundled scope.

---

## Finding 1: the Milestone 8 checkpoint no longer exists

Checked this before scoping anything, because the belief-classifier-routing
plan's own Phase 3 section presented "restart from Milestone 8" as an
option. It isn't one anymore.

Both `pkm/rl/train.py` (Milestone 8, solo) and `pkm/rl/population_train.py`
(Milestone 9, population) write numbered snapshots to the **same**
`agents/03_pult_munki/checkpoints/` directory using the **same**
`ppo_iter{it:04d}.pt` naming convention — but each script's `it` counter
restarts at 1 independently. Milestone 9 ran 2375 iterations of its own
counter, so its iteration 2000 silently overwrote the file Milestone 8's
iteration 2000 had written earlier. Verified directly: `population_train.csv`
has exactly 2375 data rows (counter really did restart at 1) and has its own
row at `iter=2000` (`2000,52,24,28,0,...,1.0000,20` — the "52 games/iter"
shape is population training's, not solo training's "16 games/iter" shape).
`ppo_iter2000.pt`'s mtime (05:04:45) sits smoothly inside Milestone 9's own
run timeline (iter1990 at 05:02:32, iter2010 at 05:07:09 — evenly paced, no
gap), not near Milestone 8's actual finish time. So `ppo_iter2000.pt` on
disk today is Milestone 9's own checkpoint, not Milestone 8's — and no
separate copy of Milestone 8's true endpoint survives locally. (Contrast
with the belief-dimension-resize transition, where `checkpoints_pre_belief_resize/`
*was* deliberately preserved before that retrain started — this same care
wasn't taken between Milestone 8 and Milestone 9.) Nothing in `AGENTS.md`'s
Hugging Face section indicates Milestone 8 was ever uploaded either — only
`ppo_latest.pt` (Milestone 9's final state) is in the bulk-upload loop.

**Consequence:** there are only two realistic starting points for Phase 3:
1. **Resume from `ppo_latest.pt`** (iter 2375, Milestone 9's actual endpoint).
2. **Full from-scratch retrain** (iteration 0) — redoing both Milestone 8's
   ~2000 iterations and Milestone 9's ~2375, at roughly the same wall-clock
   cost as both original runs combined. Not recommended unless there's a
   specific reason to want a clean, unconfounded run badly enough to pay for
   it.

This report assumes (1) unless told otherwise.

## Finding 2: the code fixes alone, without retraining, do NOT show a clear improvement — and the reason matters

Ran `pkm eval-vs-pool --agent 03_pult_munki --games 20` against the
**unretrained** iter-2375 checkpoint, now that both fix families' feature
code is live (attack-damage-estimator Phases 1+2 are unconditional in
`features.py`/`deterministic_features.py`; `eval-vs-pool` has defaulted to
live belief since the belief-classifier-routing Phase 1 fix). Result: 61.1%
overall, 50.0% vs `pool_400_mega_abomasnow_ex` specifically — up from the
35% measured right after the belief-only fix landed (before the
attack-damage fix). Read at face value, that looks like a large "free"
improvement from the attack-damage fix alone, no retrain needed.

**It isn't trustworthy at face value.** Diffed this run bot-by-bot against
the belief-only-fix run (same checkpoint both times — only the
attack-damage feature code changed in between): mean absolute swing across
all 26 bots was **10.5 percentage points**, with a 40-point swing on
`pool_315_archaludon_ex` — a bot with no plausible causal link to an
attack-damage-feature fix. That's the actual noise floor of a 20-game
sample here (consistent with the math: SE ≈ 10.7pts at n=20, i.e. a ~21pt-wide
95% CI). The Abomasnow matchup's 35%→50% swing is well inside that noise
band — not evidence of anything.

Re-ran Abomasnow specifically with a much larger sample:
`pkm eval-vs-pool --agent 03_pult_munki --games 100 --pool-glob "pool_400_mega_abomasnow_ex"`
→ **28.0%** (SE ≈ 4.5pts, ~9pt-wide 95% CI) — *lower* than both prior
small-sample reads, not higher. The honest conclusion: **the code fixes
alone, without retraining, have not moved this matchup** (the current best
estimate, 28.0% ± ~9pts, sits close to where the belief-only-fix's noisy
35% reading was, not near the noisy 50% one). This makes sense in
retrospect — the checkpoint was never trained on a world where
`attack_damage`/`lethal_this_turn` report real values for Hammer-lanche-shaped
attacks; there's no a priori reason its existing weights would already know
what to do with information they've never seen before.

**Why this matters for scoping:** Phase 3 isn't a "polish the win rate a
bit more" step — it's very plausibly the step that actually determines
whether Phases 1/2 pay off at all. Treat it as load-bearing, not optional
follow-up.

## What Phase 3 actually needs to do

1. **Resume from `ppo_latest.pt`**, since Finding 1 removes any cleaner
   option.
2. **Enable both fixes in the training run itself**, not just at eval time:
   - Attack-damage-estimator (Phases 1+2): already unconditional in the
     feature code, nothing to flip on.
   - Belief-classifier-routing (Phase 2): `pkm population-train
     --archetype-belief [--archetype-weights pkm/archetype.npz]` — this
     flag exists and is unit-tested but has never actually been run. Running
     it here also **completes belief-classifier-routing's own Phase 3**
     (that plan explicitly asked whether to resume-with-belief-newly-on vs.
     restart-from-Milestone-8; Finding 1 above answers that question by
     elimination — restart isn't available, so resume-with-belief-on is the
     only option left. Update that plan file to reflect this once this one
     is confirmed.)
3. **A modest iteration budget for a first checkpoint, not a marathon.**
   Milestone 9's `eval_win_rate` (vs. random) was already pegged at
   0.90–1.00 for its last ~150 logged iterations — it was not undertrained
   on its own original objective. The open question is adaptation to *new*
   feature values, which doesn't obviously need thousands of iterations to
   show a first signal. At the observed pace near the end of Milestone 9
   (~16–18 sec/iteration, from checkpoint mtimes: iter2340→2350→2360→2370
   each ~2.5–3 min apart), **300 iterations ≈ 75–90 minutes wall-clock**,
   **500 ≈ 125–150 minutes**. Proposing 300 as a first checkpoint — enough
   to see whether the metrics are moving in a sane direction at all — before
   committing to more.
4. **Measurement protocol, informed by Finding 2's noise audit:**
   - Baseline (already measured, reusable): 61.1% overall / **28.0% (n=100)
     vs Abomasnow specifically**.
   - After the run: re-measure the same way. For the Abomasnow-specific
     number, **use n≥100 games again**, not the 20-game default — Finding 2
     showed 20 games isn't trustworthy for a single-matchup comparison. The
     full-pool 20-games/bot aggregate is probably fine for the *overall*
     number (26 bots' independent noise partially averages out at the
     aggregate level), but don't over-read any single bot's delta without a
     bigger sample, the same way the pre-registered 50% reading turned out
     not to replicate.
   - Also worth re-running the qualitative side:
     `scripts/run_matchup_replays.py` + `scripts/analyze_matchup_replays.py`
     against Abomasnow again post-retrain, to check whether the casualty
     imbalance found in the original investigation (anchor lost 73 Pokémon
     vs. Abomasnow's 31, across the whole fragile support line, not just the
     main attacker) actually shifts — a qualitative check that a pure
     win-rate number can miss (e.g. still losing but losing less
     catastrophically, or winning via a different, more sensible line of
     play).

## Open decisions (need your call before starting)

1. **Confirm resuming from `ppo_latest.pt`** (the only available option per
   Finding 1) rather than a full from-scratch retrain.
2. **Iteration budget** — 300 as a first checkpoint (~75–90 min) is the
   proposal; say if that's the wrong ballpark given time/compute
   constraints.
3. **Enable `--archetype-belief` in this same run** — recommended (avoids a
   second retrain for belief-classifier-routing's own Phase 3), but it does
   mean this run tests *two* changes at once relative to Milestone 9's
   original config (new attack-damage features -- always on regardless --
   and now-live belief). If you want the two fixes' effects kept
   separable, that needs two runs instead of one, at roughly double the
   compute cost -- want that instead?

## Where to resume

Nothing is running. Once the open decisions above are answered, the
concrete command (pending confirmation of iteration count) is:

```bash
pkm population-train --agent 03_pult_munki \
  --iterations 300 --archetype-belief \
  # other flags default to Milestone 9's own config (games-per-pairing=2,
  # workers=8, pool_glob=pool_*, eval-every=10) unless told otherwise
```

then re-run the measurement protocol above (both the full-pool
`eval-vs-pool` sweep and the n≥100 Abomasnow-specific check) and compare
against this file's baseline (61.1% overall / 28.0% n=100 vs Abomasnow).
