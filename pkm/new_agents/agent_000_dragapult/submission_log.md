# Submission log — agent_000_dragapult

Kaggle competition: **pokemon-tcg-ai-battle** (The Pokémon Company — PTCG AI
Battle Challenge). One row per submission, newest first. Score is the Kaggle
leaderboard score; check with `pkm new_agents 000_dragapult status --watch`.

Context: the live leaderboard **does discriminate** — recent policy-only
submissions (2026-07-17 → 07-19) score roughly **407–458** (best 457.9), not the
flat 600 the older notes describe (that number came from the saturated
vs-random *training* metric, not the real leaderboard; see `TRAINING.md` §10).
So there IS headroom, and inference-time search is a direct lever on it.

Recent policy-only baseline (for comparison), newest first:
457.9 · 448.3 · 446.7 · 435.0 · 407.8 · 389.4 · 328.0 · 317.4.

| Date (UTC+8) | Checkpoint | Inference | Bundle | Message | Score |
|---|---|---|---|---|---|
| 2026-07-24 14:54 | `016_alakazam_exit_attn_large/ckpt_256.pt` (large, **attn** transformer-decoder head, n_dec_layers 3; run still training @ ~upd 268) | policy (`--inference policy`) | `submission_20260724_145359.tar.gz` | 016 alakazam ExIt ATTN head (transformer decoder, large, n_dec_layers 3) ckpt_256, POLICY | _pending — inference **verified locally**: bundle rebuilds the attn decoder, full self-play game to a decisive result, 153/153 attn decisions valid, 0 invalid. First-ever attn deploy; early ckpt (256/8192)_ |
| 2026-07-24 08:07 | `015_alakazam_exit_combo_large/ckpt_768.pt` (large, combo head) | policy (`--inference policy`) | `submission_20260724_003257.tar.gz` | 015 alakazam ExIt combo large ckpt_768, POLICY (--inference policy) (ref 54938631) | _submitted after the UTC-midnight cap reset (cron `0d9ccf51` fired); **pending**_ |
| 2026-07-24 00:32 | `010_alakazam_exit_tdlambda_medium/ckpt_4608.pt` (medium, marginal head) | policy (`--inference policy`) | `submission_20260724_003238.tar.gz` | 010 alakazam ExIt TD(lambda) medium ckpt_4608, POLICY (--inference policy) (ref 54932458) | **583.0** — same net as the 723.4 run below (identical `policy` ≡ `mcts K=0`); the ~140 pt gap is **leaderboard variance** on one checkpoint |
| 2026-07-23 23:57 | `010_alakazam_exit_tdlambda_medium/ckpt_4608.pt` (medium, marginal head) | policy (K=0) | `submission_20260723_235740.tar.gz` | 010 alakazam ExIt TD(lambda) medium ckpt_4608, POLICY only (K=0) (ref 54931869) | **723.4** — new high (prev best 639.7) |
| 2026-07-23 23:56 | `010_alakazam_exit_tdlambda_medium/ckpt_4480.pt` (medium, marginal head) | policy (K=0) | `submission_20260723_235649.tar.gz` | 010 alakazam ExIt TD(lambda) medium ckpt_4480, POLICY only (K=0) (ref 54931859) | **425.3** |
| 2026-07-22 19:50 | `010_alakazam_exit_tdlambda_medium/latest.pt` (~1581/8192 upd, still training) | **MCTS K=256, W=2** | `submission_20260722_195039.tar.gz` | 010_alakazam_exit_tdlambda_medium: latest.pt iter ~1581/8192, MCTS K=256 W=2 — real per-decision timing measured first on 2 vCPU (pinned via `taskset`), same checkpoint's own build: mean 644ms, p50 624ms, p90 985ms, **p99/max 1.14s**, safely under a 2s/decision budget (K=512,W=2 was measured first and rejected: p99 2.28s, over budget) (ref 54901809) | _pending — check `pkm new_agents 000_dragapult status --watch`_ |
| 2026-07-22 19:27 | `010_alakazam_exit_tdlambda_medium/latest.pt` (1483/8192 upd, still training) | policy (no MCTS) | `submission_20260722_192735.tar.gz` | 010_alakazam_exit_tdlambda_medium: latest.pt iter 1483/8192, ExIt TD(lambda)+w4 medium, policy (no MCTS) (ref 54901418) | _pending — check `pkm new_agents 000_dragapult status --watch`_ |
| 2026-07-21 08:08 | `009_alakazam_xl/latest.pt` (6341/8192 upd) | **MCTS K=1** | `submission_20260721_080835.tar.gz` | **alakazam · XL + base_residual · cosine LR 1e-4→1e-5 · prize_potential · MCTS K=1** (ref 54864754) | **600.0** |
| 2026-07-21 01:11 | `007_alakazam_large/latest.pt` (256 upd) | **MCTS K=1** | `submission_20260721_011110.tar.gz` | **NEW DECK: alakazam** (Mega Alakazam/Dudunsparce) · large + base_residual · prize_potential shaping · MCTS K=1 | _not submitted: HTTP 400 (daily limit); superseded by the 009 XL bundle above_ |
| 2026-07-20 21:30 | `007_xl_residual/latest.pt` (256 upd) | **MCTS K=1** | `submission_20260720_213058.tar.gz` | 007 xl + base_residual (skip conns) + swept trial-16 + prize_margin aux + MCTS K=1 | _pending_ |
| 2026-07-20 20:12 | `003_aux_loss/latest.pt` (1024 upd) | **MCTS K=1** | `submission_20260720_201252.tar.gz` | 003_aux_loss latest (large + heuristic + prize_margin aux) + MCTS K=1 | _pending_ |
| 2026-07-20 09:06 | `002_large_tuned/ckpt_512.pt` | **MCTS K=4** | `submission_20260720_090651.tar.gz` | 002_large_tuned ckpt_512 + MCTS K=4 (more search than the K=1) | **448.9** |
| 2026-07-20 08:48 | `002_large_tuned/ckpt_512.pt` | **MCTS K=1** | `submission_20260720_084819.tar.gz` | 002_large_tuned ckpt_512 + MCTS K=1 (inference-time search smoke) | **487.2** (drifted down from an early 600) |

## Notes

- **2026-07-20 — first inference-time MCTS submission (K=1).** Smoke test of the
  new inference-time MCTS path (README §9 item 1, shipped 2026-07-20). K=1 is a
  single-simulation search — deliberately minimal to (a) confirm the MCTS
  decision path runs inside Kaggle's sandbox against its own `libcg.so` search
  symbols, and (b) stay well within the per-turn + cumulative 600 s time budget.
  **Result: 600.0 — the top score, up ~150 points from the ~448 policy-only
  baseline (same net, same checkpoint).** Even a single-simulation search
  (one determinized world + a value-net leaf eval / one-ply lookahead) clearly
  beats the raw policy head. It also completed within Kaggle's time budget, so
  the inference-MCTS path runs cleanly in the sandbox against its `libcg.so`
  search symbols. **Next:** try higher K (8 / 16 / 32) — watch the per-turn +
  cumulative 600 s clock as K grows — to see how far search pushes the score.
- **2026-07-20 — scores drift; the early "600" was not stable.** The K=1 smoke
  above read 600.0 shortly after landing but has since settled to ~487 as more
  league matches played out — leaderboard scores here are a moving average, so
  treat any fresh score as provisional for a while.
- **2026-07-20 — K=4 did NOT beat K=1 (448.9 vs 487.2), same ckpt_512.** More
  search made it *worse* here. Plausible reads: with only a value-net leaf eval
  and no learned search-time exploration tuning, deeper determinized rollouts
  amplify value-head error / imperfect-info variance rather than averaging it
  out; or the extra per-decision time ate into the budget. Net: K=1 (one-ply
  lookahead) is the sweet spot so far — don't assume monotonic gains in K.
- **2026-07-20 — first `003_aux_loss` submission (K=1).** First submission of the
  new default recipe: large net + tuned low-LR PPO + heuristic rewards + the
  `prize_margin` auxiliary loss, trained 1024 updates. Packed at MCTS K=1 (the
  best-performing inference mode so far). The aux head is training-only and was
  stripped at pack time (4 tensors), so this measures whether the aux-shaped
  trunk produced a stronger policy/value — compare against the 002 K=1 (487.2)
  same-architecture baseline once the score settles.
- **2026-07-23 — first ALAKAZAM submission (010, ExIt TD(λ), K=32 W=4).**
  Bundle `submission_20260723_084117.tar.gz` (2.2 MiB), from run
  `010_alakazam_exit_tdlambda_medium` `latest.pt` (~update 3210), medium net,
  marginal head, deck **alakazam**. Inference-time MCTS **K=32, W=4**
  (c_puct=1.25, temp=0.0, determinization=sample) — the biggest search budget
  submitted so far (prior runs were dragapult at K=1/4). This is (a) the first
  *alakazam* deck on the leaderboard and (b) a test of whether more search
  (K=32) + IS-MCTS world-averaging (W=4) helps here, unlike the earlier
  dragapult finding that K=4 < K=1. Aux head stripped (4 tensors). Watch the
  per-turn + cumulative 600 s budget — 32×4 = 128 forward searches/decision is a
  lot. **Score: TBD** (moving average; treat fresh score as provisional).
- **2026-07-23 — same u3210 checkpoint, POLICY only (K=0) — the A/B control.**
  Bundle `submission_20260723_084332.tar.gz`, packed from the SAME
  `/tmp/010_latest_snapshot.pt` (~update 3210) as the K=32/W=4 entry above, but
  `--inference policy` (no search). This is the clean control: same net, same
  weights, only the inference mode differs — so the pair isolates "does search
  help at u3210?" for the alakazam deck. Prior alakazam data (u1483 policy 478.2
  > u1581 K=256/W=2 378.3) said search *hurt* at earlier checkpoints; this A/B
  re-tests at a much more-trained one. **Score: TBD.**
