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
