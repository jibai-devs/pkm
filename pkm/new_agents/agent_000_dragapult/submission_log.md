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
| 2026-07-20 08:48 | `002_large_tuned/ckpt_512.pt` | **MCTS K=1** | `submission_20260720_084819.tar.gz` | 002_large_tuned ckpt_512 + MCTS K=1 (inference-time search smoke) | **600.0** ✅ |

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
