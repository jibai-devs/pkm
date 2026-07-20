# Submission log — agent_000_dragapult

Kaggle competition: **pokemon-tcg-ai-battle** (The Pokémon Company — PTCG AI
Battle Challenge). One row per submission, newest first. Score is the Kaggle
leaderboard score; check with `pkm new_agents 000_dragapult status --watch`.

Context: every submission before 2026-07-20 landed at exactly **600.0** across
small/large/tuned nets — a real plateau for "PPO mirror self-play + greedy
policy inference" with this deck (not a hard cap; historical agents scored
319.5 / 544.2 / 600.0). See `TRAINING.md` §10 for the full diagnosis.

| Date (UTC+8) | Checkpoint | Inference | Bundle | Message | Score |
|---|---|---|---|---|---|
| 2026-07-20 08:48 | `002_large_tuned/ckpt_512.pt` | **MCTS K=1** | `submission_20260720_084819.tar.gz` | 002_large_tuned ckpt_512 + MCTS K=1 (inference-time search smoke) | _pending_ |

## Notes

- **2026-07-20 — first inference-time MCTS submission (K=1).** Smoke test of the
  new inference-time MCTS path (README §9 item 1, shipped 2026-07-20). K=1 is a
  single-simulation search — deliberately minimal to (a) confirm the MCTS
  decision path runs inside Kaggle's sandbox against its own `libcg.so` search
  symbols, and (b) stay well within the per-turn + cumulative 600 s time budget.
  If this scores and completes, higher K (8/16/32) is the next step to see
  whether real search lifts the score off the 600 plateau.
