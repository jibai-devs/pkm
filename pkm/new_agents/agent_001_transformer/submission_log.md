# agent_001_transformer — submission log

Leaderboard history for the notebook-port transformer agent. Competition:
`pokemon-tcg-ai-battle`. Append a row per Kaggle submission (date, checkpoint,
inference, bundle, message, score once it lands).

- **2026-07-20 — first submission (ref 54855590).** Bundle
  `submission_20260720_222008.tar.gz` (46.5 MiB). Checkpoint:
  `out_transformer/latest.pth` from a 20-iteration self-play/MCTS run
  (`--iters 20 --eval-games 50 --selfplay-games 100 --sims 10 --device cuda`),
  the near-verbatim port of the reference notebook
  (`../agent_000_dragapult/references/example.py`) driven through our libcg seam.
  Architecture: `EmbeddingBag` bag-of-features encoder + TransformerEncoder,
  transformer **decoder** scoring enumerated option combinations, TD(λ) value +
  advantage-style policy targets. Inference runs MCTS at **sims=10** every move
  (`SEARCH_COUNT` env, default 10). Deck = the notebook's `sample_deck`.
  **Eval-vs-random trajectory during training:** 2% → 54 → 78 → 88 → … peaked
  100% (iters 11–12), settled ~90%. Packed bundle sanity: 5/6 vs random.
  **Score: 600.0** (SubmissionStatus.COMPLETE), vs agent_000's latest 007-xl at
  426.2 — a strong first landing for a completely different architecture
  (bag-encoder + combo decoder) than agent_000's structured encoder. **Caveat:**
  leaderboard scores here are a moving average and drift as more league matches
  play out (agent_000's 007 "600" settled to ~487), so treat this as provisional
  until it stabilises. Both agents eval vs random locally, so the LB is the only
  honest head-to-head.
