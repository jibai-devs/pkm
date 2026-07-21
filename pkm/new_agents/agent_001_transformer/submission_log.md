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

- **2026-07-22 — pult_munki deck (03).** Bundle `submission_20260722_005554.tar.gz`
  (45.8 MiB). Checkpoint: `out_transformer/out/pult_munki_40/latest.pth` from a
  **40-iteration** GPU self-play/MCTS run
  (`cli train --deck pult_munki --iters 40 --device cuda`, defaults 50 eval /
  100 selfplay / sims=10). **First submission of the new multi-deck setup:** deck
  = `03_pult_munki` (Dragapult ex / Munkidori, no Dusknoir — item-disruption
  toolbox: Crushing Hammer ×4, Xerosic's Machinations, Team Rocket's Watchtower),
  now baked into the checkpoint (`deck_name`) and read back by `submit_main`.
  Inference = MCTS sims=10 every move. **Eval-vs-random trajectory:** 96% → ~88–94
  (iters 1–15) → steady **98–100%** from iter ~16 (saturates; not a true ceiling).
  **Score: pending** (check `kaggle competitions submissions -c pokemon-tcg-ai-battle`).
