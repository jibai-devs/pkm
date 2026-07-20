# Sweep results — agent_000_dragapult

Winning Optuna trials from the `dragapult_heuristic` study (SQLite at
`experiments/000_heuristic_sweep/sweeps/dragapult_heuristic.db`). Newest first.
View live: `optuna-dashboard sqlite:///<...>/dragapult_heuristic.db`.

## dragapult_heuristic — best trial 16 (2026-07-20)

- **Objective:** `curve_auc` (mean of the eval-vs-random learning curve).
- **Score:** **96.7%** (`curve_auc`). Note: eval-vs-random saturates, so this is
  "reasonable params," not a discriminating leaderboard signal.
- **Net swept on:** `--model small` (~106K params). The tuned **lr (4.68e-4) is
  only validated at small size** — it was unstable on large and is overridden to
  `1e-4` on large/xxl runs (see `scripts/005_swept_xxl/train.sh`).
- **Sweep setup:** 30 trials · 15 updates/trial · 32 games/update · `--tune-rewards`.
- **Codified in:** `scripts/004_swept_heuristic/train.sh` (small, swept lr) and
  `scripts/005_swept_xxl/train.sh` (xxl, lr forced to 1e-4).

### PPO hyperparameters

| param | value |
|---|---|
| lr | 0.0004676045518655474 |
| entropy_coef | 0.014037528236254575 |
| clip_eps | 0.12466547078861025 |
| epochs | 3 |
| minibatch_size | 64 |
| gamma | 0.9901486456977385 |
| lam | 0.929944571253055 |

### Heuristic reward weights (18 terms)

| term | weight |
|---|---|
| shaping | 0.22124198921787636 |
| board_setup | 0.26523068983353526 |
| budew_setup | 0.1470491453196494 |
| dreepy_field | 0.7337736300176154 |
| energy_penalty | 0.5804253728977349 |
| budew_bonus | 0.026423680059480703 |
| wrong_type_penalty | 0.49225355292212963 |
| dragapult_bonus | 0.26557862079433003 |
| dreepy_spread | 0.665043599126133 |
| xerosic | 0.818163371297779 |
| budew_bench_setup | 0.9973854463391766 |
| dreepy_evolve | 0.7754273038519655 |
| dreepy_bench_charge | 0.531930043580297 |
| dreepy_active_charge | 0.7958770523065606 |
| wasted_resources | 0.8677344148174203 |
| phantom_dive | 0.5263123628818893 |
| drakloak_backup_ready | 0.13432290967575222 |
| budew_redundant | 0.3262913990792372 |

The aux loss (`prize_margin=0.25`) was **not** part of this sweep (it trained
with aux off); it's carried onto the swept-recipe runs as an orthogonal add-on.
