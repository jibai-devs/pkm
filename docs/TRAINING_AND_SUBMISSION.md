# Training, Saving, and Submitting a Kaggle Bundle

End-to-end runbook: start a real training run, keep it running unattended,
save/export the resulting weights, back them up, and build+upload a Kaggle
submission. Examples use `03_pult_munki` (the Dragapult/Munkidori deck
without Dusknoir, ported in this session) — swap in any agent name.

For the architecture behind what's being trained, see `docs/ARCHITECTURE.md`.
For the underlying commands this all wraps, see `AGENTS.md` → "RL Training"
and "Kaggle Submission".

> **Note:** `AGENTS.md`/the `justfile` reference a `just <recipe>` shorthand
> for most of these. `just` is **not currently installed** on this machine
> (verified: `which just` finds nothing) — every command below is given as
> the raw command it expands to, which works regardless. Install `just`
> (e.g. `winget install --id Casey.Just` or `cargo install just`) if you want
> the shorter form later; it's a convenience, not a dependency of anything
> here.
---

## 1. Start a training run

```bash
pkm train --agent 03_pult_munki --iterations 200 --games 16 --eval-every 10
```

- `--agent <name>` resolves deck (`deck/<name>.csv`), checkpoint dir
  (`agents/<name>/checkpoints/`), metrics CSV, TensorBoard dir, and — if
  present — `agents/<name>/reward_weights.json` (falls back to
  `pkm/rl/reward_terms.py`'s `DEFAULT_WEIGHTS` if that file doesn't exist).
- `--iterations`/`--games` — total PPO iterations and self-play games per
  iteration. 200 iterations × 16 games/iter is roughly the scale of the
  existing `00_basic` run (`AGENTS.md`: 200 iters → 80% win rate vs random);
  expect on the order of 15–30 minutes depending on game length, plus eval
  overhead.
- `--eval-every 10` — play `--eval-games` (default 20) vs. the random agent
  every 10 iterations and print/log the win rate; also the cadence
  checkpoints get written at.
- **Re-running the exact same command resumes automatically.** `--agent`
  makes `train.py` call `profile.ppo_init()`, which returns
  `agents/<name>/checkpoints/ppo_latest.pt` if one exists — you don't need a
  separate "resume" command, `just resume` is literally the same recipe as
  `just train`.
  **Exception, `03_pult_munki` specifically (as of 2026-07-19):** its
  existing `ppo_latest.pt` predates the opponent-archetype belief-feature
  resize (stamped `opponent_archetype_belief` dim=4, current registry expects
  dim=26) — `profile.ppo_init()`/`check_stamp_sidecar` will raise
  `FeatureStampMismatch` rather than silently resume from an incompatible
  checkpoint. Move or delete `agents/03_pult_munki/checkpoints/` first (or
  train under a fresh agent name) to start the required full retrain from
  scratch. See `docs/opponent-archetype-classifier-plan.md` Part 3 and
  `docs/ARCHITECTURE.md` §10.

(with `just` installed, this is `just train agent=03_pult_munki iterations=200 games=16`.)

### Deck-specific reward-shaping terms

`pkm/rl/reward_terms.py` is the registry of every term (17 total: 4
potential-based + 13 direct, see the file for the current list) — its
built-in `DEFAULT_WEIGHTS` ships everything at `0.0` except prize-differential
shaping (`"shaping": 0.2`). Per-agent, `agents/<name>/reward_weights.json`
overrides that default; unknown/missing keys are ignored/defaulted, so it's
safe if the registry grows a term after the file was written.

**`agents/03_pult_munki/reward_weights.json` already exists and is already
tuned** (not the doc's job to restate the numbers — they drift; read the file
directly before a run to see current values). `pkm train --agent
03_pult_munki` picks it up automatically, no extra flag needed. To point at a
different weights file instead, use `--weights <path>`, which overrides the
agent-default lookup.

To create a fresh one for a *new* agent, `write_default_weights_file()` (same
module) writes `DEFAULT_WEIGHTS` verbatim as a starting point — every term at
`0.0` except `shaping`.

### Cross-archetype opponent sampling (Part 3c) + belief injection (Part 2a)

```bash
pkm train --agent 03_pult_munki --iterations 200 --games 16 --eval-every 10 \
           --archetype-pool --archetype-pool-prob 0.2 \
           --archetype-belief
```

`--archetype-pool` loads every trained `agents/pool_*/checkpoints/ppo_latest.pt`
(`pkm/rl/opponent_pool.py:load_pool_bots()`) — 25 pool bots as of 2026-07-19,
one per `staples.json` archetype (`AGENTS.md` → "Opponent pool decklists" /
"Pool bots"). `--archetype-pool-prob` (default 0.2) is the fraction of games
played against a random one of them **on its own deck**, instead of the
existing `--pool-size`/self-checkpoint-pool behavior (which still applies to
the remaining fraction, unchanged). Requires the pool bots to already be
trained (`agents/pool_*/checkpoints/` present); an untrained pool profile is
skipped, not an error, so a partial pool works fine too.

`--archetype-belief` (default classifier path `pkm/archetype.npz`, override
with `--archetype-weights <path>`) loads the standalone opponent-archetype
classifier once and attaches it to the trainee's `TorchPolicy` for the whole
run — that's what actually populates the encoder's `opponent_archetype_belief`
feature with a real prediction instead of zeros. **This had never happened in
any training run before 2026-07-19** — the encoder/network side of Part 2a
shipped with Parts 1-2, but `train.py` never constructed a classifier to feed
it, so the dim-4→26 resize alone didn't mean the feature carried information.

Both flags are independent and off by default — omit either (or both) for the
old single-deck-mirror-only, zero-belief behavior. The `03_pult_munki` retrain
this doc keeps pointing at should use both together (see the note above about
why it must be a fresh run, not a resume).

---

## 2. Keep it running unattended

A 200-iteration run outlives a single foreground command you'd want to
babysit. Two options on Windows:

### Option A — PowerShell, fully detached (recommended)

```powershell
Start-Process -FilePath "uv" `
  -ArgumentList "run","pkm","train","--agent","03_pult_munki","--iterations","200","--games","16","--eval-every","10" `
  -WorkingDirectory "C:\Users\Luqman\Desktop\projects\pkm" `
  -RedirectStandardOutput "agents\03_pult_munki\train_stdout.log" `
  -RedirectStandardError "agents\03_pult_munki\train_stderr.log" `
  -WindowStyle Hidden -PassThru
```

The `-PassThru` prints a process object with an `Id` — note it down.
`Start-Process` spawns a fully independent process, not a child job tied to
your shell, so it *should* survive closing the terminal window. **Verify
this once** by closing the window and running `Get-Process -Id <id>` from a
fresh terminal — some terminal apps (e.g. some Windows Terminal
configurations) still kill child processes via a Job Object on tab close.
If that happens, use Task Scheduler ("Create Basic Task" → run the same
command) for guaranteed persistence instead.

Check on it later:
```powershell
Get-Process -Id <id>                          # still running?
Get-Content agents\03_pult_munki\train_stdout.log -Tail 20
```

Stop it early if needed:
```powershell
Stop-Process -Id <id>
```

### Option B — Git Bash, `nohup` + `disown`

```bash
nohup uv run pkm train --agent 03_pult_munki --iterations 200 --games 16 --eval-every 10 \
  > agents/03_pult_munki/train.log 2>&1 &
disown
```

`nohup` ignores the hangup signal sent when the shell exits; `disown`
removes it from the shell's job table so it isn't caught up in the shell's
own cleanup. Same persistence caveat as Option A applies — MSYS2/Git Bash
process-group behavior on Windows is less predictable than on Linux, so
verify it survives a closed window before trusting it for a long run.

Check on it later:
```bash
tail -f agents/03_pult_munki/train.log
```

---

## 3. Monitor progress while it runs

- **Console / log tail** — the per-iteration line
  (`iter N | games ... | pi_loss ... | eval_win_rate ...`) is the fastest
  signal. Watch `pi_loss`/`v_loss` for NaN or blow-up, `entropy` for
  collapsing to ~0 too early (means the policy stopped exploring), and
  `eval_win_rate` trending up over time.
- **Metrics CSV** — `agents/<name>/metrics/ppo_train.csv`, one row per
  iteration: `iter,games,wins,losses,draws,decisions,samples,pi_loss,v_loss,
  entropy,clip_frac,archetype_loss,time_s,eval_win_rate,eval_games`. Load it
  into `notebooks/training_monitor.ipynb` (Plotly) for a real chart, or
  `pandas.read_csv` directly.
- **TensorBoard** — `agents/<name>/runs/ppo/` is written automatically:
  ```bash
  tensorboard --logdir=agents/03_pult_munki/runs/ppo
  ```
  then open `http://localhost:6006`.
- **wandb (optional)** — pass `--wandb-project <name>` (and optionally
  `--wandb-run-name`) to `pkm train` for a hosted live dashboard instead.

---

## 4. Save / export the model

Training already checkpoints for you every `--eval-every` iterations and at
the end, to `agents/<name>/checkpoints/`:
- `ppo_iter<NNNN>.pt`, `ppo_latest.pt` — torch state dicts.
- `ppo_latest.pt.stamp.json` — the `FeatureSpec` registry fingerprint at
  save time (`pkm/rl/features.py`). Loading a checkpoint whose stamp doesn't
  match the *current* registry raises `FeatureStampMismatch` rather than
  silently loading garbage into misaligned tensor slices — if you see that
  error, the checkpoint predates a feature-width change and can't be reused
  as-is.

**These `.pt` files are gitignored** (`checkpoints/` in `.gitignore`) — they
never leave your machine on their own. Two things you actually want to do
with a checkpoint:

### Export to `.npz` (required for Kaggle — no torch at inference time)

```bash
pkm export --agent 03_pult_munki pkm/policy.npz
```
(with `just`: `just export agent=03_pult_munki`.)

This loads `agents/03_pult_munki/checkpoints/ppo_latest.pt` (checking its
stamp), replays the same weights into a plain numpy `.npz` (also embedding
the feature stamp), and writes `pkm/policy.npz` — the file
`pkm/rl/numpy_policy.py` loads at inference time, and the one that gets
bundled into the Kaggle submission (see §6). `pkm/policy.npz` is also
gitignored, so re-export before every submission build — it doesn't persist
across sessions either.

---

## 5. Back up weights (Hugging Face)

Since checkpoints and the `.npz` export are both gitignored, the **only**
durable copy is wherever you push it. This repo already publishes to
`https://huggingface.co/TomatoCream/pkm-cabt-ppo` (public) — see `AGENTS.md`
→ "Weights on Hugging Face" for the established convention. To add this
agent:

```bash
hf auth login                                  # needs a WRITE token; a read token 403s on upload
hf upload TomatoCream/pkm-cabt-ppo agents/03_pult_munki/checkpoints/ppo_latest.pt \
  03_pult_munki/ppo_latest.pt --repo-type model
hf upload TomatoCream/pkm-cabt-ppo pkm/policy.npz \
  03_pult_munki/policy.npz --repo-type model
hf upload TomatoCream/pkm-cabt-ppo deck/03_pult_munki.csv \
  03_pult_munki/deck.csv --repo-type model
```

(`hf auth whoami` currently shows **not logged in** on this machine — you'll
need to log in with a write-scoped token before the upload commands work.)
Only `*_latest` checkpoints are worth keeping this way — per-iteration
snapshots (`ppo_iter*.pt`) were previously deleted locally and never
uploaded (`AGENTS.md`), so don't bother pushing those.

---

## 6. Build the Kaggle submission bundle

```bash
pkm export --agent 03_pult_munki pkm/policy.npz
bash submit.sh 03_pult_munki
```
(with `just`: `just build_submit agent=03_pult_munki` runs both lines.)

In order:
1. `pkm export --agent 03_pult_munki pkm/policy.npz` — freshest weights.
2. `bash submit.sh 03_pult_munki` — which:
   - copies `main.py` in,
   - flattens `deck/03_pult_munki.csv` into `submission/deck.csv`
     (`main.py` looks for `deck.csv` first, falling back to the bundled
     `deck/02_dragapult.csv` only if that's missing — see
     `pkm/agents/neural_agent.py`'s weight/deck resolution order),
   - copies the whole `pkm/` package in (including the just-exported
     `pkm/policy.npz`),
   - tars it to `submissions/submission_03_pult_munki_<timestamp>.tar.gz`.

(`submit.sh` previously hard-refused any agent but `02_dragapult` — that
restriction was stale given `main.py`/`neural_agent.py` are already
deck-agnostic; removed in this session so other agents, like this one,
can actually be submitted.)

**Sanity check before uploading** — confirm the bundled agent is actually
callable the way kaggle will call it (use a local scratch dir, not `/tmp` —
this is a Windows checkout and `/tmp` isn't guaranteed to map anywhere
useful):
```bash
mkdir -p .submission_check
tar -xzf submissions/submission_03_pult_munki_*.tar.gz -C .submission_check
cd .submission_check && uv run --project .. python -c "
from main import agent
print('agent callable, argcount:', agent.__code__.co_argcount)
"
cd .. && rm -rf .submission_check
```
(`agent.__code__.co_argcount` must be exactly 1 — kaggle inspects this
directly; a bound method or wrapped callable with the wrong arg count will
fail at submission time, see `CLAUDE.md`.)

Constraints (`AGENTS.md` → "Kaggle Submission"): **197.7 MiB max**, **5
submissions/day**, only the **latest 2** stay active.

---

## 7. Upload to Kaggle

```bash
kaggle competitions submit -c pokemon-tcg-ai-battle \
  -f submissions/submission_03_pult_munki_<timestamp>.tar.gz -m "03_pult_munki, 200 iters"
```
(with `just`: `just upload submissions/submission_03_pult_munki_<timestamp>.tar.gz` —
defaults to the newest `submissions/*.tar.gz` if you omit the path.)

`kaggle --version` / `kaggle config view` show the CLI is installed with a
saved username (`naqibl`), but that only proves a config file exists, not
that its token still works — verified by actually calling an authenticated
endpoint (`kaggle competitions submissions -c pokemon-tcg-ai-battle`), which
currently 401s. **The Kaggle CLI is not authenticated on this machine** —
generate a fresh token first (kaggle.com → Account → API → Create New Token,
drop the downloaded `kaggle.json` into `~/.kaggle/`) before any of the
commands below will work. The website upload flow is unaffected if you need
to submit before fixing this.

Poll until it finishes scoring:
```bash
while true; do
  line=$(kaggle competitions submissions -c pokemon-tcg-ai-battle --csv 2>/dev/null | head -2 | tail -1)
  status=$(echo "$line" | cut -d',' -f5); score=$(echo "$line" | cut -d',' -f6)
  echo "$(date +%H:%M:%S) — $status${score:+ (score: $score)}"
  [ "$status" != "SubmissionStatus.PENDING" ] && break
  sleep 15
done
```
(with `just`: `just poll`.)

If a submission errors out, pull its log for debugging:
```bash
kaggle competitions logs <episode_id> <agent_id> -p submissions
```
(with `just`: `just logs episode=<episode_id> agent=<agent_id>`.)

---

## Quick reference

```bash
# train (auto-resumes if agents/<name>/checkpoints/ppo_latest.pt exists)
pkm train --agent 03_pult_munki --iterations 200 --games 16 --eval-every 10

# background (Git Bash)
nohup uv run pkm train --agent 03_pult_munki --iterations 200 --games 16 --eval-every 10 \
  > agents/03_pult_munki/train.log 2>&1 & disown

# monitor
tail -f agents/03_pult_munki/train.log
tensorboard --logdir=agents/03_pult_munki/runs/ppo

# save + back up
pkm export --agent 03_pult_munki pkm/policy.npz
hf upload TomatoCream/pkm-cabt-ppo agents/03_pult_munki/checkpoints/ppo_latest.pt 03_pult_munki/ppo_latest.pt --repo-type model

# submit
pkm export --agent 03_pult_munki pkm/policy.npz
bash submit.sh 03_pult_munki
kaggle competitions submit -c pokemon-tcg-ai-battle -f submissions/submission_03_pult_munki_<timestamp>.tar.gz -m "03_pult_munki"
```
