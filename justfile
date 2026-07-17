# Pokemon TCG AI — task runner. `just` lists recipes; docs in docs/GUIDE.md.

# default: show available recipes
default:
    @just --list --unsorted

# install/sync python dependencies
sync:
    uv sync

# lint + format check
lint:
    ruff check pkm/ tests/
    ruff format --check pkm/ tests/

# auto-format
fmt:
    ruff format pkm/ tests/

# run the test suite
test:
    python -m pytest tests/ -q

# list available decks
deck:
    pkm deck list

# show deck contents
deck-show name="02_dragapult":
    pkm deck show {{name}}

# convert deck format
deck-convert name="02_dragapult" to="json":
    pkm deck convert {{name}} --to {{to}}

# dump all card data to JSON
cards-dump out="cards.json":
    pkm cards dump {{out}}

# list agent profiles
agents:
    @ls -1 agents/ 2>/dev/null || echo "no agents/ directory"

# --- training ---------------------------------------------------------------

# Phase 1: PPO self-play (agent auto-resolves deck + dirs)
train agent="02_dragapult" iterations="200" games="16" lr="3e-4":
    pkm train --agent {{agent}} --iterations {{iterations}} --games {{games}} --eval-every 10 --lr {{lr}}

# Phase 1: resume PPO from agent's latest checkpoint
resume agent="02_dragapult" iterations="200" games="16" lr="3e-4":
    pkm train --agent {{agent}} --iterations {{iterations}} --games {{games}} --eval-every 10 --lr {{lr}}

# Phase 2: expert iteration (inits from agent's ppo_latest.pt by default)
exit-train agent="02_dragapult" iterations="20" games="8" sims="32" dets="2":
    pkm exit-train --agent {{agent}} --iterations {{iterations}} --games {{games}} \
        --sims {{sims}} --dets {{dets}}

# Phase 2: resume expert iteration from agent's latest exit checkpoint
exit-resume agent="02_dragapult" iterations="20" games="8" sims="32" dets="2":
    pkm exit-train --agent {{agent}} --iterations {{iterations}} --games {{games}} \
        --sims {{sims}} --dets {{dets}}

# --- weights / evaluation / replays -----------------------------------------

# export a checkpoint to .npz for torch-free inference (default: agent's best)
export agent="03_pult_munki":
    pkm export --agent {{agent}} pkm/policy.npz

# play one rendered match and write result.html + replay.json
play p0="neural" p1="random" agent="02_dragapult":
    pkm play --agent {{agent}} --p0 {{p0}} --p1 {{p1}}

# head-to-head win rate over N games (no replay files)
eval p0="neural" p1="random" games="30" agent="02_dragapult":
    pkm play --agent {{agent}} --p0 {{p0}} --p1 {{p1}} --games {{games}}

# open the latest match replay in the browser
watch:
    xdg-open result.html

# start the replay viewer dev server (bun + vite)
replay:
    cd replay/02_vite_web_app && bun run dev

# start the React replay viewer on :5175; optionally load another replay
# (path served under its public/, or a URL). E.g. just replay-react file=/foo.json
replay-react file="":
    cd replay/05_vite_react_app && VITE_REPLAY={{file}} bun run dev

# --- submission ---------------------------------------------------------------

# export freshest weights and build submissions/submission_<agent>_<ts>.tar.gz
build_submit agent="03_pult_munki":
    pkm export --agent {{agent}} pkm/policy.npz
    ./submit.sh {{agent}}

# upload a submission bundle to Kaggle (defaults to latest submissions/*.tar.gz)
upload file=`ls -t submissions/submission_*.tar.gz 2>/dev/null | head -1`:
    kaggle competitions submit -c pokemon-tcg-ai-battle -f {{file}} -m "auto"

# download an episode's agent log into submissions/
logs episode agent:
    kaggle competitions logs {{episode}} {{agent}} -p submissions

# poll latest submission until it finishes (PENDING -> ERROR/DONE)
poll:
    @while true; do \
      line=$(kaggle competitions submissions -c pokemon-tcg-ai-battle --csv 2>/dev/null | head -2 | tail -1); \
      status=$(echo "$line" | cut -d',' -f5); \
      score=$(echo "$line" | cut -d',' -f6); \
      file=$(echo "$line" | cut -d',' -f2); \
      echo "$(date +%H:%M:%S) — $file — $status${score:+ (score: $score)}"; \
      if [ "$status" != "SubmissionStatus.PENDING" ]; then break; fi; \
      sleep 15; \
    done

# remove training/replay artifacts (keeps checkpoints)
clean:
    rm -f result.html replay.json submissions/submission_*.tar.gz
