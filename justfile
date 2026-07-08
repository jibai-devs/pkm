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
    python -m pkm.cli_deck list

# show deck contents
deck-show name="02_dragapult":
    python -m pkm.cli_deck show {{name}}

# convert deck format
deck-convert name="02_dragapult" to="json":
    python -m pkm.cli_deck convert {{name}} --to {{to}}

# dump all card data to JSON
cards-dump out="cards.json":
    python -m pkm.cli.cards dump {{out}}

# list agent profiles
agents:
    @ls -1 agents/ 2>/dev/null || echo "no agents/ directory"

# --- training ---------------------------------------------------------------

# Phase 1: PPO self-play (agent auto-resolves deck + dirs)
train agent="02_dragapult" iterations="200" games="16" lr="3e-4":
    python -m pkm.rl.train --agent {{agent}} --iterations {{iterations}} --games {{games}} --eval-every 10 --lr {{lr}}

# Phase 1: resume PPO from agent's latest checkpoint
resume agent="02_dragapult" iterations="200" games="16" lr="3e-4":
    python -m pkm.rl.train --agent {{agent}} --iterations {{iterations}} --games {{games}} --eval-every 10 --lr {{lr}}

# Phase 2: expert iteration (inits from agent's ppo_latest.pt by default)
exit-train agent="02_dragapult" iterations="20" games="8" sims="32" dets="2":
    python -m pkm.rl.exit_train --agent {{agent}} --iterations {{iterations}} --games {{games}} \
        --sims {{sims}} --dets {{dets}}

# Phase 2: resume expert iteration from agent's latest exit checkpoint
exit-resume agent="02_dragapult" iterations="20" games="8" sims="32" dets="2":
    python -m pkm.rl.exit_train --agent {{agent}} --iterations {{iterations}} --games {{games}} \
        --sims {{sims}} --dets {{dets}}

# --- weights / evaluation / replays -----------------------------------------

# export a checkpoint to .npz for torch-free inference (default: agent's best)
export agent="02_dragapult":
    python -m pkm.rl.export --agent {{agent}} pkm/policy.npz

# play one rendered match and write result.html + replay.json
play p0="neural" p1="random" agent="02_dragapult":
    python -m pkm.rl.play --agent {{agent}} --p0 {{p0}} --p1 {{p1}}

# head-to-head win rate over N games (no replay files)
eval p0="neural" p1="random" games="30" agent="02_dragapult":
    python -m pkm.rl.play --agent {{agent}} --p0 {{p0}} --p1 {{p1}} --games {{games}}

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

# export freshest weights and build submission.tar.gz
submit: export
    ./submit.sh

# remove training/replay artifacts (keeps checkpoints)
clean:
    rm -f result.html replay.json submission.tar.gz
