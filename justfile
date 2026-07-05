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
deck-show name="00_basic":
    python -m pkm.cli_deck show {{name}}

# convert deck format
deck-convert name="00_basic" to="json":
    python -m pkm.cli_deck convert {{name}} --to {{to}}

# list agent profiles
agents:
    @ls -1 agents/ 2>/dev/null || echo "no agents/ directory"

# --- training ---------------------------------------------------------------

# Phase 1: PPO self-play (agent auto-resolves deck + dirs)
train agent="00_basic" iterations="200" games="16":
    python -m pkm.rl.train --agent {{agent}} --iterations {{iterations}} --games {{games}} --eval-every 10

# Phase 1: resume PPO from agent's latest checkpoint
resume agent="00_basic" iterations="200" games="16":
    python -m pkm.rl.train --agent {{agent}} --iterations {{iterations}} --games {{games}} --eval-every 10

# Phase 2: expert iteration (inits from agent's ppo_latest.pt by default)
exit-train agent="00_basic" iterations="20" games="8" sims="32" dets="2":
    python -m pkm.rl.exit_train --agent {{agent}} --iterations {{iterations}} --games {{games}} \
        --sims {{sims}} --dets {{dets}}

# Phase 2: resume expert iteration from agent's latest exit checkpoint
exit-resume agent="00_basic" iterations="20" games="8" sims="32" dets="2":
    python -m pkm.rl.exit_train --agent {{agent}} --iterations {{iterations}} --games {{games}} \
        --sims {{sims}} --dets {{dets}}

# --- weights / evaluation / replays -----------------------------------------

# export a checkpoint to .npz for torch-free inference (default: agent's best)
export agent="00_basic":
    python -m pkm.rl.export --agent {{agent}} pkm/policy.npz

# play one rendered match and write result.html + replay.json
play p0="neural" p1="random" agent="00_basic":
    python -m pkm.rl.play --agent {{agent}} --p0 {{p0}} --p1 {{p1}}

# head-to-head win rate over N games (no replay files)
eval p0="neural" p1="random" games="30" agent="00_basic":
    python -m pkm.rl.play --agent {{agent}} --p0 {{p0}} --p1 {{p1}} --games {{games}}

# open the latest match replay in the browser
watch:
    xdg-open result.html

# --- submission ---------------------------------------------------------------

# export freshest weights and build submission.tar.gz
submit: export
    ./submit.sh

# remove training/replay artifacts (keeps checkpoints)
clean:
    rm -f result.html replay.json submission.tar.gz
