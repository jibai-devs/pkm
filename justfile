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

# --- training ---------------------------------------------------------------

# Phase 1: PPO self-play from scratch
train iterations="200" games="16" deck="deck/00_basic.csv":
    python -m pkm.rl.train --iterations {{iterations}} --games {{games}} --eval-every 10 --deck {{deck}}

# Phase 1: resume PPO from the latest checkpoint
resume iterations="200" games="16" deck="deck/00_basic.csv":
    python -m pkm.rl.train --iterations {{iterations}} --games {{games}} --eval-every 10 \
        --init checkpoints/ppo_latest.pt --deck {{deck}}

# Phase 2: expert iteration (inits from checkpoints/ppo_latest.pt by default)
exit-train iterations="20" games="8" sims="32" dets="2":
    python -m pkm.rl.exit_train --iterations {{iterations}} --games {{games}} \
        --sims {{sims}} --dets {{dets}}

# Phase 2: resume expert iteration from its own latest checkpoint
exit-resume iterations="20" games="8" sims="32" dets="2":
    python -m pkm.rl.exit_train --iterations {{iterations}} --games {{games}} \
        --sims {{sims}} --dets {{dets}} --init checkpoints/exit_latest.pt

# --- weights / evaluation / replays -----------------------------------------

# export a checkpoint to pkm/policy.npz for torch-free inference (default: best available)
export checkpoint="":
    python -m pkm.rl.export \
        "{{ if checkpoint == "" { if path_exists("checkpoints/exit_latest.pt") == "true" { "checkpoints/exit_latest.pt" } else { "checkpoints/ppo_latest.pt" } } else { checkpoint } }}" \
        pkm/policy.npz

# play one rendered match and write result.html + replay.json (agents: random|neural|mcts)
play p0="neural" p1="random" deck="deck/00_basic.csv":
    python -m pkm.rl.play --p0 {{p0}} --p1 {{p1}} --deck {{deck}}

# head-to-head win rate over N games (no replay files)
eval p0="neural" p1="random" games="30" deck="deck/00_basic.csv":
    python -m pkm.rl.play --p0 {{p0}} --p1 {{p1}} --games {{games}} --deck {{deck}}

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
