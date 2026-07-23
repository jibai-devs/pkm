# `darwinian_ml` — survival-of-the-fittest training for `default_dragapult_darwinian`

An **experiment**, deliberately isolated. Everything needed lives in this
directory; nothing here modifies the `pkm` package, so the existing agents and
training pipelines behave exactly as before whether or not this directory
exists. It imports from `pkm` read-only.

## What's different about it

The existing agent (`dragapult_default`) is trained by **PPO on mirror
self-play**: it plays itself, every decision is scored by a shaped reward, and
gradients nudge the weights. That has two properties worth questioning:

1. it never sees an opponent that isn't itself, and
2. it optimises a *proxy* (shaped per-decision reward), not the thing you
   actually want.

This experiment inverts both. There are **no gradients**. A population of
networks plays real games against a **real, different opponent** — a past
Kaggle submission, loaded and run live — and the ones that do better get to
have offspring. The selection pressure is the outcome itself.

```
population of N networks
        │
        ├── each plays G full games vs Mega Abomasnow  ──►  fitness
        │
        ├── the fittest few survive untouched          (elitism)
        ├── parents chosen by tournament               (selection)
        ├── children = uniform crossover of two parents (recombination)
        └── children perturbed by gaussian noise        (mutation)
        │
        ▼   next generation
```

## The opponent is a real past submission

`submissions/submission_04_mega_abomasnow_*.tar.gz` is self-contained: its own
`main.py`, its own copy of `pkm` (encoder included), its own `policy.npz`. So
it is playable **as-is**, even though it is architecturally incompatible with
today's network — it was trained when the encoder emitted 775 state features
and today's emits 335. Its weights never touch our network; it just answers
observations.

The one obstacle is that both packages are named `pkm`, which cannot be
imported twice in one interpreter. `opponent.py` therefore runs the bundle in
a **subprocess** with its own `sys.path`, speaking newline-JSON over stdio —
the Kaggle contract is already `agent(obs: dict) -> list[int]`, so no
adaptation is needed.

Two non-obvious details in that bridge, both found by hitting them:

- **The protocol channel is reserved at the file-descriptor level**
  (`os.dup(1)`, `os.dup2(2, 1)`), not by rebinding `sys.stdout`. The engine is
  a C library that writes banners straight to fd 1; a Python-level swap leaves
  them interleaved with the JSON, and once the pipe fills, the child blocks
  writing while the parent blocks reading — a deadlock rather than an error.
- **The opponent is rebuilt between games** (`new_game`). Its middleman owns a
  `GameContext`/`DeckTracker`; reusing one instance leaks card knowledge across
  games and quietly weakens the very opponent we measure ourselves against.

## Fitness is graded, not binary

"Beat Mega Abomasnow" is a binary target, and a population that loses every
game scores identically on all members — selection has nothing to rank and
evolution never starts. So fitness is:

```
score = 10.0 · win_rate                     # dominates once wins appear
      +  3.0 · (prize_margin / 6)           # "lost 6-5" ≫ "lost 6-0"
      +  1.0 · mean_shaping                 # the deck's known-good habits
```

The **prize margin** is what makes this workable: it gives the search a slope
to climb long before the first win. A random network currently scores about
−3.0 (shut out 6−0), so there is plenty of room to measure improvement.

`mean_shaping` reuses the default agent's own reward functions from
`pkm/rl/encoder.py`, with that agent's tuned weights — those already encode
what a Dragapult deck wants (charge the line, evolve, land Phantom Dive, don't
strand energy), so there's no reason to invent a second opinion. It is
weighted lowest on purpose: it is a hint, not the goal.

## The population is seeded, not random

342,879 parameters is far past what a genetic algorithm can discover from
noise in any realistic number of games. Generation 0 is therefore the
**existing trained policy** (`pkm/policy.npz`) plus mutated copies of it. This
is directed variation around a working agent, not a search from scratch — the
only version of this experiment that can produce a result on a laptop.

Mutation noise is scaled **per tensor** by that tensor's own standard
deviation: a flat sigma is devastating to small-magnitude layers and
imperceptible to large ones.

## Running it

```bash
darwinian_ml/evolve.sh                        # background, stoppable
darwinian_ml/evolve.sh -- --generations 50 --population 16
python -m darwinian_ml.evolve --help          # all knobs
```

Stop it with `touch darwinian_ml/runs/default_dragapult_darwinian/STOP`; it
finishes the generation in flight, keeps the best checkpoint, and exits.

Outputs land in `darwinian_ml/runs/default_dragapult_darwinian/`:

| file | what |
|---|---|
| `best.pt` | best genome so far, as a `PolicyValueNet` state dict |
| `best.json` | which generation produced it, and its fitness parts |
| `evolution.csv` | per-generation fitness / win-rate / sigma history |
| `opponent_bundle/` | the extracted submission being fought |

Roughly **4.4s per game**, so a 12-genome population at 4 games each is about
3.5 minutes per generation.

## Evaluating

```bash
python -m darwinian_ml.evaluate --games 40
```

Plays both the evolved agent **and** the full `singaporean_middleman` (first-turn
MCTS + setup + default) against the same bundle, alternating sides, and reports
win rates side by side.

## Honest caveats

- **It is not obviously going to beat PPO.** Evolutionary search is far less
  sample-efficient than gradients; its advantages here are that it optimises
  the true objective and that it can train against an opponent whose weights
  are unusable by any gradient method.
- **One opponent is a narrow target.** Optimising hard against a single fixed
  bundle invites overfitting to that specific matchup. A pool of bundles would
  be the honest next step.
- **Greedy play during evaluation** makes fitness lower-variance but hides
  whatever the policy would do stochastically.
