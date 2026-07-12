# General Agent Architecture

**Status:** Idea for discussion

## Goal

Create one interface for defining, training, evaluating, and packaging agents
without coupling local training code to Kaggle's submission protocol.

The first supported profile remains `02_dragapult` with the neural policy. The
architecture should make it possible to add other policies and decks later,
then run them against each other.

## Core Separation

Keep these concerns separate:

1. **Agent profile:** identity, deck, policy type, trainer, checkpoints, and
   output directories.
2. **Runtime policy:** a callable that accepts an observation and returns a
   legal action list.
3. **Training algorithm:** PPO, expert iteration, or a future trainer that
   produces checkpoints.
4. **Submission packaging:** exports inference weights and creates a minimal
   Kaggle entry point.

The runtime policy must remain compatible with Kaggle:

```python
action = policy(obs)
```

Training and packaging should use the same policy factory, but the Kaggle
entry point must not contain training or local battle code.

## Agent-Owned Decks

Each profile owns its deck. The deck should be stored as data in the profile
configuration or deck directory, not hardcoded into a Python policy:

```text
agents/02_dragapult/
    profile.yaml
    deck.csv
    checkpoints/
```

The profile resolves its own deck for training, local play, evaluation, and
submission packaging. This allows two players to use different decks in the
same game:

```python
p0 = AgentProfile("02_dragapult")
p1 = AgentProfile("01_psychic")
play_match(agents=(p0.make_agent(), p1.make_agent()), decks=(p0.deck(), p1.deck()))
```

The submission builder still writes the selected profile's deck as the flat
bundled `deck.csv` expected by Kaggle.

## Special Agent Logic

Deck ownership and policy behavior should be independent. A profile can select
special logic through its policy configuration:

```yaml
name: 02_dragapult
deck: deck/02_dragapult.csv
policy: neural
strategy: dragapult
trainer: ppo
```

The `strategy` hook can later select deck-specific heuristics, preprocessing,
action masks, or a hybrid policy. Policy factories should receive a resolved
profile/spec rather than only a raw deck, so special logic can inspect the
profile's deck and configuration without coupling the game runner to a named
agent.

Examples include:

- `policy: neural` with no special strategy;
- `policy: heuristic`, `strategy: dragapult`;
- `policy: hybrid`, combining a neural policy with deck-specific overrides;
- `policy: mcts`, using the profile's deck and optional neural prior.

Unknown policy and strategy names should fail during profile validation, not
during a Kaggle match.

## Proposed Profile API

```python
profile = AgentProfile("02_dragapult")

policy = profile.make_agent()
result = profile.train(iterations=200, games=16, lr=3e-4)
bundle = profile.build_submit()
```

The profile is a facade. It delegates implementation to registries rather than
implementing every algorithm itself:

```python
POLICY_FACTORIES = {
    "random": make_random_agent,
    "neural": make_neural_agent,
    "mcts": make_mcts_agent,
}

TRAINERS = {
    "ppo": train_ppo,
    "expert_iteration": train_expert_iteration,
}
```

## Profile Configuration

Each profile should eventually have explicit configuration, for example:

```yaml
name: 02_dragapult
```

The profile directory remains the stable identity and storage boundary:

```text
agents/02_dragapult/
    profile.yaml
    checkpoints/
    metrics/
    runs/
    submissions/
```

YAML should become the source of policy/trainer configuration while the
directory name remains the profile ID.

## Kaggle Entry Point

`build_submit()` should generate a minimal `main.py` equivalent to:

```python
from pathlib import Path

from pkm.agents import make_neural_agent
from pkm.data import Deck

ROOT = Path(__file__).resolve().parent
DECK = Deck.from_csv(ROOT / "deck.csv").card_ids

agent = make_neural_agent(DECK)
```

The generated entry point must:

- expose a module-level `agent` callable;
- load resources relative to `__file__`;
- contain no local battle runner;
- contain no training logic;
- package only the selected profile's deck and inference weights.

## Multi-Agent Games

Local play should accept two independently constructed agents and decks:

```python
p0 = AgentProfile("02_dragapult").make_agent()
p1 = AgentProfile("01_psychic").make_agent()

play_match(agents=(p0, p1), decks=(deck0, deck1))
```

Evaluation should load each profile's policy and deck independently, alternate
player sides, and record wins, losses, draws, and matchup metrics.

## Training And Opponents

The existing PPO checkpoint pool should become a general opponent pool. It can
contain:

- the current learner;
- past checkpoints of the learner;
- random policy;
- other profile policies;
- MCTS policies.

For example, `02_dragapult` can train against itself, older Dragapult
checkpoints, random, Psychic, and MCTS. This supports self-play, cross-play,
regression testing, and population-style training without duplicating the game
runner.

## Suggested Implementation Sequence

1. Add `AgentProfile.make_agent()` and a policy registry.
2. Move deck and weight resolution into the profile.
3. Add `AgentProfile.train()` as a facade over the existing trainers.
4. Add `AgentProfile.build_submit()` and generated Kaggle entry points.
5. Refactor play to accept two independent profiles and decks.
6. Generalize the PPO opponent pool to accept multiple policy sources.
7. Add cross-play evaluation and matchup metrics.
8. Add profile configuration validation.

Phases 1 through 4 should land first. They establish stable interfaces without
requiring an immediate rewrite of PPO. Phases 5 through 7 can then build
multi-agent training on top of those interfaces.
