"""Registry of reward-shaping terms: the single source of truth mapping a
weight name -> (kind, EncodedDecision attribute).

Add a new shaping term by adding one entry here -- compute_returns, the
training CLI, and the default weights file all pick it up automatically.
No more threading a new `..._coef: float = 0.0` parameter through
encoder.py / rollout.py / ppo.py / train.py / cli/__init__.py by hand.
"""

import json
from pathlib import Path

# potential-based terms: coef * (gamma * phi(s') - phi(s)) -- pure functions
# of state, reward *reaching* a state rather than paying out every step it
# holds.
POTENTIAL_TERMS: list[tuple[str, str]] = [
    ("shaping", "potential"),
    ("board_setup", "board_setup_potential"),
    ("budew_setup", "budew_setup_potential"),
    ("dreepy_field", "dreepy_line_field_potential"),
]

# direct, action-conditioned terms: coef * value, added straight into the
# reward at the step they're conditioned on.
DIRECT_TERMS: list[tuple[str, str]] = [
    ("energy_penalty", "energy_penalty"),
    ("budew_bonus", "budew_bonus"),
    ("wrong_type_penalty", "wrong_type_energy_penalty"),
    ("dragapult_bonus", "dragapult_attack_bonus"),
    ("dreepy_spread", "dreepy_spread_penalty"),
    ("xerosic", "xerosic_bonus"),
    ("budew_bench_setup", "budew_bench_setup_bonus"),
    ("dreepy_evolve", "dreepy_evolve_bonus"),
    ("dreepy_bench_charge", "dreepy_bench_charge_bonus"),
    ("dreepy_active_charge", "dreepy_active_charge_bonus"),
    ("wasted_resources", "wasted_resources_penalty"),
    ("phantom_dive", "phantom_dive_bonus"),
    ("drakloak_backup_ready", "drakloak_backup_ready_bonus"),
    ("budew_redundant", "budew_redundant_penalty"),
]

ALL_TERMS: list[tuple[str, str]] = POTENTIAL_TERMS + DIRECT_TERMS
TERM_NAMES: list[str] = [name for name, _ in ALL_TERMS]

# These are the *global* fallback used by any agent that has no
# reward_weights.json of its own -- deliberately conservative (matches the
# pre-refactor default of "only the prize-differential shaping is on")
# so introducing a new term here never silently changes an existing
# agent's training the next time it resumes. Deck-specific "current best
# guess" magnitudes belong in that agent's own reward_weights.json (see
# AgentProfile.reward_weights_path), not here -- e.g. 03_pult_munki's file
# carries the tuned Budew/Dreepy-line values, other agents don't.
# Every term is listed explicitly (even at 0.0) so a freshly generated
# weights file documents every knob that exists, not just the active ones.
DEFAULT_WEIGHTS: dict[str, float] = {
    "shaping": 0.2,
    "board_setup": 0.0,
    "budew_setup": 0.0,
    "dreepy_field": 0.0,
    "energy_penalty": 0.0,
    "budew_bonus": 0.0,
    "wrong_type_penalty": 0.0,
    "dragapult_bonus": 0.0,
    "dreepy_spread": 0.0,
    "xerosic": 0.0,
    "budew_bench_setup": 0.0,
    "dreepy_evolve": 0.0,
    "dreepy_bench_charge": 0.0,
    "dreepy_active_charge": 0.0,
    "wasted_resources": 0.0,
    "phantom_dive": 0.0,
    "drakloak_backup_ready": 0.0,
    "budew_redundant": 0.0,
}


def load_weights(path: str | Path | None) -> dict[str, float]:
    """DEFAULT_WEIGHTS, overridden by whatever `path` (a JSON file of
    {term_name: weight}) contains. Unknown keys in the file are ignored --
    doesn't crash on a stale file from before a term was renamed/removed."""
    weights = dict(DEFAULT_WEIGHTS)
    if path is None:
        return weights
    p = Path(path)
    if not p.is_file():
        return weights
    overrides = json.loads(p.read_text())
    for name, value in overrides.items():
        if name in DEFAULT_WEIGHTS:
            weights[name] = float(value)
    return weights


def write_default_weights_file(path: str | Path) -> None:
    """Write DEFAULT_WEIGHTS to `path` as pretty JSON, if nothing's there
    yet -- gives a fresh agent profile a real file to hand-edit rather than
    an invisible set of Python defaults."""
    p = Path(path)
    if p.is_file():
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(DEFAULT_WEIGHTS, indent=2) + "\n")
