"""Per-decision shaping, borrowed read-only from the default agent.

The existing reward terms in `pkm/rl/encoder.py` already say what a Dragapult
deck wants -- charge the line, evolve, land Phantom Dive, don't strand energy
on the wrong Pokemon. There is no reason to invent a second opinion for this
experiment, so this module just calls those functions and folds them into a
single number with the agent's own tuned weights.

Nothing here mutates `pkm`. If a term is renamed or removed upstream this
degrades to ignoring it rather than crashing, so the experiment can't be
broken by unrelated work on the main agent.
"""

from __future__ import annotations

from pkm.rl import encoder
from pkm.types.obs import Observation

# weight name (as used in reward_weights.json) -> the encoder function that
# produces it. Only the *direct*, action-conditioned terms: potential-based
# terms are differences between consecutive states, which has no meaning for
# a per-decision score summed over a game.
DIRECT_TERM_FUNCS: dict[str, str] = {
    "energy_penalty": "energy_overattach_penalty",
    "budew_bonus": "budew_first_turn_attack_bonus",
    "wrong_type_penalty": "wrong_type_energy_penalty",
    "dragapult_bonus": "dragapult_ex_attack_bonus",
    "dreepy_spread": "dreepy_energy_spread_penalty",
    "xerosic": "xerosic_machinations_bonus",
    "budew_bench_setup": "budew_turn_bench_setup_bonus",
    "dreepy_evolve": "dreepy_evolve_bonus",
    "dreepy_bench_charge": "dreepy_line_bench_charge_bonus",
    "dreepy_active_charge": "dreepy_line_active_charge_bonus",
    "wasted_resources": "wasted_resources_attack_penalty",
    "phantom_dive": "phantom_dive_attack_bonus",
    "drakloak_backup_ready": "drakloak_backup_ready_bonus",
    "budew_redundant": "budew_redundant_penalty",
}


def shaping_score(
    obs: Observation, picks: list[int], weights: dict[str, float]
) -> float:
    """Weighted sum of the default agent's direct reward terms for one pick."""
    total = 0.0
    for name, func_name in DIRECT_TERM_FUNCS.items():
        coef = weights.get(name, 0.0)
        if not coef:
            continue
        func = getattr(encoder, func_name, None)
        if func is None:
            continue  # upstream renamed it; ignore rather than crash
        try:
            total += coef * func(obs, picks)
        except Exception:
            continue
    return total
