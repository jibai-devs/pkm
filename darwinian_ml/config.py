"""Every knob for the darwinian experiment, in one place."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pkm.rl.reward_terms import load_weights

# Defaults point at the repo as it stands; nothing here is written to.
DEFAULT_BUNDLE = "submissions/submission_04_mega_abomasnow_20260718_021728.tar.gz"
DEFAULT_DECK = "deck/03_pult_munki.csv"
DEFAULT_SEED_WEIGHTS = "pkm/policy.npz"
DEFAULT_SHAPING_WEIGHTS = "agents/03_pult_munki/reward_weights.json"
OUTPUT_DIR = "darwinian_ml/runs/default_dragapult_darwinian"


@dataclass
class DarwinConfig:
    # --- what we are fighting ---------------------------------------------
    bundle: str = DEFAULT_BUNDLE
    deck_path: str = DEFAULT_DECK

    # --- population -------------------------------------------------------
    # 342k parameters is enormous for a genetic algorithm, so the population
    # is small and *seeded* from an already-trained policy (see README): this
    # is directed mutation around a working agent, not a search from noise.
    population: int = 12
    elites: int = 3  # survive untouched into the next generation
    tournament: int = 3  # parents chosen by best-of-N sampling
    generations: int = 200

    # --- variation --------------------------------------------------------
    sigma: float = 0.02  # gaussian mutation scale, relative to weight std
    sigma_decay: float = 0.995  # annealed each generation
    sigma_min: float = 0.002
    crossover_rate: float = 0.5  # fraction of genes taken from parent B

    # --- evaluation -------------------------------------------------------
    games_per_genome: int = 4  # sides alternate, so keep this even
    seed_weights: str | None = DEFAULT_SEED_WEIGHTS

    # --- fitness shape ----------------------------------------------------
    # Winning dominates, but prize margin keeps the landscape climbable while
    # the population still loses every game -- without it, early generations
    # are all tied at -1 and selection is blind.
    w_win: float = 10.0
    w_prize_margin: float = 3.0
    w_shaping: float = 1.0
    shaping_weights: dict[str, float] = field(default_factory=dict)

    # --- bookkeeping ------------------------------------------------------
    out_dir: str = OUTPUT_DIR
    max_hours: float = 0.0  # wall-clock cap; 0 = no limit
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.shaping_weights:
            self.shaping_weights = load_weights(DEFAULT_SHAPING_WEIGHTS)
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
