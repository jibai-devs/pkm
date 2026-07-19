"""Loads Part 3b pool-bot checkpoints (agents/pool_*/) as a cross-archetype
opponent pool for Part 3c self-play sampling. See
docs/opponent-archetype-classifier-plan.md Part 3c."""

from dataclasses import dataclass
from pathlib import Path

import torch

from pkm.data import Deck

from .features import check_stamp_sidecar


@dataclass
class PoolBot:
    name: str
    deck: list[int]
    state_dict: dict


def load_pool_bots(
    agents_dir: str = "agents", pattern: str = "pool_*"
) -> list[PoolBot]:
    """Load every trained pool-bot checkpoint under `agents_dir` matching
    `pattern` into memory (deck + state_dict). Skips a profile that has no
    `ppo_latest.pt` yet (not trained) rather than erroring, so a partially
    populated pool is usable during iterative bring-up. Raises
    FeatureStampMismatch for a checkpoint whose stamp sidecar disagrees with
    the current feature registry -- same fail-loud convention as
    `AgentProfile.latest_checkpoint`, not silently skipped, since a stale
    pool bot would otherwise train the anchor against a policy the network
    can't actually interpret today."""
    bots = []
    for profile_dir in sorted(Path(agents_dir).glob(pattern)):
        name = profile_dir.name
        ckpt = profile_dir / "checkpoints" / "ppo_latest.pt"
        deck_path = Path("deck") / f"{name}.csv"
        if not ckpt.is_file() or not deck_path.is_file():
            continue
        check_stamp_sidecar(ckpt)
        state_dict = torch.load(ckpt, map_location="cpu", weights_only=True)
        deck = Deck.from_csv(str(deck_path)).card_ids
        bots.append(PoolBot(name=name, deck=deck, state_dict=state_dict))
    return bots
