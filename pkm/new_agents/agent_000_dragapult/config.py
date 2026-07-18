"""Configuration for agent_000_dragapult — the single, resumable source of truth.

Groups all knobs (model dims + training hyperparameters + run settings) into
frozen dataclasses that serialize to/from plain dicts, so a full config can be
stored in every checkpoint and a run reconstructed exactly. `build_model` threads
the model dims into the network (rather than relying on the module-constant
defaults in `encoder.py` / `model.py`).

Format: dataclasses with dict (JSON) serialization. A YAML layer can wrap
`to_dict`/`from_dict` later if desired — not required. `[DECIDE]`

**TrainConfig is provisional:** the learning algorithm is not yet chosen (see
README `[DECIDE]`); those fields are placeholder PPO-style defaults, safe to
change and not a commitment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from pkm.new_agents.agent_000_dragapult.attacks import AttackEncoder
from pkm.new_agents.agent_000_dragapult.encoder import StateEncoder
from pkm.new_agents.agent_000_dragapult.features import FEATURE_VERSION
from pkm.new_agents.agent_000_dragapult.model import PolicyValueModel


@dataclass(frozen=True)
class ModelConfig:
    """Network dimensions (mirror the provisional defaults in encoder/model)."""

    d_card: int = 32
    d_entity: int = 64
    d_global: int = 64
    d_state: int = 128
    n_heads: int = 4
    d_opt: int = 64
    d_ctx: int = 16
    d_atk: int = 32  # attack (move) embedding dim


@dataclass(frozen=True)
class TrainConfig:
    """PROVISIONAL PPO-style hyperparameters (algorithm not yet chosen)."""

    seed: int = 0
    batch_size: int = 256  # decisions per optimizer batch
    minibatch_size: int = 64
    epochs_per_update: int = 4
    lr: float = 3e-4
    gamma: float = 0.997  # long horizon (~77 decisions/game)
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_workers: int = 8  # parallel self-play envs (one engine/process)
    # Target assignment (see .shaping). Defaults reproduce the v1 terminal-only,
    # plain-GAE behaviour bit-for-bit; opt into shaping via a sweep.
    advantage: str = "gae"  # key into shaping.ESTIMATORS
    # Default is potential-based prize-differential shaping (policy-invariant,
    # densifies the sparse ±1 terminal signal). Set shaping="terminal" (or
    # shaping_coef=0.0) to recover the original v1 terminal-only behaviour.
    shaping: str = "prize_potential"  # key into shaping.SHAPERS
    shaping_coef: float = 1.0  # scale on the shaping term (0.0 == terminal)
    # --- training method selector (key into trainers.TRAINERS) ---
    method: str = "ppo"
    # MCTS expert-iteration knobs (inert unless method == "exit").
    mcts_simulations: int = 32
    mcts_c_puct: float = 1.25
    mcts_temperature: float = 1.0
    determinization: str = "sample"  # key into trainers.exit determinizers


@dataclass(frozen=True)
class RunConfig:
    """Run identity + checkpoint policy."""

    name: str = "agent_000_dragapult"
    feature_version: str = FEATURE_VERSION
    checkpoint_every_updates: int = 64
    keep_last: int = 5


def _hash_dict(d: dict[str, Any]) -> str:
    """Stable 12-char sha256 of a config dict (the one hashing definition)."""
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        # Backfill missing fields with defaults (supports old checkpoints).
        train_dict = {**asdict(TrainConfig()), **d["train"]}
        return cls(
            model=ModelConfig(**d["model"]),
            train=TrainConfig(**train_dict),
            run=RunConfig(**d["run"]),
        )

    def hash(self) -> str:
        """Stable short hash of the whole config (goes in checkpoints/run dirs)."""
        return _hash_dict(self.to_dict())


def build_model(cfg: Config | ModelConfig | None = None) -> PolicyValueModel:
    """Construct the model with dims from config (single wiring point)."""
    mc = cfg.model if isinstance(cfg, Config) else (cfg or ModelConfig())
    encoder = StateEncoder(
        d_card=mc.d_card,
        d_entity=mc.d_entity,
        d_global=mc.d_global,
        d_state=mc.d_state,
        n_heads=mc.n_heads,
    )
    return PolicyValueModel(
        encoder=encoder,
        d_opt=mc.d_opt,
        d_ctx=mc.d_ctx,
        attack_enc=AttackEncoder(d_atk=mc.d_atk),
    )
