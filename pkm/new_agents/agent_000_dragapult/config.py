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
from pkm.new_agents.agent_000_dragapult.aux_tasks import (
    active_tasks,
    default_weights as _default_aux_weights,
)
from pkm.new_agents.agent_000_dragapult.deck import DEFAULT_DECK
from pkm.new_agents.agent_000_dragapult.encoder import StateEncoder
from pkm.new_agents.agent_000_dragapult.features import FEATURE_VERSION
from pkm.new_agents.agent_000_dragapult.model import PolicyValueModel
from pkm.rl.reward_terms import DEFAULT_WEIGHTS


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
    # --- trunk depth ---
    # Number of entity-attention layers in the encoder. 1 == the original v1
    # architecture (a single nn.MultiheadAttention, no residual/LayerNorm) and is
    # bit-for-bit checkpoint-compatible. >1 stacks (n_layers-1) extra pre-LN
    # transformer layers (residual + LayerNorm + FFN) on top for real depth.
    n_layers: int = 1
    ff_mult: int = 4  # FFN width = ff_mult * d_entity in the extra layers
    dropout: float = 0.0  # dropout in the extra transformer layers
    # Pre-LN residual around the base attention layer. False == v1 (no skip there;
    # only the extra transformer layers are residual). True makes the whole trunk
    # uniformly residual — recommended for deep (large/xxl) nets. Changes params
    # (adds a LayerNorm), so it's part of the config hash and checkpoint identity.
    base_residual: bool = False
    # Policy-head style. "marginal" (default, v1) scores each presented option
    # independently and leaves multi-select to the sampling layer (fixed-logit
    # Plackett–Luce). "autoreg" adds an autoregressive STOP-token head that
    # conditions each pick on the running set of already-picked options and can
    # stop early (learned count) — see model.AutoregPolicyHead. Different params
    # (a whole extra head), so it's part of the config hash and checkpoint
    # identity; old checkpoints lack the field and backfill to "marginal".
    policy_head: str = "marginal"


# Named size presets for one-word scaling from the CLI (`--model <name>`).
# "small" IS the original v1 net (unchanged defaults). Individual `--d-*`/`--n-*`
# override flags win over a preset's value. d_ctx/d_atk keep their ModelConfig
# defaults unless explicitly overridden.
MODEL_PRESETS: dict[str, dict[str, int]] = {
    "small":  {"d_card": 32, "d_entity": 64,  "d_global": 64,  "d_state": 128, "n_heads": 4, "d_opt": 64,  "n_layers": 1},  # noqa: E241
    "medium": {"d_card": 48, "d_entity": 128, "d_global": 96,  "d_state": 256, "n_heads": 8, "d_opt": 128, "n_layers": 2},  # noqa: E241
    "large":  {"d_card": 64, "d_entity": 192, "d_global": 128, "d_state": 384, "n_heads": 8, "d_opt": 192, "n_layers": 3},  # noqa: E241
    "xl":     {"d_card": 64, "d_entity": 256, "d_global": 192, "d_state": 512, "n_heads": 8, "d_opt": 256, "n_layers": 4},  # noqa: E241
}


def resolve_device(name: str = "cpu") -> str:
    """Resolve a device selector to a concrete ``'cpu'`` / ``'cuda'``.

    ``'auto'`` picks ``'cuda'`` when a CUDA build + GPU are available, else
    ``'cpu'``. An explicit ``'cuda'`` with no usable CUDA raises a clear error
    (the installed torch here is often a CPU-only build) rather than failing
    cryptically later. Device is a *runtime* choice — it is intentionally NOT
    part of ``Config`` / the config hash, so the same run is one experiment
    whether trained on CPU or GPU.
    """
    import torch

    n = (name or "cpu").lower()
    if n == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if n == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"--device cuda requested but torch.cuda is unavailable (installed "
            f"torch is '{torch.__version__}'). Install a CUDA build of torch, or "
            f"use --device cpu / --device auto."
        )
    if n not in ("cpu", "cuda"):
        raise ValueError(f"unknown device {name!r}; choose 'cpu', 'cuda', or 'auto'")
    return n


def build_model_config(
    preset: str = "small",
    overrides: dict[str, int | float | str | None] | None = None,
) -> "ModelConfig":
    """Resolve a `ModelConfig` from a named size preset plus per-field overrides.

    `overrides` maps a `ModelConfig` field name to a value; `None` values are
    ignored (so unset CLI flags fall through to the preset). Any field not named
    by the preset or overrides keeps its `ModelConfig` default.
    """
    if preset not in MODEL_PRESETS:
        raise ValueError(
            f"unknown model preset {preset!r}; choose from {sorted(MODEL_PRESETS)}"
        )
    fields: dict[str, int | float | str] = dict(MODEL_PRESETS[preset])
    if overrides:
        fields.update({k: v for k, v in overrides.items() if v is not None})
    return ModelConfig(**fields)  # type: ignore[arg-type]  # dropout float, dims int, policy_head str


@dataclass(frozen=True)
class TrainConfig:
    """PROVISIONAL PPO-style hyperparameters (algorithm not yet chosen)."""

    seed: int = 0
    batch_size: int = 256  # decisions per optimizer batch
    minibatch_size: int = 64
    epochs_per_update: int = 4
    lr: float = 3e-4
    # LR schedule over the run: "constant" (default, v1 — fixed lr) or "cosine"
    # (CosineAnnealingLR from lr down to lr_min over the planned updates; good for
    # long runs). Part of the config hash; old checkpoints backfill to constant.
    lr_schedule: str = "constant"
    lr_min: float = 0.0  # cosine floor (eta_min); e.g. 1e-5 for a 1e-4 start
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
    # Per-term shaping weights, consulted only when shaping == "heuristic" (the
    # full deck-specific reward stack ported from pkm/rl). Maps a term name in
    # reward_terms.ALL_TERMS -> coefficient. Defaults to DEFAULT_WEIGHTS (every
    # term listed, all deck-specific ones at 0.0), so this field changes nothing
    # unless shaping is switched to "heuristic" and weights are set. Serialized
    # into every checkpoint config (and folded into the config hash), so a run's
    # weights are fully reproducible.
    reward_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_WEIGHTS)
    )
    # Auxiliary-loss weights (key into aux_tasks.AUX_TASKS -> coefficient). A task
    # is active (its head is built + its loss added) iff its weight > 0. Defaults
    # to every registered task at 0.0, so this field changes nothing unless a
    # weight is set. Serialized into every checkpoint config (folded into the
    # config hash), so a run's aux setup is fully reproducible and the model
    # rebuilt on resume/eval has exactly the same heads. See .aux_tasks.
    aux_weights: dict[str, float] = field(default_factory=_default_aux_weights)
    # --- training method selector (key into trainers.TRAINERS) ---
    method: str = "ppo"
    # MCTS expert-iteration knobs (inert unless method == "exit").
    mcts_simulations: int = 32
    mcts_c_puct: float = 1.25
    mcts_temperature: float = 1.0
    determinization: str = "sample"  # key into trainers.exit determinizers
    # Determinized worlds averaged per decision during ExIt self-play (IS-MCTS).
    # 1 (default) == single-world mcts.search (v1). >1 averages the root policy
    # over W independent determinizations (mcts.search_worlds) so the π target
    # earns its rank across many possible hidden layouts, not one lucky guess.
    # Cost scales linearly in W. Old checkpoints backfill to 1.
    mcts_worlds: int = 1
    # ExIt value-target scheme. "mc" (default, v1) = the raw game outcome
    # (±1/0) for the acting seat — bit-for-bit the current behaviour. "tdlambda"
    # blends the outcome with the MCTS-refined root value along each seat's
    # trajectory (agent_001's scheme), lowering value-target variance. Part of
    # the config hash; old checkpoints backfill to "mc".
    exit_value_target: str = "mc"  # "mc" | "tdlambda"
    exit_lambda: float = 0.9  # EMA factor for the tdlambda blend (inert for "mc")


@dataclass(frozen=True)
class RunConfig:
    """Run identity + checkpoint policy."""

    name: str = "agent_000_dragapult"
    feature_version: str = FEATURE_VERSION
    checkpoint_every_updates: int = 64
    keep_last: int = 5
    # Which registered deck (deck.DECKS) both self-play seats pilot for this run.
    # The learned vocabulary spans *all* decks, so this only chooses the 60-card
    # list played — not the network shape. Part of the config hash (a run's deck
    # is part of its identity); old checkpoints without it backfill to the default.
    deck: str = DEFAULT_DECK


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
        n_layers=mc.n_layers,
        ff_mult=mc.ff_mult,
        dropout=mc.dropout,
        base_residual=mc.base_residual,
    )
    # Auxiliary heads are a *training* concern, so they're built only when a full
    # Config (with a TrainConfig) is given — i.e. training / resume / eval / the
    # rollout workers, which all rebuild from the same cfg so their state_dicts
    # match. Inference rebuilds from a bare ModelConfig (no train config), so it
    # gets no aux heads — matching the bundle whose aux keys pack.py stripped.
    aux = active_tasks(cfg.train.aux_weights) if isinstance(cfg, Config) else []
    return PolicyValueModel(
        encoder=encoder,
        d_opt=mc.d_opt,
        d_ctx=mc.d_ctx,
        attack_enc=AttackEncoder(d_atk=mc.d_atk),
        aux_tasks=aux,
        policy_head=mc.policy_head,
    )
