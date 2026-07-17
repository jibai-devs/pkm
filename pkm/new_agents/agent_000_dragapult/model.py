"""Policy + value model for agent_000_dragapult (provisional v1).

Wires the state encoder (:mod:`.encoder`) to two heads:
  * **policy** — a pointer/scorer over the *presented* legal options
    (``select.option``); the engine only ever offers legal options, so the mask
    is intrinsic (we just mask padding added for batching).
  * **value** — a scalar V(state).

Container style: ``PolicyValueModel`` holds named submodules (encoder, option
encoder, heads) so methods can swap/extend pieces (multi-net designs) later.

**Provisional v1:**
  * The scorer uses each option's (type + raw fields) + a decision-context
    embedding + the state summary. It does **not yet gather the board entity an
    option references** (the per-entity embeddings from the encoder) — that
    richer pointer is a planned upgrade. `[DECIDE]`
  * Multi-select (``maxCount > 1``) is left to the sampling layer (the agent);
    the model emits per-option logits only. `[DECIDE]`
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pkm.cabt.api import OptionType, SelectContext, SelectType
from pkm.agents.agent_000_dragapult.encoder import StateEncoder, collate_states
from pkm.agents.agent_000_dragapult.features import O, Features

# Masked-option fill: a FINITE large-negative sentinel, not -inf. exp(-1e9)
# underflows to 0 so real rows still put ~0 mass on padding, but a fully-masked
# row (a hypothetical 0-option decision collated at train time) yields a uniform
# finite distribution instead of NaN — which -inf would produce and would poison
# gradients. (Inference never hits this: the agent slices to real options.)
MASK_FILL = -1e9

# Provisional dims.
D_OPT = 64
D_CTX = 16          # select type/context embedding dim (each)
N_OPTION_TYPES = len(OptionType)      # 17
N_SELECT_TYPES = len(SelectType)      # 11
N_SELECT_CTX = len(SelectContext)     # 49


def collate(batch: list[Features]) -> dict[str, torch.Tensor]:
    """Batch Features into tensors, including padded options + option mask."""
    s = collate_states(batch)
    bsz = len(batch)
    lmax = max((f.n_options for f in batch), default=1) or 1
    otype = torch.zeros(bsz, lmax, dtype=torch.long)
    ofeat = torch.zeros(bsz, lmax, O, dtype=torch.float32)
    omask = torch.zeros(bsz, lmax, dtype=torch.float32)
    for i, f in enumerate(batch):
        n = f.n_options
        if n:
            otype[i, :n] = torch.from_numpy(f.option_type)
            ofeat[i, :n] = torch.from_numpy(f.option_feat)
            omask[i, :n] = 1.0
    s.update(
        option_type=otype,
        option_feat=ofeat,
        option_mask=omask,
        select_type=torch.tensor([f.select_type for f in batch], dtype=torch.long),
        select_context=torch.tensor([f.select_context for f in batch], dtype=torch.long),
    )
    return s


class OptionEncoder(nn.Module):
    """Encode each presented option -> a vector."""

    def __init__(self, d_opt: int = D_OPT):
        super().__init__()
        self.type_emb = nn.Embedding(N_OPTION_TYPES, d_opt)
        self.feat_proj = nn.Linear(O, d_opt)

    def forward(self, option_type: torch.Tensor, option_feat: torch.Tensor) -> torch.Tensor:
        return self.type_emb(option_type) + self.feat_proj(option_feat)   # [B,L,d_opt]


class PolicyValueModel(nn.Module):
    """Encoder + option encoder + policy scorer + value head.

    The trunk (``encoder``) and heads are separate submodules exposed via
    ``encode`` / ``policy`` / ``value`` / ``evaluate`` so consumers (a training
    loss, an MCTS driver, a swapped head) can call each independently without
    reaching into internals. Pass a custom ``encoder`` to tune its size/shape.
    """

    def __init__(
        self,
        encoder: StateEncoder | None = None,
        d_opt: int = D_OPT,
        d_ctx: int = D_CTX,
    ):
        super().__init__()
        self.encoder = encoder or StateEncoder()
        d_state = self.encoder.d_state
        self.option_enc = OptionEncoder(d_opt)
        self.sel_type_emb = nn.Embedding(N_SELECT_TYPES, d_ctx)
        self.sel_ctx_emb = nn.Embedding(N_SELECT_CTX, d_ctx)
        # policy scorer: per-option MLP over [option_vec, state, decision-context]
        self.scorer = nn.Sequential(
            nn.Linear(d_opt + d_state + 2 * d_ctx, d_opt), nn.ReLU(),
            nn.Linear(d_opt, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_state, d_state), nn.ReLU(), nn.Linear(d_state, 1),
        )

    # --- trunk ---
    def encode(self, b: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (state ``[B,d_state]``, per-entity ``[B,12,d_entity]``)."""
        return self.encoder(b)

    # --- heads (operate on a precomputed state, so the trunk runs once) ---
    def value_from_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.value_head(state).squeeze(-1)                          # [B]

    def policy_from_state(self, state: torch.Tensor, b: dict[str, torch.Tensor]) -> torch.Tensor:
        opt = self.option_enc(b["option_type"], b["option_feat"])          # [B,L,d_opt]
        ctx = torch.cat([self.sel_type_emb(b["select_type"]),
                         self.sel_ctx_emb(b["select_context"])], dim=-1)   # [B,2*d_ctx]
        bsz, lmax = opt.shape[0], opt.shape[1]
        cond = torch.cat([state, ctx], dim=-1).unsqueeze(1).expand(bsz, lmax, -1)
        logits = self.scorer(torch.cat([opt, cond], dim=-1)).squeeze(-1)   # [B,L]
        return logits.masked_fill(b["option_mask"] == 0, MASK_FILL)        # mask padding

    def forward(self, b: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        state, _entity = self.encode(b)
        return self.policy_from_state(state, b), self.value_from_state(state)

    def value(self, b: dict[str, torch.Tensor]) -> torch.Tensor:
        """Value-only (e.g. MCTS leaf eval)."""
        return self.value_from_state(self.encode(b)[0])

    @torch.no_grad()
    def evaluate(self, b: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """MCTS node eval: (priors over legal options, value). No grad."""
        state, _ = self.encode(b)
        priors = torch.softmax(self.policy_from_state(state, b), dim=-1)
        return priors, self.value_from_state(state)
