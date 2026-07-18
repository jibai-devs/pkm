"""Observation encoder for agent_000_dragapult (provisional v1).

Turns a batch of featurized observations (:mod:`.features`) into:
  * a per-observation **state embedding** ``[B, d_state]``, and
  * **per-entity embeddings** ``[B, 12, d_entity]`` for the (later) pointer
    action head to reference.

Card identity uses the **hybrid** scheme decided in the README:
  * a learned ``nn.Embedding`` over our closed 27-vocab (own cards), plus
  * an attribute channel (:mod:`.cards`) that maps any card ID — including
    open-vocab opponent cards — through its static attributes.

**Provisional:** the architecture (attention vs pooling) and all dims are
first-pass, swappable choices, NOT locked. See README `[DECIDE]` items.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from pkm.new_agents.agent_000_dragapult.cards import A, build_card_attr_table
from pkm.new_agents.agent_000_dragapult.features import F, G, Features, _VOCAB

# Provisional dimensions (tune later).
D_CARD = 32
D_ENTITY = 64
D_GLOBAL = 64
D_STATE = 128
N_HEADS = 4

# Std for the CLS token init. 0.02 is the conventional transformer
# embedding/parameter init std (as used by GPT-2 / BERT-style models).
CLS_INIT_STD = 0.02


def collate_states(batch: list[Features]) -> dict[str, torch.Tensor]:
    """Stack a list of Features into batched tensors for the encoder.

    (Board/globals are fixed-shape; option tensors are handled later by the
    action head, so they are not collated here.)
    """

    def stack(attr: str, dtype: torch.dtype) -> torch.Tensor:
        return torch.from_numpy(np.stack([getattr(f, attr) for f in batch])).to(dtype)

    return {
        "entity_id_row": stack("entity_id_row", torch.long),  # [B,12]
        "entity_card_id": stack("entity_card_id", torch.long),  # [B,12]
        "entity_feat": stack("entity_feat", torch.float32),  # [B,12,F]
        "entity_mask": stack("entity_mask", torch.float32),  # [B,12]
        "hand_hist": stack("hand_hist", torch.float32),  # [B,27]
        "discard_hist": stack("discard_hist", torch.float32),  # [B,27]
        "globals": stack("globals", torch.float32),  # [B,G]
    }


class CardEncoder(nn.Module):
    """Hybrid card encoder: learned own-vocab embedding + attribute channel."""

    def __init__(self, d_card: int = D_CARD):
        super().__init__()
        self.own_emb = nn.Embedding(_VOCAB, d_card)  # our 27-vocab (incl. UNK row)
        attr = torch.from_numpy(build_card_attr_table().astype(np.float32))
        self.register_buffer("attr", attr)  # [max_id+1, A] static
        self.attr_mlp = nn.Sequential(
            nn.Linear(A, d_card), nn.ReLU(), nn.Linear(d_card, d_card)
        )

    def forward(self, id_row: torch.Tensor, card_id: torch.Tensor) -> torch.Tensor:
        # attribute vector works for any card in the engine's card DB (opponent
        # cards included); id 0 = empty slot -> the zero attribute row
        attr_vec = self.attr_mlp(self.attr[card_id])  # [B,N,d_card]
        own_vec = self.own_emb(id_row)  # [B,N,d_card]
        return attr_vec + own_vec


class StateEncoder(nn.Module):
    """Entity/set state encoder -> (state embedding, per-entity embeddings)."""

    def __init__(
        self,
        d_card: int = D_CARD,
        d_entity: int = D_ENTITY,
        d_global: int = D_GLOBAL,
        d_state: int = D_STATE,
        n_heads: int = N_HEADS,
    ):
        super().__init__()
        self.d_state = d_state  # exposed so consumers (heads/MCTS) can size off it
        self.d_entity = d_entity
        self.card = CardEncoder(d_card)
        self.entity_proj = nn.Sequential(nn.Linear(d_card + F, d_entity), nn.ReLU())
        # A learnable CLS token, always unmasked, is prepended to the entity set.
        # It (a) pools the board via attention (its output = board summary) and
        # (b) guarantees every row has >=1 valid key, so a fully-empty board never
        # produces NaN in the attention forward OR backward pass.
        self.cls = nn.Parameter(torch.randn(1, 1, d_entity) * CLS_INIT_STD)
        self.attn = nn.MultiheadAttention(d_entity, n_heads, batch_first=True)
        self.global_mlp = nn.Sequential(nn.Linear(G + 2 * _VOCAB, d_global), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(d_entity + d_global, d_state), nn.ReLU())

    def forward(self, b: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        card_vec = self.card(b["entity_id_row"], b["entity_card_id"])  # [B,12,d_card]
        x = torch.cat([card_vec, b["entity_feat"]], dim=-1)  # [B,12,d_card+F]
        h = self.entity_proj(x)  # [B,12,d_entity]
        bsz = h.shape[0]
        seq = torch.cat([self.cls.expand(bsz, -1, -1), h], dim=1)  # [B,13,d_entity]
        cls_pad = torch.zeros(bsz, 1, dtype=torch.bool, device=h.device)
        pad = torch.cat([cls_pad, b["entity_mask"] == 0], dim=1)  # CLS never padded
        out, _ = self.attn(seq, seq, seq, key_padding_mask=pad)  # [B,13,d_entity]
        board = out[:, 0]  # CLS output = board summary
        ent = out[:, 1:]  # [B,12,d_entity] per-entity
        g_in = torch.cat([b["globals"], b["hand_hist"], b["discard_hist"]], dim=-1)
        g = self.global_mlp(g_in)  # [B,d_global]
        state = self.head(torch.cat([board, g], dim=-1))  # [B,d_state]
        return state, ent
