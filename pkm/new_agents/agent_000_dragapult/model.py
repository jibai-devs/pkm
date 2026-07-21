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
    embedding + the state summary + **the encoder's contextual embedding of the
    board entity the option references** (grounded pointer, via
    ``option_entity_slot``; :meth:`_gather_entity`). This is what makes the
    encoder's set attention pay off — the per-entity embeddings are now consumed.
  * Multi-select (``maxCount > 1``) is left to the sampling layer (the agent);
    the model emits per-option logits only. Sequential, in-network conditioning
    on already-picked options (a STOP token + running pick summary) is a planned
    upgrade. `[DECIDE]`
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pkm.new_agents.agent_000_dragapult.cabt import (
    OptionType,
    SelectContext,
    SelectType,
)
from pkm.new_agents.agent_000_dragapult.attacks import AttackEncoder
from pkm.new_agents.agent_000_dragapult.encoder import (
    CardEncoder,
    StateEncoder,
    collate_states,
)
from pkm.new_agents.agent_000_dragapult.features import O, Features

# Masked-option fill: a FINITE large-negative sentinel, not -inf. exp(-1e9)
# underflows to 0 so real rows still put ~0 mass on padding, but a fully-masked
# row (a hypothetical 0-option decision collated at train time) yields a uniform
# finite distribution instead of NaN — which -inf would produce and would poison
# gradients. (Inference never hits this: the agent slices to real options.)
MASK_FILL = -1e9

# Provisional dims.
D_OPT = 64
D_CTX = 16  # select type/context embedding dim (each)
N_OPTION_TYPES = len(OptionType)  # 17
N_SELECT_TYPES = len(SelectType)  # 11
N_SELECT_CTX = len(SelectContext)  # 49


def collate(batch: list[Features]) -> dict[str, torch.Tensor]:
    """Batch Features into tensors, including padded options + option mask."""
    s = collate_states(batch)
    bsz = len(batch)
    lmax = max((f.n_options for f in batch), default=1) or 1
    otype = torch.zeros(bsz, lmax, dtype=torch.long)
    ofeat = torch.zeros(bsz, lmax, O, dtype=torch.float32)
    omask = torch.zeros(bsz, lmax, dtype=torch.float32)
    # -1 = "no board entity" (also the fill for padded option slots).
    oslot = torch.full((bsz, lmax), -1, dtype=torch.long)
    ocard = torch.zeros(bsz, lmax, dtype=torch.long)  # raw card id (0 = none)
    ocard_row = torch.zeros(bsz, lmax, dtype=torch.long)  # own-vocab row
    oatk = torch.zeros(bsz, lmax, dtype=torch.long)  # attack id (0 = none)
    for i, f in enumerate(batch):
        n = f.n_options
        if n:
            otype[i, :n] = torch.from_numpy(f.option_type)
            ofeat[i, :n] = torch.from_numpy(f.option_feat)
            oslot[i, :n] = torch.from_numpy(f.option_entity_slot)
            ocard[i, :n] = torch.from_numpy(f.option_card_id)
            ocard_row[i, :n] = torch.from_numpy(f.option_card_row)
            oatk[i, :n] = torch.from_numpy(f.option_attack_id)
            omask[i, :n] = 1.0
    s.update(
        option_type=otype,
        option_feat=ofeat,
        option_entity_slot=oslot,
        option_card_id=ocard,
        option_card_row=ocard_row,
        option_attack_id=oatk,
        option_mask=omask,
        select_type=torch.tensor([f.select_type for f in batch], dtype=torch.long),
        select_context=torch.tensor(
            [f.select_context for f in batch], dtype=torch.long
        ),
    )
    return s


class OptionEncoder(nn.Module):
    """Encode each presented option -> a vector.

    An option is a pointer, so its vector fuses four grounded signals:
      * **what kind** of action it is (``type_emb``),
      * **which card** it acts with — run through the *shared* hybrid card
        encoder, so a card has the same identity here as on the board,
      * **which move** it is (``attack_enc``; 0 = not an attack), and
      * the remaining genuinely-numeric fields (counts/numbers) via ``feat_proj``.
    """

    def __init__(
        self,
        card_enc: CardEncoder,
        attack_enc: AttackEncoder,
        d_opt: int = D_OPT,
    ):
        super().__init__()
        self.type_emb = nn.Embedding(N_OPTION_TYPES, d_opt)
        self.feat_proj = nn.Linear(O, d_opt)
        self.card = card_enc  # SHARED with the board/state encoder
        self.card_proj = nn.Linear(card_enc.d_card, d_opt)
        self.attack = attack_enc
        self.attack_proj = nn.Linear(attack_enc.d_atk, d_opt)

    def forward(
        self,
        option_type: torch.Tensor,
        option_feat: torch.Tensor,
        option_card_row: torch.Tensor,
        option_card_id: torch.Tensor,
        option_attack_id: torch.Tensor,
    ) -> torch.Tensor:
        card_vec = self.card(option_card_row, option_card_id)  # [B,L,d_card]
        return (
            self.type_emb(option_type)
            + self.feat_proj(option_feat)
            + self.card_proj(card_vec)
            + self.attack_proj(self.attack(option_attack_id))
        )  # [B,L,d_opt]


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
        attack_enc: AttackEncoder | None = None,
        aux_tasks: list[str] | tuple[str, ...] = (),
    ):
        super().__init__()
        self.encoder = encoder or StateEncoder()
        d_state = self.encoder.d_state
        d_entity = self.encoder.d_entity
        # The option encoder SHARES the trunk's card encoder, so a card is
        # embedded identically whether it sits on the board or is being played.
        self.option_enc = OptionEncoder(
            self.encoder.card, attack_enc or AttackEncoder(), d_opt
        )
        self.sel_type_emb = nn.Embedding(N_SELECT_TYPES, d_ctx)
        self.sel_ctx_emb = nn.Embedding(N_SELECT_CTX, d_ctx)
        # Learned stand-in embedding for options that reference no board entity
        # (YES/NO, NUMBER, deck/hand picks, …). Gathered in place of a real
        # per-entity vector so the scorer input is always the same width.
        self.null_entity = nn.Parameter(torch.zeros(d_entity))
        # policy scorer: per-option MLP over
        #   [option_vec, state, decision-context, referenced-entity]
        self.scorer = nn.Sequential(
            nn.Linear(d_opt + d_state + 2 * d_ctx + d_entity, d_opt),
            nn.ReLU(),
            nn.Linear(d_opt, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(d_state, d_state),
            nn.ReLU(),
            nn.Linear(d_state, 1),
        )
        # Auxiliary heads (training-only): one per active task, keyed by name in
        # a ModuleDict so the set is config-derived and empty by default. Built
        # from the registry so adding a task never touches this file. Never
        # called by policy/value/evaluate — see .aux_from_state and aux_tasks.py.
        from pkm.new_agents.agent_000_dragapult.aux_tasks import AUX_TASKS

        self.aux_tasks = list(aux_tasks)
        self.aux_heads = nn.ModuleDict(
            {name: AUX_TASKS[name].make_head(d_state) for name in self.aux_tasks}
        )

    # --- trunk ---
    def encode(self, b: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (state ``[B,d_state]``, per-entity ``[B,12,d_entity]``)."""
        return self.encoder(b)

    # --- heads (operate on a precomputed state, so the trunk runs once) ---
    def value_from_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.value_head(state).squeeze(-1)  # [B]

    def aux_from_state(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predictions from every active auxiliary head, keyed by task name.

        Training-only: the PPO update consumes these to add the aux losses.
        Empty when no aux task is active (the default), so it costs nothing.
        """
        return {name: self.aux_heads[name](state).squeeze(-1) for name in self.aux_tasks}

    def _gather_entity(
        self, ent: torch.Tensor, b: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Per-option referenced-entity vector ``[B,L,d_entity]``.

        Each option carries ``option_entity_slot`` (the board slot it acts on,
        or -1). We gather that entity's contextual embedding from the encoder;
        options with no target — or that resolve to an empty slot — get the
        learned ``null_entity``. This is what makes the encoder's attention pay
        off: the pointer scores an option against the *entity it references*.
        """
        d_ent = ent.shape[-1]
        slot = b["option_entity_slot"]  # [B,L] long, -1 = none
        safe = slot.clamp(min=0)  # valid index for gather; masked out below
        gathered = torch.gather(
            ent, 1, safe.unsqueeze(-1).expand(-1, -1, d_ent)
        )  # [B,L,d_entity]
        occ = torch.gather(b["entity_mask"], 1, safe)  # [B,L] 1 = occupied slot
        has = (slot >= 0) & (occ > 0)
        null = self.null_entity.expand_as(gathered)
        return torch.where(has.unsqueeze(-1), gathered, null)

    def policy_from_state(
        self, state: torch.Tensor, ent: torch.Tensor, b: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        opt = self.option_enc(
            b["option_type"],
            b["option_feat"],
            b["option_card_row"],
            b["option_card_id"],
            b["option_attack_id"],
        )  # [B,L,d_opt]
        ctx = torch.cat(
            [
                self.sel_type_emb(b["select_type"]),
                self.sel_ctx_emb(b["select_context"]),
            ],
            dim=-1,
        )  # [B,2*d_ctx]
        bsz, lmax = opt.shape[0], opt.shape[1]
        cond = torch.cat([state, ctx], dim=-1).unsqueeze(1).expand(bsz, lmax, -1)
        ent_vec = self._gather_entity(ent, b)  # [B,L,d_entity]
        logits = self.scorer(torch.cat([opt, cond, ent_vec], dim=-1)).squeeze(-1)  # [B,L]
        return logits.masked_fill(b["option_mask"] == 0, MASK_FILL)  # mask padding

    def forward(self, b: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        state, ent = self.encode(b)
        return self.policy_from_state(state, ent, b), self.value_from_state(state)

    def value(self, b: dict[str, torch.Tensor]) -> torch.Tensor:
        """Value-only (e.g. MCTS leaf eval)."""
        return self.value_from_state(self.encode(b)[0])

    @torch.no_grad()
    def evaluate(self, b: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """MCTS node eval: (priors over legal options, value). No grad."""
        state, ent = self.encode(b)
        priors = torch.softmax(self.policy_from_state(state, ent, b), dim=-1)
        return priors, self.value_from_state(state)
