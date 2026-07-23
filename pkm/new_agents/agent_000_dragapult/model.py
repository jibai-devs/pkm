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
  * Multi-select (``maxCount > 1``) is handled one of three ways, chosen by
    ``ModelConfig.policy_head``:
      - ``"marginal"`` (default, v1): the model emits per-option logits only and
        multi-select is left to the sampling layer (fixed-logit Plackett–Luce,
        no conditioning on already-picked options).
      - ``"autoreg"``: :class:`AutoregPolicyHead` scores each pick **conditioned
        on the running set of already-picked options** (a pooled-pick summary
        ``g``) and emits a **STOP** logit, so the count is learned (it can pick
        fewer than ``maxCount`` once ``minCount`` is met). This is the upgrade
        the v1 docstring flagged as planned.
      - ``"combo"``: :class:`ComboPolicyHead` scores whole **option combinations**
        in one pass (a categorical over the enumerated legal sets, cap 64), the
        way agent_001's transformer decoder does. The action unit is an unordered
        set whose size lies in ``[minCount, maxCount]`` (the empty set included
        when ``minCount == 0``), so the count is learned by picking a smaller
        combo — no sequential conditioning, no STOP-legality bookkeeping.

**One meaning for the [B,L] head.** ``policy_from_state``/``forward``/
``evaluate`` always return *per-option logits* — for ``"autoreg"`` that's the
conditional scorer with an empty picked-set and STOP masked out; for ``"combo"``
that's the combination distribution **marginalized** back to per-option
inclusion logits (each option's summed combo-probability). So every consumer of
the ``[B,L]`` marginal (MCTS priors, the ExIt cross-entropy target,
inference-time MCTS) is identical across heads. The richer conditioning is an
*extra* capability exposed via :meth:`PolicyValueModel.policy_step` (autoreg) /
:meth:`PolicyValueModel.policy_combos` (combo), used only by rollout sampling,
the PPO log-prob recompute, and the non-MCTS inference pick (see ``policy.py`` /
``agent.py``).
"""

from __future__ import annotations

import itertools
import logging

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
# Max option-combinations enumerated per decision by the "combo" policy head
# (matches agent_001). k==1 decisions (the vast majority) never hit it; only
# wide multi-select nodes truncate, which is logged.
COMBO_CAP = 64
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
        # Selection-count bounds, used by the autoregressive head's STOP-legality
        # mask (inert for the marginal head). minCount<=picks<=maxCount.
        min_count=torch.tensor([f.min_count for f in batch], dtype=torch.long),
        max_count=torch.tensor([f.max_count for f in batch], dtype=torch.long),
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


class AutoregPolicyHead(nn.Module):
    """Autoregressive STOP-token multi-select head.

    Scores a pick **conditioned on the set of already-picked options** and emits
    a STOP logit so the pick count is learned. The already-picked set enters as a
    single summary vector ``g`` = a projection of the *mean of the picked
    options' encoded vectors* (an empty set pools to zero, so ``g`` is then just
    ``pick_proj``'s bias — a learned "nothing picked yet" vector, no special
    case).

    Two scorers, both fed ``g``:
      * ``opt_scorer([option_vec, state, ctx, referenced-entity, g]) -> logit``
        per option (same grounded inputs as the marginal scorer, plus ``g``);
      * ``stop_scorer([state, ctx, g]) -> logit`` — one STOP score per row.

    It owns no option/entity encoding: the model passes in the already-computed
    ``opt`` / ``ctx`` / ``ent_vec`` (shared with the value/marginal paths) so the
    trunk still runs once per decision. Padding/legality masks are applied by the
    caller, not here.
    """

    def __init__(
        self,
        d_opt: int,
        d_state: int,
        d_ctx_total: int,
        d_entity: int,
        d_g: int | None = None,
    ):
        super().__init__()
        self.d_g = d_g or d_opt
        self.pick_proj = nn.Linear(d_opt, self.d_g)  # pooled picked vecs -> g
        self.opt_scorer = nn.Sequential(
            nn.Linear(d_opt + d_state + d_ctx_total + d_entity + self.d_g, d_opt),
            nn.ReLU(),
            nn.Linear(d_opt, 1),
        )
        self.stop_scorer = nn.Sequential(
            nn.Linear(d_state + d_ctx_total + self.d_g, d_opt),
            nn.ReLU(),
            nn.Linear(d_opt, 1),
        )

    def summary(self, opt: torch.Tensor, picked_mask: torch.Tensor) -> torch.Tensor:
        """Summary ``g`` ``[B, d_g]`` of the picked set (mean of picked option vecs)."""
        w = picked_mask.unsqueeze(-1)  # [B,L,1]
        denom = picked_mask.sum(-1, keepdim=True).clamp(min=1.0)  # empty set -> 1
        pooled = (w * opt).sum(1) / denom  # [B,d_opt]; empty set -> 0
        return self.pick_proj(pooled)  # [B,d_g]

    def score(
        self,
        opt: torch.Tensor,  # [B,L,d_opt]
        state: torch.Tensor,  # [B,d_state]
        ctx: torch.Tensor,  # [B,d_ctx_total]
        ent_vec: torch.Tensor,  # [B,L,d_entity]
        g: torch.Tensor,  # [B,d_g]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (per-option logits ``[B,L]``, STOP logit ``[B]``) given ``g``."""
        bsz, lmax = opt.shape[0], opt.shape[1]
        cond = torch.cat([state, ctx], dim=-1).unsqueeze(1).expand(bsz, lmax, -1)
        g_exp = g.unsqueeze(1).expand(bsz, lmax, -1)
        opt_logits = self.opt_scorer(
            torch.cat([opt, cond, ent_vec, g_exp], dim=-1)
        ).squeeze(-1)  # [B,L]
        stop_logit = self.stop_scorer(torch.cat([state, ctx, g], dim=-1)).squeeze(-1)
        return opt_logits, stop_logit


_log = logging.getLogger(__name__)


def enumerate_combos(
    b: dict[str, torch.Tensor], cap: int = COMBO_CAP
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-row legal option-index combinations (count-bounded, capped at ``cap``).

    The ``"combo"`` head's action unit is an *unordered set* of option indices
    whose size lies in ``[minCount, maxCount]`` (clamped to the number of legal
    options). We enumerate those sets with :func:`itertools.combinations` —
    including the empty set when ``minCount == 0`` (the "select nothing" case) —
    hard-capped at ``cap`` per row (mirrors agent_001; a truncation is logged,
    not silent).

    Returns three tensors (``C`` = max retained combos over the batch, ``Kmax`` =
    max combo size), all on ``option_mask``'s device:
      * ``combo_idx``         ``[B,C,Kmax]`` long, member option indices, ``-1`` pad
      * ``combo_member_mask`` ``[B,C,Kmax]`` float, 1 = real member
      * ``combo_valid``       ``[B,C]``      float, 1 = real combo (not padding)

    Enumeration depends only on ``option_mask``/``min_count``/``max_count`` (all in
    ``b``), so it is identical between the rollout sampler and the PPO logprob
    recompute — the consistency invariant the trainer relies on.
    """
    mask = b["option_mask"]
    device = mask.device
    bsz = mask.shape[0]
    n_opts = mask.bool().sum(-1).tolist()
    min_c = b["min_count"].tolist()
    max_c = b["max_count"].tolist()

    rows: list[list[tuple[int, ...]]] = []
    truncated = 0
    for i in range(bsz):
        n = int(n_opts[i])
        kmax = min(max(int(max_c[i]), 1), max(n, 1))
        kmin = min(max(int(min_c[i]), 0), kmax)
        combos: list[tuple[int, ...]] = []
        hit = False
        for k in range(kmin, kmax + 1):
            for combo in itertools.combinations(range(n), k):
                if len(combos) >= cap:
                    hit = True
                    break
                combos.append(combo)
            if hit:
                break
        if not combos:  # optionless / degenerate row -> a single empty combo
            combos = [()]
        if hit:
            truncated += 1
        rows.append(combos)
    if truncated:
        _log.debug(
            "combo enumeration truncated to cap=%d in %d/%d rows", cap, truncated, bsz
        )

    n_combos = max((len(r) for r in rows), default=1)
    kmax_all = max((len(c) for r in rows for c in r), default=1) or 1
    combo_idx = torch.full((bsz, n_combos, kmax_all), -1, dtype=torch.long, device=device)
    member_mask = torch.zeros((bsz, n_combos, kmax_all), dtype=torch.float32, device=device)
    combo_valid = torch.zeros((bsz, n_combos), dtype=torch.float32, device=device)
    for i, r in enumerate(rows):
        for ci, combo in enumerate(r):
            combo_valid[i, ci] = 1.0
            for j, opt in enumerate(combo):
                combo_idx[i, ci, j] = opt
                member_mask[i, ci, j] = 1.0
    return combo_idx, member_mask, combo_valid


class ComboPolicyHead(nn.Module):
    """Combination-scoring multi-select head (``policy_head == "combo"``).

    Scores each enumerated option *combination* as a whole — a single categorical
    over the legal sets — the way agent_001's transformer decoder does, rather
    than scoring options one at a time. A combination is encoded by **mean-pooling
    its member option vectors** (from the shared :class:`OptionEncoder`) and their
    referenced board entities, then fusing the state summary + decision context:

        ``scorer([mean(member opt vecs), state, ctx, mean(member entities)]) -> logit``.

    The empty combination pools to zero (its "select nothing" vector is the
    scorer's response to an all-zero pooled input). It owns no option/entity
    encoding: the model passes in the already-computed ``opt`` / ``ctx`` /
    ``ent_vec`` (shared with the value/marginal paths) so the trunk runs once.
    Enumeration + padding masks come from :func:`enumerate_combos`.
    """

    def __init__(self, d_opt: int, d_state: int, d_ctx_total: int, d_entity: int):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d_opt + d_state + d_ctx_total + d_entity, d_opt),
            nn.ReLU(),
            nn.Linear(d_opt, 1),
        )

    def score(
        self,
        opt: torch.Tensor,  # [B,L,d_opt]
        ent_vec: torch.Tensor,  # [B,L,d_entity]
        state: torch.Tensor,  # [B,d_state]
        ctx: torch.Tensor,  # [B,d_ctx_total]
        combo_idx: torch.Tensor,  # [B,C,K] long, -1 pad
        member_mask: torch.Tensor,  # [B,C,K] float, 1 = real member
        combo_valid: torch.Tensor,  # [B,C] float, 1 = real combo
    ) -> torch.Tensor:
        """Per-combination logits ``[B,C]`` (invalid/padding combos -> MASK_FILL)."""
        bsz, _lopt, d_opt = opt.shape
        n_combos, kmax = combo_idx.shape[1], combo_idx.shape[2]
        d_ent = ent_vec.shape[-1]
        safe = combo_idx.clamp(min=0).reshape(bsz, n_combos * kmax)  # [B,C*K]
        opt_g = torch.gather(
            opt, 1, safe.unsqueeze(-1).expand(-1, -1, d_opt)
        ).view(bsz, n_combos, kmax, d_opt)
        ent_g = torch.gather(
            ent_vec, 1, safe.unsqueeze(-1).expand(-1, -1, d_ent)
        ).view(bsz, n_combos, kmax, d_ent)
        w = member_mask.unsqueeze(-1)  # [B,C,K,1]
        denom = member_mask.sum(-1, keepdim=True).clamp(min=1.0)  # empty combo -> 1
        combo_vec = (opt_g * w).sum(2) / denom  # [B,C,d_opt]; empty combo -> 0
        combo_ent = (ent_g * w).sum(2) / denom  # [B,C,d_entity]
        cond = torch.cat([state, ctx], dim=-1).unsqueeze(1).expand(bsz, n_combos, -1)
        logits = self.scorer(
            torch.cat([combo_vec, cond, combo_ent], dim=-1)
        ).squeeze(-1)  # [B,C]
        return logits.masked_fill(combo_valid == 0, MASK_FILL)


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
        policy_head: str = "marginal",
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
        # Policy head selection. Build ONLY the chosen head so state_dict keys are
        # unambiguous per architecture (the config hash pins which one). "marginal"
        # = the v1 per-option scorer; "autoreg" = the STOP-token multi-select head.
        if policy_head not in ("marginal", "autoreg", "combo"):
            raise ValueError(
                f"unknown policy_head {policy_head!r}; choose 'marginal', "
                f"'autoreg', or 'combo'"
            )
        self.policy_head = policy_head
        if policy_head == "autoreg":
            self.autoreg = AutoregPolicyHead(d_opt, d_state, 2 * d_ctx, d_entity)
        elif policy_head == "combo":
            self.combo = ComboPolicyHead(d_opt, d_state, 2 * d_ctx, d_entity)
        else:
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

    def _option_pieces(
        self, ent: torch.Tensor, b: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-option inputs shared by both heads: (opt ``[B,L,d_opt]``, ctx
        ``[B,2*d_ctx]``, referenced-entity ``[B,L,d_entity]``)."""
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
        ent_vec = self._gather_entity(ent, b)  # [B,L,d_entity]
        return opt, ctx, ent_vec

    def policy_from_state(
        self, state: torch.Tensor, ent: torch.Tensor, b: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Step-0 per-option logits ``[B,L]`` (padding-masked).

        For BOTH heads this is the marginal "probability each option is the first
        pick": the marginal scorer directly, or the autoregressive scorer with an
        empty picked-set (and STOP excluded). Every ``[B,L]`` consumer (MCTS
        priors, ExIt target, inference-MCTS) reads this identically.
        """
        opt, ctx, ent_vec = self._option_pieces(ent, b)
        if self.policy_head == "autoreg":
            g0 = self.autoreg.summary(opt, torch.zeros_like(b["option_mask"]))
            logits, _stop = self.autoreg.score(opt, state, ctx, ent_vec, g0)
        elif self.policy_head == "combo":
            # Score whole combinations, then marginalize back to per-option
            # inclusion logits so the [B,L] contract (MCTS priors / ExIt CE /
            # inference-MCTS) is unchanged. See ._marginalize.
            combo_idx, member_mask, combo_valid = enumerate_combos(b)
            combo_logits = self.combo.score(
                opt, ent_vec, state, ctx, combo_idx, member_mask, combo_valid
            )
            logits = self._marginalize(
                combo_logits, combo_idx, member_mask, b["option_mask"].shape
            )
        else:
            bsz, lmax = opt.shape[0], opt.shape[1]
            cond = torch.cat([state, ctx], dim=-1).unsqueeze(1).expand(bsz, lmax, -1)
            logits = self.scorer(torch.cat([opt, cond, ent_vec], dim=-1)).squeeze(-1)
        return logits.masked_fill(b["option_mask"] == 0, MASK_FILL)  # mask padding

    @staticmethod
    def _marginalize(
        combo_logits: torch.Tensor,  # [B,C]
        combo_idx: torch.Tensor,  # [B,C,K] long, -1 pad
        member_mask: torch.Tensor,  # [B,C,K] float
        shape: torch.Size,  # (B, L) of option_mask
    ) -> torch.Tensor:
        """Per-option inclusion logits ``[B,L]`` from a combination distribution.

        Softmaxes the combo logits, then scatters each combo's probability onto
        every option it contains: ``p_opt[l] = sum_{c: l in c} P(c)``. Downstream
        (``evaluate``) softmaxes these, which renormalizes the inclusion mass
        (which sums to E[picks], not 1) into a proper per-option prior — a valid
        relative option ranking for MCTS and a coherent target for the ExIt
        cross-entropy. For ``k == 1`` decisions each combo is a single option, so
        this reduces exactly to the combo categorical over options.
        """
        bsz, lmax = shape
        n_combos, kc = combo_idx.shape[1], combo_idx.shape[2]
        combo_p = torch.softmax(combo_logits, dim=-1)  # [B,C]
        safe = combo_idx.clamp(min=0).reshape(bsz, n_combos * kc)  # [B,C*K]
        contrib = (combo_p.unsqueeze(-1) * member_mask).reshape(bsz, n_combos * kc)
        p_opt = torch.zeros(bsz, lmax, device=combo_logits.device, dtype=combo_p.dtype)
        p_opt.scatter_add_(1, safe, contrib)  # pad members (mask 0) add 0 to slot 0
        return torch.log(p_opt.clamp_min(1e-9))

    def policy_combos(
        self, state: torch.Tensor, ent: torch.Tensor, b: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Combination distribution for the ``"combo"`` head.

        Returns (combo logits ``[B,C]`` padding-masked, ``combo_idx`` ``[B,C,K]``,
        ``member_mask`` ``[B,C,K]``, ``combo_valid`` ``[B,C]``) — everything the
        combo sampler / PPO logprob / entropy need. Only valid for the ``"combo"``
        head; the trunk (``state``/``ent``) is passed in so it runs once.
        """
        if self.policy_head != "combo":
            raise RuntimeError("policy_combos requires policy_head='combo'")
        opt, ctx, ent_vec = self._option_pieces(ent, b)
        combo_idx, member_mask, combo_valid = enumerate_combos(b)
        combo_logits = self.combo.score(
            opt, ent_vec, state, ctx, combo_idx, member_mask, combo_valid
        )
        return combo_logits, combo_idx, member_mask, combo_valid

    def policy_step(
        self,
        state: torch.Tensor,
        ent: torch.Tensor,
        b: dict[str, torch.Tensor],
        picked_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive step: score the next pick given ``picked_mask``.

        Returns (per-option logits ``[B,L]`` padding-masked, STOP logit ``[B]``).
        Only valid for the ``"autoreg"`` head. The caller applies the
        already-picked and STOP-legality masks; padding is masked here.
        """
        if self.policy_head != "autoreg":
            raise RuntimeError("policy_step requires policy_head='autoreg'")
        opt, ctx, ent_vec = self._option_pieces(ent, b)
        g = self.autoreg.summary(opt, picked_mask)
        opt_logits, stop_logit = self.autoreg.score(opt, state, ctx, ent_vec, g)
        opt_logits = opt_logits.masked_fill(b["option_mask"] == 0, MASK_FILL)
        return opt_logits, stop_logit

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
