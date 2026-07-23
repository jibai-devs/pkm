"""Action distribution over presented options (shared by rollout + PPO update).

A decision selects ``k = clamp(maxCount, minCount, n)`` option indices **without
replacement**, modelled as sequential draws from the masked categorical
(Plackett–Luce). For the common ``k==1`` case this is a plain categorical.

**Provisional (v1):** the *count* ``k`` is fixed to ``maxCount`` (matching the
inference agent), so "select fewer than max" (the ``minCount==0`` cases, ~4% of
decisions) is not a learned choice yet. `[DECIDE]`

All functions operate on masked logits; padding uses the finite ``MASK_FILL``
sentinel (see model.py) so degenerate rows never produce NaN.
"""

from __future__ import annotations

import torch

from pkm.new_agents.agent_000_dragapult.model import MASK_FILL


def select_count(min_count: int, max_count: int, n: int) -> int:
    """How many options to pick (fixed-count v1 policy)."""
    return max(min_count, min(max_count, n))


@torch.no_grad()
def sample_action(
    logits: torch.Tensor,  # [L] (already padding-masked)
    valid: torch.Tensor,  # [L] bool, True = real option
    k: int,
    gen: torch.Generator | None = None,
    greedy: bool = False,
) -> tuple[list[int], float]:
    """Sample an ordered set of ``k`` distinct option indices; return (picks, logprob)."""
    valid = valid.clone()
    picks: list[int] = []
    logprob = 0.0
    for _ in range(k):
        masked = logits.masked_fill(~valid, MASK_FILL)
        logp = masked - torch.logsumexp(masked, dim=-1, keepdim=True)
        if greedy:
            idx = int(torch.argmax(logp))
        else:
            idx = int(torch.multinomial(logp.exp(), 1, generator=gen))
        logprob += float(logp[idx])
        picks.append(idx)
        valid[idx] = False
    return picks, logprob


def batched_action_logprob(
    logits: torch.Tensor,  # [B,L] padding-masked
    option_mask: torch.Tensor,  # [B,L] 1 = real option
    actions: torch.Tensor,  # [B,K] padded with 0
    action_len: torch.Tensor,  # [B] number of real picks per row
) -> torch.Tensor:
    """Recompute logprob of the given ordered actions under ``logits`` -> [B]."""
    valid = option_mask.bool().clone()
    total = torch.zeros(logits.shape[0], device=logits.device)
    K = actions.shape[1]
    for j in range(K):
        masked = logits.masked_fill(~valid, MASK_FILL)
        logp = masked - torch.logsumexp(masked, dim=-1, keepdim=True)  # [B,L]
        a_j = actions[:, j].unsqueeze(1)  # [B,1]
        step_lp = logp.gather(1, a_j).squeeze(1)  # [B]
        has = (j < action_len).to(logits.dtype)  # rows picking at step j
        total = total + step_lp * has
        # remove the chosen index from valid, only for rows that actually picked
        picked = torch.zeros_like(valid).scatter(1, a_j, True) & has.bool().unsqueeze(1)
        valid = valid & ~picked
    return total


def batched_entropy(logits: torch.Tensor, option_mask: torch.Tensor) -> torch.Tensor:
    """Entropy of the base masked categorical (bonus term) -> [B]."""
    masked = logits.masked_fill(option_mask == 0, MASK_FILL)
    logp = masked - torch.logsumexp(masked, dim=-1, keepdim=True)
    p = logp.exp()
    return -(p * logp).sum(dim=-1)


# --------------------------------------------------------------------------- #
# Autoregressive STOP-token head (policy_head == "autoreg")
# --------------------------------------------------------------------------- #
#
# Generative process for one decision presenting ``n`` legal options, count
# bounds ``minCount <= m <= maxCount`` (clamped to ``n``):
#   for step j = 0, 1, ...:
#     * if j == maxCount:            forced stop  (m = maxCount, no STOP term)
#     * elif no option is available: forced stop  (m = j = n,   no STOP term)
#     * else: choose among {available options} plus {STOP if j >= minCount}.
#             STOP chosen -> stop (m = j); else pick an option and continue.
# So STOP is a *genuine decision* at step j iff ``minCount <= j < maxCount`` and
# an option is still available. logprob(action of length m) = sum of the m per-
# step pick logprobs + (the terminal STOP logprob iff ``m < maxCount and m < n``).
# The sampler and the batched recompute below implement exactly this, so a
# sampled action's returned logprob equals its recomputed logprob (tested).


def _autoreg_counts(b: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(n, min_count, max_count) per row, clamped consistently: 1<=maxCount<=n,
    0<=minCount<=maxCount."""
    n = b["option_mask"].bool().sum(-1)  # [B]
    max_c = torch.minimum(b["max_count"].clamp(min=1), n.clamp(min=1))
    min_c = b["min_count"].clamp(min=0)
    min_c = torch.minimum(min_c, max_c)
    return n, min_c, max_c


@torch.no_grad()
def sample_action_autoreg(
    model,
    state: torch.Tensor,  # [1,d_state]
    ent: torch.Tensor,  # [1,12,d_entity]
    b: dict[str, torch.Tensor],  # batch of 1
    gen: torch.Generator | None = None,
    greedy: bool = False,
) -> tuple[list[int], float]:
    """Sample an ordered set of distinct picks from the autoregressive head.

    Batch-of-1 (one live decision). Returns (picks, logprob) where the logprob is
    the full sequence logprob including the terminal STOP decision when the
    policy stops before ``maxCount``.
    """
    device = state.device
    valid = b["option_mask"][0].bool()  # [L]
    L = valid.shape[0]
    _, min_c_t, max_c_t = _autoreg_counts(b)
    min_c, max_c = int(min_c_t[0]), int(max_c_t[0])
    picked = torch.zeros(1, L, device=device)
    picks: list[int] = []
    logprob = 0.0
    for j in range(max_c):
        avail = valid & (picked[0] == 0)
        if not bool(avail.any()):
            break  # options exhausted -> forced stop, no term
        opt_logits, stop_logit = model.policy_step(state, ent, b, picked)
        ol = opt_logits[0].masked_fill(~avail, MASK_FILL)  # [L]
        offered_stop = min_c <= j  # j < max_c always holds inside range(max_c)
        cat = torch.cat([ol, stop_logit]) if offered_stop else ol
        logp = cat - torch.logsumexp(cat, dim=-1, keepdim=True)
        if greedy:
            idx = int(torch.argmax(logp))
        else:
            idx = int(torch.multinomial(logp.exp(), 1, generator=gen))
        logprob += float(logp[idx])
        if offered_stop and idx == L:  # STOP token (appended at index L)
            break
        picks.append(idx)
        picked[0, idx] = 1.0
    return picks, logprob


def batched_action_logprob_autoreg(
    model,
    state: torch.Tensor,  # [B,d_state]
    ent: torch.Tensor,  # [B,12,d_entity]
    b: dict[str, torch.Tensor],
    actions: torch.Tensor,  # [B,K] padded with 0
    action_len: torch.Tensor,  # [B] number of real picks m per row
) -> torch.Tensor:
    """Recompute logprob of the given ordered autoregressive actions -> [B]."""
    device = state.device
    valid = b["option_mask"].bool()  # [B,L]
    B = valid.shape[0]
    n, min_c, max_c = _autoreg_counts(b)
    m = action_len.to(device)
    picked = torch.zeros_like(b["option_mask"], dtype=torch.float32)
    total = torch.zeros(B, device=device)
    ar = torch.arange(B, device=device)
    kmax = int(m.max()) if B else 0
    for j in range(kmax):
        opt_logits, stop_logit = model.policy_step(state, ent, b, picked)
        avail = valid & (picked == 0)
        ol = opt_logits.masked_fill(~avail, MASK_FILL)  # [B,L]
        logz_opts = torch.logsumexp(ol, dim=-1)  # [B]
        offered = (min_c <= j) & (j < max_c) & avail.any(-1)  # [B] bool
        logz = torch.where(offered, torch.logaddexp(logz_opts, stop_logit), logz_opts)
        a_j = actions[:, j].to(device).clamp(min=0)  # [B]
        step_lp = ol[ar, a_j] - logz  # [B]
        active = j < m  # rows that actually pick at step j
        total = total + torch.where(active, step_lp, torch.zeros_like(step_lp))
        upd = torch.zeros_like(picked)
        upd[ar, a_j] = active.to(picked.dtype)
        picked = picked + upd
    # Terminal STOP decision (picked now holds every row's full action).
    opt_logits, stop_logit = model.policy_step(state, ent, b, picked)
    avail = valid & (picked == 0)
    ol = opt_logits.masked_fill(~avail, MASK_FILL)
    logz = torch.logaddexp(torch.logsumexp(ol, dim=-1), stop_logit)
    stop_lp = stop_logit - logz  # [B]
    need_stop = (m < max_c) & (m < n)
    total = total + torch.where(need_stop, stop_lp, torch.zeros_like(stop_lp))
    return total


def batched_entropy_autoreg(
    model,
    state: torch.Tensor,
    ent: torch.Tensor,
    b: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Entropy of the step-0 choice distribution (options + STOP if minCount==0) -> [B].

    A cheap, well-defined exploration bonus: the first-pick distribution (STOP is
    a legal first choice only when ``minCount == 0``). Full-sequence entropy is
    intractable; the step-0 term is the standard stand-in.
    """
    picked0 = torch.zeros_like(b["option_mask"], dtype=torch.float32)
    opt_logits, stop_logit = model.policy_step(state, ent, b, picked0)
    ol = opt_logits.masked_fill(~b["option_mask"].bool(), MASK_FILL)
    _, min_c, max_c = _autoreg_counts(b)
    offered = (min_c <= 0) & (max_c > 0)  # STOP legal at step 0 iff minCount==0
    stop_col = torch.where(offered, stop_logit, torch.full_like(stop_logit, MASK_FILL))
    cat = torch.cat([ol, stop_col.unsqueeze(-1)], dim=-1)  # [B,L+1]
    logp = cat - torch.logsumexp(cat, dim=-1, keepdim=True)
    return -(logp.exp() * logp).sum(-1)


# --------------------------------------------------------------------------- #
# Combination-scoring head (policy_head == "combo")
# --------------------------------------------------------------------------- #
#
# The action unit here is an *unordered set* of option indices (a "combo"),
# drawn from a single categorical over the legal combinations enumerated by
# model.enumerate_combos (count-bounded, cap 64, empty set included when
# minCount==0). So a decision's logprob is one categorical logprob — no
# per-step Plackett-Luce product, no STOP term. The rollout sampler and the PPO
# recompute below both call model.policy_combos on the *same* batch dict, so the
# enumeration (hence the combo distribution) is identical and a sampled combo's
# returned logprob equals its recompute (matched by set, tested).


def _combo_members(
    combo_idx: torch.Tensor, member_mask: torch.Tensor, i: int, c: int
) -> frozenset[int]:
    """The option-index set of combo ``c`` in row ``i`` (empty set for the pad/empty combo)."""
    k = combo_idx.shape[2]
    return frozenset(
        int(combo_idx[i, c, j]) for j in range(k) if member_mask[i, c, j] > 0
    )


@torch.no_grad()
def sample_action_combo(
    model,
    state: torch.Tensor,  # [1,d_state]
    ent: torch.Tensor,  # [1,12,d_entity]
    b: dict[str, torch.Tensor],  # batch of 1
    gen: torch.Generator | None = None,
    greedy: bool = False,
) -> tuple[list[int], float]:
    """Sample one option combination from the combo head -> (sorted picks, logprob).

    Batch-of-1 (one live decision). The combo head learns its own count, so no
    ``k`` is imposed: an empty combo (``minCount == 0``) yields ``[]``.
    """
    combo_logits, combo_idx, member_mask, _valid = model.policy_combos(state, ent, b)
    logits = combo_logits[0]  # [C]
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    if greedy:
        c = int(torch.argmax(logp))
    else:
        c = int(torch.multinomial(logp.exp(), 1, generator=gen))
    picks = sorted(_combo_members(combo_idx, member_mask, 0, c))
    return picks, float(logp[c])


def _match_combos(
    actions: torch.Tensor,  # [B,K] padded with 0
    action_len: torch.Tensor,  # [B]
    combo_idx: torch.Tensor,  # [B,C,Kc]
    member_mask: torch.Tensor,  # [B,C,Kc]
) -> torch.Tensor:
    """Index ``[B]`` of the enumerated combo whose member set == each row's action.

    Matching is by *set* (order-independent), so it is robust to the sampler
    returning picks in any order. Falls back to 0 if no combo matches (cannot
    happen for an action that came from the same enumeration).
    """
    bsz, n_combos = combo_idx.shape[0], combo_idx.shape[1]
    target = torch.zeros(bsz, dtype=torch.long, device=combo_idx.device)
    acts = actions.tolist()
    alen = action_len.tolist()
    for i in range(bsz):
        want = frozenset(acts[i][: int(alen[i])])
        for c in range(n_combos):
            if _combo_members(combo_idx, member_mask, i, c) == want:
                target[i] = c
                break
    return target


def batched_action_logprob_combo(
    model,
    state: torch.Tensor,  # [B,d_state]
    ent: torch.Tensor,  # [B,12,d_entity]
    b: dict[str, torch.Tensor],
    actions: torch.Tensor,  # [B,K] padded with 0
    action_len: torch.Tensor,  # [B] number of real picks per row
) -> torch.Tensor:
    """Recompute logprob of the given combos under the combo head -> [B]."""
    combo_logits, combo_idx, member_mask, _valid = model.policy_combos(state, ent, b)
    logp = combo_logits - torch.logsumexp(combo_logits, dim=-1, keepdim=True)  # [B,C]
    target = _match_combos(actions, action_len, combo_idx, member_mask)  # [B]
    return logp.gather(1, target.unsqueeze(1)).squeeze(1)


def batched_entropy_combo(
    model,
    state: torch.Tensor,
    ent: torch.Tensor,
    b: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Entropy of the combination categorical (exploration bonus) -> [B].

    Well-defined and exact for this head (the action *is* a single categorical
    draw), unlike the step-0 stand-in the autoregressive head must use. Padding
    combos carry MASK_FILL logits (prob ~0), so they contribute ~0 to the sum.
    """
    combo_logits, *_ = model.policy_combos(state, ent, b)
    logp = combo_logits - torch.logsumexp(combo_logits, dim=-1, keepdim=True)
    return -(logp.exp() * logp).sum(-1)
