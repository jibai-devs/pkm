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
