"""Returns computation (GAE + potential-based shaping) and the PPO update."""

import numpy as np
import torch

from .encoder import EncodedDecision
from .model import PolicyValueNet
from .reward_terms import DIRECT_TERMS, POTENTIAL_TERMS


def compute_returns(
    trajectory: list[EncodedDecision],
    terminal_reward: float,
    gamma: float = 0.99,
    lam: float = 0.95,
    weights: dict[str, float] | None = None,
) -> None:
    """Fill advantage/ret on each decision in place.

    `weights` maps a term name (see reward_terms.py) to its coefficient;
    a missing name means 0.0 (off). Rewards are terminal win/loss, plus two
    kinds of shaping:

    - Potential-based terms, each a pure function of state (not of what was
      picked): r_t += coef * (gamma * phi(s_{t+1}) - phi(s_t)). This leaves
      the optimal policy unchanged, and rewards *reaching* a state rather
      than paying out repeatedly for every decision made while it holds —
      the prize differential, and having Dragapult ex able to attack with a
      charged backup Drakloak on the bench.
    - Direct terms conditioned on the specific action taken at that step, so
      they're added straight into that step's reward instead of as a
      potential difference — see each `EncodedDecision` field's origin in
      encoder.py for what each one rewards/penalizes.
    """
    n = len(trajectory)
    if n == 0:
        return
    w = weights or {}
    rewards = np.zeros(n, dtype=np.float64)
    for t in range(n - 1):
        for name, attr in POTENTIAL_TERMS:
            coef = w.get(name, 0.0)
            if coef:
                rewards[t] += coef * (
                    gamma * getattr(trajectory[t + 1], attr)
                    - getattr(trajectory[t], attr)
                )
        for name, attr in DIRECT_TERMS:
            coef = w.get(name, 0.0)
            if coef:
                rewards[t] += coef * getattr(trajectory[t], attr)
    # terminal step: shaping toward final potential is folded into the outcome
    # (the terminal state's potential is implicitly 0 -- the game is over).
    rewards[n - 1] = terminal_reward
    for name, attr in POTENTIAL_TERMS:
        coef = w.get(name, 0.0)
        if coef:
            rewards[n - 1] -= coef * getattr(trajectory[n - 1], attr)
    for name, attr in DIRECT_TERMS:
        coef = w.get(name, 0.0)
        if coef:
            rewards[n - 1] += coef * getattr(trajectory[n - 1], attr)

    gae = 0.0
    for t in reversed(range(n)):
        next_value = trajectory[t + 1].value if t + 1 < n else 0.0
        delta = rewards[t] + gamma * next_value - trajectory[t].value
        gae = delta + gamma * lam * gae
        trajectory[t].advantage = gae
        trajectory[t].ret = gae + trajectory[t].value


def ppo_update(
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    decisions: list[EncodedDecision],
    epochs: int = 3,
    minibatch: int = 256,
    clip: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
) -> dict[str, float]:
    """Run PPO clip updates over the collected decisions."""
    advs = np.array([d.advantage for d in decisions], dtype=np.float32)
    adv_mean, adv_std = advs.mean(), advs.std() + 1e-8

    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0}
    n_batches = 0

    for _ in range(epochs):
        order = np.random.permutation(len(decisions))
        for start in range(0, len(decisions), minibatch):
            idx = order[start : start + minibatch]
            batch = [decisions[i] for i in idx]
            old_logprobs = torch.tensor([d.logprob for d in batch], dtype=torch.float32)
            returns = torch.tensor([d.ret for d in batch], dtype=torch.float32)
            batch_adv = torch.tensor(
                [(d.advantage - adv_mean) / adv_std for d in batch], dtype=torch.float32
            )

            logprobs, entropies, values = model.evaluate(batch)
            ratio = torch.exp(logprobs - old_logprobs)
            surr1 = ratio * batch_adv
            surr2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * batch_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = torch.nn.functional.mse_loss(values, returns)
            entropy = entropies.mean()

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            stats["policy_loss"] += float(policy_loss.detach())
            stats["value_loss"] += float(value_loss.detach())
            stats["entropy"] += float(entropy.detach())
            stats["clip_frac"] += float(
                ((ratio.detach() - 1.0).abs() > clip).float().mean()
            )
            n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in stats.items()}
