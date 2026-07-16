"""Returns computation (GAE + potential-based shaping) and the PPO update."""

import numpy as np
import torch
import torch.nn.functional as F

from .encoder import EncodedDecision
from .model import PolicyValueNet


def compute_returns(
    trajectory: list[EncodedDecision],
    terminal_reward: float,
    gamma: float = 0.99,
    lam: float = 0.95,
    shaping_coef: float = 0.2,
) -> None:
    """Fill advantage/ret on each decision in place.

    Rewards are terminal win/loss plus potential-based shaping on the prize
    differential: r_t += shaping_coef * (gamma * phi(s_{t+1}) - phi(s_t)),
    which leaves the optimal policy unchanged.
    """
    n = len(trajectory)
    if n == 0:
        return
    rewards = np.zeros(n, dtype=np.float64)
    for t in range(n - 1):
        rewards[t] = shaping_coef * (
            gamma * trajectory[t + 1].potential - trajectory[t].potential
        )
    # terminal step: shaping toward final potential is folded into the outcome
    rewards[n - 1] = terminal_reward - shaping_coef * trajectory[n - 1].potential

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
    archetype_coef: float = 0.1,
) -> dict[str, float]:
    """Run PPO clip updates over the collected decisions.

    Task 8: also adds the opponent-archetype auxiliary head's own
    cross-entropy loss, weighted by archetype_coef, at the same point
    policy_loss/value_loss combine (plan.md §8.2 rule 2) -- the actual
    combine site is here in pkm/rl/ppo.py, not pkm/rl/train.py, despite
    the plan naming train.py.

    Decisions with true_archetype == -1 (unknown -- e.g. self-play against
    a single fixed deck never stamps a real label) are masked out of the
    aux loss entirely; a batch with no labeled decisions contributes 0.
    """
    advs = np.array([d.advantage for d in decisions], dtype=np.float32)
    adv_mean, adv_std = advs.mean(), advs.std() + 1e-8

    stats = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "clip_frac": 0.0,
        "archetype_loss": 0.0,
    }
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

            labels = torch.tensor([d.true_archetype for d in batch], dtype=torch.long)
            labeled = labels >= 0
            if bool(labeled.any()):
                archetype_logits = model.evaluate_archetype(batch)
                per_sample = F.cross_entropy(
                    archetype_logits, labels.clamp(min=0), reduction="none"
                )
                archetype_loss = (
                    per_sample * labeled.float()
                ).sum() / labeled.float().sum()
            else:
                archetype_loss = torch.zeros(())

            loss = (
                policy_loss
                + value_coef * value_loss
                - entropy_coef * entropy
                + archetype_coef * archetype_loss
            )
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
            stats["archetype_loss"] += float(archetype_loss.detach())
            n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in stats.items()}
