"""Construct Kaggle-compatible callables from resolved agent profiles."""

from .neural_agent import make_neural_agent
from .profile import AgentProfile
from .registry import Agent, require_policy, require_strategy


def make_profile_agent(
    profile: AgentProfile,
    policy: str | None = None,
    weights_path: str | None = None,
) -> Agent:
    """Build a plain ``agent(obs)`` callable for a resolved profile."""
    deck = profile.load_deck()
    policy_name = policy or profile.policy
    policy_factory = require_policy(policy_name)
    if weights_path is None:
        policy_agent = policy_factory(profile, deck)
    else:
        policy_agent = policy_factory(profile, deck, weights_path)
    if profile.strategy is not None:
        policy_agent = require_strategy(profile.strategy)(profile, policy_agent)
    return policy_agent


__all__ = ["make_neural_agent", "make_profile_agent"]
