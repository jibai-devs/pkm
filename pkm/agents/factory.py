"""Construct Kaggle-compatible callables from resolved agent profiles."""

from .neural_agent import make_neural_agent
from .profile import AgentProfile
from .registry import Agent, require_policy, require_strategy


def make_profile_agent(profile: AgentProfile) -> Agent:
    """Build a plain ``agent(obs)`` callable for a resolved profile."""
    deck = profile.load_deck()
    policy = require_policy(profile.policy)(profile, deck)
    if profile.strategy is not None:
        policy = require_strategy(profile.strategy)(profile, policy)
    return policy


__all__ = ["make_neural_agent", "make_profile_agent"]
