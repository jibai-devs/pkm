import random
from typing import Callable

from .base import make_agent


def make_random_agent(deck: list[int]) -> Callable[[dict], list[int]]:
    """Create a random agent that selects random legal moves.

    Args:
        deck: List of 60 card IDs

    Returns:
        Agent function compatible with kaggle-environments
    """

    def strategy(obs: dict) -> list[int]:
        options = obs["select"]["option"]
        max_count = obs["select"]["maxCount"]
        return random.sample(range(len(options)), max_count)

    return make_agent(deck, strategy)
