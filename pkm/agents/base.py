from typing import Callable


def make_agent(
    deck: list[int], strategy_fn: Callable[[dict], list[int]]
) -> Callable[[dict], list[int]]:
    """Create an agent function from a deck and strategy function.

    Args:
        deck: List of 60 card IDs
        strategy_fn: Function that takes an observation dict and returns selected option indices

    Returns:
        Agent function compatible with kaggle-environments
    """

    def agent(obs: dict) -> list[int]:
        if obs["select"] is None:
            return deck
        return strategy_fn(obs)

    return agent
