from .base import make_agent
from .neural_agent import make_neural_agent
from .random_agent import make_random_agent
from .singaporean_middleman import make_singaporean_middleman

__all__ = [
    "make_agent",
    "make_neural_agent",
    "make_random_agent",
    "make_singaporean_middleman",
]
