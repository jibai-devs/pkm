"""Standalone next-generation agents.

Kept separate from :mod:`pkm.agents` (the original submission agents) on purpose:
these agents own their full training stack (encoder, model, PPO loop, CLI) and do
not reuse the ``pkm/rl`` infrastructure.
"""
