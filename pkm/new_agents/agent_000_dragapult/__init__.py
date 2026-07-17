"""agent_000_dragapult — a Dragapult ex specialist for the cabt competition.

Self-contained: the fixed decklist (:mod:`.deck`), the deterministic tensor
dimensioning spec (:mod:`.build_spec` -> ``spec.json``), and (to come) the
observation encoder, model, training loop, and inference agent all live here.
The agent is bound to its deck: its card vocabulary and learned embeddings are
fixed to :data:`.deck.DISTINCT_IDS`.
"""
