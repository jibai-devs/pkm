"""agent_001_transformer — a from-scratch AlphaZero-style transformer agent.

Ported (near-verbatim) from a Kaggle community notebook (see
``../agent_000_dragapult/references/example.py``) and rewired to drive this
repo's libcg seam (``pkm.new_agents.agent_000_dragapult.cabt``) instead of the
Kaggle ``cg-lib`` dataset. Architecture differs sharply from agent_000:

  * **encoder**: ``EmbeddingBag`` bag-of-features over a wide sparse index
    space -> TransformerEncoder (vs agent_000's structured per-entity encoder);
  * **action head**: a transformer *decoder* that cross-attends the encoder and
    scores enumerated option *combinations* (vs agent_000's marginal pointer);
  * **targets**: TD(lambda)-blended value + advantage-style policy targets from
    MCTS (vs agent_000's clean visit-count / GAE targets).

Shared architecture + feature builders + MCTS live in :mod:`.net`; training in
:mod:`.train`; the Kaggle entry point in :mod:`.submit_main`.
"""
