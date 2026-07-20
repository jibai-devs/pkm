"""Evaluate a trained agent against every Part 3b pool bot, on each bot's
own deck -- fills a real gap `evaluate_vs_random` (pkm/rl/train.py) doesn't
cover: a per-archetype win rate, not just vs the random agent. Also useful
later for the ablation comparing frozen-pool (Milestone 8) vs population
training (Milestone 9) -- see docs/opponent-archetype-classifier-plan.md
Part 3, Verification #5.

Usage:
    pkm eval-vs-pool --agent 03_pult_munki --games 20

Archetype belief defaults to **on** here (unlike pkm.rl.train's opt-in
--archetype-belief), because this tool's whole purpose is to measure what a
checkpoint actually does, and pkm/agents/neural_agent.py -- what pkm play
and the real Kaggle submission use -- always computes live belief by
auto-loading pkm/archetype.npz. An eval that defaults to zero belief would
be silently measuring a distribution production never runs; see
docs/superpowers/plans/2026-07-20-belief-classifier-routing.md for the full
investigation that found this. --no-archetype-belief remains available to
reproduce the old zero-belief baseline for comparison.
"""

import shutil
import tempfile
from pathlib import Path

import torch
import typer

from pkm.agents.profile import AgentProfile
from pkm.data import Deck

from .model import PolicyValueNet
from .opponent_pool import load_pool_bots
from .rollout import TorchPolicy, play_game

DEFAULT_ARCHETYPE_WEIGHTS = "pkm/archetype.npz"


def _load_archetype_classifier(weights_path: str):
    """Non-fatal: mirrors pkm/agents/neural_agent.py's philosophy -- a
    missing/corrupt archetype classifier must never take down an eval run,
    it should just fall back to zero belief with a visible warning."""
    if not Path(weights_path).is_file():
        print(
            f"warning: archetype weights not found at {weights_path!r} -- "
            "evaluating with zero belief (pass --archetype-weights or "
            "--no-archetype-belief to silence this)",
            flush=True,
        )
        return None
    from pkm.archetype.numpy_model import NumpyArchetypeClassifier

    return NumpyArchetypeClassifier.load(weights_path)


def _safe_load_state_dict(path: Path) -> dict:
    """Copy the checkpoint before loading, so reading it concurrently with
    a live training run's torch.save() (not atomic) can't hand back a
    torn/partial file."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        shutil.copyfile(path, tmp.name)
        tmp_path = tmp.name
    try:
        return torch.load(tmp_path, map_location="cpu", weights_only=True)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def eval_vs_pool(
    agent: str = "03_pult_munki",
    games: int = 20,
    pool_glob: str = "pool_*",
    agents_dir: str = "agents",
    archetype_classifier=None,
) -> dict[str, float]:
    """Win rate of `agent`'s greedy policy against each pool bot's greedy
    policy, on the pool bot's own deck, alternating sides. Returns
    {pool_bot_name: win_rate}; prints each result plus the unweighted mean
    across all pool bots as it goes.

    `archetype_classifier` (a NumpyArchetypeClassifier, see
    pkm.archetype.numpy_model), when given, is attached to **both** sides'
    TorchPolicy -- unlike pkm.rl.train's solo-training convention (trainee
    only, never the frozen opponent), there's no frozen side here: both
    checkpoints are just being played, matching how pkm play/production
    would run either one. None (the default at this layer) means zero
    belief for both, same as before this parameter existed; main() below
    defaults to loading one."""
    profile = AgentProfile(agent)
    deck = Deck.from_csv(str(profile.deck_path)).card_ids
    ckpt = profile.checkpoint_dir / "ppo_latest.pt"
    model = PolicyValueNet()
    model.load_state_dict(_safe_load_state_dict(ckpt))
    model.eval()
    policy = TorchPolicy(model, greedy=True, archetype_classifier=archetype_classifier)

    bots = load_pool_bots(agents_dir=agents_dir, pattern=pool_glob)
    results: dict[str, float] = {}
    for bot in bots:
        opp_model = PolicyValueNet()
        opp_model.load_state_dict(bot.state_dict)
        opp_model.eval()
        opp_policy = TorchPolicy(
            opp_model, greedy=True, archetype_classifier=archetype_classifier
        )

        wins = 0.0
        for g in range(games):
            side = g % 2
            policies = (policy, opp_policy) if side == 0 else (opp_policy, policy)
            decks = (deck, bot.deck) if side == 0 else (bot.deck, deck)
            result = play_game(policies, decks, collect=(False, False))
            r = result.rewards[side]
            wins += 1.0 if r > 0 else 0.5 if r == 0 else 0.0
        results[bot.name] = wins / games
        print(f"{agent} vs {bot.name}: {results[bot.name]:.1%}", flush=True)

    if results:
        overall = sum(results.values()) / len(results)
        print(f"overall vs {len(results)} pool bots: {overall:.1%}", flush=True)
    return results


app = typer.Typer(help=__doc__)


@app.command()
def main(
    agent: str = typer.Option("03_pult_munki", help="agent profile name to evaluate"),
    games: int = typer.Option(20, help="games per pool bot, alternating sides"),
    pool_glob: str = typer.Option(
        "pool_*", help="glob under agents/ for pool-bot profiles"
    ),
    use_archetype_belief: bool = typer.Option(
        True,
        "--archetype-belief/--no-archetype-belief",
        help="compute live opponent-archetype belief for both sides, "
        "matching pkm play/Kaggle (default: on). --no-archetype-belief "
        "reproduces the old always-zero-belief baseline.",
    ),
    archetype_weights: str = typer.Option(
        DEFAULT_ARCHETYPE_WEIGHTS,
        help="path to the exported NumpyArchetypeClassifier weights, used "
        "when --archetype-belief is set",
    ),
) -> None:
    archetype_classifier = (
        _load_archetype_classifier(archetype_weights) if use_archetype_belief else None
    )
    eval_vs_pool(
        agent=agent,
        games=games,
        pool_glob=pool_glob,
        archetype_classifier=archetype_classifier,
    )


if __name__ == "__main__":
    app()
