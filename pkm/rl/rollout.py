"""Self-play game driver: runs battles via battle_start/battle_select and
collects encoded decisions for both players."""

import random
from dataclasses import dataclass

from pkm.agents.first_turn_agent import make_first_turn_agent
from pkm.archetype.belief import compute_belief
from pkm.engine import (
    battle_finish,
    battle_select,
    battle_start,
)

from pkm.heuristics.context import GameContext
from pkm.heuristics.deck_tracker import DeckTracker
from pkm.types.obs import Observation

from .encoder import (
    EncodedDecision,
    budew_active_second_potential,
    budew_first_turn_attack_bonus,
    budew_redundant_play_penalty,
    budew_turn_bench_setup_bonus,
    drakloak_backup_ready_bonus,
    dragapult_backup_potential,
    dragapult_ex_attack_bonus,
    dreepy_energy_spread_penalty,
    dreepy_evolve_bonus,
    dreepy_line_active_charge_bonus,
    dreepy_line_bench_charge_bonus,
    dreepy_line_field_potential,
    encode_decision,
    energy_overattach_penalty,
    phantom_dive_attack_bonus,
    prize_potential,
    wasted_resources_attack_penalty,
    wrong_type_energy_penalty,
    xerosic_machinations_bonus,
)
from .model import PolicyValueNet
from .ppo import compute_returns

MAX_DECISIONS = 3000


class TorchPolicy:
    """Wraps a PolicyValueNet into a per-observation actor."""

    def __init__(
        self,
        model: PolicyValueNet,
        greedy: bool = False,
        temperature: float = 1.0,
        archetype_classifier=None,
    ):
        self.model = model
        self.greedy = greedy
        self.temperature = temperature
        # Optional NumpyArchetypeClassifier (pkm.archetype.numpy_model) --
        # opt-in, default None. When given, its belief replaces the trunk's
        # own dormant aux-head belief for the encoder's re-injection feature
        # (pkm/rl/features.py:_opponent_archetype_belief); see
        # docs/opponent-archetype-classifier-plan.md Part 2a.
        self.archetype_classifier = archetype_classifier

    def act(
        self, obs: dict, collect: bool, ctx: GameContext | None = None
    ) -> tuple[list[int], EncodedDecision | None]:
        parsed = Observation.model_validate(obs)
        sel = parsed.select
        assert sel is not None
        # forced decision: nothing to learn, don't run the network
        forced = sel.forced_picks()
        if forced is not None:
            return forced, None

        d = encode_decision(parsed, ctx)
        res = self.model.act(d, greedy=self.greedy, temperature=self.temperature)
        if ctx is not None and self.archetype_classifier is not None:
            # Carry the belief forward for the *next* decision's GLOBAL
            # feature read (see pkm/rl/features.py) -- one decision stale
            # by construction, never recomputed inside a pure feature
            # function. Left None (-> zeros) when no classifier is given,
            # so this is opt-in and doesn't affect existing callers.
            ctx.archetype_belief = compute_belief(obs, self.archetype_classifier)
        if not collect:
            return res.picks, None
        d.picks = res.picks
        d.stopped = res.stopped
        d.logprob = res.logprob
        d.value = res.value
        d.potential = prize_potential(parsed)
        d.board_setup_potential = dragapult_backup_potential(parsed)
        d.budew_setup_potential = budew_active_second_potential(parsed)
        d.dreepy_line_field_potential = dreepy_line_field_potential(parsed)
        d.energy_penalty = energy_overattach_penalty(parsed, res.picks)
        d.budew_bonus = budew_first_turn_attack_bonus(parsed, res.picks)
        d.wrong_type_energy_penalty = wrong_type_energy_penalty(parsed, res.picks)
        d.dragapult_attack_bonus = dragapult_ex_attack_bonus(parsed, res.picks)
        d.phantom_dive_bonus = phantom_dive_attack_bonus(parsed, res.picks)
        d.dreepy_spread_penalty = dreepy_energy_spread_penalty(parsed, res.picks)
        d.xerosic_bonus = xerosic_machinations_bonus(parsed, res.picks)
        d.budew_bench_setup_bonus = budew_turn_bench_setup_bonus(parsed, res.picks)
        d.dreepy_evolve_bonus = dreepy_evolve_bonus(parsed, res.picks)
        d.dreepy_bench_charge_bonus = dreepy_line_bench_charge_bonus(parsed, res.picks)
        d.dreepy_active_charge_bonus = dreepy_line_active_charge_bonus(
            parsed, res.picks
        )
        d.wasted_resources_penalty = wasted_resources_attack_penalty(parsed, res.picks)
        d.drakloak_backup_ready_bonus = drakloak_backup_ready_bonus(parsed, res.picks)
        d.budew_redundant_penalty = budew_redundant_play_penalty(parsed, res.picks)
        return res.picks, d


class RandomPolicy:
    def act(
        self, obs: dict, collect: bool, ctx: GameContext | None = None
    ) -> tuple[list[int], None]:
        sel = obs["select"]
        return random.sample(range(len(sel["option"])), sel["maxCount"]), None


def _is_own_first_turn(obs: dict) -> bool:
    """Whether the current decision belongs to the to-move player's own first
    turn -- setup (turn 0), turn 1 going first (or before the coin resolves),
    turn 2 going second. Mirrors `singaporean_middleman._select_agent`'s rule
    so training-time delegation matches the deployed router exactly (the
    engine's turn counter is shared across both players)."""
    cur = obs["current"]
    turn = cur["turn"]
    you = cur["yourIndex"]
    first_player = cur.get("firstPlayer", -1)
    if turn == 0:
        return True
    if turn == 1 and first_player != 1 - you:
        return True
    if turn == 2 and first_player == 1 - you:
        return True
    return False


class FirstTurnDelegatingPolicy:
    """Plays the to-move player's own first turn with the scripted first-turn
    agent, delegating every later decision to `inner`.

    First-turn picks are made by the scripted agent, so they are *never*
    collected into the trajectory (returns `record=None`, exactly like a
    forced pick) -- the policy is only trained on the turns it actually plays,
    matching how `singaporean_middleman` deploys it (turn 1 = scripted, turn
    2+ = neural)."""

    def __init__(self, inner, first_turn_agent):
        self.inner = inner
        self._first_turn = first_turn_agent

    def act(
        self, obs: dict, collect: bool, ctx: GameContext | None = None
    ) -> tuple[list[int], EncodedDecision | None]:
        if _is_own_first_turn(obs):
            return self._first_turn(obs), None
        return self.inner.act(obs, collect=collect, ctx=ctx)


def make_training_first_turn_agent(deck: list[int]):
    """A first-turn agent tuned for self-play speed (a small search budget),
    not deployment strength: it's called on both sides of every game, so the
    full 6s/decision deployment budget would throttle rollout to a crawl.
    Setup picks are priority-table lookups either way; only the in-turn MCTS
    scales with these knobs."""
    return make_first_turn_agent(
        deck, n_determinizations=1, n_simulations=6, time_budget_s=0.75
    )


@dataclass
class GameResult:
    trajectories: tuple[list[EncodedDecision], list[EncodedDecision]]
    rewards: tuple[float, float]  # terminal reward per player index
    decisions: int
    turns: int


def play_game(
    policies,
    decks: tuple[list[int], list[int]],
    collect: tuple[bool, bool] = (True, True),
) -> GameResult:
    """Play one full game; returns per-player trajectories and rewards."""
    obs, start = battle_start(list(decks[0]), list(decks[1]))
    if obs is None:
        raise RuntimeError(f"battle_start failed: errorPlayer={start.errorPlayer}")

    # One GameContext per player, each owning its own DeckTracker over its
    # own deck -- never reused across games (see pkm/heuristics/context.py).
    contexts = (
        GameContext(list(decks[0]), DeckTracker(decks[0])),
        GameContext(list(decks[1]), DeckTracker(decks[1])),
    )

    trajectories: tuple[list[EncodedDecision], list[EncodedDecision]] = ([], [])
    count = 0
    try:
        while obs["current"]["result"] < 0 and count < MAX_DECISIONS:
            p = obs["current"]["yourIndex"]
            tracker = contexts[p].tracker
            tracker.observe(obs)
            if tracker.is_search_reveal(obs):
                tracker.record_search_reveal(obs)
            picks, record = policies[p].act(obs, collect=collect[p], ctx=contexts[p])
            if record is not None:
                trajectories[p].append(record)
            obs = battle_select(picks)
            count += 1
        result = obs["current"]["result"]
        turns = obs["current"]["turn"]
    finally:
        battle_finish()

    if result == 0:
        rewards = (1.0, -1.0)
    elif result == 1:
        rewards = (-1.0, 1.0)
    else:  # draw or decision cap hit
        rewards = (0.0, 0.0)
    return GameResult(
        trajectories=trajectories, rewards=rewards, decisions=count, turns=turns
    )


@dataclass
class GameSpec:
    """One game's pre-decided matchup, so rollout (sequential or parallel)
    and aggregation always agree on who played and what to count."""

    opponent_state: dict | None  # None = mirror self-play (both sides = current)
    side: int  # which side "current" plays if opponent_state is set; -1 if mirror
    collect: tuple[bool, bool]
    # Part 3c: opponent's own deck when opponent_state came from the
    # cross-archetype pool. None = mirror `deck` (every pre-3c caller/spec),
    # so this is backward compatible by construction.
    opponent_deck: list[int] | None = None


def make_game_specs(
    games_per_iter: int,
    pool: list[dict],
    pool_prob: float,
    rng: random.Random,
    archetype_pool: list[tuple[list[int], dict]] | None = None,
    archetype_pool_prob: float = 0.0,
) -> list[GameSpec]:
    """Decide, up front, the full iteration's matchups — same logic regardless
    of whether rollout runs sequentially or across worker processes.

    Part 3c: when `archetype_pool` (a list of (deck, state_dict) pairs, e.g.
    from `pkm.rl.opponent_pool.load_pool_bots`) is given, `archetype_pool_prob`
    of games draw both an opponent deck and its matching pool-bot policy
    instead of self-mirroring `deck`. Rolled before the existing self-pool
    check, so `pool_prob`'s meaning (fraction of games vs a past checkpoint of
    this same deck) is unchanged when `archetype_pool_prob` is 0 (default)."""
    specs = []
    for _ in range(games_per_iter):
        if archetype_pool and rng.random() < archetype_pool_prob:
            opponent_deck, opponent_state = rng.choice(archetype_pool)
            side = rng.randint(0, 1)
            collect = (side == 0, side == 1)
            specs.append(GameSpec(opponent_state, side, collect, opponent_deck))
        elif rng.random() < pool_prob and len(pool) > 1:
            opponent_state = rng.choice(pool[:-1])
            side = rng.randint(0, 1)
            collect = (side == 0, side == 1)
            specs.append(GameSpec(opponent_state, side, collect))
        else:
            specs.append(GameSpec(None, -1, (True, True)))
    return specs


def play_one(
    current_model: PolicyValueNet,
    opponent_model: PolicyValueNet,
    deck: list[int],
    spec: GameSpec,
    archetype_classifier=None,
    first_turn_agent=None,
) -> GameResult:
    """Play one game per `spec`, reusing `opponent_model` as scratch space for
    the pooled-opponent case (avoids rebuilding a fresh module every game).

    Part 2a: `archetype_classifier` (a NumpyArchetypeClassifier, see
    pkm/archetype/numpy_model.py), when given, is attached only to the
    trainee's ("current") TorchPolicy -- not the frozen opponent's -- so its
    belief re-injection (docs/opponent-archetype-classifier-plan.md Part 2a)
    only ever influences the policy actually being trained this run. In the
    mirror self-play case both sides share one TorchPolicy instance, so both
    naturally get it too.

    If `first_turn_agent` is given, both sides delegate their own first turn
    to it (never collected) -- see `FirstTurnDelegatingPolicy`."""
    if spec.opponent_state is None:
        cur = TorchPolicy(current_model, archetype_classifier=archetype_classifier)
        policies = (cur, cur)
        decks = (deck, deck)
    else:
        opponent_model.load_state_dict(spec.opponent_state)
        opponent_model.eval()
        opp = TorchPolicy(opponent_model)
        cur = TorchPolicy(current_model, archetype_classifier=archetype_classifier)
        opp_deck = spec.opponent_deck if spec.opponent_deck is not None else deck
        policies = (cur, opp) if spec.side == 0 else (opp, cur)
        decks = (deck, opp_deck) if spec.side == 0 else (opp_deck, deck)
    if first_turn_agent is not None:
        policies = (
            FirstTurnDelegatingPolicy(policies[0], first_turn_agent),
            FirstTurnDelegatingPolicy(policies[1], first_turn_agent),
        )
    return play_game(policies, decks, collect=spec.collect)


def aggregate_result(
    spec: GameSpec,
    result: GameResult,
    data: list,
    gamma: float,
    lam: float,
    weights: dict[str, float] | None = None,
) -> tuple[int, int, int]:
    """Extend `data` with this game's collected trajectories and return the
    (win, loss, draw) increment for `current` — same counting rule the
    sequential loop always used, factored out so the parallel path matches."""
    wins = losses = draws = 0
    for p in range(2):
        if not spec.collect[p]:
            continue
        compute_returns(
            result.trajectories[p],
            result.rewards[p],
            gamma=gamma,
            lam=lam,
            weights=weights,
        )
        data.extend(result.trajectories[p])
        if spec.side == -1 and p == 1:
            continue  # count mirror games once
        r = result.rewards[p if spec.side == -1 else spec.side]
        wins, losses, draws = wins + (r > 0), losses + (r < 0), draws + (r == 0)
    return wins, losses, draws
