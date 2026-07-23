"""Training entry point for the Dragapult **setup** agent.

Unlike `pkm/rl/train.py`, which plays whole matches and learns from win/loss,
this trains on a deliberately truncated **2-turn episode**:

    turn 0  setup .................. scripted first-turn agent (both seats)
    turn 1  first player's opening .. scripted first-turn agent
    turn 2  second player's opening . scripted first-turn agent
    turn 3  first player's 2nd turn . SETUP AGENT  <- collected + scored
    turn 4  second player's 2nd turn  SETUP AGENT  <- collected + scored
    stop

**The turn numbers are the engine's, and it counts turns globally, not per
player.** So "each side's second turn" is engine turn 3 for whoever went
first and turn 4 for whoever went second -- not turn 2. Turn 0-2 are exactly
the span `rollout._is_own_first_turn` identifies, so the handoff here matches
the one `singaporean_middleman` performs at deployment.

**Reward.** There is no win/loss to learn from in two turns, so the terminal
reward is the end-of-turn board score from `setup_turn_score.py` (normalised
into PPO's usual O(1) range). The default agent's shaping terms still apply
underneath, scaled down by `--shaping-scale`, so they nudge without competing
with the rubric that defines this agent's whole job.

Both seats are played by the same current policy -- self-play in the literal
sense, no opponent pool: with only one scored turn per side there is no long
horizon for a stale opponent to matter.

Run it:
    python -m pkm.rl.setup_train --iterations 200 --episodes 16
    ian_tools/setup_train.sh            # background, stoppable
"""

from __future__ import annotations

import csv
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import typer

from pkm.agents.profile import AgentProfile
from pkm.data import Deck
from pkm.engine import battle_finish, battle_select, battle_start
from pkm.heuristics.context import GameContext
from pkm.heuristics.deck_tracker import DeckTracker
from pkm.types.obs import Observation, OptionType

from .encoder import EncodedDecision
from .features import write_stamp_sidecar
from .logging import MetricLog
from .model import PolicyValueNet
from .ppo import compute_returns, ppo_update
from .reward_terms import load_weights
from .rollout import (
    TorchPolicy,
    _is_own_first_turn,
    make_training_first_turn_agent,
)
from .setup_turn_score import (
    BUDEW_CARD_ID,
    CRISPIN_CARD_ID,
    DRAKLOAK_CARD_ID,
    DREEPY_CARD_ID,
    MIN_DRAKLOAK_FOR_CRISPIN,
    JUDGE_CARD_ID,
    LILLIES_DETERMINATION_CARD_ID,
    TurnScore,
    score_end_of_turn,
)

# The episode stops once the second player has finished its second turn.
LAST_SETUP_TURN = 4
MAX_EPISODE_DECISIONS = 400

# Rubric components surfaced as training metrics -- the totals alone don't say
# *why* a run is improving, and these are the ones worth watching.
TRACKED_PARTS = (
    "bench_drakloak",
    "phantom_dive_ready",
    "phantom_dive_ready_next",
    "itchy_pollen",
    "hand_size",
)

CSV_FIELDS = [
    "iter",
    "episodes",
    "decisions",
    "samples",
    "score_mean",
    "score_first",
    "score_second",
    "itchy_rate",
    *[f"part_{p}" for p in TRACKED_PARTS],
    "pi_loss",
    "v_loss",
    "entropy",
    "clip_frac",
    "time_s",
]


def seat_for_turn(turn: int, first_player: int) -> int:
    """Which seat plays `turn`. The engine alternates from turn 1 onward."""
    return first_player if turn % 2 == 1 else 1 - first_player


def setup_turn_for_seat(seat: int, first_player: int) -> int:
    """The engine turn on which `seat` takes its second (setup) turn."""
    return 3 if seat == first_player else 4


@dataclass
class SetupEpisode:
    """One 2-turn episode: what each seat did, and how its board scored."""

    trajectories: tuple[list[EncodedDecision], list[EncodedDecision]]
    scores: tuple[TurnScore, TurnScore]
    itchy: tuple[bool, bool]
    decisions: int = 0
    parts: dict[str, float] = field(default_factory=dict)


def _budew_attack_ids() -> set[int]:
    """Budew's attack ids (Itchy Pollen).

    Resolved here rather than in `setup_turn_score`, which is deliberately
    free of card-database imports; the trainer knows which option it picked
    and passes the resulting flag in.
    """
    from pkm.data.card_data import get_card_by_id

    card = get_card_by_id(BUDEW_CARD_ID)
    return {a.attack_id for a in card.attacks} if card else set()


def _judge_over_lillies(obs: dict, picks: list[int]) -> bool:
    """Judge played with Lillie's still in hand (see W_JUDGE_OVER_LILLIES)."""
    sel = obs.get("select") or {}
    options = sel.get("option") or []
    state = obs["current"]
    me = state["players"][state["yourIndex"]]
    hand = me.get("hand") or []
    for i in picks:
        if not 0 <= i < len(options):
            continue
        opt = options[i]
        if opt.get("type") != int(OptionType.PLAY):
            continue
        cid = opt.get("cardId") or 0
        if not cid and opt.get("index") is not None and 0 <= opt["index"] < len(hand):
            cid = hand[opt["index"]]["id"]
        if cid == JUDGE_CARD_ID:
            return any(
                c.get("id") == LILLIES_DETERMINATION_CARD_ID for c in hand
            )
    return False


def _crispin_played_early(obs: dict, picks: list[int]) -> bool:
    """Crispin played while the board is short of Drakloak (see W_CRISPIN_EARLY).

    Mirrors `_apply_events` in pkm/agents/turn1agent_dep/search.py so the RL
    agent trains against the same rubric the search agent optimises.
    """
    sel = obs.get("select") or {}
    options = sel.get("option") or []
    state = obs["current"]
    me = state["players"][state["yourIndex"]]
    hand = me.get("hand") or []
    for i in picks:
        if not 0 <= i < len(options):
            continue
        opt = options[i]
        if opt.get("type") != int(OptionType.PLAY):
            continue
        cid = opt.get("cardId") or 0
        if not cid and opt.get("index") is not None and 0 <= opt["index"] < len(hand):
            cid = hand[opt["index"]]["id"]
        if cid == CRISPIN_CARD_ID:
            in_play = [
                c for c in [*(me.get("active") or []), *(me.get("bench") or [])] if c
            ]
            n = sum(1 for c in in_play if c.get("id") == DRAKLOAK_CARD_ID)
            if n < MIN_DRAKLOAK_FOR_CRISPIN:
                return True
    return False


def _count_retreats(obs: dict, picks: list[int]) -> int:
    """How many of these picks are a retreat (see W_RETREAT)."""
    options = (obs.get("select") or {}).get("option") or []
    return sum(
        1
        for i in picks
        if 0 <= i < len(options)
        and options[i].get("type") == int(OptionType.RETREAT)
    )


def _excuses_meowth(obs: dict, picks: list[int]) -> bool:
    """Did this pick play a Supporter that justifies having benched Meowth ex?

    Mirrors `_apply_events` in `pkm/agents/turn1agent_dep/search.py` so the RL
    agent trains against the same rubric the search agent optimises. Without
    this the trainer scored every Meowth ex at a flat -1.00 while deployment
    waived it, which is exactly the kind of train/deploy divergence that makes
    a learned policy disagree with its own objective.

    Judge counts only when no Dreepy is in play *at this moment* -- see the
    note on W_MEOWTH_EX_IN_PLAY for why it is not tested at end of turn.
    """
    sel = obs.get("select") or {}
    options = sel.get("option") or []
    state = obs["current"]
    me = state["players"][state["yourIndex"]]
    hand = me.get("hand") or []
    for i in picks:
        if not 0 <= i < len(options):
            continue
        opt = options[i]
        if opt.get("type") != int(OptionType.PLAY):
            continue
        cid = opt.get("cardId") or 0
        if not cid and opt.get("index") is not None and 0 <= opt["index"] < len(hand):
            cid = hand[opt["index"]]["id"]
        if cid == LILLIES_DETERMINATION_CARD_ID:
            return True
        if cid == JUDGE_CARD_ID:
            in_play = [c for c in [*(me.get("active") or []), *(me.get("bench") or [])] if c]
            if not any(c.get("id") == DREEPY_CARD_ID for c in in_play):
                return True
    return False


def _used_itchy_pollen(obs: dict, picks: list[int], budew_ids: set[int]) -> bool:
    sel = obs.get("select") or {}
    options = sel.get("option") or []
    for i in picks:
        if not 0 <= i < len(options):
            continue
        opt = options[i]
        if (
            opt.get("type") == int(OptionType.ATTACK)
            and opt.get("attackId") in budew_ids
        ):
            return True
    return False


def _with_own_hand(obs: dict, seat: int, hand: list | None) -> dict:
    """`obs` with `seat`'s hand contents patched in.

    The end-of-turn board is necessarily observed from the *other* seat's
    point of view -- it's captured the moment play passes -- and the engine
    hides the non-acting player's hand (`Player.hand is None`). Scoring
    straight off it silently zeroes every hand-dependent term, so the seat's
    own last view of its hand is spliced back in.

    That hand is up to one decision stale (it predates whatever the seat did
    last), which is why `_hand_size` reads `handCount` from the unpatched
    board instead: the count stays authoritative, only the contents are
    approximate. Structurally copied rather than deep-copied -- only the one
    player dict is rewritten.
    """
    if hand is None:
        return obs
    state = obs.get("current")
    if not state or not state.get("players"):
        return obs
    players = list(state["players"])
    if not 0 <= seat < len(players):
        return obs
    players[seat] = {**players[seat], "hand": hand}
    return {**obs, "current": {**state, "players": players}}


def play_setup_episode(
    policy: TorchPolicy,
    deck: list[int],
    first_turn_agent,
    budew_ids: set[int],
) -> SetupEpisode:
    """Play turns 0-4 and score the board each seat leaves behind."""
    obs, start = battle_start(list(deck), list(deck))
    if obs is None:
        raise RuntimeError(f"battle_start failed: errorPlayer={start.errorPlayer}")

    contexts = (
        GameContext(list(deck), DeckTracker(deck)),
        GameContext(list(deck), DeckTracker(deck)),
    )
    trajectories: tuple[list[EncodedDecision], list[EncodedDecision]] = ([], [])
    itchy = [False, False]
    excused = [False, False]
    retreats = [0, 0]
    crispin_early = [False, False]
    judge_over_lil = [False, False]
    # Board snapshot taken the moment each seat's setup turn ends. Captured at
    # the turn boundary rather than reconstructed at the end, because the
    # first player's turn-3 board is already stale by the time turn 4 resolves.
    end_obs: dict[int, dict] = {}
    # Each seat's own last view of its hand -- the only place hand *contents*
    # are visible, since the end-of-turn board is seen from the other seat.
    own_hand: dict[int, list] = {}
    prev_turn: int | None = None
    count = 0

    try:
        while obs["current"]["result"] < 0 and count < MAX_EPISODE_DECISIONS:
            cur = obs["current"]
            turn = cur["turn"]
            first_player = cur.get("firstPlayer", -1)

            if prev_turn is not None and turn != prev_turn and first_player >= 0:
                ended = seat_for_turn(prev_turn, first_player)
                if prev_turn == setup_turn_for_seat(ended, first_player):
                    end_obs.setdefault(ended, obs)
            prev_turn = turn

            if turn > LAST_SETUP_TURN:
                break

            p = cur["yourIndex"]
            seen_hand = (cur.get("players") or [])[p].get("hand")
            if seen_hand is not None:
                own_hand[p] = seen_hand
            tracker = contexts[p].tracker
            tracker.observe(obs)
            if tracker.is_search_reveal(obs):
                tracker.record_search_reveal(obs)

            if _is_own_first_turn(obs):
                picks, record = first_turn_agent(obs), None
            else:
                picks, record = policy.act(obs, collect=True, ctx=contexts[p])
                if _used_itchy_pollen(obs, picks, budew_ids):
                    itchy[p] = True
                if _excuses_meowth(obs, picks):
                    excused[p] = True
                retreats[p] += _count_retreats(obs, picks)
                if _crispin_played_early(obs, picks):
                    crispin_early[p] = True
                if _judge_over_lillies(obs, picks):
                    judge_over_lil[p] = True
            if record is not None:
                trajectories[p].append(record)
            obs = battle_select(picks)
            count += 1
    finally:
        battle_finish()

    # A seat whose turn never ended (game finished early, or the loop hit a
    # cap) is scored on the final board rather than dropped.
    for seat in (0, 1):
        end_obs.setdefault(seat, obs)

    scores = []
    for seat in (0, 1):
        board = _with_own_hand(end_obs[seat], seat, own_hand.get(seat))
        parsed = Observation.model_validate(board)
        scores.append(
            score_end_of_turn(
                parsed,
                itchy_pollen_used=itchy[seat],
                seat=seat,
                meowth_excused=excused[seat],
                retreats=retreats[seat],
                crispin_early=crispin_early[seat],
                judge_over_lillies=judge_over_lil[seat],
            )
        )

    parts: dict[str, float] = {}
    for s in scores:
        for name, value in s.parts.items():
            parts[name] = parts.get(name, 0.0) + value

    return SetupEpisode(
        trajectories=trajectories,
        scores=(scores[0], scores[1]),
        itchy=(itchy[0], itchy[1]),
        decisions=count,
        parts=parts,
    )


def train_setup(
    deck_path: str = "deck/03_pult_munki.csv",
    agent: str = "03_pult_munki_setup",
    iterations: int = 200,
    episodes_per_iter: int = 16,
    lr: float = 3e-4,
    gamma: float = 0.99,
    lam: float = 0.95,
    shaping_weights_path: str = "agents/03_pult_munki/reward_weights.json",
    shaping_scale: float = 0.25,
    save_every: int = 10,
    init_checkpoint: str | None = None,
    seed: int = 0,
    max_seconds: float | None = None,
    stop_file: str | None = None,
) -> PolicyValueNet:
    random.seed(seed)
    torch.manual_seed(seed)

    profile = AgentProfile(agent)
    profile.ensure_dirs()
    deck = Deck.from_csv(deck_path).card_ids

    # The default agent's shaping, turned down: it should nudge, not compete
    # with the end-of-turn rubric that defines this agent's objective.
    weights = {
        name: value * shaping_scale
        for name, value in load_weights(shaping_weights_path).items()
    }

    model = PolicyValueNet()
    if init_checkpoint:
        model.load_state_dict(
            torch.load(init_checkpoint, map_location="cpu", weights_only=True)
        )
        print(f"resumed from {init_checkpoint}", flush=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    first_turn_agent = make_training_first_turn_agent(deck)
    budew_ids = _budew_attack_ids()

    ckpt_path = profile.checkpoint_dir / "ppo_latest.pt"
    metrics_file = profile.metrics_dir / "setup_train.csv"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    csv_f = open(metrics_file, "w", newline="")
    csv_w = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
    csv_w.writeheader()

    log = MetricLog()
    log.add_tensorboard(str(profile.runs_dir / "setup"))

    print(
        f"setup training: agent={agent} deck={deck_path} "
        f"episodes/iter={episodes_per_iter} shaping_scale={shaping_scale}",
        flush=True,
    )

    run_start = time.time()
    try:
        for it in range(1, iterations + 1):
            if max_seconds is not None and time.time() - run_start > max_seconds:
                print(f"reached max_seconds={max_seconds:.0f}s; stopping", flush=True)
                break
            if stop_file is not None and Path(stop_file).exists():
                print(f"stop file {stop_file} found; stopping", flush=True)
                break

            t0 = time.time()
            model.eval()
            policy = TorchPolicy(model)
            data: list[EncodedDecision] = []
            totals: list[float] = []
            first_scores: list[float] = []
            second_scores: list[float] = []
            itchy_count = 0
            decisions = 0
            part_sums: dict[str, float] = {}

            for _ in range(episodes_per_iter):
                ep = play_setup_episode(policy, deck, first_turn_agent, budew_ids)
                decisions += ep.decisions
                for name, value in ep.parts.items():
                    part_sums[name] = part_sums.get(name, 0.0) + value
                for seat in (0, 1):
                    score = ep.scores[seat]
                    totals.append(score.total)
                    itchy_count += int(ep.itchy[seat])
                    # The rubric replaces win/loss as the terminal reward:
                    # a 2-turn episode never reaches a result.
                    compute_returns(
                        ep.trajectories[seat],
                        score.normalized,
                        gamma=gamma,
                        lam=lam,
                        weights=weights,
                    )
                    data.extend(ep.trajectories[seat])
                first_scores.append(ep.scores[0].total)
                second_scores.append(ep.scores[1].total)

            if not data:
                print(f"iter {it}: no decisions collected; skipping update", flush=True)
                continue

            model.train()
            stats = ppo_update(model, optimizer, data)
            model.eval()

            n_seats = max(1, len(totals))
            dt = time.time() - t0
            row = {
                "iter": it,
                "episodes": episodes_per_iter,
                "decisions": decisions,
                "samples": len(data),
                "score_mean": sum(totals) / n_seats,
                "score_first": sum(first_scores) / max(1, len(first_scores)),
                "score_second": sum(second_scores) / max(1, len(second_scores)),
                "itchy_rate": itchy_count / n_seats,
                **{f"part_{p}": part_sums.get(p, 0.0) / n_seats for p in TRACKED_PARTS},
                "pi_loss": stats.get("policy_loss", 0.0),
                "v_loss": stats.get("value_loss", 0.0),
                "entropy": stats.get("entropy", 0.0),
                "clip_frac": stats.get("clip_frac", 0.0),
                "time_s": dt,
            }
            csv_w.writerow(row)
            csv_f.flush()
            log.log_dict({k: v for k, v in row.items() if k != "iter"}, it)

            print(
                f"iter {it:>5} | score {row['score_mean']:+7.2f} "
                f"(1st {row['score_first']:+6.2f} / 2nd {row['score_second']:+6.2f}) "
                f"| itchy {row['itchy_rate']:.2f} "
                f"| drakloak {row['part_bench_drakloak']:+5.2f} "
                f"| ready {row['part_phantom_dive_ready']:+5.2f}"
                f"/{row['part_phantom_dive_ready_next']:+5.2f} "
                f"| ent {row['entropy']:.3f} | {dt:.1f}s",
                flush=True,
            )

            if it % save_every == 0:
                torch.save(model.state_dict(), ckpt_path)
                write_stamp_sidecar(ckpt_path)
    finally:
        torch.save(model.state_dict(), ckpt_path)
        write_stamp_sidecar(ckpt_path)
        csv_f.close()
        log.close()
        print(f"saved {ckpt_path}", flush=True)

    return model


app = typer.Typer(add_completion=False)


@app.command()
def main(
    deck: str = typer.Option("deck/03_pult_munki.csv", help="deck CSV to train on"),
    agent: str = typer.Option(
        "03_pult_munki_setup", help="profile name for checkpoints/metrics"
    ),
    iterations: int = typer.Option(200),
    episodes: int = typer.Option(16, help="2-turn episodes per iteration"),
    lr: float = typer.Option(3e-4),
    gamma: float = typer.Option(0.99),
    lam: float = typer.Option(0.95),
    shaping_weights: str = typer.Option(
        "agents/03_pult_munki/reward_weights.json",
        help="default agent's reward weights, applied at reduced strength",
    ),
    shaping_scale: float = typer.Option(
        0.25, help="multiplier on those shaping weights (the rubric dominates)"
    ),
    save_every: int = typer.Option(10),
    init: str | None = typer.Option(None, help="checkpoint to resume from"),
    resume: bool = typer.Option(
        False, help="resume from the agent's own latest checkpoint if present"
    ),
    seed: int = typer.Option(0),
    max_seconds: float | None = typer.Option(None),
    stop_file: str | None = typer.Option(None),
) -> None:
    init_ckpt = init
    if init_ckpt is None and resume:
        p = AgentProfile(agent).checkpoint_dir / "ppo_latest.pt"
        init_ckpt = str(p) if p.is_file() else None
    train_setup(
        deck_path=deck,
        agent=agent,
        iterations=iterations,
        episodes_per_iter=episodes,
        lr=lr,
        gamma=gamma,
        lam=lam,
        shaping_weights_path=shaping_weights,
        shaping_scale=shaping_scale,
        save_every=save_every,
        init_checkpoint=init_ckpt,
        seed=seed,
        max_seconds=max_seconds,
        stop_file=stop_file,
    )


if __name__ == "__main__":
    app()
