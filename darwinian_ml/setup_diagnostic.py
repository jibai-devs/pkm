"""Side-by-side diagnostic: what does the setup agent actually buy?

Measurement says pursuing the setup rubric *loses* games, and pursuing it
harder loses more (600 games: no setup 23.8%, RL setup 18.2%, SEARCH setup
14.8%). This tool shows the boards behind those numbers.

For each sample it renders three states:

    1. BEFORE   -- the board at the start of our own second turn
    2. AFTER (setup)   -- the same turn played by the setup agent, + rubric score
    3. AFTER (default) -- the same turn played by the default net,  + rubric score

**The comparison is genuinely like-for-like.** Both continuations are forked
from the identical position *and the identical determinization*, so the deck
order, the opponent's hand and every draw are the same in both branches. Any
difference is the agent's choices, not luck. (The engine has one search
context, so the branches run sequentially: fork, play, `search_end`, fork
again with the same sampled world.)

Output is a standalone HTML page with card art, openable straight from disk --
no dev server needed.

    python -m darwinian_ml.setup_diagnostic --samples 8
"""

from __future__ import annotations

import html
import random
import shutil
from pathlib import Path

import typer

from pkm.agents.dragapult_default_agent import make_dragapult_default_agent
from pkm.agents.dragapult_setup_agent import make_dragapult_setup_agent
from pkm.agents.dragapult_setup_search_agent import (
    make_dragapult_setup_search_agent,
)
from pkm.agents.first_turn_agent import make_first_turn_agent
from pkm.agents.random_agent import make_random_agent
from pkm.data import Deck
from pkm.data.card_data import get_card_by_id
from pkm.engine import (
    battle_finish,
    battle_select,
    battle_start,
    search_begin,
    search_end,
    search_step,
)
from pkm.mcts.determinize import infer_opponent_decklist, sample_determinization
from pkm.rl.rollout import _is_own_first_turn
from pkm.agents.turn1agent_dep.search import _apply_events
from pkm.rl.setup_turn_score import score_end_of_turn
from pkm.types.obs import Observation, forced_picks

from .config import DarwinConfig

CARD_SRC = Path("replay/07_vite_react_cards/public/cards")
OUT_DIR = Path("darwinian_ml/runs/setup_diagnostic")
MAX_TURN_STEPS = 120


def _card_name(cid: int) -> str:
    c = get_card_by_id(cid)
    return c.name if c else f"Card#{cid}"


def _score(board: dict, events: dict):
    """Score an end-of-turn board using that turn's event log.

    `itchy_pollen_used` and `meowth_excused` are not recoverable from a static
    observation (see `score_end_of_turn`), so a caller that omits them silently
    reads 0.00 on the biggest bonus in the rubric.
    """
    return score_end_of_turn(
        Observation.model_validate(board),
        itchy_pollen_used=bool(events.get("itchy_pollen")),
        seat=0,
        meowth_excused=bool(
            events.get("lillies_played") or events.get("judge_no_dreepy")
        ),
        retreats=int(events.get("retreats", 0)),
    )


def _play_turn(root_obs: dict, det: dict, agent, me: int, turn: int):
    """Fork the game at `root_obs` in world `det`; play our turn with `agent`.

    Returns (final board dict, list of decisions taken as readable text).
    """
    state = search_begin(root_obs, **det)
    moves: list[str] = []
    events: dict = {}
    final = state.raw_observation
    try:
        for _ in range(MAX_TURN_STEPS):
            obs = state.raw_observation
            cur = obs.get("current") or {}
            if cur.get("result", -1) >= 0 or obs.get("select") is None:
                break
            if cur.get("turn") != turn or cur.get("yourIndex") != me:
                break
            final = obs
            forced = forced_picks(obs["select"])
            picks = forced if forced is not None else agent(obs)
            opts = obs["select"]["option"]
            for i in picks:
                if 0 <= i < len(opts):
                    moves.append(_describe(obs, opts[i]))
            events = _apply_events(events, obs, list(picks))
            state = search_step(state.search_id, list(picks))
        last = state.raw_observation
        cur = last.get("current") or {}
        if cur.get("yourIndex") == me and cur.get("result", -1) < 0:
            final = last
    finally:
        search_end()
    return final, moves, events


def _play_live(obs: dict, agent, me: int, turn: int):
    """Play our turn in the LIVE game (no fork), returning (final obs, moves).

    Used for search-based agents, which cannot run inside a `search_begin`
    fork. Because this is the real game its draws are the real ones, not the
    determinized ones the forked branches see -- so its board is comparable in
    kind but not card-for-card. The HTML says so.
    """
    moves: list[str] = []
    events: dict = {}
    final = obs
    for _ in range(MAX_TURN_STEPS):
        cur = obs.get("current") or {}
        if cur.get("result", -1) >= 0 or obs.get("select") is None:
            break
        if cur.get("turn") != turn or cur.get("yourIndex") != me:
            break
        final = obs
        forced = forced_picks(obs["select"])
        picks = forced if forced is not None else agent(obs)
        opts = obs["select"]["option"]
        for i in picks:
            if 0 <= i < len(opts):
                moves.append(_describe(obs, opts[i]))
        events = _apply_events(events, obs, list(picks))
        obs = battle_select(picks)
    return final, moves, events


_VERB = {
    0: "NUMBER",
    1: "YES",
    2: "NO",
    3: "CARD",
    4: "TOOL",
    5: "ENERGY-CARD",
    6: "ENERGY",
    7: "PLAY",
    8: "ATTACH",
    9: "EVOLVE",
    10: "ABILITY",
    11: "DISCARD",
    12: "RETREAT",
    13: "ATTACK",
    14: "END TURN",
    15: "SKILL",
    16: "CONDITION",
}


def _describe(obs: dict, opt: dict) -> str:
    t = _VERB.get(opt.get("type"), f"OPT{opt.get('type')}")
    cid = opt.get("cardId")
    if not cid and opt.get("index") is not None:
        state = obs.get("current") or {}
        players = state.get("players") or []
        seat = opt.get("playerIndex", state.get("yourIndex", 0))
        key = {
            None: "hand",
            2: "hand",
            4: "active",
            5: "bench",
            1: "deck",
            3: "discard",
        }.get(opt.get("area"))
        if key and 0 <= seat < len(players):
            zone = players[seat].get(key) or []
            if 0 <= opt["index"] < len(zone) and isinstance(zone[opt["index"]], dict):
                cid = zone[opt["index"]].get("id")
    return f"{t} {_card_name(cid)}" if cid else t


# --- HTML -------------------------------------------------------------------

CSS = """
body{font:14px/1.5 system-ui,sans-serif;margin:0;padding:24px;background:#14161a;color:#e6e8ec}
h1{font-size:20px;margin:0 0 4px}
.sub{color:#9aa3af;margin-bottom:24px}
.sample{border:1px solid #2a2f3a;border-radius:10px;margin-bottom:28px;overflow:hidden}
.sample>h2{font-size:15px;margin:0;padding:10px 14px;background:#1b1f27;border-bottom:1px solid #2a2f3a}
.cols{display:grid;grid-template-columns:repeat(3,1fr);gap:0}
.cols4{display:grid;grid-template-columns:repeat(4,1fr);gap:0}
.note{font-size:11px;color:#7c8598;margin:-6px 0 8px}
.col{padding:14px;border-right:1px solid #2a2f3a}
.col:last-child{border-right:0}
.col h3{font-size:13px;margin:0 0 10px;color:#cbd5e1}
.score{font-size:22px;font-weight:600;margin:6px 0 10px}
.win{color:#4ade80}.lose{color:#f87171}.same{color:#cbd5e1}
.zone{margin:8px 0}
.zone .lab{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#7c8598;margin-bottom:4px}
.cards{display:flex;flex-wrap:wrap;gap:4px}
.card{position:relative;width:56px}
.card img{width:56px;border-radius:4px;display:block;background:#0d0f13}
.card .en{position:absolute;bottom:2px;left:2px;font-size:9px;background:#000a;border-radius:3px;padding:0 3px}
.none{color:#5b6472;font-style:italic;font-size:12px}
table.parts{width:100%;border-collapse:collapse;margin-top:8px;font-size:12px}
table.parts td{padding:1px 0}
table.parts td:last-child{text-align:right;font-variant-numeric:tabular-nums}
.pos{color:#86efac}.neg{color:#fca5a5}
.moves{margin-top:8px;font-size:12px;color:#9aa3af}
.moves li{margin:1px 0}
"""


def _card_html(cid: int, energies=None) -> str:
    if not cid:
        return ""
    n = html.escape(_card_name(cid))
    pips = f'<span class="en">{len(energies)}E</span>' if energies else ""
    return (
        f'<div class="card" title="{n}">'
        f'<img src="cards/{cid}.png" alt="{n}" loading="lazy">{pips}</div>'
    )


def _zone_html(label: str, cards: list) -> str:
    if not cards:
        return f'<div class="zone"><div class="lab">{label}</div><div class="none">empty</div></div>'
    inner = "".join(
        _card_html(c.get("id"), c.get("energies")) for c in cards if isinstance(c, dict)
    )
    return f'<div class="zone"><div class="lab">{label} ({len(cards)})</div><div class="cards">{inner}</div></div>'


def _board_html(obs: dict, seat: int) -> str:
    state = obs.get("current") or {}
    players = state.get("players") or []
    if not 0 <= seat < len(players):
        return '<div class="none">no board</div>'
    me = players[seat]
    opp = players[1 - seat]
    active = [c for c in (me.get("active") or []) if c]
    bench = [c for c in (me.get("bench") or []) if c]
    hand = [c for c in (me.get("hand") or []) if c]
    oactive = [c for c in (opp.get("active") or []) if c]
    return (
        _zone_html("your active", active)
        + _zone_html("your bench", bench)
        + _zone_html("your hand", hand)
        + _zone_html("opponent active", oactive)
    )


def _parts_html(score) -> str:
    rows = "".join(
        f"<tr><td>{html.escape(k)}</td>"
        f'<td class="{"pos" if v > 0 else "neg"}">{v:+.2f}</td></tr>'
        for k, v in sorted(score.parts.items(), key=lambda kv: -abs(kv[1]))
    )
    return f'<table class="parts">{rows}</table>'


app = typer.Typer(add_completion=False)


@app.command()
def main(
    samples: int = typer.Option(8, help="how many setup turns to capture"),
    deck_path: str = typer.Option(DarwinConfig.deck_path),
    search_seconds: float = typer.Option(1.7, help="search budget per decision"),
    search_sims: int = typer.Option(
        0,
        help="fixed simulations/decision instead of a wall-clock budget; "
        "removes machine-load sensitivity, but does NOT make runs reproducible "
        "-- the engine deals from std::random_device (see below)",
    ),
    seed: int = typer.Option(0),
) -> None:
    random.seed(seed)
    deck = Deck.from_csv(deck_path).card_ids
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if CARD_SRC.is_dir():
        shutil.copytree(CARD_SRC, OUT_DIR / "cards", dirs_exist_ok=True)

    # The *RL* setup agent, not the search one, and that is forced rather than
    # chosen: both continuations are played inside a `search_begin` fork, and
    # the search agent runs its own `search_begin`/`search_end` internally.
    # The engine has a single search context, so its `search_end` tears down
    # the fork we are standing in ("no element with the specified search_id").
    # Both agents here are pure forward passes, so they nest safely.
    #
    # This still shows what the rubric buys: the RL agent is the deployed one
    # and measured 18.2% against the default net's 23.8% over 600 games. To
    # diagnose the search variant the fork would have to run in its own
    # process, the way `turn_planner` does.
    setup_agent = make_dragapult_setup_agent(deck, log_sink=lambda _m: None)
    # --search-sims trades the wall-clock budget for a fixed simulation count,
    # which removes one source of run-to-run variation: how many simulations
    # fit in 1.7s depends on machine load, so the search agent -- which plays
    # the LIVE game -- made different choices each run and the trajectory
    # diverged from there.
    #
    # **It does not make this tool reproducible, and cannot.** The engine seeds
    # std::mt19937 from std::random_device inside ApiBattleStart, with no seed
    # injection through the public BattleStart ABI (see AGENTS.md, "the engine
    # is nondeterministic by design"). Measured: three battle_starts under an
    # identical Python seed dealt three different hands. `--seed` therefore
    # controls only determinization and agent tie-breaking, never the deal.
    #
    # So two runs of this tool are always different boards, and its numbers can
    # never be A/B'd across a weight change. Doing that needs the `before`
    # boards captured once and persisted, then replayed through search_begin
    # forks (determinization *is* Python-seeded, so forks are reproducible) --
    # with the search agent in its own engine process, since it cannot run
    # inside a fork it would tear down.
    search_agent = make_dragapult_setup_search_agent(
        deck,
        n_simulations=search_sims if search_sims > 0 else 1_000_000,
        time_budget_s=1e9 if search_sims > 0 else search_seconds,
        seed=seed if search_sims > 0 else None,
        log_sink=lambda _m: None,
    )
    default_agent = make_dragapult_default_agent(deck)
    ft = make_first_turn_agent(deck)
    rnd = make_random_agent(deck)

    blocks = []
    got = 0
    game = 0
    while got < samples and game < samples * 6:
        game += 1
        obs, _ = battle_start(list(deck), list(deck))
        n = 0
        first = -1
        try:
            while obs["current"]["result"] < 0 and n < 150 and got < samples:
                cur = obs["current"]
                p = cur["yourIndex"]
                if first < 0:
                    first = cur.get("firstPlayer", -1)
                setup_turn = 3 if p == first else 4
                if p == 0 and not _is_own_first_turn(obs) and cur["turn"] == setup_turn:
                    before = obs
                    turn_no = cur["turn"]
                    # SEARCH agent goes first and plays the LIVE turn. It
                    # cannot run inside a fork: it calls search_begin/search_end
                    # itself, and the engine's single search context means its
                    # search_end tears down whichever fork we were holding. So
                    # it plays for real here, before any fork exists; the two
                    # search-free agents are then forked from the same board.
                    s_obs, s_moves, s_ev = _play_live(before, search_agent, 0, turn_no)
                    rng = random.Random(1000 + got)
                    det = sample_determinization(
                        before, deck, infer_opponent_decklist(before), rng
                    )
                    a_obs, a_moves, a_ev = _play_turn(
                        before, det, setup_agent, 0, turn_no
                    )
                    b_obs, b_moves, b_ev = _play_turn(
                        before, det, default_agent, 0, turn_no
                    )
                    # Score from each agent's own event log, not the bare board.
                    # Scoring statically zeroed `itchy_pollen` for everyone --
                    # the rubric's single largest bonus (+7) could never fire in
                    # this tool, which is why it read 0.00 for all three agents
                    # in every previous run -- and applied Meowth ex's -1.00
                    # even on turns that had earned the waiver.
                    ss = _score(s_obs, s_ev)
                    sa = _score(a_obs, a_ev)
                    sb = _score(b_obs, b_ev)
                    blocks.append(
                        _sample_html(
                            got + 1,
                            before,
                            s_obs,
                            ss,
                            s_moves,
                            a_obs,
                            sa,
                            a_moves,
                            b_obs,
                            sb,
                            b_moves,
                        )
                    )
                    got += 1
                    break  # one sample per game, then move on
                picks = (
                    ft(obs)
                    if (p == 0 and _is_own_first_turn(obs))
                    else (default_agent(obs) if p == 0 else rnd(obs))
                )
                obs = battle_select(picks)
                n += 1
        finally:
            battle_finish()

    page = (
        f"<!doctype html><meta charset=utf-8><title>Setup agent diagnostic</title>"
        f"<style>{CSS}</style>"
        f"<h1>Setup-turn diagnostic — {got} samples</h1>"
        f'<div class="sub">Both continuations are forked from the same position '
        f"<em>and the same determinization</em>, so deck order and draws are identical "
        f"in each pair. Score is <code>setup_turn_score.py</code>: higher = better by "
        f"the rubric the setup agent optimises.</div>" + "".join(blocks)
    )
    out = OUT_DIR / "index.html"
    out.write_text(page, encoding="utf-8")
    print(f"wrote {got} samples -> {out.resolve()}")


def _sample_html(
    idx, before, s_obs, ss, s_moves, a_obs, sa, a_moves, b_obs, sb, b_moves
) -> str:
    turn = (before.get("current") or {}).get("turn")
    best = max(ss.total, sa.total, sb.total)

    def col(title, obs, score, moves, note=""):
        cls = "win" if score.total >= best - 1e-9 else "lose"
        return (
            f'<div class="col"><h3>{title}</h3>'
            f'<div class="score {cls}">{score.total:+.2f}</div>'
            f"{f'<div class=note>{note}</div>' if note else ''}"
            f"{_board_html(obs, 0)}{_parts_html(score)}"
            f'<ul class="moves">'
            + "".join(f"<li>{html.escape(m)}</li>" for m in moves)
            + "</ul></div>"
        )

    return f"""
<div class="sample">
  <h2>Sample {idx} &middot; engine turn {turn} &middot;
      search {ss.total:+.2f} | RL net {sa.total:+.2f} | default {sb.total:+.2f}</h2>
  <div class="cols4">
    <div class="col"><h3>1. BEFORE the turn</h3>{_board_html(before, 0)}</div>
    {col("2. SEARCH setup agent", s_obs, ss, s_moves, "live game (own draws)")}
    {col("3. RL setup net", a_obs, sa, a_moves, "forked, shared world")}
    {col("4. default net", b_obs, sb, b_moves, "forked, shared world")}
  </div>
</div>"""


if __name__ == "__main__":
    app()
