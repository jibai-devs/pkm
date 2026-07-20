"""One-off diagnostic: play N games between two agent profiles, saving each
game's full JSON replay for later log analysis (not just a win-rate number).

Usage:
    python scripts/run_matchup_replays.py 03_pult_munki pool_400_mega_abomasnow_ex --games 20
"""

import argparse
import json
from pathlib import Path

from pkm.rl.play import play_match, _resolve_agent_deck_weights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("agent_a")
    ap.add_argument("agent_b")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    deck_a, weights_a = _resolve_agent_deck_weights(args.agent_a)
    deck_b, weights_b = _resolve_agent_deck_weights(args.agent_b)
    if weights_a is None or weights_b is None:
        raise SystemExit(
            f"missing exported policy.npz for {'a' if weights_a is None else 'b'}"
        )

    out_dir = Path(args.out_dir or f"runs/{args.agent_a}_vs_{args.agent_b}")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for g in range(args.games):
        a_is_p0 = g % 2 == 0
        p0_name, p1_name = (
            (args.agent_a, args.agent_b) if a_is_p0 else (args.agent_b, args.agent_a)
        )
        deck0, weights0 = (deck_a, weights_a) if a_is_p0 else (deck_b, weights_b)
        deck1, weights1 = (deck_b, weights_b) if a_is_p0 else (deck_a, weights_a)

        replay_path = out_dir / f"game_{g:02d}.json"
        env = play_match(
            "neural",
            "neural",
            deck0_path=deck0,
            deck1_path=deck1,
            weights0=weights0,
            weights1=weights1,
            html_path=None,
            replay_path=str(replay_path),
        )
        final = env.steps[-1]
        a_idx = 0 if a_is_p0 else 1
        b_idx = 1 - a_idx
        r_a, r_b = final[a_idx].reward or 0, final[b_idx].reward or 0
        result = "A_WIN" if r_a > r_b else "B_WIN" if r_b > r_a else "DRAW"
        summary.append(
            {
                "game": g,
                "a_first": a_is_p0,
                "result": result,
                "turns": len(env.steps),
                "replay": str(replay_path),
            }
        )
        print(f"game {g:02d}: a_first={a_is_p0} -> {result} ({len(env.steps)} steps)", flush=True)

    a_wins = sum(1 for s in summary if s["result"] == "A_WIN")
    b_wins = sum(1 for s in summary if s["result"] == "B_WIN")
    draws = sum(1 for s in summary if s["result"] == "DRAW")
    print(f"\n{args.agent_a}: {a_wins}/{args.games}, {args.agent_b}: {b_wins}/{args.games}, draws: {draws}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
