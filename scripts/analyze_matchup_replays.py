"""Decode a directory of pkm-play replay.json files (from
scripts/run_matchup_replays.py) into a human-readable per-game event log,
plus cross-game aggregates: which attacks/KOs actually decide the games.

A player's `observation.logs` field is a *since-last-decision* buffer: it is
only fresh at the steps where that player's status is ACTIVE (their turn to
respond). Concatenating logs from player 0's consecutive ACTIVE steps (in
order) reconstructs the full public event history without double-counting
the carried-over/repeated buffer INACTIVE steps show.

Usage:
    python scripts/analyze_matchup_replays.py runs/03_pult_munki_vs_pool_400_mega_abomasnow_ex
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from pkm.engine import all_cards, all_attacks

LOG_TYPES = {
    0: "Shuffle", 1: "HasBasicPokemon", 2: "TurnStart", 3: "TurnEnd",
    4: "Draw", 5: "DrawReverse", 6: "MoveCard", 7: "MoveCardReverse",
    8: "Switch", 9: "Change", 10: "Play", 11: "Attach", 12: "Evolve",
    13: "Devolve", 14: "MoveAttached", 15: "Attack", 16: "HpChange",
    17: "Poisoned", 18: "Burned", 19: "Asleep", 20: "Paralyzed",
    21: "Confused", 22: "Coin", 23: "Result",
}


def load_lookups():
    cards = {c["cardId"]: c["name"] for c in all_cards()}
    attacks = {a["attackId"]: (a["name"], a["damage"]) for a in all_attacks()}
    return cards, attacks


def card_name(cards: dict, card_id: int | None) -> str:
    if card_id is None:
        return "?"
    return cards.get(card_id, f"#{card_id}")


def _reconstruct_one_side(steps: list, player_idx: int) -> list[dict]:
    """Concatenate `logs` from `player_idx`'s consecutive ACTIVE steps."""
    events = []
    for step in steps:
        pstate = step[player_idx]
        if pstate["status"] in ("ACTIVE", "DONE"):
            events.extend(pstate["observation"].get("logs") or [])
    return events


def reconstruct_log(steps: list) -> list[dict]:
    """Merge both players' independent reconstructions. Each side's own
    stream is already gap-free for events *it* eventually gets an ACTIVE (or
    DONE) step to receive -- but if the game ends immediately after the
    other side's final action, that side may never get another turn to
    report it (see e.g. a game-ending attack shown only in the winner's own
    final DONE buffer). Take the union (by exact field match) of both
    sides' reconstructions so no event is lost."""
    a = _reconstruct_one_side(steps, 0)
    b = _reconstruct_one_side(steps, 1)
    remaining = Counter(tuple(sorted(e.items())) for e in a)
    merged = list(a)
    for e in b:
        k = tuple(sorted(e.items()))
        if remaining[k] > 0:
            remaining[k] -= 1
        else:
            merged.append(e)
    return merged


def decode_game(replay_path: Path, cards: dict, attacks: dict) -> dict:
    data = json.loads(replay_path.read_text())
    steps = data["steps"]
    events = reconstruct_log(steps)

    lines = []
    attacks_used = []  # (playerIndex, attacker_name, attack_name, damage)
    kos = []  # (playerIndex_of_victim, victim_name)
    energy_attach = Counter()  # playerIndex -> count
    last_hp_change: dict[int, int] = {}  # cardId -> most recent damage value seen

    for e in events:
        t = e.get("type")
        name = LOG_TYPES.get(t, f"type{t}")
        pidx = e.get("playerIndex")
        if t == 15:  # Attack
            attacker = card_name(cards, e.get("cardId"))
            atk_name, atk_dmg = attacks.get(e.get("attackId"), (f"attack#{e.get('attackId')}", None))
            attacks_used.append((pidx, attacker, atk_name, atk_dmg))
            lines.append(f"  P{pidx} {attacker} uses {atk_name} (base {atk_dmg})")
        elif t == 16:  # HpChange
            target = card_name(cards, e.get("cardId"))
            val = e.get("value")
            sign = "damage" if (val or 0) > 0 else "heal"
            last_hp_change[e.get("cardId")] = val
            lines.append(f"  P{pidx} {target} HP {sign} {val}")
        elif t == 6 and e.get("fromArea") == 4 and e.get("toArea") == 3:  # Active -> Trash = KO
            victim = card_name(cards, e.get("cardId"))
            dmg = last_hp_change.get(e.get("cardId"))
            kos.append((pidx, victim, dmg))
            lines.append(f"  P{pidx} {victim} KO'd (last hit: {dmg})")
        elif t == 9:  # Change (evolution replace mid-slot)
            before = card_name(cards, e.get("cardIdBefore"))
            after = card_name(cards, e.get("cardIdAfter"))
            lines.append(f"  P{pidx} {before} -> {after} (Change)")
        elif t == 11:  # Attach
            target = card_name(cards, e.get("cardIdTarget"))
            attached = card_name(cards, e.get("cardId"))
            energy_attach[pidx] += 1
            lines.append(f"  P{pidx} attaches {attached} to {target}")
        elif t == 10:  # Play (trainer/supporter/stadium)
            played = card_name(cards, e.get("cardId"))
            lines.append(f"  P{pidx} plays {played}")
        elif t in (17, 18, 19, 20, 21):  # status conditions
            target = card_name(cards, e.get("cardId"))
            recover = e.get("isRecover")
            lines.append(f"  P{pidx} {target} {'recovers from' if recover else 'gets'} {name}")
        elif t == 2:  # TurnStart
            lines.append(f"-- turn start P{pidx} --")

    return {
        "replay": str(replay_path),
        "lines": lines,
        "attacks_used": attacks_used,
        "kos": kos,
        "energy_attach": dict(energy_attach),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--verbose-games", type=int, default=0, help="print full per-game log for the first N games")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    cards, attacks = load_lookups()

    winning_ko_moves = Counter()  # (winner_side, attack_name) -> count, only for the LAST attack before game end
    ko_counts_by_side = Counter()  # side ('A'/'B') -> total KOs they scored
    attack_usage_by_side = Counter()  # (side, attack_name) -> times used
    turn_counts = []

    for g in summary:
        replay = Path(g["replay"])
        decoded = decode_game(replay, cards, attacks)
        a_first = g["a_first"]
        # playerIndex 0 == A if a_first else B
        side_of = {0: "A" if a_first else "B", 1: "B" if a_first else "A"}

        for pidx, attacker, atk_name, dmg in decoded["attacks_used"]:
            attack_usage_by_side[(side_of[pidx], atk_name)] += 1
        for pidx, victim, dmg in decoded["kos"]:
            # victim belongs to pidx's side; the KO credit goes to the OTHER side
            scorer = "B" if side_of[pidx] == "A" else "A"
            ko_counts_by_side[scorer] += 1
            if dmg is not None and dmg >= 200:
                winning_ko_moves[(scorer, f"finishing blow >= {dmg}dmg on {victim}")] += 1
        if decoded["attacks_used"]:
            last_attacker_pidx = decoded["attacks_used"][-1][0]
            last_atk_name = decoded["attacks_used"][-1][2]
            winning_ko_moves[(side_of[last_attacker_pidx], last_atk_name)] += 1
        turn_counts.append(len(decoded["lines"]))

        print(f"\n=== {replay.name}  a_first={a_first}  result={g['result']} ===")
        print(f"  attacks: {[(side_of[p], n, a) for p, n, a, d in decoded['attacks_used']]}")
        print(f"  KOs: {[(side_of[p], v, dmg) for p, v, dmg in decoded['kos']]}")
        print(f"  energy attaches by playerIndex: {decoded['energy_attach']}")

    if args.verbose_games:
        for g in summary[: args.verbose_games]:
            decoded = decode_game(Path(g["replay"]), cards, attacks)
            print(f"\n\n########## FULL LOG: {g['replay']} ##########")
            print("\n".join(decoded["lines"]))

    print("\n\n=== AGGREGATE ===")
    print("Attacks used, by side (top 15):")
    for (side, name), n in attack_usage_by_side.most_common(15):
        print(f"  {side}: {name} x{n}")
    print("\nTotal KOs scored, by side:", dict(ko_counts_by_side))
    print("\nLast attack before game-end, by (side, move) (proxy for finishing move):")
    for (side, name), n in winning_ko_moves.most_common(15):
        print(f"  {side} finished with: {name} x{n}")


if __name__ == "__main__":
    main()
