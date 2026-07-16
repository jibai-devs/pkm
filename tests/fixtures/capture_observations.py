"""Regenerate tests/fixtures/observations.json from the live engine.

Plays scripted random games and records one observation per distinct
(select.type, select.context) pair, plus one example of every option type and
log type seen. Run:  python tests/fixtures/capture_observations.py
"""

import json
import random
from pathlib import Path

from pkm.engine import (
    battle_finish,
    battle_select,
    battle_start,
)

from pkm.data import Deck

OUT = Path(__file__).parent / "observations.json"


def capture(seeds: tuple[int, ...] = (3, 11, 42)) -> dict:
    deck = Deck.from_csv("deck/02_dragapult.csv").card_ids
    observations: dict[str, dict] = {}
    options: dict[str, dict] = {}
    logs: dict[str, dict] = {}

    for seed in seeds:
        random.seed(seed)
        obs, _ = battle_start(deck, deck)
        try:
            for _ in range(600):
                for entry in obs["logs"]:
                    logs.setdefault(str(entry["type"]), entry)
                if obs["current"]["result"] >= 0:
                    break
                sel = obs["select"]
                observations.setdefault(f"{sel['type']}:{sel['context']}", obs)
                for opt in sel["option"]:
                    options.setdefault(str(opt["type"]), opt)
                picks = random.sample(range(len(sel["option"])), sel["maxCount"])
                obs = battle_select(picks)
        finally:
            battle_finish()

    return {
        "observations": observations,
        "options": options,
        "logs": logs,
    }


if __name__ == "__main__":
    data = capture()
    OUT.write_text(json.dumps(data, indent=1))
    print(
        f"wrote {OUT}: {len(data['observations'])} observations, "
        f"{len(data['options'])} option types, {len(data['logs'])} log types"
    )
