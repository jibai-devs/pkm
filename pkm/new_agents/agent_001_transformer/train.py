"""Self-play + MCTS training loop for agent_001_transformer.

Port of the reference notebook's training loop, factored to import the shared
architecture/featurizers/MCTS from :mod:`.net` and to write checkpoints that
carry their architecture dims (so :mod:`.submit_main` rebuilds the exact net).

Run (from the repo root)::

    python -m pkm.new_agents.agent_001_transformer.train \
        --iters 5 --eval-games 50 --selfplay-games 100 --sims 10

Checkpoints land in ``--out`` (default ``<repo>/out_transformer``) as
``model<iter>.pth`` plus ``latest.pth``; each is a dict
``{"state_dict", "dims"}`` consumed by ``submit_main`` / ``pack``.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import torch
import torch.optim

from pkm.new_agents.agent_000_dragapult.cabt import (
    battle_finish,
    battle_select,
    battle_start,
)
from pkm.new_agents.agent_001_transformer import deck as deck_registry
from pkm.new_agents.agent_001_transformer import net


class LearnInput:
    """Batch builder that concatenates SparseVectors with offset fix-up."""

    def __init__(self):
        self.index: list[int] = []
        self.value: list[float] = []
        self.offset: list[int] = []

    def add(self, sv: net.SparseVector):
        count = len(self.index)
        self.index.extend(sv.index)
        self.value.extend(sv.value)
        for o in sv.offset:
            self.offset.append(o + count)


def _progress(count: int, text: str):
    current = 0
    while True:
        percent = 100 * current // max(count, 1)
        sys.stderr.write(f"\r{text} {percent}%   ")
        sys.stderr.flush()
        if current >= count:
            sys.stderr.write("\n")
            sys.stderr.flush()
            break
        yield current
        current += 1


def _check_start(start_data):
    if start_data.errorPlayer >= 0:
        errs = {
            1: "The deck contains an invalid card ID.",
            2: "Up to four cards with the same name are allowed (except basic Energy).",
            3: "There are no Basic Pokémon in the deck.",
            4: "Only one Ace Spec card is allowed in the deck.",
        }
        raise ValueError(errs.get(start_data.errorType, "Deck error."))


def evaluate(model, deck, sims, n_games) -> int:
    results = [0, 0, 0]
    for i in _progress(n_games, "Evaluating... "):
        obs, start_data = battle_start(deck, deck)
        _check_start(start_data)
        your_index = i % 2
        while obs["current"]["result"] < 0:
            if obs["current"]["yourIndex"] == your_index:
                selected, _ = net.mcts_agent(obs, deck, model, sims)
            else:
                selected = net.random_agent(obs)
            obs = battle_select(selected)
        battle_finish()
        if obs["current"]["result"] == 2:
            results[2] += 1
        elif obs["current"]["result"] == your_index:
            results[0] += 1
        else:
            results[1] += 1
    total = results[0] + results[1]
    return 100 * results[0] // total if total else 0


def collect_selfplay(model, deck, sims, n_games) -> list[net.LearnSample]:
    sample_list: list[net.LearnSample] = []
    for _ in _progress(n_games, "Training Data Collecting... "):
        obs, _ = battle_start(deck, deck)
        samples: list[list[net.LearnSample]] = [[], []]
        while obs["current"]["result"] < 0:
            selected, sample = net.mcts_agent(obs, deck, model, sims)
            samples[obs["current"]["yourIndex"]].append(sample)
            obs = battle_select(selected)
        battle_finish()
        for i in range(2):
            lam = 0.9
            value = 1.0 if i == obs["current"]["result"] else -1.0
            for sample in reversed(samples[i]):
                label = (value + sample.value) * 0.5
                value = value * lam + sample.value * (1.0 - lam)
                sample.value = label
                sample_list.append(sample)
    return sample_list


def train_epoch(model, optimizer, sample_list, device, batch_size=128):
    loss_fn_enc = torch.nn.HuberLoss(delta=0.2)
    loss_fn_dec = torch.nn.HuberLoss(reduction="none", delta=0.1)
    random.shuffle(sample_list)
    batch_count = len(sample_list) // batch_size
    for i in range(batch_count):
        input_enc = LearnInput()
        input_dec = LearnInput()
        mask: list[float] = []
        label_enc: list[float] = []
        label_dec: list[float] = []
        start = batch_size * i
        for j in range(start, start + batch_size):
            sample = sample_list[j]
            input_enc.add(sample.sv_enc)
            input_dec.add(sample.sv_dec)
            label_enc.append(sample.value)
            label_dec.extend(sample.policy)
            for _ in range(len(sample.policy)):
                mask.append(1.0)
            for _ in range(64 - len(sample.policy)):
                mask.append(0.0)
                label_dec.append(0.0)
                input_dec.offset.append(len(input_dec.index))

        mask_t = torch.tensor(mask, dtype=torch.float32, device=device).view(batch_size, -1)
        label_enc_t = torch.tensor(label_enc, dtype=torch.float32, device=device).view(batch_size, -1)
        label_dec_t = torch.tensor(label_dec, dtype=torch.float32, device=device).view(batch_size, -1)

        optimizer.zero_grad()
        out_enc, out_dec = model(
            torch.tensor(input_enc.index, dtype=torch.int32, device=device),
            torch.tensor(input_enc.value, dtype=torch.float32, device=device),
            torch.tensor(input_enc.offset, dtype=torch.int32, device=device),
            torch.tensor(input_dec.index, dtype=torch.int32, device=device),
            torch.tensor(input_dec.value, dtype=torch.float32, device=device),
            torch.tensor(input_dec.offset, dtype=torch.int32, device=device),
        )
        loss_enc = loss_fn_enc(out_enc, label_enc_t)
        loss_dec = (loss_fn_dec(out_dec, label_dec_t) * mask_t).sum() / float(batch_size)
        loss = loss_enc + loss_dec
        loss.backward()
        optimizer.step()


def save_ckpt(model, dims, path: Path, deck: list[int] | None = None, deck_name: str | None = None):
    """Write a ``{state_dict, dims, deck, deck_name}`` checkpoint.

    The played deck is baked in so ``submit_main`` / ``pack`` are self-contained:
    inference submits exactly the deck the checkpoint was trained on. Older
    ``{state_dict, dims}`` checkpoints (no deck) fall back to ``net.sample_deck``.
    """
    blob = {"state_dict": model.state_dict(), "dims": list(dims)}
    if deck is not None:
        blob["deck"] = list(deck)
    if deck_name is not None:
        blob["deck_name"] = deck_name
    torch.save(blob, path)


def train_loop(
    *,
    deck_name: str = deck_registry.DEFAULT_DECK,
    iters: int = 5,
    eval_games: int = 50,
    selfplay_games: int = 100,
    sims: int = 10,
    lr: float = 3e-4,
    batch_size: int = 128,
    device: str | None = None,
    init: Path | None = None,
    out: Path | None = None,
    on_iter=None,
) -> Path:
    """Self-play + MCTS training loop. Returns the path to ``latest.pth``.

    ``deck_name`` selects the played 60-card list from :mod:`.deck` and is baked
    into every checkpoint. ``on_iter(counter, win)`` is an optional callback
    invoked after each iteration's eval (for CLI progress reporting).
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    deck = deck_registry.deck_60(deck_name)
    if out is None:
        out = Path(__file__).resolve().parents[3] / "out_transformer"

    if init is not None:
        blob = torch.load(init, map_location=dev, weights_only=False)
        dims = tuple(blob.get("dims", net.MODEL_DIMS))
        model = net.MyModel(*dims).to(dev)
        model.load_state_dict(blob["state_dict"])
        print(f"resumed from {init}", flush=True)
    else:
        dims = net.MODEL_DIMS
        model = net.build_model(dims, dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device={dev} dims={dims} deck={deck_name} out={out}", flush=True)

    for counter in range(iters):
        save_ckpt(model, dims, out / f"model{counter}.pth", deck, deck_name)
        save_ckpt(model, dims, out / "latest.pth", deck, deck_name)

        model.eval()
        with torch.inference_mode():
            win = evaluate(model, deck, sims, eval_games)
            print(f"[iter {counter}] evaluation win rate {win}%", flush=True)
            if on_iter is not None:
                on_iter(counter, win)
            sample_list = collect_selfplay(model, deck, sims, selfplay_games)

        print(f"[iter {counter}] training on {len(sample_list)} samples...", flush=True)
        model.train()
        train_epoch(model, optimizer, sample_list, dev, batch_size)
        print(f"[iter {counter}] training finished.", flush=True)

    latest = out / "latest.pth"
    save_ckpt(model, dims, latest, deck, deck_name)
    print(f"done. latest -> {latest}", flush=True)
    return latest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=int(os.environ.get("N_ITERS", 5)))
    ap.add_argument("--eval-games", type=int, default=int(os.environ.get("N_EVAL", 50)))
    ap.add_argument("--selfplay-games", type=int, default=int(os.environ.get("N_SELFPLAY", 100)))
    ap.add_argument("--sims", type=int, default=int(os.environ.get("SEARCH_COUNT", 10)))
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    ap.add_argument(
        "--deck",
        default=deck_registry.DEFAULT_DECK,
        choices=deck_registry.deck_names(),
        help=f"played deck (default: {deck_registry.DEFAULT_DECK})",
    )
    ap.add_argument(
        "--init",
        type=Path,
        default=None,
        help="resume from a {state_dict,dims} checkpoint instead of a fresh net",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "out_transformer",
        help="checkpoint output dir",
    )
    args = ap.parse_args()

    train_loop(
        deck_name=args.deck,
        iters=args.iters,
        eval_games=args.eval_games,
        selfplay_games=args.selfplay_games,
        sims=args.sims,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        init=args.init,
        out=args.out,
    )


if __name__ == "__main__":
    main()
