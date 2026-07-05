"""Tests for determinization and IS-MCTS over the search API."""

import random

import torch

from kaggle_environments.envs.cabt.cg.game import (
    battle_finish,
    battle_select,
    battle_start,
)

from pkm.data import Deck
from pkm.mcts.determinize import infer_opponent_decklist, sample_determinization
from pkm.mcts.search import MCTS, _forced_picks
from pkm.rl.model import PolicyValueNet
from pkm.rl.numpy_policy import NumpyPolicy
from pkm.search import search_begin, search_end, search_step


def _mid_game_obs(steps: int = 25):
    random.seed(4)
    deck = Deck.from_csv("deck.csv").card_ids
    obs, _ = battle_start(deck, deck)
    for _ in range(steps):
        if obs["current"]["result"] >= 0:
            break
        sel = obs["select"]
        obs = battle_select(random.sample(range(len(sel["option"])), sel["maxCount"]))
    return obs, deck


def test_determinization_counts():
    obs, deck = _mid_game_obs()
    try:
        state = obs["current"]
        me = state["players"][state["yourIndex"]]
        opp = state["players"][1 - state["yourIndex"]]
        det = sample_determinization(obs, deck, deck, random.Random(0))
        assert len(det["your_deck"]) == me["deckCount"]
        assert len(det["your_prize"]) == len(me["prize"])
        assert len(det["opponent_deck"]) == opp["deckCount"]
        assert len(det["opponent_prize"]) == len(opp["prize"])
        assert len(det["opponent_hand"]) == opp["handCount"]
        assert all(cid > 0 for cid in det["your_prize"])
        # the determinization must be accepted by the engine
        root = search_begin(obs, **det)
        assert root["searchId"] >= 0
        sel = root["observation"]["select"]
        nxt = search_step(root["searchId"], list(range(sel["minCount"])) or [0])
        assert nxt["observation"]["current"] is not None
        search_end()
    finally:
        battle_finish()


def test_infer_opponent_decklist():
    obs, _ = _mid_game_obs()
    try:
        est = infer_opponent_decklist(obs)
        assert len(est) == 60
        assert all(isinstance(c, int) and c > 0 for c in est)
    finally:
        battle_finish()


def test_mcts_choose_legal():
    torch.manual_seed(0)
    model = PolicyValueNet()
    policy = NumpyPolicy({k: v.detach().numpy() for k, v in model.state_dict().items()})
    mcts = MCTS(policy, n_determinizations=2, n_simulations=8, rng=random.Random(0))

    obs, deck = _mid_game_obs()
    try:
        searched = 0
        while searched < 3 and obs["current"]["result"] < 0:
            sel = obs["select"]
            forced = _forced_picks(sel)
            if forced is not None:
                obs = battle_select(forced)
                continue
            picks, agg = mcts.choose(obs, deck, deck)
            n = len(sel["option"])
            assert sel["minCount"] <= len(picks) <= sel["maxCount"]
            assert all(0 <= p < n for p in picks)
            assert len(set(picks)) == len(picks)
            assert sum(agg.values()) > 0
            obs = battle_select(picks)  # engine accepts the choice
            searched += 1
        assert searched == 3
    finally:
        battle_finish()
