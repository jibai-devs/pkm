import main as submission_main


def test_resolve_deck_prefers_bundled_deck(tmp_path, monkeypatch):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "deck.csv").write_text("1\n" * 60)

    working_dir = tmp_path / "working"
    working_dir.mkdir()
    monkeypatch.chdir(working_dir)
    monkeypatch.setattr(submission_main, "__file__", str(agent_dir / "main.py"))

    deck = submission_main._resolve_deck()

    assert deck.card_ids == [1] * 60


def test_resolve_deck_uses_dragapult_fallback(tmp_path, monkeypatch):
    agent_dir = tmp_path / "kaggle_simulations" / "agent"
    agent_dir.mkdir(parents=True)
    deck_dir = agent_dir / "deck"
    deck_dir.mkdir()
    (deck_dir / "02_dragapult.csv").write_text("2\n" * 60)

    working_dir = tmp_path / "working"
    working_dir.mkdir()
    monkeypatch.chdir(working_dir)
    monkeypatch.setattr(submission_main, "__file__", str(agent_dir / "main.py"))
    monkeypatch.setattr(submission_main, "_KAGGLE_AGENT_DIR", agent_dir)

    deck = submission_main._resolve_deck()

    assert deck.card_ids == [2] * 60


def test_module_exposes_kaggle_agent():
    assert callable(submission_main.agent)
    assert submission_main.agent({"select": None}) == submission_main.DECK
    assert len(submission_main.DECK) == 60
