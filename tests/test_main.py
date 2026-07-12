import main as submission_main


def test_resolve_deck_relative_to_submission_module(tmp_path, monkeypatch):
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "deck.csv").write_text("1\n" * 60)

    working_dir = tmp_path / "working"
    working_dir.mkdir()
    monkeypatch.chdir(working_dir)
    monkeypatch.setattr(submission_main, "__file__", str(agent_dir / "main.py"))

    deck = submission_main._resolve_deck()

    assert deck.card_ids == [1] * 60


def test_resolve_deck_without_file_attribute(tmp_path, monkeypatch):
    agent_dir = tmp_path / "kaggle_simulations" / "agent"
    agent_dir.mkdir(parents=True)
    (agent_dir / "deck.csv").write_text("2\n" * 60)

    working_dir = tmp_path / "working"
    working_dir.mkdir()
    monkeypatch.chdir(working_dir)
    monkeypatch.setattr(submission_main, "_KAGGLE_AGENT_DIR", agent_dir)
    monkeypatch.delattr(submission_main, "__file__")

    deck = submission_main._resolve_deck()

    assert deck.card_ids == [2] * 60


def test_main_returns_kaggle_agent_action(monkeypatch):
    expected_deck = list(range(60))
    calls = []

    monkeypatch.setattr(
        submission_main,
        "_resolve_deck",
        lambda path="deck.csv": type("DeckStub", (), {"card_ids": expected_deck})(),
    )

    def fake_make_neural_agent(deck):
        calls.append(deck)
        return lambda obs: deck if obs["select"] is None else [7]

    monkeypatch.setattr(submission_main, "make_neural_agent", fake_make_neural_agent)
    monkeypatch.setattr(submission_main, "_submission_agent", None, raising=False)

    assert submission_main.main({"select": None}) == expected_deck
    assert submission_main.main({"select": {"option": [1]}}) == [7]
    assert calls == [expected_deck]
