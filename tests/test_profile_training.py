from pathlib import Path

import pytest

from pkm.agents.profile import AgentProfile, TrainingResult
from pkm.agents.spec import AgentSpec, REPO_ROOT
from pkm.agents import spec as spec_module
from pkm.agents import profile as profile_module
from pkm.cli import train as cli_train
from pkm.cli import exit_train as cli_exit_train
from pkm.rl import exit_train, train


def _custom_profile(tmp_path):
    checkpoint = tmp_path / "agents/custom/checkpoints/custom_model.pt"
    exit_checkpoint = tmp_path / "agents/custom/checkpoints/custom_exit.pt"
    return AgentProfile(
        "custom",
        _spec=AgentSpec(
            name="custom",
            deck_path=REPO_ROOT / "deck/02_dragapult.csv",
            policy="random",
            trainer="ppo",
            strategy=None,
            checkpoint_path=checkpoint,
            exit_checkpoint_path=exit_checkpoint,
        ),
    )


def test_profile_train_delegates_to_registered_trainer(monkeypatch):
    profile = AgentProfile.load("02_dragapult")
    calls = {}

    def fake_train(**kwargs):
        calls.update(kwargs)
        return TrainingResult(
            checkpoint=profile.checkpoint_dir / "ppo_latest.pt", iterations=1
        )

    monkeypatch.setitem(profile_module.TRAINERS, "ppo", fake_train)
    monkeypatch.setattr(profile, "ensure_dirs", lambda: None)

    result = profile.train(iterations=1, games=1)

    assert result.checkpoint.name == "ppo_latest.pt"
    assert calls["deck_path"] == profile.deck_path
    assert calls["checkpoint_path"] == profile.checkpoint_path
    assert calls["checkpoint_dir"] == profile.checkpoint_dir
    assert calls["metrics_dir"] == profile.metrics_dir
    assert calls["runs_dir"] == profile.runs_dir


def test_profile_train_passes_existing_resume_checkpoint(monkeypatch, tmp_path):
    profile = AgentProfile.load("02_dragapult")
    resume = tmp_path / "resume.pt"
    resume.write_bytes(b"checkpoint")
    monkeypatch.setattr(profile, "ppo_init", lambda: str(resume))
    calls = {}

    def fake_train(**kwargs):
        calls.update(kwargs)
        return TrainingResult(checkpoint=resume)

    monkeypatch.setitem(profile_module.TRAINERS, "ppo", fake_train)
    monkeypatch.setattr(profile, "ensure_dirs", lambda: None)

    profile.train(iterations=2, games=3)

    assert calls["resume_path"] == resume


def test_profile_train_exit_delegates_to_expert_iteration(monkeypatch):
    profile = AgentProfile.load("02_dragapult")
    calls = {}

    def fake_exit_train(**kwargs):
        calls.update(kwargs)
        return TrainingResult(checkpoint=profile.checkpoint_dir / "exit_latest.pt")

    monkeypatch.setattr(profile_module, "EXIT_TRAINER", fake_exit_train)
    monkeypatch.setattr(profile, "ensure_dirs", lambda: None)

    result = profile.train_exit(iterations=1, games=1)

    assert result.checkpoint.name == "exit_latest.pt"
    assert calls["deck_path"] == profile.deck_path
    assert calls["checkpoint_path"] == profile.exit_checkpoint_path
    assert calls["checkpoint_dir"] == profile.checkpoint_dir
    assert calls["metrics_dir"] == profile.metrics_dir
    assert calls["runs_dir"] == profile.runs_dir


def test_profile_training_uses_separate_default_checkpoints(monkeypatch):
    profile = AgentProfile.load("02_dragapult")
    calls = {}

    def fake_ppo(**kwargs):
        calls["ppo"] = kwargs
        return TrainingResult(checkpoint=kwargs["checkpoint_path"])

    def fake_exit(**kwargs):
        calls["exit"] = kwargs
        return TrainingResult(checkpoint=kwargs["checkpoint_path"])

    monkeypatch.setitem(profile_module.TRAINERS, "ppo", fake_ppo)
    monkeypatch.setattr(profile_module, "EXIT_TRAINER", fake_exit)
    monkeypatch.setattr(profile, "ensure_dirs", lambda: None)

    ppo_result = profile.train()
    exit_result = profile.train_exit()

    assert ppo_result.checkpoint == profile.checkpoint_dir / "ppo_latest.pt"
    assert exit_result.checkpoint == profile.checkpoint_dir / "exit_latest.pt"
    assert calls["ppo"]["checkpoint_path"] == ppo_result.checkpoint
    assert calls["exit"]["checkpoint_path"] == exit_result.checkpoint
    assert ppo_result.checkpoint.parent == exit_result.checkpoint.parent


def test_profile_train_exit_forwards_resume_with_custom_checkpoint(
    monkeypatch, tmp_path
):
    profile = _custom_profile(tmp_path)
    resume = tmp_path / "resume.pt"
    resume.write_bytes(b"checkpoint")
    calls = {}

    def fake_exit_train(**kwargs):
        calls.update(kwargs)
        return TrainingResult(checkpoint=kwargs["checkpoint_path"])

    monkeypatch.setattr(profile_module, "EXIT_TRAINER", fake_exit_train)
    monkeypatch.setattr(profile, "ensure_dirs", lambda: None)

    result = profile.train_exit(resume_path=resume)

    assert calls["resume_path"] == resume
    assert calls["checkpoint_path"] == profile.exit_checkpoint_path
    assert result.checkpoint == profile.exit_checkpoint_path


def test_train_exit_initializes_from_ppo_unless_resuming(monkeypatch):
    profile = AgentProfile.load("02_dragapult")
    ppo_checkpoint = profile.checkpoint_path
    exit_checkpoint = profile.exit_checkpoint_path
    calls = []

    def fake_exit_train(**kwargs):
        calls.append(kwargs)
        return TrainingResult(checkpoint=kwargs["checkpoint_path"])

    monkeypatch.setattr(profile_module, "EXIT_TRAINER", fake_exit_train)
    monkeypatch.setattr(profile, "ensure_dirs", lambda: None)
    monkeypatch.setattr(profile, "exit_init", lambda: str(exit_checkpoint))

    profile.train_exit()
    profile.train_exit(resume=True)
    profile.train_exit(resume=True, resume_path=Path("explicit.pt"))

    assert calls[0]["resume_path"] == ppo_checkpoint
    assert calls[1]["resume_path"] == exit_checkpoint
    assert calls[2]["resume_path"] == Path("explicit.pt")


def test_register_trainer_is_used_for_profile_training(monkeypatch, tmp_path):
    profile = AgentProfile(
        "custom",
        _spec=AgentSpec(
            name="custom",
            deck_path=REPO_ROOT / "deck/02_dragapult.csv",
            policy="random",
            trainer="custom",
            strategy=None,
            checkpoint_path=tmp_path / "custom.pt",
        ),
    )
    calls = {}

    def custom_trainer(**kwargs):
        calls.update(kwargs)
        return TrainingResult(checkpoint=kwargs["checkpoint_path"])

    monkeypatch.setattr(profile, "ensure_dirs", lambda: None)
    profile_module.register_trainer("custom", custom_trainer)

    result = profile.train()

    assert calls["deck_path"] == profile.deck_path
    assert result.checkpoint == profile.checkpoint_path


def test_unknown_profile_trainer_fails_clearly(monkeypatch, tmp_path):
    profile = AgentProfile(
        "unknown",
        _spec=AgentSpec(
            name="unknown",
            deck_path=REPO_ROOT / "deck/02_dragapult.csv",
            policy="random",
            trainer="missing",
            strategy=None,
            checkpoint_path=tmp_path / "unknown.pt",
        ),
    )
    monkeypatch.setattr(profile, "ensure_dirs", lambda: None)

    with pytest.raises(ValueError, match="unknown trainer 'missing'"):
        profile.train()


def test_profile_cli_forwards_explicit_paths(monkeypatch, tmp_path):
    calls = {}
    profile = AgentProfile.load("02_dragapult")

    def fake_train(self, **kwargs):
        calls.update(kwargs)
        return TrainingResult(checkpoint=self.checkpoint_path)

    monkeypatch.setattr(AgentProfile, "train", fake_train)

    cli_train(
        agent=profile.name,
        deck="ignored.csv",
        iterations=1,
        games=1,
        lr=0.1,
        gamma=0.9,
        shaping=0.1,
        pool_size=1,
        eval_every=1,
        eval_games=1,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        metrics=str(tmp_path / "metrics.csv"),
        log_dir=str(tmp_path / "runs"),
        init=None,
        seed=3,
    )

    assert calls["checkpoint_dir"] == Path(tmp_path / "checkpoints")
    assert calls["metrics_path"] == Path(tmp_path / "metrics.csv")
    assert calls["log_dir"] == Path(tmp_path / "runs")


def test_profile_exit_cli_forwards_resume_mode(monkeypatch, tmp_path):
    calls = []
    profile = AgentProfile.load("02_dragapult")

    def fake_train_exit(self, **kwargs):
        calls.append(kwargs)
        return TrainingResult(checkpoint=self.exit_checkpoint_path)

    monkeypatch.setattr(AgentProfile, "train_exit", fake_train_exit)

    cli_exit_train(
        agent=profile.name,
        deck="ignored.csv",
        iterations=1,
        games=1,
        sims=1,
        dets=1,
        lr=0.1,
        init=str(tmp_path / "explicit.pt"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        metrics=str(tmp_path / "metrics.csv"),
        log_dir=str(tmp_path / "runs"),
        seed=3,
        resume=False,
    )
    cli_exit_train(
        agent=profile.name,
        deck="ignored.csv",
        iterations=1,
        games=1,
        sims=1,
        dets=1,
        lr=0.1,
        init=None,
        checkpoint_dir="checkpoints",
        metrics="metrics/exit_train.csv",
        log_dir="runs/exit",
        seed=3,
        resume=True,
    )

    assert calls[0]["resume"] is False
    assert calls[1]["resume"] is True
    assert calls[0]["resume_path"] == tmp_path / "explicit.pt"
    assert calls[0]["checkpoint_dir"] == tmp_path / "checkpoints"
    assert calls[0]["metrics_path"] == tmp_path / "metrics.csv"
    assert calls[0]["log_dir"] == tmp_path / "runs"


def test_ppo_facade_writes_custom_checkpoint_path(monkeypatch, tmp_path):
    checkpoint = tmp_path / "custom_ppo.pt"
    metrics = tmp_path / "custom_ppo.csv"
    calls = {}

    def fake_train(**kwargs):
        calls.update(kwargs)
        output = Path(kwargs["checkpoint_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"checkpoint")

    monkeypatch.setattr(train, "train", fake_train)

    result = train.train_profile(
        deck_path=REPO_ROOT / "deck/02_dragapult.csv",
        checkpoint_path=checkpoint,
        checkpoint_dir=tmp_path / "checkpoints",
        metrics_dir=tmp_path / "metrics",
        runs_dir=tmp_path / "runs",
        resume_path=None,
        metrics_path=metrics,
    )

    assert calls["checkpoint_path"] == str(checkpoint)
    assert calls["metrics_path"] == str(metrics)
    assert result.checkpoint == checkpoint
    assert result.metrics == metrics
    assert result.checkpoint.is_file()


def test_exit_facade_writes_custom_checkpoint_path_and_forwards_resume(
    monkeypatch, tmp_path
):
    checkpoint = tmp_path / "custom_exit.pt"
    metrics = tmp_path / "custom_exit.csv"
    resume = tmp_path / "resume.pt"
    calls = {}

    def fake_train(**kwargs):
        calls.update(kwargs)
        output = Path(kwargs["checkpoint_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"checkpoint")

    monkeypatch.setattr(exit_train, "train", fake_train)

    result = exit_train.train_profile(
        deck_path=REPO_ROOT / "deck/02_dragapult.csv",
        checkpoint_path=checkpoint,
        checkpoint_dir=tmp_path / "checkpoints",
        metrics_dir=tmp_path / "metrics",
        runs_dir=tmp_path / "runs",
        resume_path=resume,
        metrics_path=metrics,
    )

    assert calls["checkpoint_path"] == str(checkpoint)
    assert calls["init_checkpoint"] == str(resume)
    assert calls["metrics_path"] == str(metrics)
    assert result.checkpoint == checkpoint
    assert result.metrics == metrics
    assert result.checkpoint.is_file()


def test_profiles_have_isolated_training_paths():
    first = AgentProfile.load("02_dragapult")
    second = AgentProfile(
        "test_agent",
        _spec=AgentSpec(
            name="test_agent",
            deck_path=first.deck_path,
            policy="random",
            trainer="ppo",
            strategy=None,
            checkpoint_path=REPO_ROOT / "agents/test_agent/checkpoints/ppo_latest.pt",
        ),
    )

    assert first.checkpoint_dir != second.checkpoint_dir
    assert first.metrics_dir != second.metrics_dir
    assert first.runs_dir != second.runs_dir
    assert first.checkpoint_dir != Path("checkpoints").resolve()
    assert first.metrics_dir != Path("metrics").resolve()
    assert first.runs_dir != Path("runs").resolve()


def test_profile_spec_honors_explicit_exit_checkpoint(tmp_path, monkeypatch):
    profile_dir = tmp_path / "agents/configured"
    profile_dir.mkdir(parents=True)
    (profile_dir / "profile.yaml").write_text(
        "name: configured\n"
        "deck: deck.csv\n"
        "policy: random\n"
        "trainer: ppo\n"
        "strategy: null\n"
        "checkpoint: agents/configured/checkpoints/ppo.pt\n"
        "exit_checkpoint: agents/configured/checkpoints/expert.pt\n"
    )
    monkeypatch.setattr(spec_module, "REPO_ROOT", tmp_path)

    spec = AgentSpec.load("configured")

    assert spec.exit_checkpoint_path == (
        tmp_path / "agents/configured/checkpoints/expert.pt"
    )
