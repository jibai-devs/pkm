from pathlib import Path

from pkm.rl import export


def test_profile_export_defaults_to_owned_weights_path(tmp_path, monkeypatch):
    _write_profile(tmp_path)
    monkeypatch.setattr("pkm.agents.spec.REPO_ROOT", tmp_path)
    calls = {}

    def fake_export(checkpoint, out):
        calls.update(checkpoint=checkpoint, out=out)

    monkeypatch.setattr(export, "export_checkpoint", fake_export)

    export.main(checkpoint="", out=None, agent="02_dragapult", phase="ppo")

    assert calls == {
        "checkpoint": str(tmp_path / "agents/02_dragapult/checkpoints/ppo_latest.pt"),
        "out": str(tmp_path / "agents/02_dragapult/checkpoints/policy.npz"),
    }


def test_profile_export_uses_configured_checkpoint_path(tmp_path, monkeypatch):
    _write_profile(tmp_path, checkpoint_name="exit_latest.pt")
    monkeypatch.setattr("pkm.agents.spec.REPO_ROOT", tmp_path)
    calls = {}

    monkeypatch.setattr(
        export,
        "export_checkpoint",
        lambda checkpoint, out: calls.update(checkpoint=checkpoint, out=out),
    )

    export.main(checkpoint="", out=None, agent="02_dragapult", phase="ppo")

    assert calls["checkpoint"] == str(
        tmp_path / "agents/02_dragapult/checkpoints/exit_latest.pt"
    )


def test_profile_export_honors_explicit_checkpoint(tmp_path, monkeypatch):
    """--agent must not silently discard an explicitly supplied checkpoint."""
    _write_profile(tmp_path)
    monkeypatch.setattr("pkm.agents.spec.REPO_ROOT", tmp_path)
    explicit = tmp_path / "agents/02_dragapult/checkpoints/exit_latest.pt"
    explicit.write_bytes(b"exit-weights")
    calls = {}

    monkeypatch.setattr(
        export,
        "export_checkpoint",
        lambda checkpoint, out: calls.update(checkpoint=checkpoint, out=out),
    )

    export.main(checkpoint=str(explicit), out=None, agent="02_dragapult", phase="ppo")

    assert calls["checkpoint"] == str(explicit)
    # The output still defaults to the profile's own export path.
    assert calls["out"] == str(tmp_path / "agents/02_dragapult/checkpoints/policy.npz")


def test_profile_export_can_select_the_exit_phase(tmp_path, monkeypatch):
    """After expert iteration the exit checkpoint must be exportable."""
    _write_profile(tmp_path)
    monkeypatch.setattr("pkm.agents.spec.REPO_ROOT", tmp_path)
    exit_checkpoint = tmp_path / "agents/02_dragapult/checkpoints/exit_latest.pt"
    exit_checkpoint.write_bytes(b"exit-weights")
    calls = {}

    monkeypatch.setattr(
        export,
        "export_checkpoint",
        lambda checkpoint, out: calls.update(checkpoint=checkpoint, out=out),
    )

    export.main(checkpoint="", out=None, agent="02_dragapult", phase="exit")

    assert calls["checkpoint"] == str(exit_checkpoint)


def test_export_keeps_explicit_output_path(tmp_path, monkeypatch):
    _write_profile(tmp_path)
    monkeypatch.setattr("pkm.agents.spec.REPO_ROOT", tmp_path)
    calls = {}

    monkeypatch.setattr(
        export,
        "export_checkpoint",
        lambda checkpoint, out: calls.update(checkpoint=checkpoint, out=out),
    )

    export.main(
        checkpoint="checkpoint.pt", out="pkm/policy.npz", agent=None, phase="ppo"
    )

    assert calls == {"checkpoint": "checkpoint.pt", "out": "pkm/policy.npz"}


def _write_profile(tmp_path: Path, checkpoint_name: str = "ppo_latest.pt") -> None:
    profile_dir = tmp_path / "agents" / "02_dragapult"
    profile_dir.mkdir(parents=True)
    (profile_dir / "deck.csv").write_text("1\n" * 60)
    (profile_dir / "checkpoints").mkdir()
    (profile_dir / "checkpoints" / checkpoint_name).write_bytes(b"checkpoint")
    (profile_dir / "profile.yaml").write_text(
        "name: 02_dragapult\n"
        "deck: agents/02_dragapult/deck.csv\n"
        "policy: neural\n"
        "trainer: ppo\n"
        f"checkpoint: agents/02_dragapult/checkpoints/{checkpoint_name}\n"
        "strategy: null\n"
    )
