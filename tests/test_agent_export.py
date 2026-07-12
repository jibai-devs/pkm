from pathlib import Path

from pkm.rl import export


def test_profile_export_defaults_to_owned_weights_path(tmp_path, monkeypatch):
    _write_profile(tmp_path)
    monkeypatch.setattr("pkm.agents.spec.REPO_ROOT", tmp_path)
    calls = {}

    def fake_export(checkpoint, out):
        calls.update(checkpoint=checkpoint, out=out)

    monkeypatch.setattr(export, "export_checkpoint", fake_export)

    export.main(checkpoint="", out=None, agent="02_dragapult")

    assert calls == {
        "checkpoint": str(tmp_path / "agents/02_dragapult/checkpoints/ppo_latest.pt"),
        "out": str(tmp_path / "agents/02_dragapult/checkpoints/policy.npz"),
    }


def test_export_keeps_explicit_output_path(tmp_path, monkeypatch):
    _write_profile(tmp_path)
    monkeypatch.setattr("pkm.agents.spec.REPO_ROOT", tmp_path)
    calls = {}

    monkeypatch.setattr(
        export,
        "export_checkpoint",
        lambda checkpoint, out: calls.update(checkpoint=checkpoint, out=out),
    )

    export.main(checkpoint="checkpoint.pt", out="pkm/policy.npz", agent=None)

    assert calls == {"checkpoint": "checkpoint.pt", "out": "pkm/policy.npz"}


def _write_profile(tmp_path: Path) -> None:
    profile_dir = tmp_path / "agents" / "02_dragapult"
    profile_dir.mkdir(parents=True)
    (profile_dir / "deck.csv").write_text("1\n" * 60)
    (profile_dir / "checkpoints").mkdir()
    (profile_dir / "checkpoints" / "ppo_latest.pt").write_bytes(b"checkpoint")
    (profile_dir / "profile.yaml").write_text(
        "name: 02_dragapult\n"
        "deck: agents/02_dragapult/deck.csv\n"
        "policy: neural\n"
        "trainer: ppo\n"
        "checkpoint: agents/02_dragapult/checkpoints/ppo_latest.pt\n"
        "strategy: null\n"
    )
