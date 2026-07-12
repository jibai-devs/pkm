import os
import subprocess
import tarfile
from pathlib import Path
from shutil import copy2


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_submit_bundles_profile_export_as_inference_policy(tmp_path):
    (tmp_path / "deck").mkdir()
    (tmp_path / "pkm").mkdir()
    (tmp_path / "agents/02_dragapult/checkpoints").mkdir(parents=True)
    (tmp_path / "deck/02_dragapult.csv").write_text("1\n" * 60)
    (tmp_path / "main.py").write_text("# test submission\n")
    profile_policy = b"fresh profile policy"
    (tmp_path / "agents/02_dragapult/checkpoints/policy.npz").write_bytes(
        profile_policy
    )
    submit_script = tmp_path / "submit.sh"
    copy2(REPO_ROOT / "submit.sh", submit_script)
    submit_script.chmod(0o755)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    subprocess.run(
        [str(submit_script), "02_dragapult"],
        cwd=tmp_path,
        env=env,
        check=True,
    )

    archives = list((tmp_path / "submissions").glob("submission_*.tar.gz"))
    assert len(archives) == 1
    with tarfile.open(archives[0], "r:gz") as archive:
        assert archive.extractfile("./pkm/policy.npz").read() == profile_policy
