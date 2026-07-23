"""Play a *past Kaggle submission bundle* as a live opponent.

A submission tarball is completely self-contained: its own ``main.py``, its own
copy of the ``pkm`` package (encoder included), and its own ``policy.npz``. So
its weights never need to fit today's network -- the 04_mega_abomasnow bundle
was trained when the encoder emitted 775 state features and today's emits 335,
and it still plays perfectly well, because it brought the encoder those weights
were trained against.

The only real obstacle is that both packages are called ``pkm``, which cannot
be imported twice in one interpreter. That is a process-boundary problem, not a
compatibility one, so the bundle runs in a **subprocess** with its own
``sys.path`` -- the same trick `pkm/agents/turn_planner` uses to give the
planner its own engine handle.

Protocol is newline-delimited JSON over stdin/stdout, because the Kaggle agent
contract is already exactly ``agent(obs: dict) -> list[int]`` and observations
are plain JSON. stdout is swapped for stderr inside the child before importing,
since ``kaggle_environments`` prints a wall of banners that would otherwise
corrupt the protocol stream.
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import tarfile
import threading
from pathlib import Path

# The child's whole program. Kept as a string so the bundle is loaded in a
# fresh interpreter that has *its* pkm on sys.path and never sees ours.
_CHILD = r"""
import sys, json, os
sys.path.insert(0, sys.argv[1])

# Reserve the protocol channel at the *file-descriptor* level, not by
# rebinding sys.stdout. The engine is a C library that writes its banners
# straight to fd 1, so a Python-level swap leaves them interleaved with the
# JSON -- and worse, once the stdout pipe fills the child blocks writing
# while the parent blocks reading, which deadlocks instead of failing.
# Duplicating fd 1 elsewhere and pointing fd 1 at stderr sends every write,
# from Python or from C, somewhere harmless.
_protocol_fd = os.dup(1)
os.dup2(2, 1)
_protocol = os.fdopen(_protocol_fd, "w")
sys.stdout = sys.stderr

import main

def _fresh_agent():
    # Rebuild so each game starts with clean per-game memory: reusing one
    # agent across games leaks card knowledge between them and quietly
    # weakens the very opponent we are measuring ourselves against.
    #
    # Only some bundles can be rebuilt. The singaporean_middleman lineage
    # exposes its factory and deck; the agent_000_dragapult lineage exposes
    # just `agent`, and holds no per-game state, so reusing it is correct
    # there. Fall back rather than assume one shape.
    try:
        return main.make_singaporean_middleman(main.DECK)
    except Exception:
        return main.agent

def _find_deck():
    # `main.DECK` exists in some bundles but not others. The kaggle contract
    # itself is the portable answer: an agent handed an observation with no
    # `select` must return its 60-card deck. Ask it that way, so any
    # conforming bundle works regardless of internal layout.
    deck = getattr(main, "DECK", None)
    if deck:
        return list(deck)
    for probe in ({"select": None, "current": None}, {"select": None}):
        try:
            got = main.agent(dict(probe))
            if isinstance(got, list) and len(got) >= 40:
                return [int(c) for c in got]
        except Exception:
            continue
    raise RuntimeError("cannot determine this bundle's deck")

agent = _fresh_agent()

def reply(obj):
    _protocol.write(json.dumps(obj) + "\n")
    _protocol.flush()

reply({"ok": True, "deck": _find_deck()})
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except ValueError:
        continue
    if msg is None or msg.get("cmd") == "stop":
        break
    if msg.get("cmd") == "new_game":
        agent = _fresh_agent()
        reply({"ok": True})
        continue
    try:
        reply({"ok": True, "picks": agent(msg["obs"])})
    except Exception as exc:
        reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
"""


def extract_bundle(tarball: str | Path, dest: str | Path) -> Path:
    """Unpack a submission tarball once; reuse it if already unpacked.

    `dest` is treated as a *parent* and the bundle lands in a subdirectory
    named after the tarball. Extracting straight into a shared `dest` would
    silently reuse whichever bundle got there first -- the "already unpacked"
    check can't tell two bundles apart -- so a second opponent would be
    quietly replaced by the first and every result attributed to the wrong
    agent.
    """
    dest = Path(dest) / Path(tarball).name.replace(".tar.gz", "").replace(".tgz", "")
    if (dest / "main.py").is_file():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball) as tf:
        tf.extractall(dest)
    if not (dest / "main.py").is_file():
        raise RuntimeError(f"{tarball} has no main.py at its root")
    return dest


class BundleOpponent:
    """A past submission, alive in its own process, answering decisions."""

    def __init__(self, bundle_dir: str | Path, timeout_s: float = 120.0) -> None:
        self.bundle_dir = str(Path(bundle_dir).resolve())
        self.timeout_s = timeout_s
        self._proc: subprocess.Popen | None = None
        self._replies: queue.Queue = queue.Queue()
        self.deck: list[int] = []

    # --- lifecycle ---------------------------------------------------------

    def start(self) -> "BundleOpponent":
        self._proc = subprocess.Popen(
            [sys.executable, "-c", _CHILD, self.bundle_dir],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._pump, daemon=True).start()
        hello = self._await()
        if not hello or not hello.get("ok"):
            raise RuntimeError(f"opponent bundle failed to start: {hello}")
        self.deck = hello["deck"]
        return self

    def _pump(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if line:
                try:
                    self._replies.put(json.loads(line))
                except ValueError:
                    pass
        self._replies.put(None)

    def _await(self) -> dict | None:
        try:
            return self._replies.get(timeout=self.timeout_s)
        except queue.Empty:
            return None

    def _send(self, msg: dict) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(msg) + "\n")
        self._proc.stdin.flush()

    def close(self) -> None:
        try:
            if self._proc is not None and self._proc.poll() is None:
                self._send({"cmd": "stop"})
                self._proc.stdin.close()
                self._proc.wait(timeout=5)
        except Exception:
            pass
        try:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.kill()
        except Exception:
            pass
        self._proc = None

    def __enter__(self) -> "BundleOpponent":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    # --- play --------------------------------------------------------------

    def new_game(self) -> None:
        """Reset the bundle's per-game memory before a fresh battle."""
        self._send({"cmd": "new_game"})
        self._await()

    def act(self, obs: dict) -> list[int]:
        """Ask the bundle for its picks. Raises if it dies or stalls."""
        self._send({"obs": obs})
        reply = self._await()
        if reply is None:
            raise RuntimeError("opponent timed out or exited")
        if not reply.get("ok"):
            raise RuntimeError(f"opponent errored: {reply.get('error')}")
        return reply["picks"]
