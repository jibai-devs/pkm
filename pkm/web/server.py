"""HTTP bridge for browser play — the web counterpart of ``pkm/tui/app.py``.

``ThreadedEnvSession`` (``pkm/tui/session.py``) already runs kaggle's ``env.run``
on a worker thread and bridges the human via two blocking queues
(``next_event`` / ``submit``). The Textual TUI is one consumer of that four-method
protocol; this module is another — it exposes the same session over plain HTTP so
a React front-end can render the board and post the human's picks.

No async, no extra dependencies: ``ThreadingHTTPServer`` gives every request its
own thread, so a long-poll ``GET /api/event`` can simply block on
``session.next_event`` exactly like the TUI's pump does. That is the whole trick —
a blocking GET *is* ``next_event``, a POST *is* ``submit``.

Protocol
--------
* ``GET  /api/config``  -> ``{opponents: [...], decks: [...], default_*: ...}``
* ``POST /api/start``   body ``{opponent, deck, human_index}`` -> ``{ok: true}``
  (starts a fresh game, discarding any in-progress one)
* ``GET  /api/event``   -> one of (long-poll, ~25s, then a ``heartbeat``):
    - ``{type: "prompt", obs: <observation dict>, prompt: {...}, humanIndex}``
    - ``{type: "note", message}``            (opponent agent commentary)
    - ``{type: "finished", result, rewards}``
    - ``{type: "error", message}``
    - ``{type: "heartbeat"}``
* ``POST /api/submit``  body ``{picks: [int, ...]}`` -> ``{ok: true}``
* ``POST /api/quit``    -> ``{ok: true}``

Everything else is served as a static file: the built SPA in
``replay/07_vite_react_cards/dist``, plus ``/cards.json`` and ``/cards/<id>.png``
so the board art works without the Vite dev server.
"""

from __future__ import annotations

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from pkm.data import Deck
from pkm.data.deck import list_decks, resolve_deck
from pkm.tui.labels import option_label
from pkm.tui.session import (
    AgentNote,
    Failed,
    Finished,
    OpponentHand,
    Prompt,
    ThreadedEnvSession,
)
from pkm.types.obs import Observation, OptionType

REPO_ROOT = Path(__file__).resolve().parents[2]
DIST_DIR = REPO_ROOT / "replay" / "07_vite_react_cards" / "dist"
CARDS_JSON = REPO_ROOT / "replay" / "cards.json"
CARDS_DIR = REPO_ROOT / "pkm_data" / "replay" / "cards"

OPPONENTS = ["neural", "mcts", "random", "singaporean_middleman"]
DEFAULT_OPPONENT = "neural"
DEFAULT_DECK = "02_dragapult"

# Same set the TUI double-confirms: there is no undo for an attack or ending the
# turn, so the UI should ask before submitting one.
IRREVERSIBLE = {OptionType.ATTACK, OptionType.END}

# How long a GET /api/event blocks before returning a heartbeat. Keeps the HTTP
# connection from hanging indefinitely while the human (or a slow bot) thinks.
POLL_SECONDS = 25.0

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".woff2": "font/woff2",
    ".map": "application/json",
}


class GameManager:
    """Owns the single active game session. Starting a new one replaces it."""

    def __init__(self) -> None:
        self._session: ThreadedEnvSession | None = None
        self._lock = threading.Lock()

    def start(self, opponent: str, deck_name: str, human_index: int) -> None:
        if opponent not in OPPONENTS:
            raise ValueError(f"unknown opponent {opponent!r} (expected {OPPONENTS})")
        if human_index not in (0, 1):
            raise ValueError("human_index must be 0 or 1")
        deck = Deck.from_csv(resolve_deck(deck_name)).card_ids
        with self._lock:
            if self._session is not None:
                self._session.quit()
            session = ThreadedEnvSession(
                deck=deck,
                human_index=human_index,
                opponent=opponent,
                weights=None,  # neural/mcts auto-find pkm/policy.npz (same as TUI)
                html_path=str(REPO_ROOT / "result.html"),
                replay_path=str(REPO_ROOT / "replay.json"),
            )
            session.start()
            self._session = session

    @property
    def session(self) -> ThreadedEnvSession | None:
        return self._session

    def submit(self, picks: list[int]) -> None:
        with self._lock:
            if self._session is None:
                raise RuntimeError("no active game")
            self._session.submit(picks)

    def quit(self) -> None:
        with self._lock:
            if self._session is not None:
                self._session.quit()
                self._session = None


def _serialize_prompt(obs: Observation, human_index: int) -> dict:
    """Turn a typed observation into the wire prompt the front-end renders.

    ``obs`` is single-POV (the human's own side): their hand is populated, the
    opponent's is hidden by the engine. That is exactly what the board should
    show, so we dump it as-is and let the React adapter build a MergedStep.
    Option *labels* are rendered here with the same battle-tested labeller the
    TUI uses (``pkm/tui/labels.option_label``) rather than reimplemented in TS.
    """
    select = obs.select
    assert select is not None  # a Prompt event always carries a select
    options = [
        {
            "index": i,
            "label": option_label(obs, opt),
            "type": opt.type,
            "irreversible": opt.kind in IRREVERSIBLE,
        }
        for i, opt in enumerate(select.option)
    ]
    return {
        "type": "prompt",
        "humanIndex": human_index,
        "obs": obs.model_dump(mode="json"),
        "prompt": {
            "minCount": select.minCount,
            "maxCount": select.maxCount,
            "type": select.type,
            "context": select.context,
            "options": options,
        },
    }


def _serialize_finished(event: Finished, human_index: int) -> dict:
    mine = event.rewards[human_index]
    theirs = event.rewards[1 - human_index]
    # A None reward means that side never finished — kaggle marks a crashed /
    # timed-out / invalid agent that way. Report it honestly instead of dressing
    # up a forfeit as a real result: if the *opponent* errored, you didn't win
    # the game, the bot fell over (e.g. stale policy weights that no longer match
    # the encoder — see AGENTS.md). If *you* errored, likewise say so.
    if theirs is None and mine is not None:
        result = "opponent_crashed"
    elif mine is None:
        result = "you_errored"
    elif mine > 0:
        result = "win"
    elif mine < 0:
        result = "lose"
    else:
        result = "draw"
    return {"type": "finished", "result": result, "rewards": list(event.rewards)}


class Handler(BaseHTTPRequestHandler):
    manager: GameManager  # set on the class in serve()

    # Quieter logging than the default one-line-per-request to stderr.
    def log_message(self, format: str, *args) -> None:  # noqa: A002 (matches base)
        pass

    # -- helpers -----------------------------------------------------------

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw) if raw else {}

    def _send_file(self, path: Path, content_type: str | None = None) -> None:
        if not path.is_file():
            self._send_json({"error": "not found"}, status=404)
            return
        data = path.read_bytes()
        ctype = content_type or _CONTENT_TYPES.get(
            path.suffix, "application/octet-stream"
        )
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    # -- routing -----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        route = urlparse(self.path).path
        if route == "/api/config":
            self._handle_config()
        elif route == "/api/event":
            self._handle_event()
        elif route == "/cards.json":
            self._send_file(CARDS_JSON)
        elif route.startswith("/cards/"):
            self._send_file(CARDS_DIR / Path(route).name)
        else:
            self._handle_static(route)

    def do_POST(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        try:
            if route == "/api/start":
                self._handle_start()
            elif route == "/api/submit":
                self._handle_submit()
            elif route == "/api/quit":
                self.manager.quit()
                self._send_json({"ok": True})
            else:
                self._send_json({"error": "unknown endpoint"}, status=404)
        except Exception as exc:  # noqa: BLE001 - report to the client, don't 500 silently
            self._send_json({"error": str(exc)}, status=400)

    def do_OPTIONS(self) -> None:  # noqa: N802 (CORS preflight for direct dev use)
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # -- handlers ----------------------------------------------------------

    def _handle_config(self) -> None:
        decks = [p.stem for p in list_decks()]
        # de-dup while preserving order (a deck can exist as both .csv and .json)
        seen: dict[str, None] = {}
        for d in decks:
            seen.setdefault(d, None)
        self._send_json(
            {
                "opponents": OPPONENTS,
                "decks": list(seen),
                "defaultOpponent": DEFAULT_OPPONENT,
                "defaultDeck": DEFAULT_DECK
                if DEFAULT_DECK in seen
                else next(iter(seen), ""),
            }
        )

    def _handle_start(self) -> None:
        body = self._read_body()
        self.manager.start(
            opponent=body.get("opponent", DEFAULT_OPPONENT),
            deck_name=body.get("deck", DEFAULT_DECK),
            human_index=int(body.get("humanIndex", 0)),
        )
        self._send_json({"ok": True})

    def _handle_submit(self) -> None:
        body = self._read_body()
        picks = [int(p) for p in body.get("picks", [])]
        self.manager.submit(picks)
        self._send_json({"ok": True})

    def _handle_event(self) -> None:
        session = self.manager.session
        if session is None:
            self._send_json({"type": "error", "message": "no active game"})
            return
        human_index = session.human_index
        # Drain until we have something the client should see. OpponentHand is a
        # spy view (the human's real obs never contains the opponent's hand) —
        # drop it for fair play. Bound the loop by wall-clock via POLL_SECONDS.
        deadline_pulls = 0
        while True:
            try:
                event = session.next_event(timeout=POLL_SECONDS)
            except queue.Empty:
                self._send_json({"type": "heartbeat"})
                return
            if isinstance(event, Prompt):
                self._send_json(_serialize_prompt(event.obs, human_index))
                return
            if isinstance(event, Finished):
                self._send_json(_serialize_finished(event, human_index))
                return
            if isinstance(event, Failed):
                self._send_json({"type": "error", "message": str(event.error)})
                return
            if isinstance(event, AgentNote):
                self._send_json({"type": "note", "message": event.message})
                return
            if isinstance(event, OpponentHand):
                # spy view — never surface to the human's own screen
                deadline_pulls += 1
                if deadline_pulls > 1000:  # pathological guard, should never hit
                    self._send_json({"type": "heartbeat"})
                    return
                continue

    def _handle_static(self, route: str) -> None:
        # Serve the built SPA; unknown routes fall back to index.html so client
        # routing / query-param modes work on reload.
        rel = route.lstrip("/") or "index.html"
        candidate = (DIST_DIR / rel).resolve()
        # Contain path traversal within DIST_DIR.
        if DIST_DIR not in candidate.parents and candidate != DIST_DIR:
            self._send_json({"error": "forbidden"}, status=403)
            return
        if candidate.is_file():
            self._send_file(candidate)
        else:
            index = DIST_DIR / "index.html"
            if index.is_file():
                self._send_file(index)
            else:
                self._send_json(
                    {
                        "error": "SPA not built — run `bun run build` in "
                        "replay/07_vite_react_cards (or use the Vite dev server)"
                    },
                    status=404,
                )


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    Handler.manager = GameManager()
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"pkm web-play server on http://{host}:{port}")
    print("  dev:  bun run dev  in replay/07_vite_react_cards  (proxies /api here)")
    print(f"  prod: open http://{host}:{port}/?mode=play  (after `bun run build`)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        httpd.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Browser play server for cabt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(host=args.host, port=args.port)
