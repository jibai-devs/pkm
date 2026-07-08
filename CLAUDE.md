# pkm — project instructions

Full project guide (structure, RL training, decks, submission): @AGENTS.md

## Replay viewer

Step-by-step viewer for match replays. The maintained one is React/TS at
`replay/05_vite_react_app/`.

```bash
just play                            # generate a replay.json + result.html
just replay-react                    # view it at http://localhost:5175
just replay-react file=/foo.json     # view a different replay
```

Load a different replay three ways (precedence: picker > `?replay=` > `VITE_REPLAY`
> default): the in-app **"Load replay…"** button / drag-drop (any local `.json`),
a `?replay=/foo.json` URL param, or the `VITE_REPLAY` env var. `?step=N`
deep-links a step. Controls: **Space** play/pause, **←/→** step, **Home/End**
first/last, scrubber to jump.

**Full usage:** `docs/REPLAY_VIEWER.md` · **data contract & code layout:**
`replay/05_vite_react_app/README.md`.
