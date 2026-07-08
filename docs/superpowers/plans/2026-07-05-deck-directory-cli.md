# Deck Directory + Deck CLI Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `deck.csv` into a `deck/` directory, add JSON deck format with `{id, name, count}` entries, and create a CLI to list/show/convert decks.

**Architecture:** The existing `Deck` class gets `from_json()` / `to_json()` methods. A new `pkm/cli_deck.py` typer app provides `list`, `show`, and `convert` subcommands. All default deck paths change from `deck.csv` to `deck/00_basic.csv`. The `submit.sh` script generates a flat CSV for Kaggle compatibility.

**Tech Stack:** typer (already installed), rich (ships with typer), json (stdlib)

---

## Files to Modify

| File | Change |
|------|--------|
| `deck.csv` | Delete (moved to `deck/00_basic.csv`) |
| `deck/00_basic.csv` | Create (moved from `deck.csv`) |
| `pkm/data/deck.py` | Add `from_json()`, `to_json()`, `list_decks()`, `resolve_deck()` |
| `pkm/cli_deck.py` | Create — typer app with `list`, `show`, `convert` commands |
| `pkm/rl/train.py:207` | Default `deck` option: `deck.csv` -> `deck/00_basic.csv` |
| `pkm/rl/exit_train.py:287` | Default `deck` option: `deck.csv` -> `deck/00_basic.csv` |
| `pkm/rl/play.py:41,78,108` | Default `deck` option: `deck.csv` -> `deck/00_basic.csv` |
| `main.py:9,34` | Default deck path: `deck.csv` -> `deck/00_basic.csv` |
| `submit.sh:16` | Generate flat CSV from deck dir for kaggle bundle |
| `justfile` | Add `deck` recipe, update any `deck.csv` references |
| `tests/test_rl.py:30,63,80` | Update `deck.csv` -> `deck/00_basic.csv` |
| `tests/test_mcts.py:23` | Update `deck.csv` -> `deck/00_basic.csv` |

---

## Task 1: Create deck directory and move deck.csv

**Files:**
- Delete: `deck.csv`
- Create: `deck/00_basic.csv`

- [ ] **Step 1: Create the deck directory and move the file**

```bash
mkdir -p deck
mv deck.csv deck/00_basic.csv
```

- [ ] **Step 2: Verify the file exists**

```bash
ls deck/
cat deck/00_basic.csv | head -5
```

Expected: `00_basic.csv` listed, first 5 lines are card IDs.

- [ ] **Step 3: Commit**

```bash
git add deck/00_basic.csv
git rm deck.csv
git commit -m "refactor: move deck.csv -> deck/00_basic.csv"
```

---

## Task 2: Add JSON methods to Deck class

**Files:**
- Modify: `pkm/data/deck.py`
- Modify: `pkm/data/__init__.py`

- [ ] **Step 1: Add `from_json` and `to_json` to Deck**

Add these methods to the `Deck` class in `pkm/data/deck.py`:

```python
@classmethod
def from_json(cls, path: str | Path) -> "Deck":
    """Load a deck from a JSON file ([{"id": N, "name": "...", "count": N}, ...])."""
    import json
    with open(path) as f:
        entries = json.load(f)
    card_ids = []
    for entry in entries:
        card_ids.extend([entry["id"]] * entry["count"])
    return cls(card_ids)

def to_json(self, path: str | Path) -> None:
    """Save deck to a JSON file with id, name, count format."""
    import json
    from .card_data import get_card_by_id
    counts: dict[int, int] = {}
    for cid in self.card_ids:
        counts[cid] = counts.get(cid, 0) + 1
    entries = []
    for cid, count in sorted(counts.items()):
        card = get_card_by_id(cid)
        name = card.name if card else f"Unknown#{cid}"
        entries.append({"id": cid, "name": name, "count": count})
    with open(path, "w") as f:
        json.dump(entries, f, indent=2)
```

Also add a `DECK_DIR` constant and helper functions at the module level:

```python
from pathlib import Path

DECK_DIR = Path(__file__).parent.parent.parent / "deck"

def list_decks() -> list[Path]:
    """List all deck files in the deck directory."""
    if not DECK_DIR.is_dir():
        return []
    return sorted(DECK_DIR.glob("*.csv")) + sorted(DECK_DIR.glob("*.json"))

def resolve_deck(name: str) -> Path:
    """Resolve a deck name to a path. Accepts filenames with or without extension."""
    p = Path(name)
    if p.is_file():
        return p
    # Try in deck directory
    for ext in (".csv", ".json"):
        candidate = DECK_DIR / f"{name}{ext}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Deck not found: {name!r} (searched deck/ directory)")
```

- [ ] **Step 2: Update `pkm/data/__init__.py` to export new names**

```python
from .card_data import get_card_data, get_attack_data
from .deck import Deck, DECK_DIR, list_decks, resolve_deck

__all__ = ["get_card_data", "get_attack_data", "Deck", "DECK_DIR", "list_decks", "resolve_deck"]
```

- [ ] **Step 3: Test it works**

```bash
python -c "
from pkm.data import Deck, list_decks, resolve_deck
print('Decks:', list_decks())
d = Deck.from_csv(resolve_deck('00_basic'))
d.to_json('/tmp/test_deck.json')
import json
print(json.dumps(json.load(open('/tmp/test_deck.json')), indent=2))
"
```

Expected: JSON output with `[{id, name, count}, ...]` entries, counts summing to 60.

- [ ] **Step 4: Commit**

```bash
git add pkm/data/deck.py pkm/data/__init__.py
git commit -m "feat: add JSON deck format and deck directory helpers"
```

---

## Task 3: Create deck CLI

**Files:**
- Create: `pkm/cli_deck.py`

- [ ] **Step 1: Create the CLI module**

```python
"""Deck management CLI.

Usage:
    python -m pkm.cli_deck list
    python -m pkm.cli_deck show 00_basic
    python -m pkm.cli_deck convert 00_basic --to json
"""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from pkm.data import Deck, list_decks, resolve_deck

app = typer.Typer(help=__doc__)
console = Console()


@app.command()
def list() -> None:
    """List available decks in the deck/ directory."""
    decks = list_decks()
    if not decks:
        console.print("[yellow]No decks found in deck/ directory[/yellow]")
        raise typer.Exit(1)

    table = Table(title="Available Decks")
    table.add_column("Name", style="cyan")
    table.add_column("Format", style="green")
    table.add_column("Cards", justify="right")

    for path in decks:
        fmt = path.suffix.lstrip(".")
        if path.suffix == ".csv":
            count = sum(1 for line in path.open() if line.strip())
        else:
            count = sum(e["count"] for e in json.load(path.open()))
        table.add_row(path.stem, fmt, str(count))

    console.print(table)


@app.command()
def show(
    name: str = typer.Argument(help="Deck name (with or without extension)"),
) -> None:
    """Show the contents of a deck."""
    path = resolve_deck(name)
    if path.suffix == ".json":
        with open(path) as f:
            entries = json.load(f)
    else:
        deck = Deck.from_csv(path)
        deck.to_json("/tmp/_deck_show.json")
        with open("/tmp/_deck_show.json") as f:
            entries = json.load(f)

    table = Table(title=f"Deck: {path.name}")
    table.add_column("ID", justify="right", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Count", justify="right")

    total = 0
    for entry in entries:
        table.add_row(str(entry["id"]), entry["name"], str(entry["count"]))
        total += entry["count"]

    console.print(table)
    console.print(f"Total: {total} cards, {len(entries)} unique")


@app.command()
def convert(
    name: str = typer.Argument(help="Deck name (with or without extension)"),
    to: str = typer.Option("json", help="Output format: json or csv"),
    out: str | None = typer.Option(None, help="Output path (default: same name, new extension)"),
) -> None:
    """Convert a deck between CSV and JSON formats."""
    path = resolve_deck(name)
    deck = Deck.from_csv(path) if path.suffix == ".csv" else Deck.from_json(path)

    if out:
        out_path = Path(out)
    else:
        out_path = path.with_suffix(".json" if to == "json" else ".csv")

    if to == "json":
        deck.to_json(out_path)
    else:
        deck.to_csv(out_path)

    console.print(f"Converted [cyan]{path.name}[/cyan] -> [green]{out_path}[/green]")


if __name__ == "__main__":
    app()
```

- [ ] **Step 2: Test the CLI**

```bash
python -m pkm.cli_deck list
python -m pkm.cli_deck show 00_basic
python -m pkm.cli_deck convert 00_basic --to json
```

Expected:
- `list` shows a table with `00_basic.csv`
- `show` shows a table with card IDs, names, counts
- `convert` creates `deck/00_basic.json`

- [ ] **Step 3: Commit**

```bash
git add pkm/cli_deck.py
git commit -m "feat: add deck CLI (list, show, convert)"
```

---

## Task 4: Update all default deck paths

**Files:**
- Modify: `pkm/rl/train.py:207`
- Modify: `pkm/rl/exit_train.py:287`
- Modify: `pkm/rl/play.py:41,78,108`
- Modify: `main.py:9,34`

- [ ] **Step 1: Update train.py default**

Change line 207:
```python
    deck: str = typer.Option("deck.csv", help="path to deck CSV"),
```
To:
```python
    deck: str = typer.Option("deck/00_basic.csv", help="path to deck CSV"),
```

Also update line 59 (function default):
```python
    deck_path: str = "deck.csv",
```
To:
```python
    deck_path: str = "deck/00_basic.csv",
```

- [ ] **Step 2: Update exit_train.py default**

Change line 287:
```python
    deck: str = typer.Option("deck.csv", help="path to deck CSV"),
```
To:
```python
    deck: str = typer.Option("deck/00_basic.csv", help="path to deck CSV"),
```

Also update line 193:
```python
    deck_path: str = "deck.csv",
```
To:
```python
    deck_path: str = "deck/00_basic.csv",
```

- [ ] **Step 3: Update play.py defaults**

Change lines 41, 78, 108 from `"deck.csv"` to `"deck/00_basic.csv"`.

- [ ] **Step 4: Update main.py defaults**

Change lines 9 and 34 from `"deck.csv"` to `"deck/00_basic.csv"`.

- [ ] **Step 5: Update tests**

Update `tests/test_rl.py:30,63,80` and `tests/test_mcts.py:23` from `"deck.csv"` to `"deck/00_basic.csv"`.

- [ ] **Step 6: Run tests to verify**

```bash
just test
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add pkm/rl/train.py pkm/rl/exit_train.py pkm/rl/play.py main.py tests/
git commit -m "refactor: update default deck path to deck/00_basic.csv"
```

---

## Task 5: Update submit.sh and justfile

**Files:**
- Modify: `submit.sh`
- Modify: `justfile`

- [ ] **Step 1: Update submit.sh to generate flat CSV**

Replace line 16 (`cp deck.csv submission/`) with:

```bash
# Generate flat deck.csv from deck directory for kaggle
python -c "
from pkm.data import Deck
Deck.from_csv('deck/00_basic.csv').to_csv('submission/deck.csv')
"
```

- [ ] **Step 2: Add `deck` recipe to justfile**

Add after the `test` recipe:

```just
# list available decks
deck:
    python -m pkm.cli_deck list

# show deck contents
deck-show name="00_basic":
    python -m pkm.cli_deck show {{name}}

# convert deck format
deck-convert name="00_basic" to="json":
    python -m pkm.cli_deck convert {{name}} --to {{to}}
```

- [ ] **Step 3: Test justfile recipes**

```bash
just deck
just deck-show 00_basic
just deck-convert 00_basic json
```

Expected: table of decks, table of card contents, conversion message.

- [ ] **Step 4: Commit**

```bash
git add submit.sh justfile
git commit -m "chore: update submit.sh and justfile for deck directory"
```

---

## Task 6: Update AGENTS.md and TODO.md

**Files:**
- Modify: `AGENTS.md`
- Modify: `TODO.md`

- [ ] **Step 1: Update AGENTS.md project structure**

Update the `Project Structure` section to mention `deck/` directory:

```markdown
- `deck/` — deck files (CSV: one card ID per line; JSON: id/name/count)
- `deck/00_basic.csv` — default 60-card deck
```

Update `Build & Run` section:
```bash
uv sync                    # install deps
python main.py             # run a battle
python -m pkm.cli_deck list  # list decks
./submit.sh                # create Kaggle submission bundle
```

- [ ] **Step 2: Update TODO.md**

Mark deck-related items and add the deck CLI task as done.

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md TODO.md
git commit -m "docs: update AGENTS.md and TODO.md for deck directory"
```
