"""Tests for pkm.archetype.archetypes staple name -> card_id resolution."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pkm.archetype.archetypes import (
    _normalize_name,
    _parse_pct,
    get_archetypes,
    load_archetypes_with_report,
)


def test_archetypes_load():
    archetypes = get_archetypes()
    assert len(archetypes) == 25, "staples.json should have 25 archetypes"


def test_resolution_report_mostly_resolved():
    _, report = load_archetypes_with_report()
    assert report.total > 0
    resolved_pct = (report.auto + report.alias) / report.total
    assert resolved_pct > 0.9, (
        f"resolution rate {resolved_pct:.1%} regressed below 90% -- "
        "either a staples.json change or an aliases.py regression"
    )


def test_resolution_report_no_new_unresolved():
    """Locks in the known-remaining unresolved set (see aliases.py TODO) so a
    silent regression (e.g. breaking the apostrophe normalization) is caught,
    without failing the suite over the pre-existing, documented gap."""
    _, report = load_archetypes_with_report()
    unique_unresolved = {(name, set_, num) for (_, name, set_, num) in report.unresolved}
    assert len(unique_unresolved) <= 24, (
        f"unresolved set grew to {len(unique_unresolved)} unique cards -- "
        "investigate before adding more aliases blindly"
    )


def test_apostrophe_normalization():
    assert _normalize_name("Boss’s Orders") == "Boss's Orders"
    assert _normalize_name("Boss's Orders") == "Boss's Orders"


def test_parse_pct():
    assert _parse_pct("99.80%") == 0.998
    assert _parse_pct("100%") == 1.0


def test_staple_card_ids_resolved_are_valid():
    from pkm.data.card_data import get_card_data

    cards = get_card_data()
    archetypes = get_archetypes()
    for archetype in archetypes:
        for staple in archetype.staples:
            if staple.card_id is not None:
                assert staple.card_id in cards, (
                    f"{archetype.name}: {staple.name} resolved to card_id "
                    f"{staple.card_id}, not present in engine card DB"
                )
