"""Hand-maintained overrides for staples.json name -> engine card_id resolution.

pkm.data.card_data.CardData has no set/number field, only name -- so when a
staple's name matches zero or multiple cards in the engine's card DB, the
staple's own (set, number) from staples.json is the authoritative
disambiguator. Populate this table by running
`pkm.archetype.archetypes.load_archetypes_with_report()` and adding an entry
for each unresolved (name, set, number) tuple it reports.
"""

ALIASES: dict[tuple[str, str, str], int] = {
    # Basic energies: staples.json names them "<Type> Energy"; the engine
    # names them "Basic {<code>} Energy" (card_ids 1-8). Confirmed by direct
    # lookup against pkm.data.card_data.get_energy_cards().
    ("Grass Energy", "MEE", "1"): 1,
    ("Fire Energy", "MEE", "2"): 2,
    ("Water Energy", "MEE", "3"): 3,
    ("Lightning Energy", "MEE", "4"): 4,
    ("Psychic Energy", "MEE", "5"): 5,
    ("Fighting Energy", "MEE", "6"): 6,
    ("Darkness Energy", "MEE", "7"): 7,
    ("Metal Energy", "MEE", "8"): 8,
    # Special energies: staples.json's scraped name differs slightly from
    # the engine's card name for the same card.
    ("Growing Grass Energy", "POR", "86"): 18,  # engine: "Grow Grass Energy"
    ("Telepathic Psychic Energy", "POR", "88"): 19,  # engine: "Telepath Psychic Energy"
    ("Rocky Fighting Energy", "POR", "87"): 20,  # engine: "Rock Fighting Energy"
}

# TODO: unresolved as of 2026-07-19's resolution report run (see
# pkm.archetype.archetypes.load_archetypes_with_report()). Two distinct
# categories, neither fixable by more normalization:
#
# 1. Zero matches -- the card does not exist under ANY name in the engine's
#    card pool (pkm.engine.all_cards()), confirmed by substring search:
#    ("Special Red Card", "CRI", "82"), ("Weedle", "CRI", "1"),
#    ("Kakuna", "CRI", "2"), ("Beedrill ex", "CRI", "3"),
#    ("Prism Tower", "CRI", "80"). These cards are genuinely not modeled by
#    the engine -- notably this knocks out 3 of "Beedrill ex" archetype's
#    core pieces (its own evolution line). Not resolvable via aliasing;
#    these staples should be dropped from the trainable feature set for
#    their archetype, not guessed at.
#
# 2. Multiple matches -- the name matches 2-3 reprints with DIFFERENT
#    gameplay stats (weakness/energyType/attacks differ between prints), and
#    CardData has no set/number field to disambiguate automatically. Needs a
#    real card-database lookup (by set+number) to pick the correct card_id:
#    ("Dwebble", "DRI", "11"), ("Crustle", "DRI", "12"),
#    ("Slowpoke", "SCR", "57"), ("Dipplin", "TWM", "18"),
#    ("Bayleef", "MEG", "9"), ("Abra", "MEG", "54"),
#    ("Alakazam", "MEG", "56"), ("Dunsparce", "JTG", "120"),
#    ("Genesect", "SFA", "40"), ("Chien-Pao", "SSP", "56"),
#    ("Shaymin", "DRI", "10"), ("Riolu", "MEG", "76"),
#    ("Victini", "SSP", "21"), ("Beldum", "TEF", "113"),
#    ("Metang", "TEF", "114"), ("Buneary", "PFL", "83"),
#    ("Snorunt", "ASC", "46"), ("Froakie", "CRI", "20"),
#    ("Duraludon", "SCR", "106").
