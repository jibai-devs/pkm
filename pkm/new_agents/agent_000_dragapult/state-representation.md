# State representation — coverage map (agent_000_dragapult)

How each part of the board becomes numbers, and **where in the code** it happens.
Reflects the current v1 implementation. See `README.md` for design/decisions and
`../../../../docs/engine-enum-sources.md` for the enum sources.

## Two "embedding" mechanisms (read this first)

1. **Card identity → a vector.** Two channels, **summed** in
   `encoder.py :: CardEncoder.forward`:
   - **Learned** `nn.Embedding` — `encoder.py :: CardEncoder.own_emb`, indexed by
     `entity_id_row`. Only our **27 vocab** cards (`deck.py :: DISTINCT_IDS`,
     via `deck.py :: row_of`) get a trainable row; everything else → shared `UNK`.
   - **Attribute** — `encoder.py :: CardEncoder.attr_mlp` over the static table
     `cards.py :: build_card_attr_table` (cols in `cards.py :: ATTR_COLS`),
     indexed by raw `entity_card_id`. Works for **any** card (opponent included).
2. **Everything else → plain numeric features** (HP, energy counts, flags, zone
   counts). Not lookup embeddings; just numbers consumed by learned MLP/attention
   weights.

## Board coverage

Featurization: `features.py :: featurize` (helpers `_entity_row`, `_hist`,
`_option_row`; layouts `ENTITY_FEAT_COLS`, `GLOBAL_COLS`, `OPTION_FEAT_COLS`).
Encoding: `encoder.py :: StateEncoder` (+ `CardEncoder`), batched by
`encoder.py :: collate_states`.

| Board part | Featurized (`features.py`) | Encoded (`encoder.py`) | Representation | Learned ID emb? |
|---|---|---|---|---|
| Our active | `featurize`→`_entity_row`, slot 0 | `CardEncoder`→`StateEncoder` attn | card vec + 26 feats | ✅ `own_emb` + attribute |
| Our bench ≤5 | `_entity_row`, slots 1–5 | same | card vec + feats | ✅ `own_emb` + attribute |
| Opp active | `_entity_row`, slot 6 (`is_own=0`) | same | card vec + feats | ⚠️ **attribute only** (`UNK` in `own_emb`) |
| Opp bench ≤5 | `_entity_row`, slots 7–11 | same | card vec + feats | ⚠️ attribute only |
| Our hand | `_hist(me.hand)` → `hand_hist[27]` | `StateEncoder.global_mlp` | count histogram | ❌ counts only |
| Opp hand | `handCount` → `globals[12]` | `global_mlp` | 1 scalar | ❌ hidden |
| Our discard | `_hist(me.discard)` → `discard_hist[27]` | `global_mlp` | count histogram | ❌ counts only |
| Opp discard | — | — | **nothing** | ❌ **gap** |
| Our deck | `deckCount` → `globals[9]` | `global_mlp` | scalar | n/a (hidden contents) |
| Opp deck | `deckCount` → `globals[10]` | `global_mlp` | scalar | n/a |
| Prizes (both) | `globals[7],[8]` | `global_mlp` | 2 scalars | n/a (face-down) |
| Stadium | `stadium_present` → `globals[13]` | `global_mlp` | **boolean only** | ❌ identity not embedded |
| Turn / flags | `globals[0..6],[14],[15]` | `global_mlp` | scalars | n/a |
| Abilities / attack effects | — | — | **nothing** | ❌ **gap** |
| Legal actions (options) | `_option_row` → `option_type`,`option_feat` | *(action head — not built)* | type + raw fields | pending `model.py` |

## Per-Pokémon sub-features

All in one entity vector: `features.py :: _entity_row`, columns
`features.py :: ENTITY_FEAT_COLS` (F=26). Card vector from `CardEncoder`; the
26 feats below are concatenated and projected by `StateEncoder.entity_proj`.

| Sub-feature | Column(s) | Representation | Embedded? |
|---|---|---|---|
| Current / max HP | `hp_norm, maxhp_norm, hp_frac` | normalized scalars | numeric |
| Played this turn | `appear_this_turn` | boolean | numeric |
| Attached energy | `energy_0..11` (12-d type histogram) + `n_energy_cards` | counts by type | ❌ energy card identity not embedded |
| Attached tools | `n_tools` | count | ❌ tool identity not embedded |
| Evolution stack | `evo_depth` | count | ❌ pre-evo identity not embedded |
| Ownership / slot | `is_own, is_active` | booleans | numeric |
| Status conditions | `poisoned,burned,asleep,paralyzed,confused` (active only) | booleans | numeric |

## Card attribute vector (the attribute channel)

`cards.py :: build_card_attr_table` → `[max_id+1, A=54]`, columns
`cards.py :: ATTR_COLS`: `hp_norm, retreat_norm`, one-hot `cardType` (7), one-hot
`energyType` (12), one-hot `weakness` (12), one-hot `resistance` (12), stage/kind
flags `basic,stage1,stage2,ex,megaEx,tera,aceSpec`, `max_atk_dmg_norm,
n_attacks_norm`. **Note:** stats only — **no ability/attack effect text**.

## Gaps (not represented yet)

Priority order; where each fix would live:

1. **Ability / attack effects** — the model sees damage numbers, not what a card
   *does*. Biggest gap. Would add: attack-ID embedding + effect tags/text
   (`cards.py` attribute table and/or a new attacks encoder in `encoder.py`).
   `[DECIDE]`
2. **Opponent discard** — currently unfeaturized; add an attribute-pooled or
   histogram summary in `features.py :: featurize`. *(cheap)*
3. **Stadium identity** — only a presence boolean; embed the stadium card via
   `CardEncoder` in `features.py`/`encoder.py`. *(cheap)*
4. **Attached energy / tool / evolution identities** — collapsed to counts; embed
   per-attached-card if it matters. `[DECIDE]`
5. **Opponent hand contents** — hidden; not embeddable, belief-modeling territory
   (infer from `logs`). `[DECIDE]`
