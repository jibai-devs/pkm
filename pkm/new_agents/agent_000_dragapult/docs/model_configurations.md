# Model configurations — agent_000_dragapult

The living reference for **how the network is shaped and which head it uses**.
Every model/architecture feature and every CLI flag that changes the net should
be recorded here (feature → what/why, flag → options + default + effect).

- Config source of truth: `ModelConfig` in `config.py`. Every field below is in
  the **config hash** and baked into each checkpoint, so a checkpoint is tied to
  the exact architecture it was trained with. Old checkpoints that lack a newer
  field **backfill to that field's default** on load (`Config.from_dict`).
- The CLI resolves a **size preset** (`--model`) and then applies **per-dim
  overrides** (`--d-*` / `--n-*`), an unset flag falling through to the preset
  (`build_model_config`). Wiring point: `build_model` in `config.py`.
- Flags below are on `train` (and, where noted, `sweep`). Non-flag defaults are
  the `ModelConfig` field defaults.

---

## Size presets — `--model`

`--model {small,medium,large,xl}` · **default `small`** (== v1, checkpoint-compatible).
Each preset sets the dims below; individual `--d-*/--n-*` flags override it.

| preset | n_layers | d_state | d_entity | n_heads | d_opt | d_card | d_global |
|---|--:|--:|--:|--:|--:|--:|--:|
| **small** (v1) | 1 | 128 | 64 | 4 | 64 | 32 | 64 |
| **medium** | 2 | 256 | 128 | 8 | 128 | 48 | 96 |
| **large** | 3 | 384 | 192 | 8 | 192 | 64 | 128 |
| **xl** | 4 | 512 | 256 | 8 | 256 | 64 | 192 |

## Per-dimension overrides (win over the preset)

| flag | default* | meaning |
|---|---|---|
| `--n-layers` | 1 | Entity-attention layers in the **encoder** trunk. 1 = v1 (single `MultiheadAttention`, no residual/LN). >1 stacks (n−1) extra pre-LN transformer blocks. |
| `--d-state` | 128 | State (trunk output) embedding dim — width of the pooled board summary the value head + heads read. |
| `--d-entity` | 64 | Per-entity embedding dim = the encoder's attention width. **`--n-heads` must divide this.** |
| `--n-heads` | 4 | Attention heads (encoder, and the `attn` decoder head). Must divide `d_entity`. |
| `--d-opt` | 64 | Option embedding / scorer width (the per-option token). |
| `--d-card` | 32 | Card embedding dim (shared hybrid card encoder: own-vocab embedding + attribute MLP). |
| `--dropout` | 0.0 | Dropout in the extra transformer layers (encoder depth + `attn` decoder). Regularization. |
| `--base-residual` / `--no-base-residual` | off | Pre-LN residual **around the base encoder attention** (uniform-residual trunk). Recommended for deep large/xl. Changes params → not interchangeable across the flag. |

\* default = the `ModelConfig` field default (i.e. the `small`/v1 value). A
preset overrides these unless you also pass the flag.

Not exposed as flags (kept at `ModelConfig` defaults unless edited in code):
`d_ctx=16` (select-type/context embedding), `d_atk=32` (attack/move embedding),
`ff_mult=4` (FFN width = ff_mult·d_entity in the transformer layers).

---

## Model size & what drives complexity

Total trainable parameters by preset × policy head (measured; `attn` at its
default `n_dec_layers=2`):

| preset | d_entity | n_heads | n_layers | marginal | autoreg | combo | **attn** |
|---|--:|--:|--:|--:|--:|--:|--:|
| small (v1) | 64 | 4 | 1 | 92,002 | 114,723 | 92,002 | **225,634** |
| medium | 128 | 8 | 2 | 501,970 | 588,371 | 501,970 | **1,031,378** |
| large | 192 | 8 | 3 | 1,532,610 | 1,723,651 | 1,532,610 | **2,719,938** |
| xl | 256 | 8 | 4 | 3,486,530 | 3,823,171 | 3,486,530 | **5,593,922** |

`attn` depth sensitivity (large preset, varying `--n-dec-layers`): 1 →
2,126,466 · 2 → 2,719,938 · 3 → 3,313,410 · 4 → 3,906,882 (≈ **+0.59 M per
decoder layer** at d_entity=192).

**What each knob does to the param count:**

| knob | effect on size | notes |
|---|---|---|
| `--d-entity` | **largest lever** — attention + FFN scale ~quadratically (projections d_entity², FFN width `ff_mult·d_entity`). | Also the encoder's + attn decoder's working width. |
| `--d-state` | large (linear) — value head + all head-conditioning read it. | |
| `--d-opt`, `--d-card`, `--d-global` | moderate (linear). | |
| `--n-layers` | +1 encoder transformer block per step. | Trunk depth. |
| `--n-dec-layers` | +1 attn-decoder block per step (`attn` only; ~+0.59 M at large). | Inert for other heads. |
| `--ff-mult` (not a flag; `ModelConfig.ff_mult=4`) | scales FFN width in every transformer layer. | Edit in code to change. |
| `--n-heads` | **zero param change** — only partitions `d_entity` into heads. | Must divide `d_entity`. Affects the attention factorization, not size. |
| `--policy-head` | `marginal`==`combo` (same-shape MLP scorer); `autoreg` +~20-35 %; **`attn` roughly 1.5-2.5× the marginal total** (adds a full transformer decoder). | Combo's power is enumeration, not params — hence same count as marginal. |
| `--dropout`, `--base-residual` | negligible / tiny (`base_residual` adds one LayerNorm). | Regularization / gradient-flow, not capacity. |

**Rule of thumb:** to scale capacity, move `--model` preset first (co-scales
d_entity/d_state/d_opt/n_layers sensibly), then fine-tune with `--d-entity`
(biggest single lever) and depth (`--n-layers`, `--n-dec-layers`). `--n-heads` is
free capacity-wise — tune it for the attention pattern, not the size.

The `attn` head config we'd likely train (large + 3 decoder layers) is **~3.31 M
params** — still small; inference cost, not param count, is the real deployment
constraint (one decoder pass per decision, cheap).

---

## Policy head — `--policy-head`

`--policy-head {marginal,autoreg,combo,attn}` · **default `marginal`**.
Chooses how the net turns the encoded state into a decision over the presented
legal options. **All four emit the same per-option `[B,L]` logits** from
`policy_from_state`/`forward`/`evaluate` (the "one meaning for [B,L]" doctrine),
so **MCTS priors, the ExIt cross-entropy target, and inference-time MCTS are
identical across heads** — the differences are how multi-select is modelled and
(for autoreg/combo) an extra capability used only by rollout sampling + the PPO
recompute. Changing the head changes params → part of the config hash; a
checkpoint is tied to its head.

| value | what it does | multi-select | extra sampling code |
|---|---|---|---|
| **marginal** (v1, default) | Scores each option independently with an MLP over `[option_vec, state, decision-ctx, referenced-entity]`. | left to the sampling layer (fixed-logit Plackett–Luce) | none |
| **autoreg** | Scores each pick **conditioned on the running set of already-picked options** + a learned **STOP** logit, so the count is learned (can stop once minCount met). | learned via STOP | `AutoregPolicyHead`; `policy.*_autoreg` |
| **combo** | Scores whole option **combinations** (unordered sets of size minCount..maxCount, empty set if minCount=0, **cap 64**) as one categorical; learns the count by picking a smaller set. agent_001's idea. Its `[B,L]` is the combo distribution **marginalized** to per-option inclusion logits. | learned via set size | `ComboPolicyHead`; `policy.*_combo` |
| **attn** *(new, see below)* | Transformer **decoder** over options: options **self-attend** (see each other) and **cross-attend to the board** tokens, then read out one logit each. Same `[B,L]` contract as marginal. | left to the sampling layer (reuses the marginal path) | none |

### `--n-dec-layers`
`--n-dec-layers <int>` · **default `2`**. Transformer-decoder depth for
`--policy-head attn` (number of option-self-attn + cross-attn-to-board layers).
**Inert for every other head.** In the config hash.

---

## Feature: `attn` — per-option transformer-decoder head *(added 2026-07-24)*

**What.** A fourth policy head, `--policy-head attn`, implemented as
`model.OptionDecoderHead`. The presented options become decoder tokens that
**self-attend** to each other and **cross-attend** to the encoder's per-entity
board tokens (`ent [B,12,d_entity]`), then a linear reads out one logit per
option → `[B,L]`.

**Why.** agent_000's *encoder* is already a set-transformer (CLS + self-attention
over the 12 board entities). What it lacked — and what agent_001 has — is a
transformer *decoder*: the marginal MLP scores each option in isolation from only
its single referenced entity. The `attn` head lets options be scored **jointly**
(each sees the rest of the slate) and against the **whole board** (cross-attention),
which is agent_001's board-attending decoder idea grafted onto our card/option
encoder. Tests `test_options_are_coupled_unlike_marginal` /
`test_logit_depends_on_board` lock in exactly this difference vs `marginal`.

**Why it was cheap (no refactor).**
- Reuses the existing encoder unchanged — `encode()` already returns the
  per-entity board tokens the decoder cross-attends to.
- Emits the standard `[B,L]` logits, so **sampling, PPO (logprob/entropy), MCTS,
  ExIt, pack, submit, and inference are all unchanged** — the head plugs into the
  *marginal* path verbatim (no `policy.py` / `ppo.py` / `agent.py` edits).
- Additive: new module + a branch in `PolicyValueModel.__init__` and
  `policy_from_state`, a `ModelConfig.n_dec_layers` field, and two CLI flags.

**Decoder width.** Runs in `d_model = d_entity` (option tokens are projected
`d_opt→d_entity`) so `--n-heads` must divide `d_entity` (true for all presets).
Uses pre-LN `nn.TransformerDecoderLayer`; a `nan_to_num` guards the degenerate
all-padded row before the caller's `MASK_FILL` (same finite-fill doctrine as the
other heads).

**Status.** Code + `test_attn_head.py` (16 tests) landed and green; **not yet
trained** — needs a fresh run (new head → new config hash; old checkpoints
incompatible), e.g.
`train --deck alakazam --model large --policy-head attn --n-dec-layers 3 …`.

---

## Future idea (not built): combo-level decoder

A **variant of the `attn` head whose decoder tokens are whole combinations**
instead of individual options — i.e. agent_001's combo-scoring decoder, but on
our encoder. It's the same transformer-decoder core as `attn`; only the token
set and a marginalize step differ. Sketch (all the pieces already exist):

1. `enumerate_combos(b)` → `combo_idx [B,C,K]`, `member_mask [B,C,K]`,
   `combo_valid [B,C]` (`model.py:521`). (C capped at 64.)
2. Build **combo tokens** by pooling member-option vectors: gather `opt` by
   `combo_idx`, mask with `member_mask`, sum/mean → `[B,C,d]` (same pooling the
   current `ComboPolicyHead.score` already does at `model.py:355-364`).
3. Run the **same decoder core** as `attn` (combo tokens self-attend +
   cross-attend to the board tokens) → `[B,C]` combo logits.
4. `_marginalize(...)` → `[B,L]` for the `[B,L]` contract (`model.py:534-558`),
   so MCTS/ExIt/inference stay unchanged.
5. Extra capability + sampling come for free via the existing `policy_combos` /
   `sample_action_combo` (`policy.py`).

Net: this is really **upgrading the current `combo` head's MLP scorer
(`self.combo.score`, `model.py:522`) into a transformer decoder** — it reuses the
combo head's enumerate/marginalize/sampler machinery and the `attn` head's
decoder block. More expressive (combinations attend to each other + the board)
but more machinery than `attn`; build `attn` first, then graft this on if the
per-option decoder proves out. Would be a new `policy_head` value (e.g.
`"attn_combo"`) in the config hash → its own retrain.

## How this maps to the code

- `config.py` — `ModelConfig` fields, `MODEL_PRESETS`, `build_model_config`, `build_model`.
- `encoder.py` — `CardEncoder` (shared hybrid card embedding), `StateEncoder` (CLS set-transformer → `state` + per-entity tokens).
- `model.py` — `OptionEncoder`, the heads (`AutoregPolicyHead`, `ComboPolicyHead`, `OptionDecoderHead`), `PolicyValueModel.policy_from_state` / `evaluate`.
- `cli.py` — `train` flags (`--model`, `--d-*/--n-*`, `--policy-head`, `--n-dec-layers`, `--base-residual`).
- Tests — `test_autoreg_head.py`, `test_combo_head.py`, `test_attn_head.py`.
