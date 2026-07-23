# rl — C++ MCTS self-play trainer

A **direct, 1:1 C++ port** of `engine/references/mcts.py` (the competition's
Transformer + MCTS + self-play RL sample), built on libtorch and the engine's
C ABI.

## Why C++ / how it talks to the engine

- The trainer is built under the **GCC/libstdc++** stdenv (to match
  `libtorch-bin`'s ABI). The engine `cg.so` is built under **libc++**.
- They communicate only through the engine's `extern "C"` boundary (POD structs
  + UTF-8 JSON strings) — exactly what Kaggle's `libcg.so` exports and what
  `pkm/engine/api.py` binds via ctypes. No STL crosses the boundary, so the
  ABI split is a non-issue. We declare the 13 symbols ourselves and link
  `cg.so` directly (no dlopen).
- The observation only exists as the engine's `ToJsonApi` JSON, so — like the
  Python reference's `to_observation_class` — we parse that JSON (nlohmann) into
  mirror structs. That is the faithful port, not a shortcut.

## Files

| File | Mirrors (mcts.py) |
|---|---|
| `obs.hpp` | `to_observation_class` + the observation dataclasses; enum values |
| `engine.hpp` | `cg.api` / `cg.game` bindings (battle + search) |
| `sparse.hpp` | `SparseVector`, `LearnInput` |
| `features.hpp` | `add_*`, `get_encoder_input`, `get_decoder_input`, `get_card` |
| `model.hpp` | `DecoderLayer`, `MyModel` (EmbeddingBag + Transformer) |
| `mcts.hpp` | `eval_nn`, `Node`/`Child`, `create_node`, `mcts_agent` |
| `train.cpp` | the `__main__` self-play/eval/train loop |

## Build & run

```bash
# CPU (fast to build, slow to run):
nix build .#torch-driver && ./result/bin/trainer

# CUDA (uses the 3090; ~2.5 GB libtorch download):
nix build .#torch-driver-cuda && ./result/bin/trainer
```

`trainer` writes `out/model<N>.pt` each iteration. The device is auto-selected
(`torch::cuda::is_available()`); the CUDA build lights up the GPU.

## Status / caveats

- Verified: compiles, links `cg.so` + libtorch, and runs the full loop
  end-to-end on CPU (real battles, MCTS search, self-play collection, model
  save).
- **Not** verified: training convergence across iterations — that needs a long
  run, which is what the CUDA build is for. This is a local-training tool; it is
  not part of any Kaggle submission.
