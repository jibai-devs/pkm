// SparseVector / LearnInput — direct port of the EmbeddingBag input builders
// from references/mcts.py. Pure containers, no engine dependency.
#pragma once

#include <cstdint>
#include <vector>

namespace rl {

// torch.nn.EmbeddingBag input (index / per-sample value / offset), plus a
// running write position so feature blocks can be laid out sequentially.
struct SparseVector {
  std::vector<int32_t> index;
  std::vector<float> value;
  std::vector<int32_t> offset;
  int32_t pos = 0;

  void add(int32_t idx, float v) {
    if (v != 0.0f) {
      index.push_back(pos + idx);
      value.push_back(v);
    }
  }

  void add_pos(int32_t n) { pos += n; }

  void add_single(float v) {
    if (v != 0.0f) {
      index.push_back(pos);
      value.push_back(v);
    }
    pos += 1;
  }

  void word_start() { offset.push_back(static_cast<int32_t>(index.size())); }
};

// Batch builder: concatenates several SparseVectors, rebasing their offsets.
struct LearnInput {
  std::vector<int32_t> index;
  std::vector<float> value;
  std::vector<int32_t> offset;

  void add(const SparseVector& sv) {
    const int32_t count = static_cast<int32_t>(index.size());
    index.insert(index.end(), sv.index.begin(), sv.index.end());
    value.insert(value.end(), sv.value.begin(), sv.value.end());
    for (int32_t o : sv.offset) offset.push_back(o + count);
  }
};

}  // namespace rl
