// MCTS + eval_nn — direct port of references/mcts.py (Node/Child/create_node/
// mcts_agent). Uses the engine search API for forward simulation and the
// Transformer for value/policy priors.
#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "engine.hpp"
#include "features.hpp"
#include "model.hpp"
#include "obs.hpp"
#include "sparse.hpp"

namespace rl {

inline int g_search_count = 10;  // SEARCH_COUNT (runtime-configurable via CLI)

// Process-wide RNG (varied by seed at startup). std::random_device seed.
inline std::mt19937& rng() {
  static std::mt19937 gen{std::random_device{}()};
  return gen;
}

// Pick k values from `deck` without replacement (random.sample semantics).
inline std::vector<int> sample_k(const std::vector<int>& deck, int k) {
  std::vector<int> idx(deck.size());
  for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int>(i);
  std::shuffle(idx.begin(), idx.end(), rng());
  std::vector<int> out;
  out.reserve(k);
  for (int i = 0; i < k && i < (int)idx.size(); ++i) out.push_back(deck[idx[i]]);
  return out;
}

// Training sample: value target + per-action policy target + the sparse inputs.
struct LearnSample {
  float value;
  std::vector<float> policy;
  SparseVector sv_enc;
  SparseVector sv_dec;
};

// Evaluate the model on a single (encoder, decoder) input pair.
inline std::pair<float, std::vector<float>> eval_nn(const SparseVector& enc,
                                                    const SparseVector& dec,
                                                    MyModel& model,
                                                    torch::Device device) {
  auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(device);
  auto f32 = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto [value, policy] = model->forward(
      torch::tensor(enc.index, i32), torch::tensor(enc.value, f32),
      torch::tensor(enc.offset, i32), torch::tensor(dec.index, i32),
      torch::tensor(dec.value, f32), torch::tensor(dec.offset, i32));
  float v = value.cpu()[0][0].item<float>();
  auto prow = policy.cpu()[0].contiguous();
  std::vector<float> pol(prow.data_ptr<float>(),
                         prow.data_ptr<float>() + prow.numel());
  return {v, pol};
}

struct Node;

// A child edge: an action (selected option indices) + prior probability.
struct Child {
  std::unique_ptr<Node> node;  // owned lazily on expansion
  std::vector<int> select;
  double prob;
  Child(std::vector<int> s, double p) : select(std::move(s)), prob(p) {}
};

struct Node {
  double value = -2.0;
  double total = 0.0;
  int visit = 0;
  Node* parent = nullptr;
  std::vector<Child> children;
  SearchState state;

  Node(Node* parent_, SearchState state_)
      : parent(parent_), state(std::move(state_)) {}

  void backprop(double v) {
    total += v;
    visit += 1;
    if (parent) parent->backprop(v);
  }
};

// Enumerate up to 64 combinations of size maxCount from range(len(option)).
inline std::vector<std::vector<int>> enumerate_actions(int max_count,
                                                       int option_count) {
  std::vector<std::vector<int>> actions;
  std::vector<int> indices(max_count);
  for (int i = 0; i < max_count; ++i) indices[i] = i;
  for (int iter = 0; iter < 64; ++iter) {
    actions.push_back(indices);
    bool advanced = false;
    const int n = static_cast<int>(indices.size());
    for (int i = 0; i < n; ++i) {
      const int index = n - i - 1;
      if (indices[index] < option_count - i - 1) {
        indices[index] += 1;
        for (int j = index + 1; j < n; ++j) indices[j] = indices[j - 1] + 1;
        advanced = true;
        break;
      }
    }
    if (!advanced) break;
  }
  return actions;
}

// Create (and evaluate) an MCTS node. Returns the node plus a training sample
// (null for terminal states).
inline std::pair<std::unique_ptr<Node>, std::unique_ptr<LearnSample>> create_node(
    Node* parent, SearchState search_state, int your_index,
    const std::vector<int>& your_deck, MyModel& model, torch::Device device) {
  auto node = std::make_unique<Node>(parent, std::move(search_state));
  const Observation& obs = node->state.observation;
  const GameState& state = obs.current;

  std::unique_ptr<LearnSample> sample;
  if (state.result >= 0) {
    if (state.result == 2)
      node->value = 0;
    else if (state.result == your_index)
      node->value = 1;
    else
      node->value = -1;
    node->backprop(node->value);
  } else {
    auto actions = enumerate_actions(obs.select.maxCount,
                                     static_cast<int>(obs.select.option.size()));
    SparseVector sv_enc = get_encoder_input(obs, your_deck);
    SparseVector sv_dec = get_decoder_input(obs, actions);
    auto [value, policy] = eval_nn(sv_enc, sv_dec, model, device);

    double v = value;
    if (state.yourIndex != your_index) v = -v;
    node->value = v;
    node->backprop(v);

    double sum = 0.0;
    const int n = std::min(policy.size(), actions.size());
    for (int i = 0; i < n; ++i) {
      double p = std::exp(policy[i] * 10.0);
      node->children.emplace_back(actions[i], p);
      sum += p;
    }
    for (auto& c : node->children) c.prob /= sum;

    sample = std::make_unique<LearnSample>(
        LearnSample{value, std::move(policy), std::move(sv_enc), std::move(sv_dec)});
  }
  return {std::move(node), std::move(sample)};
}

// Run MCTS from the current battle observation and return (selected option
// indices, training sample).
inline std::pair<std::vector<int>, std::unique_ptr<LearnSample>> mcts_agent(
    engine::Engine& eng, const Observation& obs,
    const std::vector<int>& your_deck, MyModel& model, torch::Device device) {
  const int your_index = obs.current.yourIndex;
  const GameState& state = obs.current;
  const Player& me = state.players[your_index];
  const Player& opp = state.players[1 - your_index];
  const auto& active = opp.active;

  std::vector<int> opponent_active;
  if (!active.empty() && !active[0].has_value()) opponent_active = {1072};

  SearchState root_state = eng.search_begin(
      obs,
      /*your_deck=*/sample_k(your_deck, me.deckCount),
      /*your_prize=*/sample_k(your_deck, static_cast<int>(me.prize.size())),
      /*opponent_deck=*/std::vector<int>(opp.deckCount, 1072),
      /*opponent_prize=*/std::vector<int>(opp.prize.size(), 1),
      /*opponent_hand=*/std::vector<int>(opp.handCount, 1),
      /*opponent_active=*/opponent_active);

  auto [root_ptr, sample] =
      create_node(nullptr, std::move(root_state), your_index, your_deck, model, device);
  Node* root = root_ptr.get();

  for (int s = 0; s < g_search_count; ++s) {
    Node* current = root;
    while (true) {
      double best = -1e9;
      const double c = 0.4 * std::sqrt(static_cast<double>(current->visit));
      Child* next = nullptr;
      for (auto& child : current->children) {
        int visit = 0;
        double v;
        if (!child.node) {
          v = current->total / current->visit;
        } else {
          v = child.node->total / child.node->visit;
          visit = child.node->visit;
        }
        if (current->state.observation.current.yourIndex != your_index) v = -v;
        v += c * child.prob / (1 + visit);
        if (best < v) {
          best = v;
          next = &child;
        }
      }
      if (!next) break;  // terminal-with-no-children guard

      if (!next->node) {
        SearchState ss = eng.search_step(current->state.searchId, next->select);
        auto [child_node, _] =
            create_node(current, std::move(ss), your_index, your_deck, model, device);
        next->node = std::move(child_node);
        break;
      } else {
        current = next->node.get();
        if (current->state.observation.current.result >= 0) {
          current->backprop(current->value);
          break;
        }
      }
    }
  }

  // Most-visited child + minimum child value (for policy targets).
  Child* max_child = nullptr;
  int max_visit = -1;
  double min_value = 10.0;
  for (auto& child : root->children) {
    if (child.node) {
      if (max_visit < child.node->visit) {
        max_child = &child;
        max_visit = child.node->visit;
      }
      double v = child.node->total / child.node->visit;
      if (min_value > v) min_value = v;
    }
  }

  // Training targets.
  if (sample) {
    sample->value = static_cast<float>(root->total / root->visit);
    for (size_t i = 0; i < root->children.size() && i < sample->policy.size();
         ++i) {
      Child& child = root->children[i];
      double v;
      if (!child.node)
        v = min_value - sample->value - 0.03;
      else
        v = child.node->total / child.node->visit - sample->value;
      sample->policy[i] = static_cast<float>(std::max(-1.0, std::min(1.0, v)));
    }
  }

  eng.search_end();
  std::vector<int> selected = max_child ? max_child->select : std::vector<int>{};
  return {std::move(selected), std::move(sample)};
}

}  // namespace rl
