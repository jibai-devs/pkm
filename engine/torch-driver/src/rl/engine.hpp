// Direct bindings to the engine's extern "C" ABI (the same 13 symbols Kaggle's
// libcg.so exports), plus thin C++ wrappers mirroring pkm/engine/api.py.
//
// We declare the symbols ourselves and link cg.so directly — no dlopen, no
// ctypes. Only POD/pointers and UTF-8 JSON strings cross the boundary, so this
// is ABI-safe even though cg.so is built with libc++ and this driver with
// libstdc++ (see torch-driver/README).
#pragma once

#include <nlohmann/json.hpp>

#include <stdexcept>
#include <string>
#include <vector>

#include "obs.hpp"

namespace rl::engine {

// Return-by-value structs — field order/layout must match engine/src/api/Api.h.
extern "C" {
struct StartData {
  void* battlePtr;
  int errorPlayer;
  int errorType;
};
struct SerialData {
  const char8_t* json;
  const char* data;
  int count;
  int selectPlayer;
};

void GameInitialize();
StartData BattleStart(int* cards);
void* AgentStart();
void BattleFinish(void* data);
SerialData GetBattleData(void* data);
int Select(void* data, int* select, int selectCount);
const char8_t* SearchBegin(void* data, const char* serialized, int count,
                           int* myDeck, int* myPrize, int* enemyDeck,
                           int* enemyPrize, int* enemyHand, int* enemyActive,
                           int manualCoin);
const char8_t* SearchStep(void* data, long long searchId, int* select,
                          int selectCount);
void SearchEnd(void* data);
void SearchRelease(void* data, long long searchId);
const char8_t* AllCard();
const char8_t* AllAttack();
}  // extern "C"

inline const char* cstr(const char8_t* s) {
  return reinterpret_cast<const char*>(s);
}

// --- card / attack metadata -------------------------------------------------
inline nlohmann::json all_cards() {
  return nlohmann::json::parse(cstr(AllCard()));
}
inline nlohmann::json all_attacks() {
  return nlohmann::json::parse(cstr(AllAttack()));
}

// --- a single live battle + a persistent search agent -----------------------
// Mirrors pkm/engine/api.py's module-level Battle + _agent_ptr state, but
// instance-scoped so nothing is global.
class Engine {
 public:
  Engine() {
    GameInitialize();
    agent_ptr_ = AgentStart();
  }

  // Start a battle from two 60-card decks. Returns the first observation.
  Observation battle_start(const std::vector<int>& deck0,
                           const std::vector<int>& deck1) {
    if (deck0.size() != 60 || deck1.size() != 60)
      throw std::invalid_argument("The deck must contain 60 cards.");
    std::vector<int> cards = deck0;
    cards.insert(cards.end(), deck1.begin(), deck1.end());
    StartData sd = BattleStart(cards.data());
    battle_ptr_ = sd.battlePtr;
    if (!battle_ptr_) throw std::runtime_error("BattleStart failed (deck error)");
    return get_battle_data();
  }

  // Apply option indices; return the next observation.
  Observation battle_select(const std::vector<int>& select) {
    std::vector<int> sel = select;
    int err = Select(battle_ptr_, sel.data(), static_cast<int>(sel.size()));
    if (err != 0) throw std::runtime_error("Select error " + std::to_string(err));
    return get_battle_data();
  }

  void battle_finish() { BattleFinish(battle_ptr_); battle_ptr_ = nullptr; }

  // --- search (forward simulation) ------------------------------------------
  SearchState search_begin(const Observation& obs,
                           const std::vector<int>& your_deck,
                           const std::vector<int>& your_prize,
                           const std::vector<int>& opponent_deck,
                           const std::vector<int>& opponent_prize,
                           const std::vector<int>& opponent_hand,
                           const std::vector<int>& opponent_active,
                           bool manual_coin = false) {
    if (obs.search_begin_input.empty())
      throw std::runtime_error("observation has no search_begin_input");
    // Copy: the engine's serialized blob may be overwritten by the next call.
    std::string sbi = obs.search_begin_input;
    auto vec = [](const std::vector<int>& v) {
      // SearchBegin takes int*; make a mutable copy (never mutated by engine).
      return std::vector<int>(v);
    };
    std::vector<int> yd = vec(your_deck), yp = vec(your_prize),
                     od = vec(opponent_deck), op = vec(opponent_prize),
                     oh = vec(opponent_hand), oa = vec(opponent_active);
    const char8_t* raw = SearchBegin(
        agent_ptr_, sbi.data(), static_cast<int>(sbi.size()), yd.data(),
        yp.data(), od.data(), op.data(), oh.data(), oa.data(), manual_coin ? 1 : 0);
    return parse_search(cstr(raw));
  }

  SearchState search_step(long long search_id, const std::vector<int>& select) {
    std::vector<int> sel = select;
    const char8_t* raw = SearchStep(agent_ptr_, search_id, sel.data(),
                                    static_cast<int>(sel.size()));
    return parse_search(cstr(raw));
  }

  void search_end() { SearchEnd(agent_ptr_); }
  void search_release(long long id) { SearchRelease(agent_ptr_, id); }

 private:
  Observation get_battle_data() {
    SerialData sd = GetBattleData(battle_ptr_);
    auto j = nlohmann::json::parse(cstr(sd.json));
    Observation obs = Observation::parse(j);
    // search_begin_input = raw base64 blob (ascii), copied out immediately.
    if (sd.data && sd.count > 0)
      obs.search_begin_input.assign(sd.data, static_cast<size_t>(sd.count));
    return obs;
  }

  static SearchState parse_search(const char* raw) {
    auto result = nlohmann::json::parse(raw);
    int error = result.value("error", 0);
    if (error != 0)
      throw std::runtime_error("Search error " + std::to_string(error));
    SearchState ss;
    const auto& st = result.at("state");
    ss.searchId = st.value("searchId", 0LL);
    ss.observation = Observation::parse(st.at("observation"));
    return ss;
  }

  void* battle_ptr_ = nullptr;
  void* agent_ptr_ = nullptr;
};

}  // namespace rl::engine
