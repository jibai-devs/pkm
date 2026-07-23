// C++ mirror of the engine observation, parsed from the ToJsonApi JSON with
// nlohmann/json. Faithful to pkm/types/obs.py / references/mcts.py's
// to_observation_class: lenient (missing key -> default, null -> "None").
//
// Only the fields the reference actually reads are modelled; everything else is
// tolerated and dropped (extra="allow" semantics).
#pragma once

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

namespace rl {

using json = nlohmann::json;

// --- enum values (wire values; see AGENTS.md / the schema map) -------------
// SelectOptionType — NOT offset on the wire.
namespace opt {
inline constexpr int Number = 0;
inline constexpr int Yes = 1;
inline constexpr int No = 2;
inline constexpr int Card = 3;
inline constexpr int ToolCard = 4;
inline constexpr int EnergyCard = 5;
inline constexpr int Energy = 6;
inline constexpr int Play = 7;
inline constexpr int Attach = 8;
inline constexpr int Evolve = 9;
inline constexpr int Ability = 10;
inline constexpr int Discard = 11;
inline constexpr int Retreat = 12;
inline constexpr int Attack = 13;
inline constexpr int End = 14;
inline constexpr int Skill = 15;
inline constexpr int SpecialCondition = 16;
}  // namespace opt

// AreaType — NOT offset. Trash(3) is the discard pile.
namespace area {
inline constexpr int Deck = 1;
inline constexpr int Hand = 2;
inline constexpr int Trash = 3;  // discard
inline constexpr int Active = 4;
inline constexpr int Bench = 5;
inline constexpr int Prize = 6;
inline constexpr int Stadium = 7;
inline constexpr int Looking = 12;
}  // namespace area

// SelectContext — wire value (raw C++ minus 1). RecoverSpecialCondition == 48.
inline constexpr int kRecoverSpecialCondition = 48;

// --- small json helpers -----------------------------------------------------
inline int jint(const json& j, const char* key, int def = 0) {
  auto it = j.find(key);
  if (it == j.end() || it->is_null()) return def;
  return it->get<int>();
}
inline double jdouble(const json& j, const char* key, double def = 0.0) {
  auto it = j.find(key);
  if (it == j.end() || it->is_null()) return def;
  return it->get<double>();
}
inline bool jbool(const json& j, const char* key, bool def = false) {
  auto it = j.find(key);
  if (it == j.end() || it->is_null()) return def;
  return it->get<bool>();
}

// --- structs ----------------------------------------------------------------
// A card reference or a full Pokémon. `id` is the cardId. Pokémon-only fields
// (hp/tools/energyCards) stay empty/zero for plain cards.
struct Card {
  int id = 0;
  double hp = 0.0;
  std::vector<Card> tools;
  std::vector<Card> energyCards;

  static Card parse(const json& j) {
    Card c;
    c.id = jint(j, "id");
    c.hp = jdouble(j, "hp");
    if (auto it = j.find("tools"); it != j.end() && it->is_array()) {
      for (const auto& t : *it)
        if (!t.is_null()) c.tools.push_back(parse(t));
    }
    if (auto it = j.find("energyCards"); it != j.end() && it->is_array()) {
      for (const auto& e : *it)
        if (!e.is_null()) c.energyCards.push_back(parse(e));
    }
    return c;
  }
};

// A board slot: either a card/Pokémon or "None" (face-down / empty).
using Slot = std::optional<Card>;

inline std::vector<Slot> parse_slots(const json& j, const char* key) {
  std::vector<Slot> out;
  auto it = j.find(key);
  if (it == j.end() || it->is_null()) return out;
  for (const auto& e : *it) {
    if (e.is_null())
      out.emplace_back(std::nullopt);
    else
      out.emplace_back(Card::parse(e));
  }
  return out;
}

struct Player {
  std::vector<Slot> active;
  std::vector<Slot> bench;
  std::vector<Slot> hand;     // empty when hidden (opponent)
  std::vector<Slot> discard;  // from "discard" (ps.trash)
  std::vector<Slot> prize;
  int deckCount = 0;
  int handCount = 0;
  bool poisoned = false, burned = false, asleep = false, paralyzed = false,
       confused = false;

  static Player parse(const json& j) {
    Player p;
    p.active = parse_slots(j, "active");
    p.bench = parse_slots(j, "bench");
    p.hand = parse_slots(j, "hand");
    p.discard = parse_slots(j, "discard");
    p.prize = parse_slots(j, "prize");
    p.deckCount = jint(j, "deckCount");
    p.handCount = jint(j, "handCount");
    p.poisoned = jbool(j, "poisoned");
    p.burned = jbool(j, "burned");
    p.asleep = jbool(j, "asleep");
    p.paralyzed = jbool(j, "paralyzed");
    p.confused = jbool(j, "confused");
    return p;
  }
};

struct GameState {
  int turn = 0;
  int yourIndex = 0;
  int firstPlayer = -1;
  int result = -1;
  std::vector<Slot> stadium;
  std::vector<Slot> looking;
  std::vector<Player> players;  // exactly 2

  static GameState parse(const json& j) {
    GameState s;
    s.turn = jint(j, "turn");
    s.yourIndex = jint(j, "yourIndex");
    s.firstPlayer = jint(j, "firstPlayer", -1);
    s.result = jint(j, "result", -1);
    s.stadium = parse_slots(j, "stadium");
    s.looking = parse_slots(j, "looking");
    if (auto it = j.find("players"); it != j.end() && it->is_array())
      for (const auto& p : *it) s.players.push_back(Player::parse(p));
    return s;
  }
};

struct Option {
  int type = 0;
  int index = 0;
  int area = 0;
  int playerIndex = 0;
  int inPlayArea = 0;
  int inPlayIndex = 0;
  int attackId = 0;
  int number = 0;
  int specialConditionType = 0;
  int cardId = 0;
  int toolIndex = 0;
  int energyIndex = 0;

  static Option parse(const json& j) {
    Option o;
    o.type = jint(j, "type");
    o.index = jint(j, "index");
    o.area = jint(j, "area");
    o.playerIndex = jint(j, "playerIndex");
    o.inPlayArea = jint(j, "inPlayArea");
    o.inPlayIndex = jint(j, "inPlayIndex");
    o.attackId = jint(j, "attackId");
    o.number = jint(j, "number");
    o.specialConditionType = jint(j, "specialConditionType");
    o.cardId = jint(j, "cardId");
    o.toolIndex = jint(j, "toolIndex");
    o.energyIndex = jint(j, "energyIndex");
    return o;
  }
};

struct Select {
  int type = 0;
  int context = 0;
  int minCount = 0;
  int maxCount = 0;
  std::vector<Option> option;
  bool hasDeck = false;
  std::vector<Slot> deck;

  static Select parse(const json& j) {
    Select s;
    s.type = jint(j, "type");
    s.context = jint(j, "context");
    s.minCount = jint(j, "minCount");
    s.maxCount = jint(j, "maxCount");
    if (auto it = j.find("option"); it != j.end() && it->is_array())
      for (const auto& o : *it) s.option.push_back(Option::parse(o));
    if (auto it = j.find("deck"); it != j.end() && !it->is_null()) {
      s.hasDeck = true;
      s.deck = parse_slots(j, "deck");
    }
    return s;
  }
};

struct Observation {
  GameState current;
  Select select;
  std::string search_begin_input;

  static Observation parse(const json& j) {
    Observation o;
    if (auto it = j.find("current"); it != j.end() && !it->is_null())
      o.current = GameState::parse(*it);
    if (auto it = j.find("select"); it != j.end() && !it->is_null())
      o.select = Select::parse(*it);
    if (auto it = j.find("search_begin_input");
        it != j.end() && it->is_string())
      o.search_begin_input = it->get<std::string>();
    return o;
  }
};

// Result of a search_begin/step call: an observation + its search node id.
struct SearchState {
  Observation observation;
  long long searchId = 0;
};

}  // namespace rl
