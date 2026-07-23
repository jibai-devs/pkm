// Sparse feature builders — direct port of the add_* / get_encoder_input /
// get_decoder_input functions in references/mcts.py.
//
// Dimension globals (card_count / attack_count) are resolved once from engine
// card data via init_dims(), mirroring the module-level globals in the Python.
#pragma once

#include <algorithm>
#include <optional>
#include <vector>

#include "model.hpp"   // FeatureDims + layout constants
#include "obs.hpp"
#include "sparse.hpp"

namespace rl {

// Set once at startup from AllCard()/AllAttack() (see train.cpp).
inline FeatureDims g_dims;
inline void init_dims(int64_t card_count, int64_t attack_count) {
  g_dims.card_count = card_count;
  g_dims.attack_count = attack_count;
}
inline int card_count() { return static_cast<int>(g_dims.card_count); }

// --- encoder ----------------------------------------------------------------
inline void add_card(SparseVector& sv, const Slot& card) {
  if (card) sv.add(card->id, 1.0f);
  sv.add_pos(card_count());
}

inline void add_cards(SparseVector& sv, const std::vector<Slot>& cards,
                      float value) {
  for (const auto& c : cards)
    if (c) sv.add(c->id, value);
  sv.add_pos(card_count());
}

// Overload for present-only card lists (Pokémon tools/energyCards).
inline void add_cards(SparseVector& sv, const std::vector<Card>& cards,
                      float value) {
  for (const auto& c : cards) sv.add(c.id, value);
  sv.add_pos(card_count());
}

inline void add_pokemon(SparseVector& sv, const Slot& poke) {
  if (!poke) {
    sv.add_single(1.0f);
    sv.add_pos(1 + 3 * card_count());
  } else {
    sv.add_single(0.0f);
    sv.add_single(static_cast<float>(poke->hp / 400.0));
    add_card(sv, poke);
    add_cards(sv, poke->tools, 1.0f);
    add_cards(sv, poke->energyCards, 0.5f);
  }
}

inline void add_player(SparseVector& sv, const Player& ps) {
  sv.add_single(ps.deckCount / 60.0f);
  sv.add_single(static_cast<float>(ps.discard.size()) / 60.0f);
  sv.add_single(ps.handCount / 8.0f);
  sv.add_single(static_cast<float>(ps.bench.size()) / 5.0f);
  sv.add(static_cast<int>(ps.prize.size()), 1.0f);
  sv.add_pos(7);

  sv.add_single(ps.poisoned ? 1.0f : 0.0f);
  sv.add_single(ps.burned ? 1.0f : 0.0f);
  sv.add_single(ps.asleep ? 1.0f : 0.0f);
  sv.add_single(ps.paralyzed ? 1.0f : 0.0f);
  sv.add_single(ps.confused ? 1.0f : 0.0f);

  add_cards(sv, ps.discard, 0.25f);
}

inline SparseVector get_encoder_input(const Observation& obs,
                                      const std::vector<int>& your_deck) {
  const int your_index = obs.current.yourIndex;
  const GameState& state = obs.current;

  SparseVector sv;
  for (int i = 0; i < 2; ++i) {
    const Player& ps = state.players[i ^ your_index];
    for (int j = 0; j < 8; ++j) {  // bench slots
      sv.word_start();
      const int pos = sv.pos;
      if (j < static_cast<int>(ps.bench.size()))
        add_pokemon(sv, ps.bench[j]);
      else
        add_pokemon(sv, std::nullopt);
      if (j != 7) sv.pos = pos;  // not last -> rewind
    }
  }

  for (int i = 0; i < 2; ++i) {
    const Player& ps = state.players[i ^ your_index];
    sv.word_start();
    if (!ps.active.empty())
      add_pokemon(sv, ps.active[0]);
    else
      add_pokemon(sv, std::nullopt);
  }

  for (int i = 0; i < 2; ++i) {
    const Player& ps = state.players[i ^ your_index];
    sv.word_start();
    add_player(sv, ps);
  }

  sv.word_start();
  add_cards(sv, state.players[your_index].hand, 0.25f);

  sv.word_start();
  for (int id : your_deck) sv.add(id, 0.25f);
  sv.add_pos(card_count());

  sv.word_start();
  add_cards(sv, state.stadium, 1.0f);

  sv.word_start();
  sv.add_single(1.0f);
  sv.add_single(state.turn / 10.0f);
  sv.add_single(state.firstPlayer == your_index ? 1.0f : 0.0f);
  return sv;
}

// --- decoder ----------------------------------------------------------------
inline Slot get_card(const Observation& obs, int area_type, int index,
                     int player_index) {
  auto at = [](const std::vector<Slot>& v, int i) -> Slot {
    if (i < 0 || i >= static_cast<int>(v.size())) return std::nullopt;
    return v[i];
  };
  const Player& ps = obs.current.players[player_index];
  switch (area_type) {
    case area::Deck:
      return obs.select.hasDeck ? at(obs.select.deck, index) : std::nullopt;
    case area::Hand:
      return at(ps.hand, index);
    case area::Trash:
      return at(ps.discard, index);
    case area::Active:
      return at(ps.active, index);
    case area::Bench:
      return at(ps.bench, index);
    case area::Prize:
      return at(ps.prize, index);
    case area::Stadium:
      return at(obs.current.stadium, index);
    case area::Looking:
      return at(obs.current.looking, index);
    default:
      return std::nullopt;
  }
}

inline void decoder_main(SparseVector& sv, int feature_index, const Slot& card) {
  if (card)
    sv.add(g_dims.decoder_card_offset() + feature_index * card_count() + card->id,
           1.0f);
}

inline void decoder_card_id(SparseVector& sv, int context, int card_id) {
  sv.add(g_dims.decoder_card_offset() +
             (kDecoderMainFeature + context) * card_count() + card_id,
         1.0f);
}

inline void decoder_card(SparseVector& sv, int context, const Slot& card) {
  if (card) decoder_card_id(sv, context, card->id);
}

inline SparseVector get_decoder_input(
    const Observation& obs, const std::vector<std::vector<int>>& actions) {
  SparseVector sv;
  const int your_index = obs.current.yourIndex;
  const Player& ps = obs.current.players[your_index];
  const int context = obs.select.context;

  for (const auto& action : actions) {
    sv.word_start();
    if (action.empty()) {
      sv.add(0, 1.0f);
      continue;
    }
    for (int i : action) {
      const Option& o = obs.select.option[i];
      switch (o.type) {
        case opt::End:
          sv.add(1, 1.0f);
          break;
        case opt::Yes:
          sv.add(2, 1.0f);
          break;
        case opt::No:
          sv.add(3, 1.0f);
          break;
        case opt::SpecialCondition:
          sv.add(4 + o.specialConditionType, 1.0f);
          break;
        case opt::Number:
          sv.add(9 + std::min(o.number, 4), 1.0f);
          break;
        case opt::Attack:
          sv.add(static_cast<int>(kDecoderAttackOffset) + o.attackId, 1.0f);
          break;
        case opt::Play:
          decoder_main(sv, 0, (o.index >= 0 && o.index < (int)ps.hand.size())
                                  ? ps.hand[o.index]
                                  : std::nullopt);
          break;
        case opt::Attach:
          decoder_main(sv, 1, get_card(obs, o.area, o.index, your_index));
          decoder_main(sv, 2,
                       get_card(obs, o.inPlayArea, o.inPlayIndex, your_index));
          break;
        case opt::Evolve:
          decoder_main(sv, 3, get_card(obs, o.area, o.index, your_index));
          decoder_main(sv, 4,
                       get_card(obs, o.inPlayArea, o.inPlayIndex, your_index));
          break;
        case opt::Ability:
          decoder_main(sv, 5, get_card(obs, o.area, o.index, your_index));
          break;
        case opt::Discard:
          decoder_main(sv, 6, get_card(obs, o.area, o.index, your_index));
          break;
        case opt::Retreat:
          decoder_main(sv, 7, ps.active.empty() ? std::nullopt : ps.active[0]);
          break;
        case opt::Card:
          decoder_card(sv, context,
                       get_card(obs, o.area, o.index, o.playerIndex));
          break;
        case opt::ToolCard: {
          Slot card = get_card(obs, o.area, o.index, o.playerIndex);
          if (card && o.toolIndex >= 0 &&
              o.toolIndex < (int)card->tools.size())
            decoder_card(sv, context, card->tools[o.toolIndex]);
          break;
        }
        case opt::EnergyCard:
        case opt::Energy: {
          Slot card = get_card(obs, o.area, o.index, o.playerIndex);
          if (card && o.energyIndex >= 0 &&
              o.energyIndex < (int)card->energyCards.size())
            decoder_card(sv, context, card->energyCards[o.energyIndex]);
          break;
        }
        case opt::Skill:
          decoder_card_id(sv, context, o.cardId);
          break;
        default:
          break;
      }
    }
  }
  return sv;
}

}  // namespace rl
