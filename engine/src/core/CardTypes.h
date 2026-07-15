// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and
// character names are trademarks of Nintendo. SPDX-License-Identifier:
// LicenseRef-PTCG-ABC-Competition-Use-Only Part of the Pokémon TCG AI Battle
// Challenge. Provided for Competition use only; the full license is in the
// LICENSES/ folder and incorporates the Competition Rules. Competition Rules:
// https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

enum class PokemonType : unsigned char {
  NotPokemon, // not pokemon
  Normal,
  PokemonItem, // 化石や人形。明示的に書かれていなければ対戦準備では出せない
  Ex,          // Pokemon ex
  MegaEx,      // メガシンカex
};

enum class AreaType : unsigned char {
  All,
  Deck,
  Hand,
  Trash,
  Active, // バトル場
  Bench,
  Prize,
  Stadium,
  Energy,       // 場のエネルギー
  Tool,         // 場のどうぐ
  PreEvolution, // 進化前
  Player,
  Looking,
  Playing,    // Playing item and supporter
  DeckBottom, // デッキの下。移動元では使わない

  Me,
  Effected,
  EffectedPreTarget, // 前の前の対象
  SelectedList,      // selectedList
  TriggerSubject,
  TriggerObject,
  Attach,          // 付いているポケモン
  TurnPlay,        // この番に使ったカード
  AttackPreMyTurn, // 前の自分のターンにワザを使ったポケモン
  Temporary,       // 一時エリア
};

// 特殊状態
enum class BadStatusType : unsigned char {
  None,      // no bad status
  Asleep,    // ねむり
  Paralyzed, // マヒ
  Confused,  // こんらん
};

// 特殊状態の選択
enum class SelectSpecialConditionType : unsigned char {
  Poison,   // どく
  Burn,     // やけど
  Sleep,    // ねむり
  Paralyze, // マヒ
  Confuse,  // こんらん
};

enum class CardType : unsigned char {
  Pokemon,
  Item, // グッズ
  Tool, // ポケモンのどうぐ
  Supporter,
  Stadium,
  BasicEnergy,   // 基本エネルギー
  SpecialEnergy, // 特殊エネルギー
};

inline constexpr bool IsTrainer(CardType type) {
  return type == CardType::Item || type == CardType::Tool ||
         type == CardType::Supporter || type == CardType::Stadium;
}

inline constexpr bool IsEnergy(CardType type) {
  return type == CardType::BasicEnergy || type == CardType::SpecialEnergy;
}

enum class EvolutionType : unsigned char {
  NoEvolutionType, // ポケモンでない
  Basic,           // たね
  Stage1,          // 1進化
  Stage2,          // 2進化
};
