// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and
// character names are trademarks of Nintendo. SPDX-License-Identifier:
// LicenseRef-PTCG-ABC-Competition-Use-Only Part of the Pokémon TCG AI Battle
// Challenge. Provided for Competition use only; the full license is in the
// LICENSES/ folder and incorporates the Competition Rules. Competition Rules:
// https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include <cassert>

// 能力発動タイミング
enum class TriggerType : unsigned char {
  None,
  TurnEnd,
  PokemonCheckup,
  HandToBench,
  ActiveToBench,
  ToBenchMyTurn, // バトル場からベンチは含まない
  BenchToActive,
  DeckToTrashEnemyEffect, // 相手のワザ・特性・グッズ・サポートの効果で山札からトラッシュされたとき
  EvolveFromHand,
  EnergyAttachFromHand, // 主体はポケモン
  DamagedEnemyAttack,   // 相手のポケモンからワザのダメージを受けたとき
  DamagedEnemyAttackActive, // バトル場で相手のポケモンからワザのダメージを受けたとき
  PreKo,                    // ワザのダメージを受けて【きぜつ】するとき
  PreKoFull, // HPがまんたんの状態で、ワザのダメージを受けて【きぜつ】するとき
  PreKoFullEnemy, // HPがまんたんの状態で、相手のポケモンからワザのダメージを受けて【きぜつ】するとき
  Ko,
  KoEnemyAttackDamage,
  KoEnemyExAttackDamage, // 相手の「ポケモン【ex】」からワザのダメージを受けて【きぜつ】したとき
  KoEnemyAttackDamageActive, // バトル場で相手のポケモンからワザのダメージを受けて【きぜつ】したとき。ワザのダメージをバトル場で受けてHP0になってさえいれば、きぜつのタイミングでベンチにいてもいい
  PreRetreat,                // 【にげる】とき
  Attach,                    // PullTriggerを介さない
};

// Comparator types
enum class ComparatorType : unsigned char {
  Equal,
  GreaterEqual,
  LessEqual,
  NotEqual,
  Greater,
  Less,
};

inline bool Compare(int val, int effectVal, ComparatorType type) {
  switch (type) {
  case ComparatorType::Equal:
    return val == effectVal;
  case ComparatorType::GreaterEqual:
    return val >= effectVal;
  case ComparatorType::LessEqual:
    return val <= effectVal;
  case ComparatorType::NotEqual:
    return val != effectVal;
  case ComparatorType::Greater:
    return val > effectVal;
  case ComparatorType::Less:
    return val < effectVal;
  default:
    assert(false);
    return false;
  }
}

inline constexpr bool BoolCompare(bool flag, ComparatorType type) {
  if (type == ComparatorType::Equal) {
    return flag;
  } else {
    return !flag;
  }
}

enum class SkillType : unsigned char {
  Ability,
  Play,
  Attach,
};
