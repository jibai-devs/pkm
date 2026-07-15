// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and
// character names are trademarks of Nintendo. SPDX-License-Identifier:
// LicenseRef-PTCG-ABC-Competition-Use-Only Part of the Pokémon TCG AI Battle
// Challenge. Provided for Competition use only; the full license is in the
// LICENSES/ folder and incorporates the Competition Rules. Competition Rules:
// https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

// 対象タイプ
enum class TargetType : unsigned char {
  All,

  Hp,
  MaxHp,
  RetreatCost,
  EnergyType,
  EnergyType2, // 2つのどちらかのタイプ
  Resistance,  // 抵抗力タイプ

  PokemonCard,
  BasicPokemon,   // たねポケモン
  EvolvedPokemon, // 進化ポケモン
  Stage1,         // 1進化ポケモン
  Stage2,         // 2進化ポケモン
  BasicEnergy,
  SpecialEnergy,
  EnergyCard,
  Item,
  Tool,
  Supporter,
  Stadium,
  Trainer,
  CardId,

  PokemonOrBasicEnergy,
  BasicPokemonOrBasicEnergy,
  NotRulePokemonCardOrBasicEnergy,
  EnergyTypePokemonOrStadium,
  ItemOrTool,
  EnemyToolOrSpecialEnergyOrStadium,
  EthanPokemonOrBasicFireEnergy,

  HasAbility,
  HasAbilityName, // 特定のとくせいを持っているか
  HasAttackName,  // 特定のワザを持っているか
  RulePokemon,
  NotRulePokemon, // 化石含む
  Ex,
  MegaEx,
  Terastal,
  Ancient,
  Future,
  Hop,
  Lillie,
  Iono,
  N,
  Ethan,
  Cynthia,
  Misty,
  Arven,
  Steven,
  Marnie,
  Erika,
  Larry,
  TeamRocket,

  SilcoonOrCascoon,             // 「カラサリス」または「マユルド」
  KoffingOrWeezing,             // 名前に「ドガース」または「マタドガス」とつく
  HonedgeOrDoubladeOrAegislash, // 「ヒトツキ」「ニダンギル」「ギルガルド」

  Name,
  NameContains, // 「○○」とつくカード

  CanEvolve,                       // 進化可能
  CanEvolve2,                      // can evolve Basic -> Stage2
  CanEvolveMe,                     // 自身から進化可能
  CanEvolveContextCard,            // contextCardから進化可能
  CanEvolvesToContextCard,         // contextCardに進化可能
  CanEvolveField,                  // 場のポケモンから進化可能
  CanEvolveFieldNotAppearThisTurn, // 出したばかりではない場のポケモンから進化可能
  Evolved,                         // 進化している（下に進化前カードが存在する）
  EvolvedThisTurnName,             // この番「○○」から進化している
  NotAppearThisTurn,               // この番に出したばかりのカードではない
  HealThisTurn,                    // この番にHPを回復している
  AttachedMe,
  AttachedEffected, // 前の効果対象に付いている
  AttachedTriggerSubject,
  AttachedTriggerObject,
  AttachedContextCard,
  AttachedActivePokemon,         // バトル場のポケモンに付いている
  AttachedBenchPokemon,          // ベンチのポケモンに付いている
  IsAttachedEnergy,              // エネルギーが付いている
  AttachedEnergyCount,           // ～個エネルギーが付いているポケモン
  IsAttachedSpecialEnergy,       // 特殊エネルギーが付いている
  IsAttachedEnergyType,          // 特定のタイプのエネルギーが付いている
  IsAttachedEnergy2Type,         // 特定のタイプのエネルギーが2個以上付いている
  IsAttachedEnergyName,          // 特定のエネルギーが付いている
  IsAttachedTool,                // どうぐが付いている
  IsAttachedToolName,            // 特定のどうぐが付いている
  IsAttachedToolOrSpecialEnergy, // どうぐか特殊エネルギーが付いている
  NotContextCardAttachedPokemon, // contextCardが付いているポケモンでない
  NotSelectedListAttachedPokemon, // selectedListのカードが付いているポケモンでない
  EnergyTypeAttached, // 特定タイプのエネルギーとして扱われている、付けられたエネルギー
  Reverse,            // 裏向き
  Area,
  TriggerSubject,
  TriggerObject,
  DamageCounter, // のっているダメカンの数
  MinHp, // おたがいの場のポケモン（このポケモンをのぞく）の中から、残りHPが一番少ないポケモン
  SameTypeEnemy,             // 相手の場のポケモンと同じタイプ
  SpecialCondition,          // 特殊状態
  SpecialConditionOrDamaged, // 特殊状態またはダメカンがのっている
  Poison,                    // どく
  Burn,                      // やけど
  Confuse,                   // こんらん
  PoisonOrBurn,              // 【どく】または【やけど】
  BenchToActiveThisTurn,     // この番にベンチからバトル場に出ている
  SameNameEnemyField,        // 相手の場のポケモンと同じ名前のポケモン
  NotChecked,                // checkListに含まれていない(カードIDで判定)
};

// 事前条件タイプ
enum class ConditionType : unsigned char {
  Always,

  AnyTargetAfterEffect, // Skill中のこれより後のEffectのTargetのどれかが1つ以上の対象カードを持つか
  CountTarget,               // 対象の数
  CountTarget2,              // 対象の数2つ
  CountTargetMeOrEnemy,      // 自分と相手のどちらかの条件が満たされていればOK
  CompareCountTargetMeEnemy, // 左辺が自分で右辺が相手。TargetPlayerはBothにする
  CountEnergy,               // 場のエネルギー全ての個数
  CountEnergyType,           // 指定タイプのエネルギーの個数
  CompareCountEnergyMeEnemy, // エネルギー個数。左辺が自分で右辺が相手。TargetPlayerはBothにする
  AttackEnergyExtra, // このワザを使うためのエネルギーより、N個多くエネルギーがついているなら
  NotFullBench,

  MyTurn,                   // Equalなら自分ターン、NotEqualなら相手ターン
  Turn,                     // ターン数
  KoPreEnemyTurn,           // 前の相手の番に自分のポケモンがきぜつしたか
  KoPreEnemyTurnTeamRocket, // 前の相手の番に自分の「ロケット団のポケモン」がきぜつしたか
  KoAttackDamagePreEnemyTurn, // 前の相手の番にワザのダメージで自分のポケモンがきぜつしたか
  KoAttackDamageEthanPreEnemyTurn, // 前の相手の番にワザのダメージで自分の「ヒビキのポケモン」が【きぜつ】したか
  KoAttackDamageHopPreEnemyTurn, // 前の相手の番にワザのダメージで自分の「ホップのポケモン」が【きぜつ】したか
  NoSameNameSkillThisTurn, // 同名のスキルをこの番に使っていない
  SameAttackPreMyTurn, // 前の自分の番に、このポケモンがこのワザを使っているか
  CoinHeadCount,       // コインでオモテが出た回数
  AttachActive,        // バトルポケモンにつけたか

  MysteryGarden, // 自分の手札の枚数が、自分の場の【超】ポケモンの数以下
  LoveBall,      // ラブラブボール
};

// ビット演算使用
enum class TargetPlayer : unsigned char {
  None = 0,
  Me = 1,
  Enemy = 2,
  Both = 3
};

inline constexpr bool IsTargetPlayer(int myPlayerIndex, int targetPlayerIndex,
                                     TargetPlayer targetPlayer) {
  return ((1 + (myPlayerIndex ^ targetPlayerIndex)) &
          static_cast<int>(targetPlayer)) != 0;
}
// 選択数タイプ

enum class EffectSelectType : unsigned char {
  All,

  // ～枚指定
  CardCount,
  MaxCardCount, // 1以上N枚まで
  CardUntil,    // N枚になるように（選択肢数 - N）
  MaxCardUntil, // 0～N枚になるように。

  Energy,        // N個選ぶ
  MaxEnergyCard, // 1以上N個まで
  ToolCard,

  CardOrAttachedCardCount,

  Evolve,
  Evolve2, // ふしぎなアメ
};
