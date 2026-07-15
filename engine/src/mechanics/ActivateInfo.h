// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and
// character names are trademarks of Nintendo. SPDX-License-Identifier:
// LicenseRef-PTCG-ABC-Competition-Use-Only Part of the Pokémon TCG AI Battle
// Challenge. Provided for Competition use only; the full license is in the
// LICENSES/ folder and incorporates the Competition Rules. Competition Rules:
// https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "card/Card.h"

struct ActivateAbilityInfo {
  int skillId = 0;
  AreaRef effectCard{}; // 能力を発動したと見なされるカード
  signed char usePlayerIndex =
      0; // 使用したプレイヤー。カードの持ち主とは限らない
  bool isEffectStack = false;
  signed char effectStackIndex = 0;
  bool isSpecialCondition = false;

  ActivateAbilityInfo() = default;
  ActivateAbilityInfo(int skillId, AreaRef effectCard, int usePlayerIndex)
      : skillId(skillId), effectCard(effectCard),
        usePlayerIndex(static_cast<signed char>(usePlayerIndex)),
        isEffectStack(false), effectStackIndex(0), isSpecialCondition(false) {}
};

struct TriggerInfo {
  TriggerType type = TriggerType::None;
  signed char depth = 0;
  int value = 0;
  AreaRef subject{};
  AreaRef object{};

  bool isNull() const { return type == TriggerType::None; }
};

struct TriggeredAbility {
  ActivateAbilityInfo activateInfo;
  TriggerInfo trigger;
};

struct EffectState {
  ActivateAbilityInfo ability;
  signed char effectIndex = 0;
  bool onEffect = false; // エフェクト処理中ならtrue
  signed char selectedListIndex = 0;
  signed char eachListIndex = 0;
  short effectRate = 1; // 効果倍率
  int damageChange = 0;

  void init() { effectRate = 1; }

  int usePlayerIndex() const { return ability.usePlayerIndex; }
};