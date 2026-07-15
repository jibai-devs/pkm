// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and
// character names are trademarks of Nintendo. SPDX-License-Identifier:
// LicenseRef-PTCG-ABC-Competition-Use-Only Part of the Pokémon TCG AI Battle
// Challenge. Provided for Competition use only; the full license is in the
// LICENSES/ folder and incorporates the Competition Rules. Competition Rules:
// https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include <array>
#include <cassert>

enum class EnergyType : unsigned short {
  Colorless = 0,
  Grass = 1 << 0,
  Fire = 1 << 1,
  Water = 1 << 2,
  Lightning = 1 << 3,
  Psychic = 1 << 4,
  Fighting = 1 << 5,
  Darkness = 1 << 6,
  Metal = 1 << 7,
  Dragon = 1 << 8,
  All = (1 << 9) - 1,
};

constexpr EnergyType operator|(EnergyType t0, EnergyType t1) {
  return static_cast<EnergyType>(static_cast<int>(t0) | static_cast<int>(t1));
}

constexpr std::array<EnergyType, 12> EnergyTypes = {
    EnergyType::Colorless, EnergyType::Grass,
    EnergyType::Fire,      EnergyType::Water,
    EnergyType::Lightning, EnergyType::Psychic,
    EnergyType::Fighting,  EnergyType::Darkness,
    EnergyType::Metal,     EnergyType::Dragon,
    EnergyType::All,       EnergyType::Psychic | EnergyType::Darkness,
};

inline constexpr bool ContainsEnergyType(EnergyType t0, EnergyType t1) {
  return (static_cast<int>(t0) & static_cast<int>(t1)) != 0;
}

// 無色をタイプの一つとしてとらえる場合
inline constexpr bool MatchEnergyType(EnergyType t0, EnergyType t1) {
  if (t1 == EnergyType::Colorless) {
    return t0 == EnergyType::Colorless;
  } else {
    return (static_cast<int>(t0) & static_cast<int>(t1)) != 0;
  }
}

inline int EnergyTypeIndex(EnergyType type) {
  switch (type) {
  case EnergyType::Colorless:
    return 0;
  case EnergyType::Grass:
    return 1;
  case EnergyType::Fire:
    return 2;
  case EnergyType::Water:
    return 3;
  case EnergyType::Lightning:
    return 4;
  case EnergyType::Psychic:
    return 5;
  case EnergyType::Fighting:
    return 6;
  case EnergyType::Darkness:
    return 7;
  case EnergyType::Metal:
    return 8;
  case EnergyType::Dragon:
    return 9;
  case EnergyType::All:
    return 10;
  case EnergyType::Psychic | EnergyType::Darkness:
    return 11;
  default:
    assert(false);
    return 0;
  }
}

inline const char *EnergyText(EnergyType type) {
  switch (type) {
  case EnergyType::Colorless:
    return "Colorless";
  case EnergyType::Grass:
    return "Grass";
  case EnergyType::Fire:
    return "Fire";
  case EnergyType::Water:
    return "Water";
  case EnergyType::Lightning:
    return "Lightning";
  case EnergyType::Psychic:
    return "Psychic";
  case EnergyType::Fighting:
    return "Fighting";
  case EnergyType::Darkness:
    return "Darkness";
  case EnergyType::Metal:
    return "Metal";
  case EnergyType::Dragon:
    return "Dragon";
  case EnergyType::All:
    return "Rainbow";
  case EnergyType::Psychic | EnergyType::Darkness:
    return "PsychicAndDarkness";
  default:
    assert(false);
    return "";
  }
}
