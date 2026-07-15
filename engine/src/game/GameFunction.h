// SPDX-FileCopyrightText: © Pokémon/Nintendo/Creatures/GAME FREAK TM, ®, and
// character names are trademarks of Nintendo. SPDX-License-Identifier:
// LicenseRef-PTCG-ABC-Competition-Use-Only Part of the Pokémon TCG AI Battle
// Challenge. Provided for Competition use only; the full license is in the
// LICENSES/ folder and incorporates the Competition Rules. Competition Rules:
// https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/rules

#pragma once

#include "core/Core.h"

struct State;

using GameFunctionNoArg = void (*)(State &);
using GameFunctionI = void (*)(State &, int);
using GameFunctionB = void (*)(State &, bool);
using GameFunctionII = void (*)(State &, int, int);
using GameFunctionIII = void (*)(State &, int, int, int);

inline std::vector<void *> FunctionTable;
inline std::unordered_map<long long, int> FunctionIndexTable;

enum class ArgType : unsigned char {
  None = 0, // 無し
  I,        // int
  B,        // bool
  II,       // int, int
  III,      // int, int, int
};

// 呼び出し予約情報
struct GameFunction {
  int functionIndex = 0;
  int arg0 = 0;
  int arg1 = 0;
  int arg2 = 0;
  ArgType argType = ArgType::None; // 引数タイプ
  unsigned char callCount = 0;
  unsigned char calledCount = 0;

  GameFunction() = default;
  GameFunction(void *functionPointer, ArgType argType)
      : arg0(0), arg1(0), arg2(0), argType(argType), callCount(1),
        calledCount(0) {
    functionIndex =
        FunctionIndexTable.at(reinterpret_cast<long long>(functionPointer));
  }

  void setCallCount(int callCount) {
    this->callCount = static_cast<unsigned char>(callCount);
  }
};
