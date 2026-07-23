cmake_minimum_required(VERSION 3.20)
project(pkm_torch_driver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(nlohmann_json REQUIRED)
find_package(CLI11 REQUIRED)

# --- torch (manual wiring) --------------------------------------------------
# We deliberately do NOT use find_package(Torch): for the CUDA libtorch,
# Caffe2's config does find_package(CUDAToolkit REQUIRED) at configure time,
# which would force the whole (unfree, multi-GB) CUDA toolkit into this build
# even though we compile no CUDA code ourselves. libtorch-bin statically links
# the CUDA runtime, so we only need its headers + shared objects. TORCH_INCLUDE
# and TORCH_LIBDIR are passed from nix; TORCH_CXX_FLAGS is empty for this build
# (default libstdc++ cxx11 ABI), so no extra compile flags are required.
if(NOT DEFINED TORCH_INCLUDE OR NOT DEFINED TORCH_LIBDIR)
  message(FATAL_ERROR "TORCH_INCLUDE and TORCH_LIBDIR must be set")
endif()
set(TORCH_INCLUDE_DIRS ${TORCH_INCLUDE} ${TORCH_INCLUDE}/torch/csrc/api/include)
set(TORCH_LIBS "")
foreach(_l torch torch_cpu c10)
  find_library(_lib_${_l} NAMES ${_l} PATHS ${TORCH_LIBDIR} NO_DEFAULT_PATH REQUIRED)
  list(APPEND TORCH_LIBS ${_lib_${_l}})
endforeach()
# Force-link the CUDA libs when present (CPU build simply won't have them), so
# runtime CUDA is actually registered even with no direct symbol reference.
foreach(_l torch_cuda c10_cuda)
  find_library(_lib_${_l} NAMES ${_l} PATHS ${TORCH_LIBDIR} NO_DEFAULT_PATH)
  if(_lib_${_l})
    list(APPEND TORCH_LIBS -Wl,--no-as-needed ${_lib_${_l}} -Wl,--as-needed)
  endif()
endforeach()

# ENGINE_LIB = absolute path to the engine's cg.so (passed from nix). We declare
# its extern "C" symbols ourselves (src/rl/engine.hpp), so no engine headers are
# needed — only the shared object to link against.
if(NOT DEFINED ENGINE_LIB)
  message(FATAL_ERROR "ENGINE_LIB (path to cg.so) must be set")
endif()

# Toolchain proof (torch + cg.so C-ABI smoke test).
add_executable(driver src/main.cpp)
target_include_directories(driver PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(driver PRIVATE ${TORCH_LIBS} ${CMAKE_DL_LIBS})
target_compile_features(driver PRIVATE cxx_std_20)

# The RL/MCTS self-play trainer (direct port of references/mcts.py).
add_executable(trainer src/rl/train.cpp)
target_include_directories(trainer PRIVATE src/rl ${TORCH_INCLUDE_DIRS})
target_link_libraries(trainer PRIVATE ${TORCH_LIBS}
                                      nlohmann_json::nlohmann_json
                                      CLI11::CLI11
                                      "${ENGINE_LIB}" ${CMAKE_DL_LIBS})
target_compile_features(trainer PRIVATE cxx_std_20)

install(TARGETS driver trainer RUNTIME DESTINATION bin)
