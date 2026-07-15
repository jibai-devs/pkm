cmake_minimum_required(VERSION 3.20)
project(ptcg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(game SHARED src/Export.cpp)

target_include_directories(game PUBLIC src)

target_compile_definitions(game PRIVATE GAME_EXPORTS)

set_target_properties(
  game
  PROPERTIES CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON
             POSITION_INDEPENDENT_CODE ON
             PREFIX ""
             OUTPUT_NAME "cg")

if(WIN32)
  target_compile_definitions(game PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX _WINDOWS
                                          _USRDLL)
endif()

include(GNUInstallDirs)

install(
  TARGETS game
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
