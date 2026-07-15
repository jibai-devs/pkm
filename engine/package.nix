{
  lib,
  stdenv,
  cmake,
  ninja,
}:

stdenv.mkDerivation {
  pname = "ptcg";
  version = "0.1.0";

  src = lib.sourceByRegex ./. [
    "^src(/.*)?$"
    "^CMakeLists\\.txt$"
  ];

  nativeBuildInputs = [
    cmake
    ninja
  ];
}
