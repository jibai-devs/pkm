{
  stdenv, # plain GCC/libstdc++ stdenv — matches libtorch-bin's ABI (NOT libc++)
  cmake,
  ninja,
  libtorch-bin,
  nlohmann_json,
  cli11,
  engine, # the engine package (builds lib/cg.so); passed from flake.nix
  cudaSupport ? false, # flip to true for the CUDA build that uses the 3090
}:

let
  torch = libtorch-bin.override { inherit cudaSupport; };
in
stdenv.mkDerivation {
  pname = "pkm-torch-driver";
  version = "0.1.0";

  src = ./.;

  nativeBuildInputs = [
    cmake
    ninja
  ];
  buildInputs = [
    torch
    nlohmann_json
    cli11
  ];

  # libtorch's CMake config lives in the `dev` output. Build the binary with
  # its final (install) rpath so CMake never has to rewrite it at install time
  # — libtorch's config injects bogus Intel/MKL rpath entries that otherwise
  # break the install-time RPATH_CHANGE. Also add the engine's lib dir so the
  # linked cg.so is found at runtime.
  cmakeFlags = [
    "-DTORCH_INCLUDE=${torch.dev}/include"
    "-DTORCH_LIBDIR=${torch}/lib"
    "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON"
    "-DCMAKE_INSTALL_RPATH=${torch}/lib;${engine}/lib"
    "-DENGINE_LIB=${engine}/lib/cg.so"
  ];
}
