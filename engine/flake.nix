{
  description = "Pokémon TCG AI Battle Simulator - C++20 development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    treefmt-nix.url = "github:numtide/treefmt-nix";
  };

  outputs =
    {
      self,
      nixpkgs,
      treefmt-nix,
      ...
    }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
      treefmtEval = forAllSystems (
        system: treefmt-nix.lib.evalModule nixpkgs.legacyPackages.${system} ./treefmt.nix
      );
    in
    {
      formatter = forAllSystems (system: treefmtEval.${system}.config.build.wrapper);

      checks = forAllSystems (system: {
        formatting = treefmtEval.${system}.config.build.check self;
      });
      packages = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          llvm = pkgs.llvmPackages_21;
        in
        {
          default = pkgs.callPackage ./package.nix { stdenv = llvm.libcxxStdenv; };
          # libtorch C++ driver (NN inference / MCTS self-play trainer). Built
          # with the DEFAULT GCC/libstdc++ stdenv to match libtorch-bin's ABI —
          # it talks to cg.so only through the engine's extern "C" boundary.
          torch-driver = pkgs.callPackage ./torch-driver/torch.nix {
            engine = pkgs.callPackage ./package.nix { stdenv = llvm.libcxxStdenv; };
          };
          torch-driver-cuda = pkgs.callPackage ./torch-driver/torch.nix {
            engine = pkgs.callPackage ./package.nix { stdenv = llvm.libcxxStdenv; };
            cudaSupport = true;
          };
        }
      );

      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          llvm = pkgs.llvmPackages_21;
        in
        {
          default = pkgs.mkShell.override { stdenv = llvm.libcxxStdenv; } {
            name = "ptcg-dev";

            nativeBuildInputs = [
              llvm.clang-tools
              pkgs.cmake
              pkgs.ninja
              pkgs.pkg-config
            ];

            buildInputs = [
              llvm.llvm
              llvm.lld
            ];

            packages =
              with pkgs;
              [
                pkgs.mold
                llvm.lldb
                gdb
                ccache
                cmake-format
              ]
              ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [
                pkgs.valgrind
              ];

            CMAKE_GENERATOR = "Ninja";
            CMAKE_EXPORT_COMPILE_COMMANDS = "ON";
            CMAKE_BUILD_TYPE = "Debug";
            LDFLAGS = "-fuse-ld=mold";

            shellHook = ''
              echo "ptcg dev: $(c++ --version | head -1)"
            '';
          };

          # Separate shell for the libtorch driver: DEFAULT gcc/libstdc++
          # stdenv (matches libtorch-bin), torch's CMake config on
          # CMAKE_PREFIX_PATH. Set cudaSupport via `torch.override`.
          torch =
            let
              libtorch = pkgs.libtorch-bin;
            in
            pkgs.mkShell {
              name = "pkm-torch-dev";
              nativeBuildInputs = [
                pkgs.cmake
                pkgs.ninja
                pkgs.pkg-config
              ];
              buildInputs = [
                libtorch
                pkgs.nlohmann_json
                pkgs.cli11
              ];
              CMAKE_GENERATOR = "Ninja";
              CMAKE_EXPORT_COMPILE_COMMANDS = "ON";
              CMAKE_PREFIX_PATH = "${libtorch.dev}";
              shellHook = ''
                echo "pkm-torch dev: $(c++ --version | head -1) + libtorch ${libtorch.version}"
              '';
            };
        }
      );
    };
}
