{
  description = "Pokemon TCG AI Battle Challenge - Simulation agent";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    treefmt-nix.url = "github:numtide/treefmt-nix";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    treefmt-nix,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = nixpkgs.legacyPackages.${system};
        treefmtEval = treefmt-nix.lib.evalModule pkgs ./treefmt.nix;
      in {
        formatter = treefmtEval.config.build.wrapper;

        checks = {
          formatting = treefmtEval.config.build.check self;
        };

        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.python312
            pkgs.uv
            pkgs.ruff
            pkgs.pyright
            pkgs.cacert
            pkgs.just
            # prebuilt cabt/kaggle wheels (libcg.so, numpy) need a system C++ runtime
            pkgs.stdenv.cc.cc.lib
          ];

          # expose libstdc++.so.6 / libgcc_s for ctypes-loaded native libs, and
          # the NVIDIA driver libs (libcuda.so.1) so the uv/pip torch wheel can
          # use CUDA. On NixOS the driver lives at /run/opengl-driver/lib, not
          # /usr/lib — without this torch.cuda.is_available() is False and
          # training silently falls back to CPU.
          LD_LIBRARY_PATH =
            pkgs.lib.makeLibraryPath [pkgs.stdenv.cc.cc.lib]
            + ":/run/opengl-driver/lib";

          shellHook = ''
            export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
            export NIX_SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"

            if [ ! -d .venv ]; then
              echo "Creating virtual environment with uv..."
              uv venv .venv --python 3.12
            fi
            source .venv/bin/activate

            echo "Python: $(python --version)"
            echo "uv: $(uv --version)"
          '';
        };
      }
    );
}
