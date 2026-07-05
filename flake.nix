{
  description = "Pokemon TCG AI Battle Challenge - Simulation agent";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.python312
            pkgs.uv
            pkgs.ruff
            pkgs.pyright
            pkgs.cacert
          ];

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
