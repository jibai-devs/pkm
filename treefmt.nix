{ pkgs, ... }:
{
  projectRootFile = "flake.nix";

  programs.ruff-check.enable = true;
  programs.ruff-format.enable = true;

  settings.global.excludes = [
    "*.lock"
    ".venv/*"
    "__pycache__/*"
    "checkpoints/*"
    "metrics/*"
    "runs/*"
    "result.html"
    "replay.json"
  ];
}
