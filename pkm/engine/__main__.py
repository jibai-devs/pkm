"""Print a report of the selected engine backend and its capabilities.

python -m pkm.engine              # report the default (kaggle) backend
PKM_ENGINE=vendored python -m pkm.engine
"""

from __future__ import annotations

from pkm.engine.loader import available_backends, capabilities


def main() -> None:
    caps = capabilities()
    print("cabt engine")
    print(f"  backend            : {caps.backend}")
    print(f"  lib path           : {caps.lib_path}")
    print(f"  available backends : {', '.join(available_backends()) or '(none)'}")
    print(f"  kaggle available   : {caps.kaggle_available}")
    print(f"  vendored built     : {caps.vendored_built}")
    print(f"  core ABI symbols   : {len(caps.present_symbols)}/13 present")
    if caps.missing_symbols:
        print(f"  MISSING symbols    : {', '.join(caps.missing_symbols)}")
    print(f"  seed injection     : {caps.supports_seed_injection}")
    print(f"  deterministic      : {caps.deterministic}")


if __name__ == "__main__":
    main()
