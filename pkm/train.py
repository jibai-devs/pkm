"""Superseded by pkm.rl.train (PPO self-play). Kept as an entry-point shim.

Run: python -m pkm.train  (equivalent to python -m pkm.rl.train)
"""

from pkm.rl.train import main, train  # noqa: F401

if __name__ == "__main__":
    main()
