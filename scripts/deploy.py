#!/usr/bin/env python3
"""Dependency-free entry; installation runs before importing the ML runtime."""
from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info < (3, 10):
    raise SystemExit("Bootstrap requires Python 3.10+. D4RL gets its own Python 3.10 runtime.")
sys.dont_write_bytecode = True  # Even --dry-run must not create source-tree caches.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from arlab.deployment.manager import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(source=ROOT))
