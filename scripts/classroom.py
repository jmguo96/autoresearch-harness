#!/usr/bin/env python3
"""Thin, dependency-free source distribution entry for students."""
from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info < (3, 10):
    raise SystemExit("需要 Python 3.10+；D4RL 的私有 Python 环境由安装程序准备。")
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from arlab.classroom.bootstrap import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(source=ROOT))
