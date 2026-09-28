#!/usr/bin/env python3
"""Batch data entry; run with the deployment's Python or after installing arlab."""
from __future__ import annotations

import argparse
from pathlib import Path

from arlab.deployment.tasks import prepare_many
from arlab.core.io import read_json
from arlab.tasks.iql.environments import D4RL_IDS


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--env", choices=D4RL_IDS, default="hopper-medium-v2")
    selection.add_argument("--suite12", action="store_true")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--raw-dir", type=Path)
    p.add_argument("--checksums", type=Path)
    p.add_argument("--offline", action="store_true")
    args = p.parse_args()
    prepare_many(args.data_root, list(D4RL_IDS) if args.suite12 else [args.env], raw_dir=args.raw_dir,
                 offline=args.offline, checksums=read_json(args.checksums) if args.checksums else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
