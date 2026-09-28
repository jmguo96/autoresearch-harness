#!/usr/bin/env python3
"""Optional owner command: prepare ONLY native libraries under this project.

No model, D4RL download, Python environment installation or research is started.
Normally start.sh setup invokes the same code automatically.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / 'src'))

from arlab.deployment.common import Runner, atomic_json, exclusive_lock
from arlab.deployment.local import contained_path, local_process, project_environment, clean_path
from arlab.deployment.native import ensure_native_prerequisites


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=SOURCE / '.arlab/d4rl9')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        from arlab.deployment.manager import parser as deployment_parser, validate_options
        root = validate_options(deployment_parser().parse_args(['--root', str(args.root)]), SOURCE)
        if args.dry_run:
            print(json.dumps({'root': str(root), 'mode': 'check' if args.check_only else 'auto',
                              'offline': args.offline, 'uses_sudo': False, 'system_install': False,
                              'writes_now': False, 'calls_llm': False}, ensure_ascii=False, indent=2))
            return 0
        root.mkdir(parents=True, exist_ok=True)
        # Mark a fresh native-only workspace as managed, but never claim ready.
        marker = root / 'deployment.json'
        if not marker.exists():
            atomic_json(marker, {'schema': 1, 'status': 'prepared_native_only', 'steps': [],
                                 'created_by': 'scripts/prepare_native.py', 'research_started': False})
        with exclusive_lock(root / '.deploy.lock'), local_process(root, SOURCE):
            attempt = root / 'attempts' / (time.strftime('%Y%m%d-%H%M%S') + '-native-' + uuid.uuid4().hex[:6])
            env = project_environment(root)
            env['PATH'] = clean_path(root)
            result = ensure_native_prerequisites(root, Runner(attempt, env),
                       mode='check' if args.check_only else 'auto', offline=args.offline)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f'NATIVE ERROR {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
