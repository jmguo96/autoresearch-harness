#!/usr/bin/env bash
# Student entry: no activation, JSON editing or shell evaluation of configuration.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON="${ARLAB_BOOTSTRAP_PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || {
  printf '%s\n' '需要 Python 3.10+；请先使用课程 Linux/WSL2 环境。' >&2
  exit 2
}
exec "$PYTHON" -B "$ROOT/scripts/classroom.py" "$@"
