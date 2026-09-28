#!/usr/bin/env bash
# Source distribution entry point: no curl|sh and no implicit research launch.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON="${ARLAB_BOOTSTRAP_PYTHON:-python3}"
command -v "$PYTHON" >/dev/null 2>&1 || {
  printf '%s\n' 'Python 3.10+ is required to run the bootstrap (Ubuntu 22.04/24.04 includes it).' >&2
  exit 2
}
exec "$PYTHON" -B "$ROOT/scripts/deploy.py" "$@"
