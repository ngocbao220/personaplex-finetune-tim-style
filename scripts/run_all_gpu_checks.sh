#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON:-python}" "$ROOT/scripts/_run_all.py" --config "${1:?Usage: bash scripts/run_all_gpu_checks.sh /absolute/config.full.yaml}"