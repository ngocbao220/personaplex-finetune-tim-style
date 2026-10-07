#!/usr/bin/env bash
# Export prepared synthetic sources, then launch the existing single-GPU bridge.
set -euo pipefail
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
STAGE="${1:-all}"
case "$STAGE" in
  prepare|train|all) ;;
  *) echo 'Usage: bash scripts/train_synthetic_server.sh [prepare|train|all]' >&2; exit 2 ;;
esac
PYTHON="${PYTHON:-python}"
MODEL_ROOT="${MODEL_ROOT:-/home/voice/data/voice/personaplex-7b-v1}"
PREPARED_MANIFEST="${PREPARED_MANIFEST:-/home/voice/data/voice/vdt/data_ready/synthetic_500h/train.jsonl}"
EXPORT_DIR="${EXPORT_DIR:-/home/voice/data/voice/personaplex-exports/synthetic-vi}"
TRAIN_CONFIG="${TRAIN_CONFIG:-$PROJECT/configs/train_single_gpu.yaml}"
FREE_RUNNING_CONFIG="${FREE_RUNNING_CONFIG:-$PROJECT/config.full.yaml}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

if [[ "$STAGE" == prepare || "$STAGE" == all ]]; then
  echo "[prepare] $PREPARED_MANIFEST -> $EXPORT_DIR"
  "$PYTHON" "$PROJECT/prepare_data.py" \
    --manifest "$PREPARED_MANIFEST" --output "$EXPORT_DIR" \
    --config "$TRAIN_CONFIG" --workers "${EXPORT_WORKERS:-0}"
fi
if [[ "$STAGE" == train || "$STAGE" == all ]]; then
  # Validate inference assets before loading the 7B model; frequency=0 disables it.
  "$PYTHON" - "$FREE_RUNNING_CONFIG" <<'PY'
import sys
from pathlib import Path
import yaml
from tim_compat.free_running import frequency
from tim_compat.inference_input import select_source, conditioning
values = yaml.safe_load(Path(sys.argv[1]).read_text())
acceptance = values.get('acceptance', {})
settings = acceptance.get('inference', {})
if frequency(settings):
    select_source(settings, acceptance.get('prepared_manifest'))
    conditioning(settings, acceptance.get('prepared_manifest'))
PY
  echo "[train] config=$EXPORT_DIR/train.yaml GPU=$CUDA_VISIBLE_DEVICES"
  args=(--standalone --nproc-per-node=1 "$PROJECT/train_local.py"
    --model-root "$MODEL_ROOT" --config "$EXPORT_DIR/train.yaml"
    --free-running-config "$FREE_RUNNING_CONFIG"
    --token-cache "$EXPORT_DIR/token_cache")
  if [[ -n "${RESUME_FROM:-}" ]]; then
    args+=(--resume-from "$RESUME_FROM")
  fi
  # Use torchrun from the same active interpreter as preparation.
  "$PYTHON" -m torch.distributed.run "${args[@]}" 2>&1 | tee -a "$EXPORT_DIR/train.log"
fi
