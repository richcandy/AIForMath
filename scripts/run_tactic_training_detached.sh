#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-$ROOT_DIR/outputs/qwen2_5_math_algebra_lora}"
shift || true

LOG_DIR="$ROOT_DIR/outputs/logs"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="$LOG_DIR/train_tactic_${TIMESTAMP}.log"
PID_PATH="$LOG_DIR/train_tactic_${TIMESTAMP}.pid"

CONDA_BIN="$(command -v conda || true)"
if [[ -z "$CONDA_BIN" ]]; then
  printf 'Could not find a working conda executable in PATH\n' >&2
  exit 1
fi

TRAIN_CMD=(
  "$CONDA_BIN" run --no-capture-output -n ty env
  PATH="$HOME/.elan/bin:$PATH"
  CUDA_VISIBLE_DEVICES=0,1
  TORCH_NCCL_BLOCKING_WAIT=1
  python -m torch.distributed.run --nproc_per_node=2
  "$ROOT_DIR/scripts/train_qwen_lean_dojo.py"
  --data-path "$ROOT_DIR/training_data/highschool_algebra_mathlib/train.json"
  --data-format traced_tactics
  --eval-data-path "$ROOT_DIR/training_data/highschool_algebra_mathlib/val.json"
  --eval-data-format traced_tactics
  --output-dir "$OUTPUT_DIR"
  --epochs 2
  --batch-size 2
  --grad-accum 4
  --max-length 1024
  --learning-rate 2e-5
  --warmup-ratio 0.03
  --max-grad-norm 0.5
  --logging-steps 10
  --eval-steps 500
  --save-steps 500
  --save-total-limit 3
)

if [[ $# -gt 0 ]]; then
  TRAIN_CMD+=("$@")
fi

setsid -f bash -lc '
  exec </dev/null
  exec >>"$1" 2>&1
  shift
  printf "started_at=%s\n" "$(date -Is)"
  printf "cmd=%q " "$@"
  printf "\n"
  exec "$@"
' _ "$LOG_PATH" "${TRAIN_CMD[@]}"

sleep 1
TRAIN_PID="$(pgrep -n -f "train_qwen_lean_dojo.py.*$OUTPUT_DIR" || true)"
if [[ -n "$TRAIN_PID" ]]; then
  printf '%s\n' "$TRAIN_PID" > "$PID_PATH"
fi

printf 'Started detached training\n'
printf 'output_dir=%s\n' "$OUTPUT_DIR"
printf 'log=%s\n' "$LOG_PATH"
printf 'pid=%s\n' "${TRAIN_PID:-unknown}"
printf 'pid_file=%s\n' "$PID_PATH"
printf '\n'
printf 'Watch log: tail -f %s\n' "$LOG_PATH"
