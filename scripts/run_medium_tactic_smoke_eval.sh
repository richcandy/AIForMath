#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAMPLE_COUNT="${1:-500}"

MODEL_ROOT="$ROOT_DIR/Qwen2.5-Math-1.5B"
ADAPTER_DIR="$ROOT_DIR/outputs/qwen2_5_math_algebra_medium_lora"
EVAL_OUTPUT_DIR="$ROOT_DIR/outputs/evaluation"

mkdir -p "$EVAL_OUTPUT_DIR"

CONDA_BIN="$(command -v conda || true)"
if [[ -z "$CONDA_BIN" ]]; then
  printf 'Could not find a working conda executable in PATH\n' >&2
  exit 1
fi

LATEST_CHECKPOINT="$(python - <<'PY' "$ADAPTER_DIR"
from pathlib import Path
import sys

adapter_dir = Path(sys.argv[1])
candidates = []
for path in adapter_dir.glob("checkpoint-*"):
    suffix = path.name.removeprefix("checkpoint-")
    if path.is_dir() and suffix.isdigit():
        candidates.append((int(suffix), path))
if not candidates:
    raise SystemExit(f"No checkpoint-* directories found under {adapter_dir}")
print(max(candidates)[1])
PY
)"

printf 'Using checkpoint: %s\n' "$LATEST_CHECKPOINT"

"$CONDA_BIN" run --no-capture-output -n ty python "$ROOT_DIR/evaluation/smoke_tactic_accuracy.py" \
  --input "$ROOT_DIR/training_data/highschool_algebra_mathlib_medium/val.json" \
  --model-root "$MODEL_ROOT" \
  --adapter-root "$LATEST_CHECKPOINT" \
  --output "$EVAL_OUTPUT_DIR/medium_val_tactic_smoke_$(basename "$LATEST_CHECKPOINT").jsonl" \
  --summary-path "$EVAL_OUTPUT_DIR/medium_val_tactic_smoke_$(basename "$LATEST_CHECKPOINT").summary.json" \
  --max-samples "$SAMPLE_COUNT" \
  --num-return-sequences 3 \
  --max-goal-tokens 2800

"$CONDA_BIN" run --no-capture-output -n ty python "$ROOT_DIR/evaluation/smoke_tactic_accuracy.py" \
  --input "$ROOT_DIR/training_data/highschool_algebra_mathlib_medium/test.json" \
  --model-root "$MODEL_ROOT" \
  --adapter-root "$LATEST_CHECKPOINT" \
  --output "$EVAL_OUTPUT_DIR/medium_test_tactic_smoke_$(basename "$LATEST_CHECKPOINT").jsonl" \
  --summary-path "$EVAL_OUTPUT_DIR/medium_test_tactic_smoke_$(basename "$LATEST_CHECKPOINT").summary.json" \
  --max-samples "$SAMPLE_COUNT" \
  --num-return-sequences 3 \
  --max-goal-tokens 2800
