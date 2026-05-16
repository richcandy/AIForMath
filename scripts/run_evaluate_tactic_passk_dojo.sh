#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_PATH="${1:-$ROOT_DIR/outputs/evaluation/deepseek_highschool_test_pass32_ckpt384.jsonl}"
shift || true

CONDA_BIN="$(command -v conda || true)"
if [[ -z "$CONDA_BIN" ]]; then
  printf 'Could not find a working conda executable in PATH\n' >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_PATH")"

EVAL_CMD=(
  "$CONDA_BIN" run --no-capture-output -n ty env
  PATH="$HOME/.elan/bin:$PATH"
  python "$ROOT_DIR/evaluation/evaluate_tactic_passk_dojo.py"
  --output "$OUTPUT_PATH"
)

if [[ $# -gt 0 ]]; then
  EVAL_CMD+=("$@")
fi

printf 'Starting tactic pass@k evaluation\n'
printf 'output=%s\n' "$OUTPUT_PATH"
printf 'cmd=%q ' "${EVAL_CMD[@]}"
printf '\n'

exec "${EVAL_CMD[@]}"
