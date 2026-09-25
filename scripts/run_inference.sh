#!/usr/bin/env bash
# Run one model over the benchmark and write structured predictions.
#
# Usage:
#   bash scripts/run_inference.sh MODEL [INPUT_JSONL] [OUTPUT_DIR]
#
# Example:
#   bash scripts/run_inference.sh gemini_25 data/OmniCapBench/eval.jsonl output/eval
#
# Every parameter is passed explicitly on the command line so a run can be
# reproduced from the shell history alone.

set -euo pipefail

PYTHON="${PYTHON:-python3}"

MODEL="${1:-}"
INPUT_JSONL="${2:-examples/eval_sample.jsonl}"
OUTPUT_DIR="${3:-output/eval}"
VIDEO_ROOT="${VIDEO_ROOT:-}"
WORKERS="${WORKERS:-8}"

if [ -z "$MODEL" ]; then
    echo "Usage: bash scripts/run_inference.sh MODEL [INPUT_JSONL] [OUTPUT_DIR]" >&2
    echo >&2
    echo "Available models:" >&2
    "$PYTHON" -c "from omnicapbench_eval.run_eval import MODEL_PRESETS; print('  ' + '\n  '.join(sorted(MODEL_PRESETS)))" >&2
    exit 1
fi

echo "Model:      $MODEL"
echo "Input:      $INPUT_JSONL"
echo "Output:     $OUTPUT_DIR/$MODEL"
echo "Video root: ${VIDEO_ROOT:-<paths used as-is>}"
echo "Workers:    $WORKERS"
echo

"$PYTHON" -m omnicapbench_eval.run_eval \
    --model "$MODEL" \
    --input "$INPUT_JSONL" \
    --video-root "$VIDEO_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --workers "$WORKERS"

echo
echo "Predictions: $OUTPUT_DIR/$MODEL/results.jsonl"
echo "Next: bash scripts/eval_all_metrics.sh all $INPUT_JSONL $OUTPUT_DIR/$MODEL"
