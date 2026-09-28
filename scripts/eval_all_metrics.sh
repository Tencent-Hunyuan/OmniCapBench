#!/usr/bin/env bash
# Score a prediction file: rule-based structural metrics, then the localized LLM judge.
#
# Usage:
#   bash scripts/eval_all_metrics.sh MODE GT_JSONL PRED_DIR
#
#   MODE = all         rule-based metrics + LLM judge + rendered tables
#          rule-based  rule-based metrics only
#          llm-only    LLM judge only
#          convert-only convert a free-form captioner output into MTSS, then stop
#
# Example:
#   bash scripts/eval_all_metrics.sh all my_benchmark/eval.jsonl output/eval/gemini_25
#
# The judge endpoint is configured through these variables, which are forwarded
# explicitly to each Python entry point rather than being read from inside the
# library code:
#
#   JUDGE_BASE_URL   OpenAI-compatible endpoint  (default http://localhost:8000/v1)
#   JUDGE_MODEL      judge model name            (default gpt-4o)
#   JUDGE_API_KEY    API key                     (default EMPTY)
#   JUDGE_WORKERS    parallel judge requests     (default 8)
#
# The same endpoint also drives LLM-assisted reference matching inside the
# rule-based metrics, so Ref, RefUse and CCC stay consistent with the judge.

set -euo pipefail

PYTHON="${PYTHON:-python3}"

MODE="${1:-all}"
GT_JSONL="${2:-examples/eval_sample.jsonl}"
PRED_DIR="${3:-output/eval/demo}"

JUDGE_BASE_URL="${JUDGE_BASE_URL:-http://localhost:8000/v1}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o}"
JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
JUDGE_WORKERS="${JUDGE_WORKERS:-8}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.7}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-4096}"
METRICS_WORKERS="${METRICS_WORKERS:-4}"
CONVERT_MODEL_KIND="${CONVERT_MODEL_KIND:-}"

PRED_FILE="$PRED_DIR/results.jsonl"
CONVERTED_FILE="$PRED_DIR/results_mtss_converted.jsonl"

run_convert() {
    if [ -z "$CONVERT_MODEL_KIND" ]; then
        echo "CONVERT_MODEL_KIND must name the source captioner (e.g. avocado)" >&2
        exit 1
    fi
    echo "[convert] $PRED_FILE -> $CONVERTED_FILE (source: $CONVERT_MODEL_KIND)"
    OMNICAP_EVAL_CONVERT_BASE_URL="$JUDGE_BASE_URL" \
    OMNICAP_EVAL_CONVERT_MODEL="$JUDGE_MODEL" \
    OMNICAP_EVAL_CONVERT_API_KEY="$JUDGE_API_KEY" \
        "$PYTHON" -m omnicapbench_eval.mtss_transformer_convert \
            --model "$CONVERT_MODEL_KIND" \
            --input "$PRED_FILE" \
            --output "$CONVERTED_FILE"
}

resolve_pred_file() {
    if [ -f "$CONVERTED_FILE" ]; then
        echo "$CONVERTED_FILE"
    else
        echo "$PRED_FILE"
    fi
}

run_rule_metrics() {
    pred="$(resolve_pred_file)"
    echo "[rule-based] $pred"
    REF_MATCH_BASE_URL="$JUDGE_BASE_URL" \
    REF_MATCH_MODEL="$JUDGE_MODEL" \
    REF_MATCH_API_KEY="$JUDGE_API_KEY" \
    REF_MATCH_TEMPERATURE="0.0" \
    REF_MATCH_MAX_TOKENS="$JUDGE_MAX_TOKENS" \
        "$PYTHON" -m omnicapbench_eval.mtss_metrics \
            --gt "$GT_JSONL" \
            --pred "$pred" \
            --output "$PRED_DIR/metrics_summary.json" \
            --workers "$METRICS_WORKERS"
}

run_llm_judge() {
    pred="$(resolve_pred_file)"
    echo "[llm-judge] $pred (judge: $JUDGE_MODEL)"
    "$PYTHON" -m omnicapbench_eval.mtss_llm_judge \
        --gt "$GT_JSONL" \
        --pred "$pred" \
        --output "$PRED_DIR/llm_judge_scores.json" \
        --base-url "$JUDGE_BASE_URL" \
        --model "$JUDGE_MODEL" \
        --api-key "$JUDGE_API_KEY" \
        --temperature "$JUDGE_TEMPERATURE" \
        --max-tokens "$JUDGE_MAX_TOKENS" \
        --workers "$JUDGE_WORKERS"
}

render_tables() {
    "$PYTHON" -m omnicapbench_eval.report_tables \
        "$(basename "$PRED_DIR")=$PRED_DIR" \
        --output "$PRED_DIR"
}

if [ ! -f "$PRED_FILE" ] && [ ! -f "$CONVERTED_FILE" ]; then
    echo "No predictions found at $PRED_FILE" >&2
    exit 1
fi

case "$MODE" in
    all)
        [ -n "$CONVERT_MODEL_KIND" ] && run_convert
        run_rule_metrics
        run_llm_judge
        render_tables
        ;;
    rule-base|rule-based)
        run_rule_metrics
        render_tables
        ;;
    llm-only)
        run_llm_judge
        render_tables
        ;;
    convert-only)
        run_convert
        ;;
    *)
        echo "Unknown mode: $MODE (expected all, rule-based, llm-only or convert-only)" >&2
        exit 1
        ;;
esac

echo
echo "Results in $PRED_DIR:"
echo "  metrics_summary.json         rule-based structural metrics"
echo "  metrics_summary_detail.jsonl per-video rule-based breakdown"
echo "  llm_judge_scores.json        localized semantic metrics"
echo "  report_tables.md             both result tables"
