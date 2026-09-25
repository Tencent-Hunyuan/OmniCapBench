#!/usr/bin/env bash
# Serve a local omnimodal checkpoint through vLLM's OpenAI-compatible API.
#
# Usage:
#   bash scripts/serve_vllm.sh MODEL_PATH [PORT]
#
# Example:
#   bash scripts/serve_vllm.sh omnicapbench_eval/models/Qwen3-Omni-30B-A3B-Instruct 8903
#
# Once the server is up, point the evaluation at it:
#   OMNICAP_EVAL_QWEN3_BASE_URL=http://localhost:8903/v1 \
#     bash scripts/run_inference.sh qwen3_omni_instruct

set -euo pipefail

PYTHON="${PYTHON:-python3}"

MODEL_PATH="${1:-}"
PORT="${2:-8903}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

if [ -z "$MODEL_PATH" ]; then
    echo "Usage: bash scripts/serve_vllm.sh MODEL_PATH [PORT]" >&2
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "Model directory not found: $MODEL_PATH" >&2
    echo "Download it first: python -m omnicapbench_eval.download_models --all" >&2
    exit 1
fi

echo "Serving $MODEL_PATH on port $PORT (tp=$TENSOR_PARALLEL_SIZE, max_len=$MAX_MODEL_LEN)"

exec "$PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$(basename "$MODEL_PATH")" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code
