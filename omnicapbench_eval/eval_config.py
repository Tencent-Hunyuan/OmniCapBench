#!/usr/bin/env python3
"""
MTSS Eval - Configuration
All configurable parameters for the evaluation inference pipeline.

Model families:
  - closed_source: gemini, openai_compatible (video_url, fps=2, no max_frames)
  - qwen3_omni:    qwen3_omni (remote API), qwen3_omni_instruct/thinking/captioner (local)
                   (base64/local, fps=2, max_frames=192, 2-pass for >1min)
  - qwen25_omni:   qwen25_omni_7b (local base),
                   ugc_videocaptioner, asid_caption_7b, asid_caption_3b, avocado, timechat_captioner (local fine-tuned)
                   (base64/local, fps=2, only <1min videos)

Video sampling strategies differ by model family:
  - closed_source: fps=2, no max_frames limit
  - qwen3_omni:    fps=2, max_frames=192; for >1min videos, 2-pass (ref_bank + shot groups)
  - qwen25_omni:   fps=2, max_frames=192; only test <1min videos (also used for local models)
"""

import os

# ============================================================
# Base directory (used by all path definitions below)
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Model Presets
# ============================================================

PRESET_GEMINI_31 = {
    "model_type": "gemini",
    "model_family": "closed_source",
    "api_key": os.environ.get("GEMINI_API_KEY", ""),
    "model_name": os.environ.get("OMNICAP_EVAL_GEMINI31_MODEL", "gemini-1.5-pro"),
    "max_output_tokens": 65536,
    "timeout": 600,
    "request_timeout": 3600,
}

PRESET_GEMINI_25 = {
    "model_type": "gemini",
    "model_family": "closed_source",
    "api_key": os.environ.get("GEMINI_API_KEY", ""),
    "model_name": os.environ.get("OMNICAP_EVAL_GEMINI25_MODEL", "gemini-2.5-pro"),
    "max_output_tokens": 65536,
    "timeout": 600,
    "request_timeout": 3600,
}

PRESET_QWEN35_OMNI_PLUS = {
    "model_type": "openai_compatible",
    "model_family": "closed_source",
    "api_key": os.environ.get("OMNICAP_EVAL_QWEN35_API_KEY", ""),
    "base_url": os.environ.get("OMNICAP_EVAL_QWEN35_BASE_URL", ""),
    "model_name": os.environ.get("OMNICAP_EVAL_QWEN35_MODEL", "qwen3.5-omni-plus"),
    "max_completion_tokens": 32768,
    "temperature": 0.9,
    "request_timeout": 3600,
}

PRESET_QWEN35_OMNI_FLASH = {
    "model_type": "openai_compatible",
    "model_family": "closed_source",
    "api_key": os.environ.get(
        "OMNICAP_EVAL_QWEN35_FLASH_API_KEY",
        os.environ.get("OMNICAP_EVAL_QWEN35_API_KEY", ""),
    ),
    "base_url": os.environ.get(
        "OMNICAP_EVAL_QWEN35_FLASH_BASE_URL",
        os.environ.get("OMNICAP_EVAL_QWEN35_BASE_URL", ""),
    ),
    "model_name": os.environ.get("OMNICAP_EVAL_QWEN35_FLASH_MODEL", "qwen3.5-omni-flash"),
    "max_completion_tokens": 32768,
    "temperature": 0.9,
    "request_timeout": 3600,
}

PRESET_OPENROUTER = {
    "model_type": "openai_compatible",
    "model_family": "closed_source",
    "api_key": os.environ.get("OPENROUTER_API_KEY", ""),
    "base_url": "https://openrouter.ai/api/v1",
    "model_name": os.environ.get("OMNICAP_EVAL_OPENROUTER_MODEL", "google/gemini-2.5-pro"),
    "max_completion_tokens": 65536,
    "temperature": 0.7,
    "request_timeout": 3600,
}

PRESET_QWEN3_OMNI = {
    "model_type": "qwen_omni",
    "model_family": "qwen3_omni",
    "base_url": os.environ.get("OMNICAP_EVAL_QWEN3_BASE_URL", "http://localhost:8903/v1"),
    "api_key": os.environ.get("OMNICAP_EVAL_QWEN3_API_KEY", "EMPTY"),
    "model_name": os.environ.get("OMNICAP_EVAL_QWEN3_MODEL", "Omni_qwen3"),
    "timeout": 3000,
}

# Qwen2.5-Omni: placeholder — update base_url/model_name when deploying
PRESET_QWEN25_OMNI = {
    "model_type": "qwen_omni",
    "model_family": "qwen25_omni",
    "base_url": os.environ.get("OMNICAP_EVAL_QWEN25_BASE_URL", "http://localhost:8903/v1"),
    "api_key": os.environ.get("OMNICAP_EVAL_QWEN25_API_KEY", "EMPTY"),
    "model_name": os.environ.get("OMNICAP_EVAL_QWEN25_MODEL", "Omni_qwen25"),
    "timeout": 300,
}

# ============================================================
# Local models
# Two inference backends:
#   1. vLLM serving → OpenAI-compatible API (for large models: Qwen3-Omni 30B, Qwen2.5-Omni-7B)
#   2. Transformers native (for smaller fine-tuned models: ASID, AVoCaDO, TimeChat, UGC)
# ============================================================

MODELS_DIR = os.environ.get("OMNICAP_EVAL_MODELS_DIR", os.path.join(BASE_DIR, "models"))
# Shared inference parameters for ALL local models
# max_new_tokens set to model's max_position_embeddings (32768)
# For Qwen2.5-Omni (7B/3B) and all fine-tuned variants: context window = 32768
# For Qwen3-Omni-30B-A3B: thinker context = 65536, but vLLM max_model_len = 32768
_LOCAL_INFERENCE_PARAMS = {
    "max_new_tokens": 32768,
    "temperature": 0.7,
    "top_p": 0.9,
    "timeout": 600,
    "max_retries": 5,       # Local models: low retries (failures are deterministic)
    "retry_delay": 5.0,
}

# vLLM server config (used by eval.sh to start the server)
VLLM_PORT = 8903
VLLM_TP = 8                # tensor-parallel across 8 GPUs per node
VLLM_MAX_MODEL_LEN = 32768
VLLM_GPU_MEMORY_UTILIZATION = 0.90
# Concurrency: how many requests vLLM processes simultaneously per node
# Qwen3-Omni-30B-A3B: ~15GB BF16, TP=8 → ~2GB/card, plenty of KV cache room for 4 concurrent
# Qwen2.5-Omni-7B:    ~14GB BF16, TP=8 → ~2GB/card, even more room for 8 concurrent
VLLM_MAX_NUM_SEQS_30B = 4   # Qwen3-Omni-30B-A3B (3 variants)
VLLM_MAX_NUM_SEQS_7B = 8    # Qwen2.5-Omni-7B

# ---- Qwen3-Omni 30B-A3B series (3 variants) ----

PRESET_QWEN3_OMNI_INSTRUCT = {
    "model_type": "qwen_omni",
    "model_family": "qwen3_omni",
    "inference_backend": "vllm",
    "model_path": os.path.join(MODELS_DIR, "Qwen3-Omni-30B-A3B-Instruct"),
    "base_url": f"http://localhost:{VLLM_PORT}/v1",
    "api_key": "mtss_eval",
    "model_name": "Qwen3-Omni-30B-A3B-Instruct",
    **_LOCAL_INFERENCE_PARAMS,
}

PRESET_QWEN3_OMNI_THINKING = {
    "model_type": "qwen_omni",
    "model_family": "qwen3_omni",
    "inference_backend": "vllm",
    "model_path": os.path.join(MODELS_DIR, "Qwen3-Omni-30B-A3B-Thinking"),
    "base_url": f"http://localhost:{VLLM_PORT}/v1",
    "api_key": "mtss_eval",
    "model_name": "Qwen3-Omni-30B-A3B-Thinking",
    **_LOCAL_INFERENCE_PARAMS,
}

PRESET_QWEN3_OMNI_CAPTIONER = {
    "model_type": "qwen_omni",
    "model_family": "qwen3_omni",
    "inference_backend": "vllm",
    "model_path": os.path.join(MODELS_DIR, "Qwen3-Omni-30B-A3B-Captioner"),
    "base_url": f"http://localhost:{VLLM_PORT}/v1",
    "api_key": "mtss_eval",
    "model_name": "Qwen3-Omni-30B-A3B-Captioner",
    **_LOCAL_INFERENCE_PARAMS,
}

# ---- Qwen2.5-Omni-7B (base model) ----

PRESET_QWEN25_OMNI_7B = {
    "model_type": "qwen_omni",
    "model_family": "qwen25_omni",
    "inference_backend": "vllm",
    "model_path": os.path.join(MODELS_DIR, "Qwen2.5-Omni-7B"),
    "base_url": f"http://localhost:{VLLM_PORT}/v1",
    "api_key": "mtss_eval",
    "model_name": "Qwen2.5-Omni-7B",
    **_LOCAL_INFERENCE_PARAMS,
}

# ---- Qwen2.5-Omni fine-tuned variants ----

PRESET_UGC_VIDEOCAPTIONER = {
    "model_type": "qwen_omni",
    "model_family": "qwen25_omni",
    "inference_backend": "transformers",
    "model_path": os.path.join(MODELS_DIR, "UGC-VideoCaptioner"),
    "model_name": "UGC-VideoCaptioner",
    **_LOCAL_INFERENCE_PARAMS,
}

PRESET_ASID_CAPTION_7B = {
    "model_type": "qwen_omni",
    "model_family": "qwen25_omni",
    "inference_backend": "transformers",
    "model_path": os.path.join(MODELS_DIR, "ASID-Captioner-7B"),
    "model_name": "ASID-Captioner-7B",
    **_LOCAL_INFERENCE_PARAMS,
}

PRESET_ASID_CAPTION_3B = {
    "model_type": "qwen_omni",
    "model_family": "qwen25_omni",
    "inference_backend": "transformers",
    "model_path": os.path.join(MODELS_DIR, "ASID-Captioner-3B"),
    "model_name": "ASID-Captioner-3B",
    **_LOCAL_INFERENCE_PARAMS,
}

PRESET_AVOCADO = {
    "model_type": "qwen_omni",
    "model_family": "qwen25_omni",
    "inference_backend": "transformers",
    "model_path": os.path.join(MODELS_DIR, "AVoCaDO"),
    "model_name": "AVoCaDO",
    **_LOCAL_INFERENCE_PARAMS,
}

PRESET_TIMECHAT_CAPTIONER = {
    "model_type": "qwen_omni",
    "model_family": "qwen25_omni",
    "inference_backend": "transformers",
    "model_path": os.path.join(MODELS_DIR, "TimeChat-Captioner-GRPO-7B"),
    "model_name": "TimeChat-Captioner-GRPO-7B",
    **_LOCAL_INFERENCE_PARAMS,
}

# ============================================================
# Video Sampling Strategy per Model Family
# ============================================================

SAMPLING_STRATEGY = {
    "closed_source": {
        "fps": 2,
        "max_frames": None,       # No limit
        "resolution": 448,        # Compress to 448x448
        "skip_long_video": False,  # Process all durations
        "two_pass_threshold": None,  # No 2-pass needed (large context)
    },
    "qwen3_omni": {
        "fps": 2,
        "max_frames": 192,
        "resolution": 448,
        "skip_long_video": False,
        "two_pass_threshold": 60,  # >1min => 2-pass inference
    },
    "qwen25_omni": {
        "fps": 2,
        "max_frames": 192,
        "resolution": 448,
        "skip_long_video": True,   # Only test <1min videos
        "max_duration": 60,        # Skip videos > 60s
        "two_pass_threshold": None,
    },
}

# ============================================================
# Paths
# ============================================================

# Input data
INPUT_JSONL = os.environ.get(
    "OMNICAP_EVAL_INPUT_JSONL",
    os.path.join(os.path.dirname(BASE_DIR), "examples", "eval_sample.jsonl"),
)

# Media root — relative video_path entries in the annotations are resolved against
# this directory. Leave empty when the annotations already carry absolute paths.
VIDEO_ROOT = os.environ.get("OMNICAP_EVAL_VIDEO_ROOT", "")

# Prompt templates
PROMPT_DIR = os.environ.get("OMNICAP_EVAL_PROMPT_DIR", os.path.join(BASE_DIR, "mtss_generate_prompts"))

# Output directory — results/errors/progress per model
OUTPUT_DIR = os.environ.get("OMNICAP_EVAL_OUTPUT_DIR", os.path.join(os.path.dirname(BASE_DIR), "output", "eval"))

# Video cache directory — preprocessed videos (separate from output)
VIDEO_CACHE_DIR = os.environ.get("OMNICAP_EVAL_VIDEO_CACHE_DIR", os.path.join(os.path.dirname(BASE_DIR), "video_cache"))

# Log directory
LOG_DIR = os.environ.get("OMNICAP_EVAL_LOG_DIR", os.path.join(os.path.dirname(BASE_DIR), "logs", "eval"))

# FFmpeg / FFprobe paths
FFMPEG_PATH = os.environ.get("OMNICAP_EVAL_FFMPEG_PATH", "ffmpeg")
FFPROBE_PATH = os.environ.get("OMNICAP_EVAL_FFPROBE_PATH", "ffprobe")

# ============================================================
# Pipeline Control
# ============================================================

# Parallel workers for concurrent inference
# - API models (gemini, qwen35_omni_plus): use 50-100 workers
# - Local vLLM models: use 4 workers (30B) or 8 workers (7B) matching VLLM_MAX_NUM_SEQS
# - Local Transformers models: use 1 worker per GPU (8 GPUs per node)
NUM_WORKERS = 50

# API retry parameters
# Remote API models (gemini, qwen35_omni_plus): high retries for rate limits/transient errors
API_MAX_RETRIES = 200
API_RETRY_DELAY = 10.0
# Local vLLM models: low retries (failures are usually deterministic, not transient)
LOCAL_MAX_RETRIES = 5
LOCAL_RETRY_DELAY = 5.0

# Shot group duration target for 2-pass long video inference (seconds)
# When splitting shots into groups, each group targets ~60s of video
SHOT_GROUP_TARGET_DURATION = 60

# Rounds for single-pass (<1min or closed_source)
ROUNDS_SINGLE = ["reference_bank", "event", "shot"]

# Rounds for 2-pass long video
ROUND_PASS1 = "reference_bank"
ROUNDS_PASS2 = ["event", "shot"]
