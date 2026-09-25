#!/usr/bin/env python3
"""
OmniCapBench Pipeline configuration.

Credentials, service URLs, and local file paths are read from environment
variables. The repository intentionally ships with empty or public-safe defaults.
"""

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent


def _env(name: str, default: str = "") -> str:
    """Return an environment variable value with a safe default."""
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    """Return an integer environment variable value."""
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    """Return a float environment variable value."""
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Return a boolean environment variable value."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


# ============================================================
# Model presets
# ============================================================

PRESET_QWEN_OMNI = {
    "model_type": "qwen_omni",
    "base_url": _env("OMNICAP_QWEN_OMNI_BASE_URL", "http://localhost:8000/v1"),
    "api_key": _env("OMNICAP_QWEN_OMNI_API_KEY", "EMPTY"),
    "model_name": _env("OMNICAP_QWEN_OMNI_MODEL", "qwen-omni"),
    "timeout": _env_int("OMNICAP_QWEN_OMNI_TIMEOUT", 300),
}

PRESET_GEMINI_31 = {
    "model_type": "gemini",
    "api_key": _env("GEMINI_API_KEY"),
    "model_name": _env("OMNICAP_GEMINI31_MODEL", "gemini-1.5-pro"),
    "max_output_tokens": _env_int("OMNICAP_GEMINI31_MAX_TOKENS", 8192),
    "timeout": _env_int("OMNICAP_GEMINI31_TIMEOUT", 600),
    "request_timeout": _env_int("OMNICAP_GEMINI31_REQUEST_TIMEOUT", 3600),
}

PRESET_GEMINI_25 = {
    "model_type": "gemini",
    "api_key": _env("GEMINI_API_KEY"),
    "model_name": _env("OMNICAP_GEMINI25_MODEL", "gemini-2.5-pro"),
    "max_output_tokens": _env_int("OMNICAP_GEMINI25_MAX_TOKENS", 65536),
    "timeout": _env_int("OMNICAP_GEMINI25_TIMEOUT", 600),
    "request_timeout": _env_int("OMNICAP_GEMINI25_REQUEST_TIMEOUT", 3600),
}

PRESET_QWEN35_OMNI_PLUS = {
    "model_type": "openai_compatible",
    "api_key": _env("OMNICAP_QWEN35_API_KEY"),
    "base_url": _env("OMNICAP_QWEN35_BASE_URL"),
    "model_name": _env("OMNICAP_QWEN35_MODEL", "qwen3.5-omni-plus"),
    "max_completion_tokens": _env_int("OMNICAP_QWEN35_MAX_COMPLETION_TOKENS", 16384),
    "temperature": _env_float("OMNICAP_QWEN35_TEMPERATURE", 0.9),
    "request_timeout": _env_int("OMNICAP_QWEN35_REQUEST_TIMEOUT", 3600),
}

PRESET_OPENROUTER = {
    "model_type": "openai_compatible",
    "api_key": _env("OPENROUTER_API_KEY"),
    "base_url": "https://openrouter.ai/api/v1",
    "model_name": _env("OMNICAP_OPENROUTER_MODEL", "google/gemini-2.5-pro"),
    "max_completion_tokens": _env_int("OMNICAP_OPENROUTER_MAX_TOKENS", 65536),
    "temperature": _env_float("OMNICAP_OPENROUTER_TEMPERATURE", 0.7),
    "request_timeout": _env_int("OMNICAP_OPENROUTER_REQUEST_TIMEOUT", 3600),
}

# ============================================================
# Per-stage model assignment
# ============================================================

STAGE1_MODEL = {**PRESET_QWEN_OMNI}
STAGE2_MODELS = [{**PRESET_QWEN_OMNI}, {**PRESET_QWEN_OMNI}, {**PRESET_QWEN_OMNI}]
STAGE3_MODELS = [{**PRESET_QWEN_OMNI}, {**PRESET_QWEN_OMNI}]
STAGE4_MODEL = {**PRESET_QWEN_OMNI}

STAGE2_FALLBACK_MAP = {
    "qwen35_omni_plus": [PRESET_GEMINI_25, PRESET_GEMINI_31],
    "qwen_omni": [PRESET_GEMINI_25, PRESET_GEMINI_31],
}

# ============================================================
# Pipeline control
# ============================================================

STAGE2_MAX_RETRY = _env_int("OMNICAP_STAGE2_MAX_RETRY", 4)
STAGE3_MAX_RETRY = _env_int("OMNICAP_STAGE3_MAX_RETRY", 3)

SHOT_CHUNK_SIZE = _env_int("OMNICAP_SHOT_CHUNK_SIZE", 10)
SHOT_CHUNK_THRESHOLD = _env_int("OMNICAP_SHOT_CHUNK_THRESHOLD", 22)

API_MAX_RETRIES = _env_int("OMNICAP_API_MAX_RETRIES", 200)
API_RETRY_DELAY = _env_float("OMNICAP_API_RETRY_DELAY", 10.0)

SAVE_INTERMEDIATE = _env_bool("OMNICAP_SAVE_INTERMEDIATE", True)
NUM_WORKERS = _env_int("OMNICAP_NUM_WORKERS", 16)

# ============================================================
# Paths
# ============================================================

PROMPT_DIR = str(PACKAGE_DIR / "prompts")
ORIGINAL_PROMPT_DIR = _env(
    "OMNICAP_ORIGINAL_PROMPT_DIR",
    str(PACKAGE_DIR / "mtss_generate_prompts"),
)

DEFAULT_INPUT_JSON = _env(
    "OMNICAP_INPUT_JSON",
    str(PROJECT_ROOT / "examples" / "generation_input_sample.json"),
)
DEFAULT_OUTPUT_DIR = _env("OMNICAP_OUTPUT_DIR", str(PROJECT_ROOT / "output"))
DEBUG_DIR = _env("OMNICAP_DEBUG_DIR", str(PROJECT_ROOT / "debug"))
LOG_DIR = _env("OMNICAP_LOG_DIR", str(PROJECT_ROOT / "logs"))

FFMPEG_PATH = _env("OMNICAP_FFMPEG_PATH", "ffmpeg")
FFPROBE_PATH = _env("OMNICAP_FFPROBE_PATH", "ffprobe")

# ============================================================
# Round definitions
# ============================================================

ROUNDS = ["reference_bank", "event", "shot"]
