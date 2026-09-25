#!/usr/bin/env python3
"""
MTSS Pipeline - Model Client
Unified model client abstraction supporting multiple backends:
  - qwen_omni:          OpenAI-compatible API (local vLLM, base64 video/audio)
  - gemini:             Google Gemini via official google-genai SDK
  - openai_compatible:  Any OpenAI-compatible API (Qwen/DashScope, OpenRouter, OpenAI, etc.)

All clients expose a unified `inference(content, system_prompt)` interface.
"""

import json
import logging
import os
import random
import time as _time

from json_repair import repair_json

from . import config as _config

logger = logging.getLogger("omnicapbench_pipeline")

# Default retry parameters from config
_DEFAULT_MAX_RETRIES = getattr(_config, 'API_MAX_RETRIES', 100)
_DEFAULT_RETRY_DELAY = getattr(_config, 'API_RETRY_DELAY', 10.0)


# HTTP 400 error patterns that are deterministic and should NOT be retried
_NON_RETRYABLE_PATTERNS = [
    "InvalidParameter",       # Input length exceeds model limit
    "Range of input length",  # Same as above, more specific
]

# HTTP 400 error patterns that are semi-deterministic: retry a few times (content safety)
_LIMITED_RETRY_PATTERNS = [
    "DataInspectionFailed",   # Content safety filter triggered
]
_LIMITED_RETRY_MAX = 3  # Max retries for semi-deterministic errors

# Sentinel return value when input exceeds model's token limit
INPUT_TOO_LONG_SENTINEL = "__INPUT_TOO_LONG__"


def _is_non_retryable_error(error_text: str) -> bool:
    """Check if an HTTP error is deterministic and should not be retried."""
    for pattern in _NON_RETRYABLE_PATTERNS:
        if pattern in error_text:
            return True
    return False


def _is_limited_retry_error(error_text: str) -> bool:
    """Check if an HTTP error should only be retried a limited number of times."""
    for pattern in _LIMITED_RETRY_PATTERNS:
        if pattern in error_text:
            return True
    return False


def _backoff_sleep(attempt: int, base_delay: float, max_delay: float = 120.0):
    """Exponential backoff with jitter. Cap at max_delay seconds."""
    delay = min(base_delay * (1.5 ** attempt), max_delay)
    jitter = random.uniform(0, delay * 0.3)
    actual = delay + jitter
    logger.info(f"[Retry] Sleeping {actual:.1f}s before retry (attempt {attempt+1})")
    _time.sleep(actual)


# ======================================================================
# Base class — unified interface
# ======================================================================

class BaseModelClient:
    """Abstract base for all model clients."""

    def __init__(self, tag: str = ""):
        self.tag = tag

    def inference(self, content, system_prompt: str = "",
                  max_retries: int = None, retry_delay: float = None) -> str:
        """
        Call the model with multimodal content.

        Args:
            content:       list of content dicts (video_url / audio_url / text)
            system_prompt: optional system prompt string
            max_retries:   max retry attempts on failure
            retry_delay:   base delay between retries in seconds

        Returns:
            Raw model response text, or "" on failure.
        """
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(tag={self.tag})"


# ======================================================================
# QwenOmni — OpenAI-compatible (existing, local vLLM endpoint)
#    Input: base64 data-URL for video & audio
# ======================================================================

class QwenOmniClient(BaseModelClient):
    """OpenAI-compatible client for local Qwen-Omni vLLM endpoint."""

    def __init__(self, base_url: str, api_key: str, model_name: str,
                 timeout: int = 300, tag: str = ""):
        super().__init__(tag=tag or model_name)
        self.base_url = base_url
        self.model_name = model_name
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def inference(self, content, system_prompt: str = "",
                  max_retries: int = None, retry_delay: float = None) -> str:
        max_retries = max_retries or _DEFAULT_MAX_RETRIES
        retry_delay = retry_delay or _DEFAULT_RETRY_DELAY
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )
                result = response.choices[0].message.content
                logger.info(f"[{self.tag}] Inference OK, length={len(result)}")
                return result
            except Exception as e:
                logger.warning(
                    f"[{self.tag}] Attempt {attempt+1}/{max_retries} FAILED: {e}"
                )
                if attempt < max_retries - 1:
                    _backoff_sleep(attempt, retry_delay)
                else:
                    logger.error(f"[{self.tag}] All {max_retries} attempts failed")
        return ""


# ======================================================================
# Gemini — Google official google-genai SDK
# ======================================================================

class GeminiClient(BaseModelClient):
    """
    Google Gemini client using the official google-genai SDK.
    Supports video input via file upload + generate_content.
    """

    def __init__(self, api_key: str = "", model_name: str = "gemini-2.5-pro",
                 max_output_tokens: int = 65536, timeout: int = 600,
                 request_timeout: int = 3600, tag: str = ""):
        super().__init__(tag=tag or model_name)
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.request_timeout = request_timeout
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)

    def _extract_video_and_text(self, content):
        """Extract video path/URL and text from content list."""
        video_path = ""
        text_parts = []
        for item in content:
            if item.get("type") == "video_url":
                vu = item.get("video_url", {})
                video_path = vu.get("_local_path", "") or vu.get("_http_url", "") or vu.get("url", "")
            elif item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return video_path, "\n".join(text_parts)

    def inference(self, content, system_prompt: str = "",
                  max_retries: int = None, retry_delay: float = None) -> str:
        from google.genai import types

        max_retries = max_retries or _DEFAULT_MAX_RETRIES
        retry_delay = retry_delay or _DEFAULT_RETRY_DELAY
        self._ensure_client()

        video_path, prompt = self._extract_video_and_text(content)
        if not video_path:
            logger.error(f"[{self.tag}] No video path/URL found in content")
            return ""

        _limited_retry_count = 0
        for attempt in range(max_retries):
            try:
                # Upload video file
                video_file = self._client.files.upload(file=video_path)
                # Poll until processing is complete
                poll_count = 0
                while video_file.state.name == "PROCESSING":
                    _time.sleep(3)
                    video_file = self._client.files.get(name=video_file.name)
                    poll_count += 1
                    if poll_count > 200:
                        logger.warning(f"[{self.tag}] Video processing timeout")
                        break

                if video_file.state.name == "FAILED":
                    logger.warning(f"[{self.tag}] Video processing failed")
                    if attempt < max_retries - 1:
                        _backoff_sleep(attempt, retry_delay)
                    continue

                # Generate content
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=[video_file, prompt],
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt if system_prompt else None,
                        max_output_tokens=self.max_output_tokens,
                    ),
                )
                result = response.text
                if result:
                    logger.info(f"[{self.tag}] Inference OK, length={len(result)}")
                    return result
                logger.warning(f"[{self.tag}] Empty response from Gemini")
                if attempt < max_retries - 1:
                    _backoff_sleep(attempt, retry_delay)

            except Exception as e:
                error_str = str(e)
                logger.warning(f"[{self.tag}] Attempt {attempt+1}/{max_retries} FAILED: {e}")
                if _is_non_retryable_error(error_str):
                    return INPUT_TOO_LONG_SENTINEL
                if _is_limited_retry_error(error_str):
                    _limited_retry_count += 1
                    if _limited_retry_count >= _LIMITED_RETRY_MAX:
                        return ""
                if attempt < max_retries - 1:
                    _backoff_sleep(attempt, retry_delay)
        return ""


# ======================================================================
# OpenAI-compatible — works with Qwen/DashScope, OpenRouter, OpenAI, etc.
# ======================================================================

class OpenAICompatibleClient(BaseModelClient):
    """
    Generic OpenAI-compatible client for multimodal video inference.
    Works with any provider that supports the OpenAI chat completions API:
      - Qwen / DashScope (https://dashscope.aliyuncs.com/compatible-mode/v1)
      - OpenRouter (https://openrouter.ai/api/v1)
      - OpenAI (https://api.openai.com/v1)
      - Self-hosted vLLM / TGI endpoints
    """

    def __init__(self, api_key: str = "", base_url: str = "",
                 model_name: str = "", max_completion_tokens: int = 32768,
                 temperature: float = 0.7, request_timeout: int = 3600,
                 tag: str = ""):
        super().__init__(tag=tag or model_name)
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=request_timeout,
        )
        self.model_name = model_name
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature

    def _extract_video_url_and_text(self, content):
        video_url = ""
        text_parts = []
        for item in content:
            if item.get("type") == "video_url":
                vu = item.get("video_url", {})
                video_url = vu.get("_http_url", "") or vu.get("url", "")
            elif item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        return video_url, "\n".join(text_parts)

    def inference(self, content, system_prompt: str = "",
                  max_retries: int = None, retry_delay: float = None) -> str:
        max_retries = max_retries or _DEFAULT_MAX_RETRIES
        retry_delay = retry_delay or _DEFAULT_RETRY_DELAY
        video_url, prompt = self._extract_video_url_and_text(content)
        if not video_url:
            logger.error(f"[{self.tag}] No video URL found in content")
            return ""

        user_parts = [
            {"type": "video_url", "video_url": {"url": video_url}},
            {"type": "text", "text": prompt},
        ]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_parts},
        ]

        _limited_retry_count = 0
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_completion_tokens=self.max_completion_tokens,
                    temperature=self.temperature,
                )
                msg = response.choices[0].message
                content_val = msg.content
                if isinstance(content_val, str):
                    result = content_val
                elif isinstance(content_val, list):
                    parts = [b.get("text", "") for b in content_val
                             if isinstance(b, dict) and b.get("type") == "text"]
                    result = "\n".join(parts)
                else:
                    result = str(content_val) if content_val else ""
                if result:
                    logger.info(f"[{self.tag}] Inference OK, length={len(result)}")
                    return result
                logger.warning(f"[{self.tag}] Empty response")
                if attempt < max_retries - 1:
                    _backoff_sleep(attempt, retry_delay)
            except Exception as e:
                error_str = str(e)
                logger.warning(f"[{self.tag}] Attempt {attempt+1}/{max_retries} FAILED: {e}")
                if _is_non_retryable_error(error_str):
                    return INPUT_TOO_LONG_SENTINEL
                if _is_limited_retry_error(error_str):
                    _limited_retry_count += 1
                    if _limited_retry_count >= _LIMITED_RETRY_MAX:
                        return ""
                if attempt < max_retries - 1:
                    _backoff_sleep(attempt, retry_delay)
        return ""


# ======================================================================
# Backward-compatible alias
# ======================================================================
ModelClient = QwenOmniClient


# ======================================================================
# Factory function — create the right client from config dict
# ======================================================================

def create_model_client(cfg: dict, tag: str = "") -> BaseModelClient:
    """
    Create a model client based on the 'model_type' field in config.

    Supported model_type values:
      - "qwen_omni"          (default) — local vLLM OpenAI-compatible
      - "gemini"             — Google Gemini via official SDK
      - "openai_compatible"  — Any OpenAI-compatible API (Qwen, OpenRouter, OpenAI, etc.)
    """
    model_type = cfg.get("model_type", "qwen_omni")

    if model_type == "qwen_omni":
        return QwenOmniClient(
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
            model_name=cfg["model_name"],
            timeout=cfg.get("timeout", 300),
            tag=tag,
        )
    elif model_type == "gemini":
        return GeminiClient(
            api_key=cfg.get("api_key", ""),
            model_name=cfg.get("model_name", "gemini-2.5-pro"),
            max_output_tokens=cfg.get("max_output_tokens", 65536),
            timeout=cfg.get("timeout", 600),
            request_timeout=cfg.get("request_timeout", 3600),
            tag=tag,
        )
    elif model_type == "openai_compatible":
        return OpenAICompatibleClient(
            api_key=cfg.get("api_key", ""),
            base_url=cfg.get("base_url", ""),
            model_name=cfg.get("model_name", ""),
            max_completion_tokens=cfg.get("max_completion_tokens", 32768),
            temperature=cfg.get("temperature", 0.7),
            request_timeout=cfg.get("request_timeout", 3600),
            tag=tag,
        )
    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            f"Supported: qwen_omni, gemini, openai_compatible"
        )


# ======================================================================
# JSON parsing utilities
# ======================================================================

def parse_response_to_json(response: str) -> dict:
    """
    Parse model response text into a Python dict.
    Tries multiple strategies: direct parse -> code-block extraction ->
    json_repair -> brace extraction.
    """
    if not response:
        return {"raw_response": "", "parse_error": True}

    def _ensure_dict(obj):
        """If obj is a list, wrap it in a dict with an inferred key."""
        if isinstance(obj, list):
            if obj and isinstance(obj[0], dict):
                if "ref_id" in obj[0]:
                    return {"references": obj}
                elif "event_id" in obj[0]:
                    return {"events": obj}
                elif "shot_id" in obj[0]:
                    return {"shots": obj}
            return {"items": obj}
        if isinstance(obj, dict):
            return obj
        return {"raw_response": str(obj), "parse_error": True}

    # 1. Direct parse
    try:
        parsed = json.loads(response)
        return _ensure_dict(parsed)
    except json.JSONDecodeError:
        pass

    # 2. Extract from ```json ... ``` or ``` ... ```
    try:
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            parsed = json.loads(response[start:end].strip())
            return _ensure_dict(parsed)
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            parsed = json.loads(response[start:end].strip())
            return _ensure_dict(parsed)
    except Exception:
        pass

    # 3. json_repair
    try:
        repaired = repair_json(response)
        parsed = json.loads(repaired)
        return _ensure_dict(parsed)
    except Exception:
        pass

    # 4. Brace extraction (try object first, then array)
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(response[start:end])
            return _ensure_dict(parsed)
    except Exception:
        pass

    # 5. Array extraction
    try:
        start = response.find("[")
        end = response.rfind("]") + 1
        if start != -1 and end > start:
            parsed = json.loads(response[start:end])
            return _ensure_dict(parsed)
    except Exception:
        pass

    return {"raw_response": response, "parse_error": True}
