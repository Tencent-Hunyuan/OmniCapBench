#!/usr/bin/env python3
"""
MTSS Eval - Model Client
Unified model client abstraction for evaluation inference.

Supported model_type values:
  - "qwen_omni"          — OpenAI-compatible API (base64 video+audio), also used for local vLLM
  - "gemini"             — Google Gemini via official google-genai SDK
  - "openai_compatible"  — Any OpenAI-compatible API (Qwen/DashScope, OpenRouter, OpenAI, etc.)
"""

import logging
import os
import random
import time as _time

from . import eval_config as config

logger = logging.getLogger("mtss_eval")

_DEFAULT_MAX_RETRIES = config.API_MAX_RETRIES
_DEFAULT_RETRY_DELAY = config.API_RETRY_DELAY

# Non-retryable error patterns
_NON_RETRYABLE_PATTERNS = [
    "InvalidParameter",
    "Range of input length",
]

_LIMITED_RETRY_PATTERNS = [
    "DataInspectionFailed",
]
_LIMITED_RETRY_MAX = 3

INPUT_TOO_LONG_SENTINEL = "__INPUT_TOO_LONG__"


def _is_non_retryable_error(error_text: str) -> bool:
    for pattern in _NON_RETRYABLE_PATTERNS:
        if pattern in error_text:
            return True
    return False


def _is_limited_retry_error(error_text: str) -> bool:
    for pattern in _LIMITED_RETRY_PATTERNS:
        if pattern in error_text:
            return True
    return False


def _backoff_sleep(attempt: int, base_delay: float, max_delay: float = 120.0):
    delay = min(base_delay * (1.5 ** attempt), max_delay)
    jitter = random.uniform(0, delay * 0.3)
    actual = delay + jitter
    logger.info(f"[Retry] Sleeping {actual:.1f}s before retry (attempt {attempt+1})")
    _time.sleep(actual)


# ======================================================================
# Base class
# ======================================================================

class BaseModelClient:
    def __init__(self, tag: str = ""):
        self.tag = tag

    def inference(self, content, system_prompt: str = "",
                  max_retries: int = None, retry_delay: float = None) -> str:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(tag={self.tag})"


# ======================================================================
# QwenOmni — OpenAI-compatible (base64 video+audio)
# ======================================================================

class QwenOmniClient(BaseModelClient):
    def __init__(self, base_url: str, api_key: str, model_name: str,
                 timeout: int = 300, max_new_tokens: int = 32768,
                 temperature: float = 0.7, top_p: float = 0.9,
                 max_retries: int = None, retry_delay: float = None,
                 tag: str = ""):
        super().__init__(tag=tag or model_name)
        self.base_url = base_url
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._default_max_retries = max_retries
        self._default_retry_delay = retry_delay
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def inference(self, content, system_prompt: str = "",
                  max_retries: int = None, retry_delay: float = None) -> str:
        max_retries = max_retries or self._default_max_retries or _DEFAULT_MAX_RETRIES
        retry_delay = retry_delay or self._default_retry_delay or _DEFAULT_RETRY_DELAY
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        current_max_tokens = min(self.max_new_tokens, 12288)
        _input_too_long_count = 0
        _INPUT_TOO_LONG_MAX = 3

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_completion_tokens=current_max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                result = response.choices[0].message.content
                logger.info(f"[{self.tag}] Inference OK, length={len(result)}, "
                            f"max_tokens_used={current_max_tokens}")
                return result
            except Exception as e:
                error_str = str(e)
                logger.warning(f"[{self.tag}] Attempt {attempt+1}/{max_retries} FAILED: {e}")

                if "max_tokens" in error_str and "is too large" in error_str:
                    import re
                    m = re.search(r'has (\d+) input tokens', error_str)
                    if m:
                        reported_input = int(m.group(1))
                        new_max = 32768 - reported_input - 64
                        if new_max > 256:
                            current_max_tokens = new_max
                            logger.info(f"[{self.tag}] Auto-reducing max_completion_tokens to "
                                        f"{current_max_tokens} (input={reported_input})")
                            continue
                        else:
                            return INPUT_TOO_LONG_SENTINEL

                if ("maximum context length" in error_str and "input tokens" in error_str
                        and "reduce the length" in error_str):
                    _input_too_long_count += 1
                    if _input_too_long_count >= _INPUT_TOO_LONG_MAX:
                        return INPUT_TOO_LONG_SENTINEL
                    if attempt < max_retries - 1:
                        _backoff_sleep(attempt, retry_delay)
                    continue

                if "decoder prompt" in error_str and "longer than the maximum model length" in error_str:
                    return INPUT_TOO_LONG_SENTINEL

                if attempt < max_retries - 1:
                    _backoff_sleep(attempt, retry_delay)
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
                video_path = vu.get("_local_path", "") or vu.get("url", "")
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
                    if poll_count > 200:  # ~10 min max wait
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
                video_url = vu.get("url", "")
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
# Qwen2.5-Omni Transformers — local native inference (no vLLM)
# For smaller fine-tuned models: ASID, AVoCaDO, TimeChat, UGC
# ======================================================================

class Qwen25OmniTransformersClient(BaseModelClient):
    """
    Local Qwen2.5-Omni inference using Transformers native pipeline.
    Uses Qwen2_5OmniForConditionalGeneration + qwen_omni_utils.process_mm_info.
    Loads model once, runs inference on GPU directly.
    """

    def __init__(self, model_path: str, model_name: str = "",
                 max_new_tokens: int = 32768, temperature: float = 0.7,
                 top_p: float = 0.9, tag: str = ""):
        super().__init__(tag=tag or model_name or os.path.basename(model_path))
        self.model_path = model_path
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._model = None
        self._processor = None

    def _ensure_loaded(self):
        """Lazy-load model and processor on first call."""
        if self._model is not None:
            return
        import torch

        logger.info(f"[{self.tag}] Loading model from {self.model_path} ...")

        try:
            from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
            ModelClass = Qwen2_5OmniForConditionalGeneration
            ProcessorClass = Qwen2_5OmniProcessor
            logger.info(f"[{self.tag}] Using Qwen2_5OmniForConditionalGeneration")
        except ImportError:
            from transformers import AutoModelForCausalLM, AutoProcessor
            ModelClass = AutoModelForCausalLM
            ProcessorClass = AutoProcessor
            logger.info(f"[{self.tag}] Falling back to AutoModelForCausalLM")

        self._processor = ProcessorClass.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self._model = ModelClass.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        if hasattr(self._model, 'disable_talker'):
            self._model.disable_talker()
        self._model.eval()
        logger.info(f"[{self.tag}] Model loaded successfully.")

    def inference(self, content, system_prompt: str = "",
                  max_retries: int = None, retry_delay: float = None) -> str:
        import torch
        try:
            from qwen_omni_utils import process_mm_info
        except ImportError:
            raise ImportError(
                "qwen_omni_utils is required for Transformers-based Qwen2.5-Omni inference. "
                "Install it with: pip install qwen-omni-utils[decord]"
            )
        from . import video_utils as _video_utils

        self._ensure_loaded()

        user_content = []
        for item in content:
            item_type = item.get("type", "")
            if item_type == "video":
                user_content.append({"type": "video", "video": item["video"]})
            elif item_type == "text":
                text = item.get("text", "")
                if text:
                    user_content.append({"type": "text", "text": text})

        conversation = []
        if system_prompt:
            conversation.append({
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            })
        conversation.append({"role": "user", "content": user_content})

        try:
            text_input = self._processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )

            _has_audio = True
            for item in content:
                if item.get("type") == "video" and item.get("video"):
                    _has_audio = _video_utils.has_audio_track(item["video"])
                    break
            USE_AUDIO_IN_VIDEO = _has_audio
            if not USE_AUDIO_IN_VIDEO:
                logger.info(f"[{self.tag}] Video has no audio track, using use_audio_in_video=False")
            if USE_AUDIO_IN_VIDEO:
                import shutil
                if not shutil.which("ffmpeg"):
                    USE_AUDIO_IN_VIDEO = False
                    logger.warning(f"[{self.tag}] ffmpeg not found, forcing use_audio_in_video=False")
            audios, images, videos = process_mm_info(
                conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO
            )

            inputs = self._processor(
                text=text_input,
                audio=audios if audios else None,
                images=images if images else None,
                videos=videos if videos else None,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=USE_AUDIO_IN_VIDEO,
            )
            inputs = inputs.to(self._model.device).to(self._model.dtype)

            with torch.no_grad():
                text_ids = self._model.generate(
                    **inputs,
                    use_audio_in_video=USE_AUDIO_IN_VIDEO,
                    return_audio=False,
                    thinker_max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True if self.temperature > 0 else False,
                )

            input_len = inputs["input_ids"].shape[1]
            generated_ids = text_ids[:, input_len:]
            result = self._processor.batch_decode(
                generated_ids, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            if result.startswith("\nassistant\n"):
                result = result.split("\nassistant\n", 1)[-1].strip()

            logger.info(f"[{self.tag}] Inference OK, length={len(result)}")
            return result

        except Exception as e:
            logger.error(f"[{self.tag}] Transformers inference error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""


# ======================================================================
# Factory function
# ======================================================================

def create_model_client(cfg: dict, tag: str = "") -> BaseModelClient:
    """Create a model client based on the 'model_type' and 'inference_backend' fields in config."""
    model_type = cfg.get("model_type", "qwen_omni")
    inference_backend = cfg.get("inference_backend", "")

    # Transformers native backend (for local fine-tuned qwen2.5-omni models)
    if inference_backend == "transformers":
        return Qwen25OmniTransformersClient(
            model_path=cfg["model_path"],
            model_name=cfg.get("model_name", ""),
            max_new_tokens=cfg.get("max_new_tokens", 32768),
            temperature=cfg.get("temperature", 0.7),
            top_p=cfg.get("top_p", 0.9),
            tag=tag,
        )

    if model_type == "qwen_omni":
        return QwenOmniClient(
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
            model_name=cfg["model_name"],
            timeout=cfg.get("timeout", 300),
            max_new_tokens=cfg.get("max_new_tokens", 32768),
            temperature=cfg.get("temperature", 0.7),
            top_p=cfg.get("top_p", 0.9),
            max_retries=cfg.get("max_retries"),
            retry_delay=cfg.get("retry_delay"),
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
        raise ValueError(f"Unknown model_type: '{model_type}'")
