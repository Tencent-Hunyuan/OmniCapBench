#!/usr/bin/env python3
"""Unified text-generation client for MTSS judge and conversion.

Works with any OpenAI-compatible endpoint:
  - OpenAI (https://api.openai.com/v1)
  - OpenRouter (https://openrouter.ai/api/v1)
  - DashScope / Qwen (https://dashscope.aliyuncs.com/compatible-mode/v1)
  - Self-hosted vLLM / TGI
  - Any other OpenAI-compatible provider
"""

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)


class UnifiedTextClient:
    """Text-only client wrapper used by both LLM judge and converter.

    This client uses the OpenAI Python SDK to communicate with any
    OpenAI-compatible chat completions endpoint. Configure it with
    an API key, base URL, and model name.

    Examples:
        # OpenAI
        client = UnifiedTextClient(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
        )

        # OpenRouter (access 200+ models with one key)
        client = UnifiedTextClient(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            model="google/gemini-2.5-pro",
        )

        # Qwen / DashScope
        client = UnifiedTextClient(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-plus",
        )

        # Local vLLM
        client = UnifiedTextClient(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1",
            model="Qwen3-32B",
        )
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "http://localhost:8000/v1",
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 300,
        max_retries: int = 5,
    ):
        self.model = model
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
            base_url=base_url,
            timeout=timeout,
        )

    def complete(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """Generate a text completion.

        Args:
            prompt: The user message text.
            system_prompt: Optional system message.

        Returns:
            The model's response text, or None on failure after all retries.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                content = response.choices[0].message.content
                if content:
                    return content
                logger.warning("Empty response content from model")
                if attempt < self.max_retries - 1:
                    time.sleep(min(2 ** (attempt + 1), 30))
            except Exception as exc:
                logger.warning(
                    "Request failed (attempt %d/%d): %s",
                    attempt + 1, self.max_retries, str(exc)[:200],
                )
                if attempt < self.max_retries - 1:
                    time.sleep(min(2 ** (attempt + 1), 30))
        return None
