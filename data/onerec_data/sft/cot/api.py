"""
LLM client (OpenAI-compatible only).

Config JSON format (path passed via --api_config_path):

    {
        "base_url": "https://api.openai.com/v1",
        "api_key": "your_token_here",
        "model": "selected_model_name"
    }

`base_url` can point to any OpenAI-compatible endpoint (OpenAI / DeepSeek /
vLLM OpenAI server / local vLLM / etc.). `model` in the config overrides the
`--model` CLI arg if present.

Usage:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from api import init_llm_client, TokenStats

    llm = init_llm_client(model="...", api_config_path="/path/to/config.json")
    text = llm.call("your prompt", verbose=True)
    llm.close()
"""

from __future__ import annotations

import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except ModuleNotFoundError as exc:
    raise ImportError(
        "openai library is required. Install with: pip install openai"
    ) from exc


DEFAULT_SYSTEM_PROMPT = "你是一个热心的助手，会帮助人类解决各种各样的问题。"


# =============================================================================
# Token usage tracker
# =============================================================================
class TokenStats:
    """Thread-safe token usage tracker."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.prompt_tokens = 0
        self.response_tokens = 0
        self.thoughts_tokens = 0
        self.cached_tokens = 0
        self.total_tokens = 0
        self.request_count = 0

    def add_usage(self, usage: Any) -> None:
        if usage is None:
            return
        with self.lock:
            self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.response_tokens += int(getattr(usage, "completion_tokens", 0) or 0)

            details = getattr(usage, "completion_tokens_details", None)
            self.thoughts_tokens += int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0

            pdetails = getattr(usage, "prompt_tokens_details", None)
            self.cached_tokens += int(getattr(pdetails, "cached_tokens", 0) or 0) if pdetails else 0

            self.total_tokens += int(getattr(usage, "total_tokens", 0) or 0)
            self.request_count += 1

    def snapshot(self) -> Dict[str, int]:
        with self.lock:
            return {
                "request_count": self.request_count,
                "prompt_tokens": self.prompt_tokens,
                "response_tokens": self.response_tokens,
                "thoughts_tokens": self.thoughts_tokens,
                "cached_tokens": self.cached_tokens,
                "total_tokens": self.total_tokens,
            }

    def delta(self, previous: Optional[Dict[str, int]] = None) -> Dict[str, int]:
        current = self.snapshot()
        if previous is None:
            return current
        return {
            key: int(current.get(key, 0) or 0) - int(previous.get(key, 0) or 0)
            for key in current
        }


# =============================================================================
# Helpers
# =============================================================================
def _normalize_text_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(_normalize_text_content(v) for v in value if v).strip()
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        return ""
    text = getattr(value, "content", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    return ""


def _build_messages(system_prompt: Optional[str], prompt: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


# =============================================================================
# Base client
# =============================================================================
class BaseLLMClient(ABC):
    def __init__(
        self,
        *,
        client: Any,
        model: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.0,
        max_output_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = "none",
        max_retries: int = 3,
        retry_initial_delay: float = 1.0,
        retry_backoff_factor: float = 2.0,
        retry_max_delay: float = 60.0,
        timeout: Optional[int] = None,
    ) -> None:
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self.retry_initial_delay = retry_initial_delay
        self.retry_backoff_factor = retry_backoff_factor
        self.retry_max_delay = retry_max_delay
        self.timeout = timeout

    def call(
        self,
        prompt: str,
        *,
        verbose: bool = False,
        token_stats: Optional[TokenStats] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        max_retries: Optional[int] = None,
        retry_initial_delay: Optional[float] = None,
        retry_backoff_factor: Optional[float] = None,
        retry_max_delay: Optional[float] = None,
    ) -> str:
        final_temperature = self.temperature if temperature is None else temperature
        final_max_output_tokens = self.max_output_tokens if max_output_tokens is None else max_output_tokens
        final_system_prompt = self.system_prompt if system_prompt is None else system_prompt
        final_reasoning_effort = self.reasoning_effort if reasoning_effort is None else reasoning_effort

        attempts = self.max_retries if max_retries is None else max_retries
        delay = self.retry_initial_delay if retry_initial_delay is None else retry_initial_delay
        backoff = self.retry_backoff_factor if retry_backoff_factor is None else retry_backoff_factor
        max_delay = self.retry_max_delay if retry_max_delay is None else retry_max_delay

        if attempts <= 0:
            raise ValueError("max_retries must be > 0")

        last_error: Optional[Exception] = None
        next_delay = max(0.0, delay)
        for attempt in range(1, attempts + 1):
            try:
                text, usage = self._call_once(
                    prompt=prompt,
                    temperature=final_temperature,
                    max_output_tokens=final_max_output_tokens,
                    system_prompt=final_system_prompt,
                    reasoning_effort=final_reasoning_effort,
                )
                if token_stats is not None:
                    token_stats.add_usage(usage)
                if verbose:
                    snap = token_stats.snapshot() if token_stats is not None else {}
                    logging.info(
                        "usage prompt=%s response=%s thoughts=%s cached=%s total=%s",
                        snap.get("prompt_tokens", 0),
                        snap.get("response_tokens", 0),
                        snap.get("thoughts_tokens", 0),
                        snap.get("cached_tokens", 0),
                        snap.get("total_tokens", 0),
                    )
                return text
            except Exception as exc:
                last_error = exc
                if attempt >= attempts:
                    break
                logging.warning(
                    "LLM call failed, retrying: model=%s attempt=%s/%s error=%s",
                    self.model, attempt, attempts, exc,
                )
                sleep_seconds = min(next_delay, max_delay)
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                next_delay = min(max_delay, next_delay * backoff) if next_delay > 0 else 0.0

        raise RuntimeError(
            f"LLM call failed after {attempts} retries: model={self.model}"
        ) from last_error

    def close(self) -> None:
        close_fn = getattr(self.client, "close", None)
        if callable(close_fn):
            close_fn()

    @abstractmethod
    def _call_once(
        self,
        *,
        prompt: str,
        temperature: float,
        max_output_tokens: Optional[int],
        system_prompt: str,
        reasoning_effort: Optional[str],
    ) -> Tuple[str, Any]:
        raise NotImplementedError


# =============================================================================
# OpenAI-compatible client
# =============================================================================
class OpenAIClient(BaseLLMClient):
    def _call_once(
        self,
        *,
        prompt: str,
        temperature: float,
        max_output_tokens: Optional[int],
        system_prompt: str,
        reasoning_effort: Optional[str],
    ) -> Tuple[str, Any]:
        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": _build_messages(system_prompt, prompt),
        }
        if self.timeout is not None:
            request_kwargs["timeout"] = self.timeout
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        if max_output_tokens is not None:
            request_kwargs["max_completion_tokens"] = max_output_tokens
        if reasoning_effort not in (None, "", "none"):
            request_kwargs["reasoning_effort"] = reasoning_effort

        response = self.client.chat.completions.create(**request_kwargs)
        usage = getattr(response, "usage", None)
        message = response.choices[0].message
        text = _normalize_text_content(getattr(message, "content", None))
        if not text:
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            raise RuntimeError(
                f"OpenAI-compatible API returned empty text: finish_reason={finish_reason!r}"
            )
        return text, usage


# =============================================================================
# Factory
# =============================================================================
def init_llm_client(
    *,
    model: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.0,
    max_output_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = "none",
    max_retries: int = 3,
    retry_initial_delay: float = 1.0,
    retry_backoff_factor: float = 2.0,
    retry_max_delay: float = 60.0,
    api_config_path: Optional[str] = None,
    timeout: Optional[int] = None,
) -> BaseLLMClient:
    """Initialize an OpenAI-compatible LLM client from a JSON config.

    Config JSON fields:
        base_url (str): OpenAI-compatible endpoint URL.
        api_key  (str): API token.
        model    (str): Model name (overrides the ``model`` arg if present).

    Args:
        model: Fallback model name if not specified in the config.
        api_config_path: Path to the JSON config file (required).
    """
    if not api_config_path:
        raise ValueError("api_config_path is required (JSON with base_url, api_key, model)")

    with open(api_config_path, "r", encoding="utf-8") as f:
        api_config = json.load(f)

    effective_model = api_config.get("model") or model
    if not effective_model:
        raise ValueError("model must be provided in the config JSON or via the model arg")

    client_kwargs: Dict[str, Any] = {}
    if api_config.get("base_url"):
        client_kwargs["base_url"] = api_config["base_url"]
    if api_config.get("api_key"):
        client_kwargs["api_key"] = api_config["api_key"]
    if api_config.get("organization"):
        client_kwargs["organization"] = api_config["organization"]

    openai_client = OpenAI(**client_kwargs)

    common_kwargs: Dict[str, Any] = dict(
        system_prompt=system_prompt,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        max_retries=max_retries,
        retry_initial_delay=retry_initial_delay,
        retry_backoff_factor=retry_backoff_factor,
        retry_max_delay=retry_max_delay,
    )
    if timeout is not None:
        common_kwargs["timeout"] = timeout

    logging.info(
        "init OpenAI-compatible client: model=%s api_config_path=%s base_url=%s",
        effective_model, api_config_path, api_config.get("base_url"),
    )
    return OpenAIClient(client=openai_client, model=effective_model, **common_kwargs)
