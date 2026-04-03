"""LLM client wrapper for vLLM's OpenAI-compatible API.

Shared by ALL agents — Simple RAG and Agentic RAG use the same client.
Returns raw response with token usage stats for cost tracking.
"""

import re
from dataclasses import dataclass
from typing import List, Dict

from openai import OpenAI


def strip_think_tags(text: str) -> tuple[str, str]:
    """Strip <think>...</think> tags from model output.

    Returns:
        (clean_answer, thinking_content) tuple.
    """
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    thinking = think_match.group(1).strip() if think_match else ""
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return clean, thinking


@dataclass
class LLMResponse:
    """Structured response from the LLM."""
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMClient:
    """Thin wrapper around vLLM's OpenAI-compatible API."""

    def __init__(self, base_url: str, model: str, temperature: float = 0, max_tokens: int = 512):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(base_url=base_url, api_key="not-needed")

    def generate(self, messages: List[Dict[str, str]], temperature: float = None, max_tokens: int = None) -> LLMResponse:
        """Send a chat completion request and return structured response.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            temperature: Override default temperature (optional).
            max_tokens: Override default max_tokens (optional).

        Returns:
            LLMResponse with content and token usage.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
        )

        usage = response.usage
        return LLMResponse(
            content=response.choices[0].message.content.strip(),
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )
