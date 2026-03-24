"""
Abstract base for LLM providers.

All providers implement the same interface so agents can swap
between llama.cpp, OpenAI, Anthropic, DeepSeek transparently.

Migrated from neutron/agents/providers/base.py.
"""

from __future__ import annotations

import abc
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""

    base_url: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    timeout: float = 30.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""

    content: str
    model: str = ""
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


class LLMProvider(abc.ABC):
    """Abstract LLM provider interface."""

    def __init__(self, config: ProviderConfig | None = None, **kwargs):
        self.config = config or ProviderConfig(**kwargs)

    @abc.abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        """Generate a completion from a prompt."""
        ...

    @abc.abstractmethod
    async def generate_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        """Generate from a chat message list."""
        ...

    async def embed(self, text: str) -> list[float]:
        """Generate embeddings for text. Not all providers support this."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support embeddings"
        )

    async def health(self) -> bool:
        """Check if the provider is reachable."""
        try:
            resp = await self.generate("ping", max_tokens=1)
            return len(resp.content) > 0
        except Exception:
            return False
