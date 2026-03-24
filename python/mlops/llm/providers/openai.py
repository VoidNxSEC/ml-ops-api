"""
OpenAI provider.

Connects to OpenAI API for GPT models or OpenAI-compatible endpoints (DeepSeek, etc.).
Requires: openai Python package.

Migrated from neutron/agents/providers/openai.py.
"""

from __future__ import annotations

import os
from typing import Any

from .base import LLMProvider, LLMResponse, ProviderConfig


class OpenAIProvider(LLMProvider):
    """Provider for OpenAI API and OpenAI-compatible endpoints.

    Supports both OpenAI GPT models and compatible APIs like DeepSeek.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        **kwargs,
    ):
        config = ProviderConfig(
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
            base_url=base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=model or os.getenv("OPENAI_MODEL", "gpt-4"),
            **kwargs,
        )
        super().__init__(config)

        try:
            import openai
            self._openai = openai
            self._client = openai.AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
        except ImportError as e:
            raise ImportError(
                "openai package required for OpenAIProvider. "
                "Install with: pip install openai"
            ) from e

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        """Generate completion via chat endpoint."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        return await self.generate_chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )

    async def generate_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        """Generate from chat messages via /v1/chat/completions."""
        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }

        if stop:
            params["stop"] = stop

        response = await self._client.chat.completions.create(**params)

        choice = response.choices[0] if response.choices else None
        content = choice.message.content if choice and choice.message else ""

        return LLMResponse(
            content=content or "",
            model=response.model,
            finish_reason=choice.finish_reason if choice else "stop",
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )

    async def embed(self, text: str) -> list[float]:
        """Generate embeddings via /v1/embeddings."""
        response = await self._client.embeddings.create(
            model=self.config.extra.get("embedding_model", "text-embedding-3-small"),
            input=text,
        )

        if response.data:
            return response.data[0].embedding
        return []

    async def health(self) -> bool:
        """Check OpenAI API health."""
        try:
            response = await self.generate("ping", max_tokens=10)
            return len(response.content) > 0
        except Exception:
            return False

    def __repr__(self) -> str:
        api_key_masked = self.config.api_key[:12] + "..." if len(self.config.api_key) > 12 else "***"
        return f"OpenAIProvider(model={self.config.model!r}, base_url={self.config.base_url!r}, api_key={api_key_masked!r})"


class DeepSeekProvider(OpenAIProvider):
    """Provider for DeepSeek API (OpenAI-compatible)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        **kwargs,
    ):
        super().__init__(
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com",
            model=model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            **kwargs,
        )

    def __repr__(self) -> str:
        api_key_masked = self.config.api_key[:12] + "..." if len(self.config.api_key) > 12 else "***"
        return f"DeepSeekProvider(model={self.config.model!r}, api_key={api_key_masked!r})"
