"""
Anthropic Claude provider.

Connects to Anthropic API for Claude models.
Requires: anthropic Python package.

Migrated from neutron/agents/providers/anthropic.py.
"""

from __future__ import annotations

import os
from typing import Any

from .base import LLMProvider, LLMResponse, ProviderConfig


class AnthropicProvider(LLMProvider):
    """Provider for Anthropic Claude API.

    Usage:
        provider = AnthropicProvider(api_key="sk-ant-...", model="claude-3-5-sonnet-20241022")
        response = await provider.generate("Hello, world!")
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        **kwargs,
    ):
        config = ProviderConfig(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY", ""),
            model=model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            **kwargs,
        )
        super().__init__(config)

        try:
            import anthropic
            self._anthropic = anthropic
            self._client = anthropic.AsyncAnthropic(api_key=self.config.api_key)
        except ImportError as e:
            raise ImportError(
                "anthropic package required for AnthropicProvider. "
                "Install with: pip install anthropic"
            ) from e

    def _convert_to_anthropic_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Convert OpenAI-style messages to Anthropic format."""
        system = None
        anthropic_messages = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system = content
            elif role in ("user", "assistant"):
                anthropic_messages.append({"role": role, "content": content})
            else:
                anthropic_messages.append({"role": "user", "content": content})

        return system, anthropic_messages

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        """Generate completion via messages API."""
        messages = [{"role": "user", "content": prompt}]

        return await self.generate_chat(
            messages,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )

    async def generate_chat(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
    ) -> LLMResponse:
        """Generate from chat messages via Anthropic Messages API."""
        extracted_system, anthropic_messages = self._convert_to_anthropic_messages(messages)
        final_system = system or extracted_system

        params: dict[str, Any] = {
            "model": self.config.model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
        }

        if final_system:
            params["system"] = final_system

        if stop:
            params["stop_sequences"] = stop

        response = await self._client.messages.create(**params)

        content = ""
        if response.content:
            content = "".join(
                block.text for block in response.content if hasattr(block, "text")
            )

        return LLMResponse(
            content=content,
            model=response.model,
            finish_reason=response.stop_reason or "stop",
            prompt_tokens=response.usage.input_tokens if response.usage else 0,
            completion_tokens=response.usage.output_tokens if response.usage else 0,
            total_tokens=(
                response.usage.input_tokens + response.usage.output_tokens
                if response.usage
                else 0
            ),
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )

    async def health(self) -> bool:
        """Check Anthropic API health."""
        try:
            response = await self.generate("ping", max_tokens=10)
            return len(response.content) > 0
        except Exception:
            return False

    def __repr__(self) -> str:
        api_key_masked = self.config.api_key[:12] + "..." if len(self.config.api_key) > 12 else "***"
        return f"AnthropicProvider(model={self.config.model!r}, api_key={api_key_masked!r})"
