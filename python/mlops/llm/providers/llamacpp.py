"""
llama.cpp OpenAI-compatible provider.

Connects to a running llama-server (or llama-swap) via the
OpenAI-compatible /v1/chat/completions endpoint.

Migrated from neutron/agents/providers/llamacpp.py.
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError

from .base import LLMProvider, LLMResponse, ProviderConfig


class LlamaCppProvider(LLMProvider):
    """Provider for llama.cpp server (llama-server / llama-swap).

    Uses the OpenAI-compatible API exposed by llama.cpp.
    No external dependencies required - uses urllib.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        **kwargs,
    ):
        config = ProviderConfig(
            base_url=base_url or os.getenv("LLAMACPP_URL", "http://localhost:8081"),
            model=model or os.getenv("LLAMACPP_MODEL", "current-model"),
            **kwargs,
        )
        super().__init__(config)

    def _request(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """Make a synchronous HTTP request to llama.cpp server."""
        url = f"{self.config.base_url}{path}"
        data = json.dumps(body).encode("utf-8")
        req = Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        timeout = self.config.timeout
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _build_params(
        self,
        temperature: float | None,
        max_tokens: int | None,
        stop: list[str] | None,
    ) -> dict[str, Any]:
        """Build common request parameters."""
        params: dict[str, Any] = {
            "model": self.config.model,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
        }
        if self.config.top_p is not None:
            params["top_p"] = self.config.top_p
        if stop:
            params["stop"] = stop
        return params

    def _parse_response(self, raw: dict[str, Any]) -> LLMResponse:
        """Parse OpenAI-format response into LLMResponse."""
        choice = raw.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = raw.get("usage", {})

        return LLMResponse(
            content=message.get("content", "").strip(),
            model=raw.get("model", self.config.model),
            finish_reason=choice.get("finish_reason", "stop"),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            raw=raw,
        )

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
        params = self._build_params(temperature, max_tokens, stop)
        params["messages"] = messages

        raw = self._request("/v1/chat/completions", params)
        return self._parse_response(raw)

    async def embed(self, text: str) -> list[float]:
        """Generate embeddings via /v1/embeddings (if supported)."""
        raw = self._request(
            "/v1/embeddings",
            {"input": text, "model": self.config.model},
        )
        data = raw.get("data", [{}])
        if data:
            return data[0].get("embedding", [])
        return []

    async def health(self) -> bool:
        """Check server health via /health endpoint."""
        try:
            url = f"{self.config.base_url}/health"
            req = Request(url, method="GET")
            with urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body.get("status") == "ok"
        except (URLError, OSError, json.JSONDecodeError):
            return False

    def __repr__(self) -> str:
        return f"LlamaCppProvider(url={self.config.base_url!r}, model={self.config.model!r})"
