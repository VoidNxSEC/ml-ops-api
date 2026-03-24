"""
LLM Client with fallback chain, retry logic, and circuit breakers.

Provides a unified interface for calling LLMs with automatic failover
between providers (Anthropic → DeepSeek → OpenAI → llama.cpp).

Migrated from neutron/agents/llm_client.py.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from mlops.llm.config import LLMConfig, ProviderType, get_config
from mlops.llm.providers import (
    LLMProvider,
    LLMResponse,
    MLOffloadProvider,
    AnthropicProvider,
    OpenAIProvider,
    DeepSeekProvider,
    LlamaCppProvider,
)

logger = logging.getLogger("mlops.llm.client")


@dataclass
class CircuitBreakerState:
    """Circuit breaker state for a provider."""

    failures: int = 0
    last_failure_time: float = 0.0
    is_open: bool = False
    threshold: int = 5
    timeout: float = 60.0

    def record_failure(self) -> None:
        """Record a failure and potentially open circuit."""
        self.failures += 1
        self.last_failure_time = time.time()

        if self.failures >= self.threshold:
            self.is_open = True
            logger.warning(
                f"Circuit breaker opened after {self.failures} failures"
            )

    def record_success(self) -> None:
        """Record success and reset circuit."""
        self.failures = 0
        self.is_open = False

    def can_attempt(self) -> bool:
        """Check if we can attempt a call."""
        if not self.is_open:
            return True

        # Check if timeout has passed
        if time.time() - self.last_failure_time > self.timeout:
            logger.info("Circuit breaker timeout passed, attempting half-open state")
            self.is_open = False
            return True

        return False


class LLMClient:
    """LLM client with fallback chain and resilience features.

    Features:
    - Automatic fallback between providers
    - Exponential backoff retry
    - Circuit breakers per provider
    - Request/response logging
    - Token usage tracking

    Usage:
        client = LLMClient()
        response = await client.generate("Hello, world!")
    """

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or get_config()
        self._providers: dict[ProviderType, LLMProvider] = {}
        self._circuit_breakers: dict[ProviderType, CircuitBreakerState] = {}
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Initialize LLM providers based on configuration."""
        provider_classes = {
            ProviderType.ML_OFFLOAD: MLOffloadProvider,
            ProviderType.ANTHROPIC: AnthropicProvider,
            ProviderType.OPENAI: OpenAIProvider,
            ProviderType.DEEPSEEK: DeepSeekProvider,
            ProviderType.LLAMACPP: LlamaCppProvider,
        }

        for provider_type in self.config.get_enabled_providers():
            provider_config = self.config.get_provider_config(provider_type)

            if not provider_config.enabled:
                continue

            try:
                provider_class = provider_classes[provider_type]
                kwargs: dict[str, Any] = {
                    "temperature": provider_config.temperature,
                    "max_tokens": provider_config.max_tokens,
                    "timeout": provider_config.timeout,
                }

                if provider_config.api_key:
                    kwargs["api_key"] = provider_config.api_key
                if provider_config.base_url:
                    kwargs["base_url"] = provider_config.base_url
                if provider_config.model:
                    kwargs["model"] = provider_config.model

                provider = provider_class(**kwargs)
                self._providers[provider_type] = provider

                # Initialize circuit breaker
                self._circuit_breakers[provider_type] = CircuitBreakerState(
                    threshold=provider_config.circuit_breaker_threshold,
                    timeout=provider_config.circuit_breaker_timeout,
                )

                logger.info(f"Initialized {provider_type.value} provider: {provider}")

            except ImportError as e:
                logger.warning(
                    f"Failed to initialize {provider_type.value} provider: {e}. "
                    f"Install required package to enable."
                )
            except Exception as e:
                logger.error(
                    f"Error initializing {provider_type.value} provider: {e}"
                )

    async def _retry_with_backoff(
        self,
        func,
        *args,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        **kwargs,
    ) -> LLMResponse:
        """Retry function with exponential backoff."""
        delay = initial_delay

        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise

                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff

        # Should not reach here, but satisfy type checker
        raise RuntimeError("Max retries exceeded")

    async def _call_provider(
        self,
        provider_type: ProviderType,
        method: str,
        *args,
        **kwargs,
    ) -> LLMResponse:
        """Call a provider with circuit breaker and retry logic."""
        # Check circuit breaker
        circuit_breaker = self._circuit_breakers.get(provider_type)
        if circuit_breaker and not circuit_breaker.can_attempt():
            raise RuntimeError(
                f"{provider_type.value} circuit breaker is open, skipping"
            )

        provider = self._providers.get(provider_type)
        if not provider:
            raise ValueError(f"Provider {provider_type.value} not available")

        # Get method (generate or generate_chat)
        provider_method = getattr(provider, method)

        # Get retry settings
        provider_config = self.config.get_provider_config(provider_type)

        try:
            if self.config.enable_retries:
                response = await self._retry_with_backoff(
                    provider_method,
                    *args,
                    max_retries=provider_config.max_retries,
                    initial_delay=provider_config.retry_delay,
                    **kwargs,
                )
            else:
                response = await provider_method(*args, **kwargs)

            # Record success
            if circuit_breaker:
                circuit_breaker.record_success()

            if self.config.log_requests:
                logger.info(
                    f"{provider_type.value} call succeeded: "
                    f"{response.total_tokens} tokens, "
                    f"finish_reason={response.finish_reason}"
                )

            return response

        except Exception as e:
            # Record failure
            if circuit_breaker:
                circuit_breaker.record_failure()

            logger.error(f"{provider_type.value} call failed: {e}")
            raise

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        provider_hint: ProviderType | None = None,
    ) -> LLMResponse:
        """Generate completion with automatic fallback.

        Args:
            prompt: User prompt
            system: Optional system message
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            stop: Stop sequences
            provider_hint: Prefer this provider if available

        Returns:
            LLMResponse from first successful provider

        Raises:
            RuntimeError: If all providers fail
        """
        if provider_hint and provider_hint in self._providers:
            providers_to_try = [provider_hint] + [
                p for p in self.config.get_enabled_providers() if p != provider_hint
            ]
        else:
            providers_to_try = self.config.get_enabled_providers()

        last_error = None

        for provider_type in providers_to_try:
            if provider_type not in self._providers:
                continue

            try:
                logger.debug(f"Attempting {provider_type.value} provider...")
                response = await self._call_provider(
                    provider_type,
                    "generate",
                    prompt,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                )
                logger.info(f"✓ {provider_type.value} succeeded")
                return response

            except Exception as e:
                last_error = e
                logger.warning(f"✗ {provider_type.value} failed: {e}")
                continue

        raise RuntimeError(
            f"All LLM providers failed. Last error: {last_error}"
        )

    async def generate_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        provider_hint: ProviderType | None = None,
    ) -> LLMResponse:
        """Generate from chat messages with automatic fallback."""
        if provider_hint and provider_hint in self._providers:
            providers_to_try = [provider_hint] + [
                p for p in self.config.get_enabled_providers() if p != provider_hint
            ]
        else:
            providers_to_try = self.config.get_enabled_providers()

        last_error = None

        for provider_type in providers_to_try:
            if provider_type not in self._providers:
                continue

            try:
                logger.debug(f"Attempting {provider_type.value} provider...")
                response = await self._call_provider(
                    provider_type,
                    "generate_chat",
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                )
                logger.info(f"✓ {provider_type.value} succeeded")
                return response

            except Exception as e:
                last_error = e
                logger.warning(f"✗ {provider_type.value} failed: {e}")
                continue

        raise RuntimeError(
            f"All LLM providers failed. Last error: {last_error}"
        )

    async def health_check(self) -> dict[str, bool]:
        """Check health of all providers."""
        health = {}
        for provider_type, provider in self._providers.items():
            try:
                is_healthy = await provider.health()
                health[provider_type.value] = is_healthy
            except Exception as e:
                logger.warning(f"Health check failed for {provider_type.value}: {e}")
                health[provider_type.value] = False

        return health

    def get_circuit_breaker_status(self) -> dict[str, dict[str, Any]]:
        """Get circuit breaker status for all providers."""
        status = {}
        for provider_type, cb in self._circuit_breakers.items():
            status[provider_type.value] = {
                "is_open": cb.is_open,
                "failures": cb.failures,
                "last_failure_time": cb.last_failure_time,
                "threshold": cb.threshold,
                "timeout": cb.timeout,
            }
        return status
