"""
LLM provider configuration for MLOps.

Migrated from neutron/core/config.py — stripped of NEXUS/SOPS/compliance deps.
Uses plain os.getenv for secrets.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("mlops.llm.config")


class ProviderType(Enum):
    """Available LLM providers."""

    ML_OFFLOAD = "ml_offload"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    LLAMACPP = "llamacpp"


@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""

    provider_type: ProviderType
    enabled: bool = True
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    temperature: float = 0.3
    max_tokens: int = 1024
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, provider_type: ProviderType) -> ProviderConfig:
        """Create provider config from environment variables."""
        prefix = provider_type.value.upper()

        return cls(
            provider_type=provider_type,
            enabled=os.getenv(f"{prefix}_ENABLED", "true").lower() == "true",
            api_key=os.getenv(f"{prefix}_API_KEY", ""),
            base_url=os.getenv(f"{prefix}_BASE_URL", ""),
            model=os.getenv(f"{prefix}_MODEL", cls._default_model(provider_type)),
            temperature=float(os.getenv(f"{prefix}_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv(f"{prefix}_MAX_TOKENS", "1024")),
            timeout=float(os.getenv(f"{prefix}_TIMEOUT", "30.0")),
            max_retries=int(os.getenv(f"{prefix}_MAX_RETRIES", "3")),
            retry_delay=float(os.getenv(f"{prefix}_RETRY_DELAY", "1.0")),
        )

    @staticmethod
    def _default_model(provider_type: ProviderType) -> str:
        """Get default model for provider."""
        defaults = {
            ProviderType.ML_OFFLOAD: "current-model",
            ProviderType.ANTHROPIC: "claude-3-5-sonnet-20241022",
            ProviderType.OPENAI: "gpt-4",
            ProviderType.DEEPSEEK: "deepseek-chat",
            ProviderType.LLAMACPP: "current-model",
        }
        return defaults.get(provider_type, "")


@dataclass
class LLMConfig:
    """LLM configuration with fallback chain."""

    primary_provider: ProviderType = ProviderType.ML_OFFLOAD

    fallback_chain: list[ProviderType] = field(
        default_factory=lambda: [
            ProviderType.ANTHROPIC,
            ProviderType.DEEPSEEK,
            ProviderType.OPENAI,
            ProviderType.LLAMACPP,
        ]
    )

    providers: dict[ProviderType, ProviderConfig] = field(default_factory=dict)

    enable_retries: bool = True
    enable_circuit_breaker: bool = True
    log_requests: bool = True

    @classmethod
    def from_env(cls) -> LLMConfig:
        """Create LLM config from environment variables."""
        primary = os.getenv("LLM_PRIMARY_PROVIDER", "ml_offload").lower()
        provider_map = {
            "ml_offload": ProviderType.ML_OFFLOAD,
            "ml-offload": ProviderType.ML_OFFLOAD,
            "offload": ProviderType.ML_OFFLOAD,
            "anthropic": ProviderType.ANTHROPIC,
            "claude": ProviderType.ANTHROPIC,
            "openai": ProviderType.OPENAI,
            "gpt": ProviderType.OPENAI,
            "deepseek": ProviderType.DEEPSEEK,
            "llamacpp": ProviderType.LLAMACPP,
            "llama": ProviderType.LLAMACPP,
        }
        primary_provider = provider_map.get(primary, ProviderType.ML_OFFLOAD)

        fallback_str = os.getenv("LLM_FALLBACK_CHAIN", "anthropic,deepseek,openai,llamacpp")
        fallback_chain = []
        for name in fallback_str.split(","):
            name = name.strip().lower()
            if name in provider_map:
                provider_type = provider_map[name]
                if provider_type != primary_provider:
                    fallback_chain.append(provider_type)

        providers = {}
        for provider_type in ProviderType:
            providers[provider_type] = ProviderConfig.from_env(provider_type)

        return cls(
            primary_provider=primary_provider,
            fallback_chain=fallback_chain,
            providers=providers,
            enable_retries=os.getenv("LLM_ENABLE_RETRIES", "true").lower() == "true",
            enable_circuit_breaker=os.getenv("LLM_ENABLE_CIRCUIT_BREAKER", "true").lower() == "true",
            log_requests=os.getenv("LLM_LOG_REQUESTS", "true").lower() == "true",
        )

    def get_provider_config(self, provider_type: ProviderType) -> ProviderConfig:
        """Get configuration for a provider."""
        return self.providers.get(
            provider_type, ProviderConfig(provider_type=provider_type)
        )

    def get_enabled_providers(self) -> list[ProviderType]:
        """Get list of enabled providers in fallback order."""
        enabled = [self.primary_provider]

        for provider in self.fallback_chain:
            config = self.get_provider_config(provider)
            if config.enabled and provider not in enabled:
                enabled.append(provider)

        return enabled


# Global config instance (lazy loaded)
_global_config: LLMConfig | None = None


def get_config() -> LLMConfig:
    """Get global LLM configuration."""
    global _global_config
    if _global_config is None:
        _global_config = LLMConfig.from_env()
        logger.info("Loaded LLM configuration from environment")
    return _global_config
