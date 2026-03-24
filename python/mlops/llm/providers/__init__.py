from .base import LLMProvider, LLMResponse, ProviderConfig
from .llamacpp import LlamaCppProvider
from .anthropic import AnthropicProvider
from .openai import OpenAIProvider, DeepSeekProvider
from .ml_offload import MLOffloadProvider

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "ProviderConfig",
    "MLOffloadProvider",
    "LlamaCppProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "DeepSeekProvider",
    "create_provider",
]


def create_provider(
    provider_type: str = "llamacpp",
    **kwargs,
) -> LLMProvider:
    """Factory to create LLM providers.

    Args:
        provider_type: One of "llamacpp", "openai", "anthropic", "deepseek", "ml_offload"
        **kwargs: Passed to provider constructor

    Returns:
        LLMProvider instance
    """
    providers = {
        "ml_offload": MLOffloadProvider,
        "llamacpp": LlamaCppProvider,
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "deepseek": DeepSeekProvider,
    }

    if provider_type not in providers:
        raise ValueError(
            f"Unknown provider: {provider_type}. "
            f"Available: {list(providers.keys())}"
        )

    return providers[provider_type](**kwargs)
