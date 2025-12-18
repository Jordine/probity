"""
Model providers for debate infrastructure.
"""

from .base import BaseModelProvider, GenerationResult
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .openrouter import OpenRouterProvider

# Lazy imports for torch-dependent providers
def __getattr__(name):
    """Lazy load local providers only when accessed"""
    if name == 'LocalModelProvider':
        from .local import LocalModelProvider
        return LocalModelProvider
    elif name == 'FastLocalModelProvider':
        from .fast_local import FastLocalModelProvider
        return FastLocalModelProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def create_provider(config):
    """Factory function to create appropriate provider"""
    from ..types import ProviderType

    if config.provider == ProviderType.LOCAL:
        from .local import LocalModelProvider
        return LocalModelProvider(config)
    elif config.provider == ProviderType.FAST_LOCAL:
        from .fast_local import FastLocalModelProvider
        return FastLocalModelProvider(config)
    elif config.provider == ProviderType.OPENAI:
        return OpenAIProvider(config)
    elif config.provider == ProviderType.ANTHROPIC:
        return AnthropicProvider(config)
    elif config.provider == ProviderType.OPENROUTER:
        return OpenRouterProvider(config)
    else:
        raise ValueError(f"Unknown provider type: {config.provider}")

__all__ = [
    "BaseModelProvider",
    "GenerationResult",
    "LocalModelProvider",
    "FastLocalModelProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OpenRouterProvider",
    "create_provider"
]