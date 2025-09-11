"""
Model providers for debate infrastructure.
"""

from .base import BaseModelProvider, GenerationResult
from .local import LocalModelProvider
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .openrouter import OpenRouterProvider

def create_provider(config):
    """Factory function to create appropriate provider"""
    from ..types import ProviderType
    
    if config.provider == ProviderType.LOCAL:
        return LocalModelProvider(config)
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
    "OpenAIProvider",
    "AnthropicProvider",
    "OpenRouterProvider",
    "create_provider"
]