### 2. **Create Missing Provider Implementations**

Create **/debate/providers/__init__.py** (replace current one):

```python
"""
Model providers for debate infrastructure.
"""

from .base import BaseModelProvider, GenerationResult
from .local import LocalModelProvider

def create_provider(config):
    """Factory function to create appropriate provider"""
    from ..types import ProviderType
    
    if config.provider == ProviderType.LOCAL:
        return LocalModelProvider(config)
    elif config.provider == ProviderType.OPENAI:
        from .openai import OpenAIProvider
        return OpenAIProvider(config)
    elif config.provider == ProviderType.ANTHROPIC:
        from .anthropic import AnthropicProvider
        return AnthropicProvider(config)
    elif config.provider == ProviderType.OPENROUTER:
        from .openrouter import OpenRouterProvider
        return OpenRouterProvider(config)
    else:
        raise ValueError(f"Unknown provider type: {config.provider}")

__all__ = [
    "BaseModelProvider", 
    "GenerationResult", 
    "LocalModelProvider",
    "create_provider"
]