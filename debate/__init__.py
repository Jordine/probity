#__init__.py
"""
Debate package for AI oversight with probe-based deception detection
"""

from .types import (
    DebateType,
    ProviderType,
    DebateRole,
    ModelConfig,
    ProbeConfig,
    DebateConfig,
    DebateTurn,
    DebateResult,
    APPSProblem,
    QuALITYProblem,
    ProbeAccessConfig
)

# Lazy imports for providers to avoid torch dependency when not needed
def __getattr__(name):
    """Lazy load providers only when accessed"""
    if name in ('BaseModelProvider', 'BaseProvider'):
        from .providers import BaseModelProvider
        return BaseModelProvider
    elif name in ('LocalModelProvider', 'LocalProvider'):
        from .providers import LocalModelProvider
        return LocalModelProvider
    elif name == 'OpenAIProvider':
        from .providers import OpenAIProvider
        return OpenAIProvider
    elif name == 'AnthropicProvider':
        from .providers import AnthropicProvider
        return AnthropicProvider
    elif name == 'OpenRouterProvider':
        from .providers import OpenRouterProvider
        return OpenRouterProvider
    elif name == 'create_provider':
        from .providers import create_provider
        return create_provider
    elif name == 'DebateManager':
        from .debate_manager import DebateManager
        return DebateManager
    elif name == 'DebateStatementLabeler':
        from .labeling import DebateStatementLabeler
        return DebateStatementLabeler
    elif name == 'label_debate_transcripts':
        from .labeling import label_debate_transcripts
        return label_debate_transcripts
    # Dataset loaders - lazy to avoid 'datasets' dependency
    elif name == 'APPSDatasetLoader':
        from .dataset_loader import APPSDatasetLoader
        return APPSDatasetLoader
    elif name == 'SolutionGenerator':
        from .dataset_loader import SolutionGenerator
        return SolutionGenerator
    elif name == 'FlexibleDatasetLoader':
        from .dataset_loader import FlexibleDatasetLoader
        return FlexibleDatasetLoader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Types
    "DebateType",
    "ProviderType",
    "DebateRole",
    "ModelConfig",
    "ProbeConfig",
    "DebateConfig",
    "DebateTurn",
    "DebateResult",
    "APPSProblem",
    "QuALITYProblem",
    "ProbeAccessConfig",

    # Providers (lazy loaded)
    "BaseProvider",
    "BaseModelProvider",
    "LocalProvider",
    "LocalModelProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OpenRouterProvider",
    "create_provider",

    # Core (lazy loaded)
    "DebateManager",

    # Dataset (lazy loaded)
    "APPSDatasetLoader",
    "SolutionGenerator",
    "FlexibleDatasetLoader",

    # Labeling (lazy loaded)
    "DebateStatementLabeler",
    "label_debate_transcripts",
]