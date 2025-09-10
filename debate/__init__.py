"""
Debate package for AI oversight with probe-based deception detection

This package implements debate-based oversight with optional integration
of deception probes to improve debate outcomes.
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
    ProbeAccessConfig
)

from .providers import (
    BaseProvider,
    LocalProvider,
    OpenAIProvider,
    AnthropicProvider,
    OpenRouterProvider,
    create_provider
)

from .debate_manager import (
    DebateManager,
    ProbeScorer
)

from .dataset_loader import (
    APPSDatasetLoader,
    SolutionGenerator,
    FlexibleDatasetLoader
)

from .analyze_results import DebateAnalyzer

__version__ = "0.1.0"
__author__ = "Probity Debate System"

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
    "ProbeAccessConfig",
    
    # Providers
    "BaseProvider",
    "LocalProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OpenRouterProvider",
    "create_provider",
    
    # Core
    "DebateManager",
    "ProbeScorer",
    
    # Dataset
    "APPSDatasetLoader",
    "SolutionGenerator",
    "FlexibleDatasetLoader",
    
    # Analysis
    "DebateAnalyzer"
]