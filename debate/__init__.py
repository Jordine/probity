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

from .providers import (
    BaseModelProvider,
    LocalModelProvider,
    OpenAIProvider,
    AnthropicProvider,
    OpenRouterProvider,
    create_provider
)

from .debate_manager import DebateManager
from .dataset_loader import APPSDatasetLoader, SolutionGenerator, FlexibleDatasetLoader
from .analyze_results import DebateAnalyzer

__version__ = "0.1.0"
__author__ = "Probity Debate System"

# Create aliases for a cleaner public API to match the original intent
BaseProvider = BaseModelProvider
LocalProvider = LocalModelProvider

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
    
    # Providers (using aliases)
    "BaseProvider",
    "LocalProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OpenRouterProvider",
    "create_provider",
    
    # Core
    "DebateManager",
    
    # Dataset
    "APPSDatasetLoader",
    "SolutionGenerator",
    "FlexibleDatasetLoader",
    
    # Analysis
    "DebateAnalyzer"
]