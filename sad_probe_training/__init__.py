"""
SAD Probe Training Scripts

Binary token-level training pipeline for SAD deployment vs evaluation classification.
Reuses the efficient NTML infrastructure for activation caching and probe training.
"""

from .config import SADBinaryTrainingConfig
from .data_loading import SADBinaryDataset

# Import existing infrastructure from NTML
import sys
from pathlib import Path

# Add ntml_efficient_scripts to path
ntml_path = Path(__file__).parent.parent / "ntml_efficient_scripts"
sys.path.insert(0, str(ntml_path))

from activation_utils import NTMLActivationCache, collect_all_layers_activations, extract_layer_training_data
from training import NTMLBinaryTrainer

# Create aliases for SAD-specific use
SADActivationCache = NTMLActivationCache
SADBinaryTrainer = NTMLBinaryTrainer

__all__ = [
    "SADBinaryTrainingConfig",
    "SADBinaryDataset",
    "SADActivationCache", 
    "SADBinaryTrainer",
    "collect_all_layers_activations",
    "extract_layer_training_data",
]