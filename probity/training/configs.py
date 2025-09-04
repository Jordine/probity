import torch
from typing import Dict, Optional
from transformer_lens import HookedTransformer
from tqdm import tqdm

from probity.probes import (
    LogisticProbe, LogisticProbeConfig,
    PCAProbe, PCAProbeConfig,
    MeanDifferenceProbe, MeanDiffProbeConfig,
    KMeansProbe, KMeansProbeConfig,
    LinearProbe, LinearProbeConfig,
    AttentionProbe, AttentionProbeConfig, 
    MLPProbe, MLPProbeConfig              
)
from probity.training.trainer import (
    SupervisedProbeTrainer, SupervisedTrainerConfig,
    DirectionalProbeTrainer, DirectionalTrainerConfig
)



def get_probe_config(probe_type: str, hidden_size: int, model_name: str, 
                    hook_point: str, layer: int, dtype: torch.dtype,
                    hyperparams: Optional[Dict] = None) -> Dict:
    """Get probe configuration based on type with optional hyperparameter override."""
    
    # Base configuration
    base_params = {
        'input_size': hidden_size,
        'model_name': model_name,
        'hook_point': hook_point,
        'hook_layer': layer,
        'name': f"lie_truth_{probe_type}_layer_{layer}",
        'dtype': 'bfloat16' if dtype == torch.bfloat16 else 'float32'
    }
    
    # Merge with provided hyperparameters
    if hyperparams:
        base_params.update(hyperparams)
    
    configs = {
        'logistic': LogisticProbeConfig(**base_params),
        'linear': LinearProbeConfig(**base_params),
        'pca': PCAProbeConfig(**base_params),
        'meandiff': MeanDiffProbeConfig(**base_params),
        'kmeans': KMeansProbeConfig(**base_params),
        'attention': AttentionProbeConfig(**base_params),
        'mlp': MLPProbeConfig(**base_params),
    }
    
    return configs.get(probe_type)

def get_probe_class(probe_type: str):
    """Get probe class based on type"""
    classes = {
        'logistic': LogisticProbe,
        'linear': LinearProbe,
        'pca': PCAProbe,
        'meandiff': MeanDifferenceProbe,
        'kmeans': KMeansProbe,
        'attention': AttentionProbe,
        'mlp': MLPProbe,
    }
    return classes.get(probe_type)


    
def get_trainer_config(probe_type: str, device: str, batch_size: int) -> Dict:
    """Get trainer configuration based on probe type"""
    if probe_type in ['logistic', 'linear', 'attention', 'mlp']:  # Add new types here if they use supervised training
        return SupervisedTrainerConfig(
            batch_size=batch_size,
            learning_rate=1e-3,
            num_epochs=10,
            weight_decay=0.01,
            train_ratio=0.8,
            handle_class_imbalance=True,
            show_progress=True,
            device=device,
            standardize_activations=True
        )
    else:
        return DirectionalTrainerConfig(
            batch_size=batch_size,
            device=device,
            standardize_activations=True
        )

def get_trainer_class(probe_type: str):
    """Get trainer class based on probe type"""
    if probe_type in ['logistic', 'linear', 'attention', 'mlp']:  # Add new types here if they use supervised training
        return SupervisedProbeTrainer
    else:
        return DirectionalProbeTrainer

def get_probe_class(probe_type: str):
    """Get probe class based on type"""
    classes = {
        'attention': AttentionProbe,
        'mlp': MLPProbe,
        'logistic': LogisticProbe,
        'linear': LinearProbe,
        'pca': PCAProbe,
        'meandiff': MeanDifferenceProbe,
        'kmeans': KMeansProbe
    }
    probe_cls = classes.get(probe_type)
    if probe_cls is None:
        raise ValueError(f"Unknown probe type: '{probe_type}'. Available types: {list(classes.keys())}")
    return probe_cls


def get_trainer_config(probe_type: str, device: str, batch_size: int) -> Dict:
    """Get trainer configuration based on probe type"""
    if probe_type in ['logistic', 'linear']:
        return SupervisedTrainerConfig(
            batch_size=batch_size,
            learning_rate=1e-3,
            num_epochs=10,
            weight_decay=0.01,
            train_ratio=0.8,
            handle_class_imbalance=True,
            show_progress=True,
            device=device,
            standardize_activations=True
        )
    else:
        return DirectionalTrainerConfig(
            batch_size=batch_size,
            device=device,
            standardize_activations=True
        )


def get_trainer_class(probe_type: str):
    """Get trainer class based on probe type"""
    if probe_type in ['logistic', 'linear']:
        return SupervisedProbeTrainer
    else:
        return DirectionalProbeTrainer
