# probity/training/configs.py
import torch
from typing import Dict, Optional

from probity.probes import (
    LogisticProbe, LogisticProbeConfig,
    LinearProbe,  LinearProbeConfig,
    PCAProbe,     PCAProbeConfig,
    MeanDifferenceProbe, MeanDiffProbeConfig,
    KMeansProbe,  KMeansProbeConfig,
    AttentionProbe, AttentionProbeConfig,
    MLPProbe,      MLPProbeConfig,
    SklearnLogisticProbe, SklearnLogisticProbeConfig,
)
from probity.training.trainer import (
    SupervisedProbeTrainer,  SupervisedTrainerConfig,
    DirectionalProbeTrainer, DirectionalTrainerConfig,
)

# --------------------------------------------------------------------------- #
# Probe ↔ Config ↔ Trainer lookup tables
# --------------------------------------------------------------------------- #

_SUPERVISED_STD   = {"logistic", "linear"}          # needs std + un-scaling
_SUPERVISED_RAW   = {"attention", "mlp"}            # no std – avoids NotImpl
_DIRECTIONAL      = {"pca", "meandiff", "kmeans", "sklearn_logistic"}

# ---------- PROBE CONFIG ---------------------------------------------------- #
def get_probe_config(
    probe_type: str,
    hidden_size: int,
    model_name: str,
    hook_point: str,
    layer: int,
    dtype: torch.dtype,
    hyperparams: Optional[Dict] = None,
):
    """Return the correct *Config* object for the requested probe type."""
    base = dict(
        input_size  = hidden_size,
        model_name  = model_name,
        hook_point  = hook_point,
        hook_layer  = layer,
        name        = f"lie_truth_{probe_type}_layer_{layer}",
        dtype       = "bfloat16" if dtype == torch.bfloat16 else "float32",
    )
    if hyperparams:
        base.update(hyperparams)

    configs = {
        "logistic":  LogisticProbeConfig(**base),
        "linear":    LinearProbeConfig(**base),
        "pca":       PCAProbeConfig(**base),
        "meandiff":  MeanDiffProbeConfig(**base),
        "kmeans":    KMeansProbeConfig(**base),
        "attention": AttentionProbeConfig(**base),
        "mlp":       MLPProbeConfig(**base),
        "sklearn_logistic": SklearnLogisticProbeConfig(**base),
    }
    if probe_type not in configs:
        raise ValueError(f"Unknown probe type '{probe_type}'")
    return configs[probe_type]

# ---------- PROBE CLASS ----------------------------------------------------- #
def get_probe_class(probe_type: str):
    probes = {
        "logistic":  LogisticProbe,
        "linear":    LinearProbe,
        "pca":       PCAProbe,
        "meandiff":  MeanDifferenceProbe,
        "kmeans":    KMeansProbe,
        "attention": AttentionProbe,
        "mlp":       MLPProbe,
        "sklearn_logistic": SklearnLogisticProbe,
    }
    if probe_type not in probes:
        raise ValueError(f"Unknown probe type '{probe_type}'")
    return probes[probe_type]

# ---------- TRAINER CONFIG -------------------------------------------------- #
def get_trainer_config(probe_type: str, device: str, batch_size: int):
    if probe_type in _SUPERVISED_STD:
        return SupervisedTrainerConfig(
            batch_size              = batch_size,
            learning_rate           = 1e-3,
            num_epochs              = 10,
            weight_decay            = 0.01,
            train_ratio             = 0.8,
            handle_class_imbalance  = True,
            show_progress           = True,
            device                  = device,
            standardize_activations = True,   # will un-scale afterwards
        )

    if probe_type in _SUPERVISED_RAW:
        return SupervisedTrainerConfig(
            batch_size              = batch_size,
            learning_rate           = 1e-3,
            num_epochs              = 10,
            weight_decay            = 0.01,
            train_ratio             = 0.8,
            handle_class_imbalance  = True,
            show_progress           = True,
            device                  = device,
            standardize_activations = False,  # ← disables un-scaling step
        )

    if probe_type in _DIRECTIONAL:
        return DirectionalTrainerConfig(
            batch_size              = batch_size,
            device                  = device,
            standardize_activations = True,
        )

    raise ValueError(f"Unknown probe type '{probe_type}'")

# ---------- TRAINER CLASS --------------------------------------------------- #
def get_trainer_class(probe_type: str):
    if probe_type in _SUPERVISED_STD | _SUPERVISED_RAW:
        return SupervisedProbeTrainer
    if probe_type in _DIRECTIONAL:
        return DirectionalProbeTrainer
    raise ValueError(f"Unknown probe type '{probe_type}'")