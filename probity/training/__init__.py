"""Probity training module."""

from probity.training.losses import (
    LossMode,
    ProbeLoss,
    compute_probe_bce_loss,
    compute_max_aggregation_loss,
    compute_max_aggregation_loss_vectorized,
    precompute_span_masks,
    compute_annealed_loss,
    compute_sparsity_loss,
)

from probity.training.trainer import (
    BaseTrainerConfig,
    BaseProbeTrainer,
    SupervisedTrainerConfig,
    SupervisedProbeTrainer,
    DirectionalTrainerConfig,
    DirectionalProbeTrainer,
)

__all__ = [
    # Loss modes and unified loss class
    "LossMode",
    "ProbeLoss",
    # Legacy loss functions
    "compute_probe_bce_loss",
    "compute_max_aggregation_loss",
    "compute_max_aggregation_loss_vectorized",
    "precompute_span_masks",
    "compute_annealed_loss",
    "compute_sparsity_loss",
    # Trainers
    "BaseTrainerConfig",
    "BaseProbeTrainer",
    "SupervisedTrainerConfig",
    "SupervisedProbeTrainer",
    "DirectionalTrainerConfig",
    "DirectionalProbeTrainer",
]
