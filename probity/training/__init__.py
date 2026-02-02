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

from probity.training.parallel import (
    train_probes_parallel,
    ProbeTrainTask,
    ProbeTrainResult,
    _ensure_spawn_start_method,
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
    # Parallel training
    "train_probes_parallel",
    "ProbeTrainTask",
    "ProbeTrainResult",
    "_ensure_spawn_start_method",
]
