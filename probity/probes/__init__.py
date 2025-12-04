# __init__.py for probity.probes
# Import probe configuration classes
from .config import (
    ProbeConfig,
    LinearProbeConfig,
    LogisticProbeConfig,
    MultiClassLogisticProbeConfig,
    KMeansProbeConfig,
    PCAProbeConfig,
    MeanDiffProbeConfig,
    LogisticProbeConfigBase,  # Base for sklearn
    SklearnLogisticProbeConfig,
    MLPProbeConfig,
    AttentionProbeConfig
)
# Import the base probe class
from .base import BaseProbe
# Import concrete probe implementations
from .linear import LinearProbe
from .logistic import LogisticProbe, MultiClassLogisticProbe
from .directional import (
    DirectionalProbe,  # Base for non-learned direction probes
    KMeansProbe,
    PCAProbe,
    MeanDifferenceProbe,
)
from .sklearn_logistic import SklearnLogisticProbe
from .mlp import MLPProbe  # Add this line
from .attention import AttentionProbe  # Add this line
# Import the ProbeSet class
from .probe_set import ProbeSet
# Define __all__ for explicit public API
__all__ = [
    # Configs
    "ProbeConfig",
    "LinearProbeConfig",
    "LogisticProbeConfig",
    "MultiClassLogisticProbeConfig",
    "KMeansProbeConfig",
    "PCAProbeConfig",
    "MeanDiffProbeConfig",
    "LogisticProbeConfigBase",
    "SklearnLogisticProbeConfig",
    "MLPProbeConfig",  # Add this line
    "AttentionProbeConfig",  # Add this line
    # Base Class
    "BaseProbe",
    # Concrete Probes
    "LinearProbe",
    "LogisticProbe",
    "MultiClassLogisticProbe",
    "DirectionalProbe",
    "KMeansProbe",
    "PCAProbe",
    "MeanDifferenceProbe",
    "SklearnLogisticProbe",
    "MLPProbe",  # Add this line
    "AttentionProbe",  # Add this line
    # Probe Collection
    "ProbeSet",
]
from .apollo_probe import ApolloProbe, ApolloProbeConfig
