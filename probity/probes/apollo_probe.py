# probity/probes/apollo_probe.py
"""Apollo Research deception probe adapter.

This allows using Apollo's pre-trained deception probes within probity's
probe inference system.

Apollo's LogisticRegressionDetector format (pickle):
{
    "layers": list[int],           # e.g., [22]
    "directions": Tensor[layer, emb],  # learned LR weights
    "scaler_mean": Tensor[layer, emb], # StandardScaler mean
    "scaler_scale": Tensor[layer, emb], # StandardScaler scale
    "normalize": bool,             # whether to apply normalization
    "reg_coeff": float,            # regularization coefficient
}

Scoring: einsum(normalized_acts, directions, "toks layer emb, layer emb -> toks layer").mean(-1)
Output: sigmoid applied to get [0, 1] range where higher = more deceptive
"""

import torch
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union


@dataclass
class ApolloProbeConfig:
    """Config for Apollo probe wrapper"""
    hidden_dim: int
    layer: int  # Primary layer (for probity compatibility)
    layers: List[int] = None  # Apollo can have multiple layers
    normalize: bool = True
    device: str = "cuda"

    def __post_init__(self):
        if self.layers is None:
            self.layers = [self.layer]


class ApolloProbe(torch.nn.Module):
    """
    Wrapper to make Apollo probes compatible with probity's BaseProbe interface.

    Apollo probes use logistic regression trained on "instructed pairs" - pairs
    where the model is explicitly told to lie vs tell truth.

    The scoring is:
    1. Optionally normalize: (acts - scaler_mean) / scaler_scale
    2. Dot product: einsum("toks layer emb, layer emb -> toks layer", acts, directions)
    3. Average across layers: .mean(dim=-1)
    4. Sigmoid to get [0, 1] range (higher = more deceptive)
    """

    def __init__(self, config: Union[ApolloProbeConfig, Dict[str, Any]]):
        super().__init__()
        
        # Handle both dict and dataclass configs
        if isinstance(config, dict):
            self.hidden_dim = config['hidden_dim']
            self.layer = config['layer']
            self.layers = config.get('layers', [config['layer']])
            self.normalize_input = config.get('normalize', True)
            self._device = config.get('device', 'cuda')
            self.config = config  # Keep as dict
        else:
            self.hidden_dim = config.hidden_dim
            self.layer = config.layer
            self.layers = config.layers if config.layers else [config.layer]
            self.normalize_input = config.normalize
            self._device = config.device
            self.config = config
        
        self.n_layers = len(self.layers)

        # These will be loaded from checkpoint
        # Apollo format: [n_layers, hidden_dim]
        self.register_buffer('directions', torch.zeros(self.n_layers, self.hidden_dim))
        self.register_buffer('scaler_mean', torch.zeros(self.n_layers, self.hidden_dim))
        self.register_buffer('scaler_scale', torch.ones(self.n_layers, self.hidden_dim))

        self.has_fit = True  # Pre-trained, always ready

    @property
    def dtype(self):
        return self.directions.dtype

    def to(self, device):
        """Override to handle device movement properly"""
        super().to(device)
        self._device = str(device)
        if isinstance(self.config, dict):
            self.config['device'] = str(device)
        elif hasattr(self.config, 'device'):
            self.config.device = str(device)
        return self

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Score activations using Apollo's method.

        Args:
            activations: [seq_len, hidden_dim] - probity sends single layer activations

        Returns:
            scores: [seq_len] in [0, 1] range - higher = more deceptive
        """
        # Probity sends [seq_len, hidden_dim] for single layer
        # We need to add layer dim to match Apollo's format
        if activations.dim() == 2:
            # [seq_len, hidden_dim] -> [seq_len, 1, hidden_dim]
            activations = activations.unsqueeze(1)

        # Move to same device/dtype as directions
        activations = activations.to(device=self.directions.device, dtype=self.directions.dtype)

        # Normalize (like sklearn StandardScaler)
        if self.normalize_input:
            # scaler_mean/scale: [n_layers, hidden_dim]
            # We only use the first layer since probity sends single layer
            mean = self.scaler_mean[:activations.shape[1]]  # [1, hidden_dim] or [n_layers, hidden_dim]
            scale = self.scaler_scale[:activations.shape[1]]
            activations = (activations - mean) / (scale + 1e-8)

        # Dot product with direction vector
        # directions: [n_layers, hidden_dim]
        # activations: [seq_len, n_layers, hidden_dim]
        # Result: [seq_len, n_layers]
        dirs = self.directions[:activations.shape[1]]  # Match layer count
        scores = torch.einsum('slh,lh->sl', activations, dirs)

        # Average across layers (Apollo does this when multiple layers)
        scores = scores.mean(dim=-1)  # [seq_len]

        # Apply sigmoid to get [0, 1] range (higher = more deceptive)
        scores = torch.sigmoid(scores)

        return scores

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """Load from converted Apollo format"""
        if 'directions' in state_dict:
            dirs = state_dict['directions']
            # Handle shape mismatch
            if dirs.dim() == 1:
                dirs = dirs.unsqueeze(0)
            # Use copy_ for registered buffers
            if dirs.shape == self.directions.shape:
                self.directions.copy_(dirs)
            else:
                # Resize buffer if needed
                self.directions = dirs.to(self.directions.device)

        if 'scaler_mean' in state_dict:
            mean = state_dict['scaler_mean']
            if mean.dim() == 1:
                mean = mean.unsqueeze(0)
            if mean.shape == self.scaler_mean.shape:
                self.scaler_mean.copy_(mean)
            else:
                self.scaler_mean = mean.to(self.scaler_mean.device)

        if 'scaler_scale' in state_dict:
            scale = state_dict['scaler_scale']
            if scale.dim() == 1:
                scale = scale.unsqueeze(0)
            if scale.shape == self.scaler_scale.shape:
                self.scaler_scale.copy_(scale)
            else:
                self.scaler_scale = scale.to(self.scaler_scale.device)

        self.has_fit = True

    def state_dict(self) -> Dict[str, Any]:
        return {
            'directions': self.directions,
            'scaler_mean': self.scaler_mean,
            'scaler_scale': self.scaler_scale,
        }
