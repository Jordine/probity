import torch
import torch.nn as nn
from typing import List, Optional
from .base import BaseProbe
from .config import MLPProbeConfig

class MLPProbe(BaseProbe[MLPProbeConfig]):
    """Multi-layer perceptron probe."""
    
    def __init__(self, config: MLPProbeConfig):
        super().__init__(config)
        
        layers = []
        input_dim = config.input_size
        
        # Build hidden layers
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim, bias=config.use_bias))
            
            # Activation
            if config.activation == 'relu':
                layers.append(nn.ReLU())
            elif config.activation == 'gelu':
                layers.append(nn.GELU())
            elif config.activation == 'tanh':
                layers.append(nn.Tanh())
            
            # Dropout
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, config.output_size, bias=config.use_bias))
        
        self.mlp = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                if self.config.init_method == 'xavier':
                    nn.init.xavier_uniform_(module.weight)
                elif self.config.init_method == 'kaiming':
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP."""
        # Handle both [batch, seq_len, dim] and [seq_len, dim] inputs
        original_shape = x.shape
        if x.dim() == 3:
            # For sequence data, average pool over sequence dimension
            x = x.mean(dim=1)
        elif x.dim() == 2 and len(original_shape) == 2:
            # Single sequence, average over tokens
            x = x.mean(dim=0, keepdim=True)
        
        return self.mlp(x).squeeze()
    
    # Required methods for BaseProbe compatibility
    def _get_raw_direction_representation(self) -> torch.Tensor:
        """Return concatenated parameters."""
        params = []
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                params.append(module.weight.data.flatten())
                if module.bias is not None:
                    params.append(module.bias.data)
        return torch.cat(params)
    
    def _set_raw_direction_representation(self, vector: torch.Tensor) -> None:
        """Not implemented for MLP probes."""
        raise NotImplementedError("Use save/load methods for MLPProbe")
    
    def get_direction(self, normalized: bool = True) -> torch.Tensor:
        """Return first layer weights as proxy direction."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                direction = module.weight.data.mean(dim=0)
                if normalized:
                    norm = torch.norm(direction)
                    direction = direction / (norm + 1e-8)
                return direction
        return torch.zeros(self.config.input_size)
    
    def get_loss_fn(self, pos_weight: Optional[torch.Tensor] = None) -> nn.Module:
        """Get loss function for training."""
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)