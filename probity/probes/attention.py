import torch
import torch.nn as nn
from typing import Optional
from .base import BaseProbe
from .config import AttentionProbeConfig

class AttentionProbe(BaseProbe[AttentionProbeConfig]):
    """Attention-based probe following EleutherAI/LASR architecture."""
    
    def __init__(self, config: AttentionProbeConfig):
        super().__init__(config)
        
        # Query projection: d_model -> n_heads
        self.query_proj = nn.Linear(
            config.input_size, 
            config.n_heads, 
            bias=config.query_bias
        )
        
        # Value projection: d_model -> n_heads * n_outputs
        self.value_proj = nn.Linear(
            config.input_size,
            config.n_heads * config.output_size,
            bias=config.value_bias
        )
        
        # Learnable position weights (like ALiBi)
        if config.use_position_weights:
            self.position_weights = nn.Parameter(
                torch.zeros(config.n_heads)
            )
        else:
            self.register_buffer('position_weights', torch.zeros(config.n_heads))
        
        self.temperature = config.temperature
        
        # Initialize weights
        nn.init.zeros_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with attention mechanism."""
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        batch_size, seq_len, _ = x.shape
        
        # Compute attention logits: [batch, seq_len, n_heads]
        attention_logits = self.query_proj(x) / self.temperature
        
        # Add position bias if enabled
        if self.config.use_position_weights:
            position_bias = self.position_weights.unsqueeze(0) * torch.arange(
                seq_len, device=x.device, dtype=x.dtype
            ).unsqueeze(-1)
            attention_logits = attention_logits + position_bias.unsqueeze(0)
        
        # Apply softmax to get attention weights: [batch, seq_len, n_heads]
        attention_weights = torch.softmax(attention_logits, dim=1)
        
        # Compute values: [batch, seq_len, n_heads * n_outputs]
        values = self.value_proj(x)
        values = values.view(batch_size, seq_len, self.config.n_heads, self.config.output_size)
        
        # Apply attention and aggregate: [batch, n_outputs]
        output = (attention_weights.unsqueeze(-1) * values).sum(dim=[1, 2])
        
        return output.squeeze() if output.shape[0] == 1 else output
    
    # Required methods for BaseProbe compatibility
    def _get_raw_direction_representation(self) -> torch.Tensor:
        """Return a representative vector - for attention probes, we concatenate parameters."""
        # For compatibility, concatenate query and value projections
        params = []
        params.append(self.query_proj.weight.data.flatten())
        params.append(self.value_proj.weight.data.flatten())
        if self.config.use_position_weights:
            params.append(self.position_weights.data)
        return torch.cat(params)
    
    def _set_raw_direction_representation(self, vector: torch.Tensor) -> None:
        """Set parameters from a vector - mainly for loading."""
        # This is complex for attention probes, so we override save/load instead
        raise NotImplementedError("Use save/load methods for AttentionProbe")
    
    def get_direction(self, normalized: bool = True) -> torch.Tensor:
        """Get a representative direction - not meaningful for attention probes."""
        # Return the query projection weights as a proxy
        direction = self.query_proj.weight.data.mean(dim=0)
        if normalized and self.config.normalize_weights:
            norm = torch.norm(direction)
            direction = direction / (norm + 1e-8)
        return direction
    
    def get_loss_fn(self, pos_weight: Optional[torch.Tensor] = None) -> nn.Module:
        """Get loss function for training."""
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)