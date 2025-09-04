import torch
import torch.nn as nn
from typing import Optional, List
from .base import BaseProbe
from .config import MLPProbeConfig


class MLPProbe(BaseProbe[MLPProbeConfig]):
    """Simple multi-layer perceptron probe."""

    _ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh}

    def __init__(self, config: MLPProbeConfig):
        super().__init__(config)

        layers: List[nn.Module] = []
        in_dim = config.input_size

        # hidden layers -------------------------------------------------------
        for h in config.hidden_dims:
            layers.append(nn.Linear(in_dim, h, bias=config.use_bias))
            layers.append(self._ACTIVATIONS[config.activation]())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = h

        # output layer --------------------------------------------------------
        layers.append(nn.Linear(in_dim, config.output_size, bias=config.use_bias))

        # build Sequential & cast to dtype ------------------------------------
        self.mlp = nn.Sequential(*layers).to(dtype=self.dtype)

        self._init_weights()

    # -------------------------------------------------------------------------
    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                if self.config.init_method == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif self.config.init_method == "kaiming":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        • [B, S, d] → mean-pool over S → [B, d]
        • [B, d]    → unchanged
        • [S, d]    → batch=1, mean-pool over S → [1, d]

        Returns [B, output_size].  For binary probes output_size = 1.
        """
        x = x.to(dtype=self.dtype)

        if x.dim() == 3:          # [B, S, d]
            x = x.mean(dim=1)
        elif x.dim() == 2:        # [B, d]  or  [S, d]
            pass
        else:
            raise ValueError(f"Unexpected input shape {x.shape}")

        out = self.mlp(x)         # [B, O]

        # keep the second dim if O == 1 so it matches target shape [B, 1]
        return out

    # -------------------------------------------------------------------------
    # BaseProbe requirements
    # -------------------------------------------------------------------------
    def _get_raw_direction_representation(self) -> torch.Tensor:
        vecs = []
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                vecs.append(m.weight.data.flatten())
                if m.bias is not None:
                    vecs.append(m.bias.data)
        return torch.cat(vecs)

    def _set_raw_direction_representation(self, vector: torch.Tensor) -> None:
        raise NotImplementedError("Use save/load methods for MLPProbe")

    def get_direction(self, normalized: bool = True) -> torch.Tensor:
        # proxy: average weights of first linear layer
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                d = m.weight.data.mean(dim=0)
                if normalized:
                    d = d / (d.norm() + 1e-8)
                return d
        return torch.zeros(self.config.input_size, dtype=self.dtype)

    def get_loss_fn(self, pos_weight: Optional[torch.Tensor] = None) -> nn.Module:
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)