import torch
import torch.nn as nn
from typing import Optional
from .base import BaseProbe
from .config import AttentionProbeConfig


class AttentionProbe(BaseProbe[AttentionProbeConfig]):
    """
    Attention-style probe (loosely following LASR).

    • query_proj  : d_model → n_heads  
    • value_proj  : d_model → n_heads · output_size  
    """

    def __init__(self, config: AttentionProbeConfig):
        super().__init__(config)

        # ------------------------------------------------------------------
        # Layers – immediately moved to the probe’s dtype (bf16/float32)
        # ------------------------------------------------------------------
        self.query_proj = nn.Linear(
            config.input_size, config.n_heads, bias=config.query_bias
        ).to(dtype=self.dtype)

        self.value_proj = nn.Linear(
            config.input_size,
            config.n_heads * config.output_size,
            bias=config.value_bias,
        ).to(dtype=self.dtype)

        # optional positional bias per head
        if config.use_position_weights:
            self.position_weights = nn.Parameter(
                torch.zeros(config.n_heads, dtype=self.dtype)
            )
        else:
            self.register_buffer(
                "position_weights",
                torch.zeros(config.n_heads, dtype=self.dtype),
                persistent=False,
            )

        self.temperature = float(config.temperature)

        # ------------------------------------------------------------------
        # Weight initialisation
        # ------------------------------------------------------------------
        nn.init.zeros_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)

    # ======================================================================
    # Forward
    # ======================================================================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accepts
          • [B, S, d]  – standard sequence
          • [B, d]     – a single position per example
          • [S, d]     – treat as batch = 1, seq = S
        Returns: [B, output_size]  (or scalar if B==1 & output_size==1)
        """
        x = x.to(dtype=self.dtype)

        if x.dim() == 2:
            if x.shape[0] == self.config.input_size:      # unlikely – safeguard
                x = x.unsqueeze(0)                        # [1, S, d]
            else:
                x = x.unsqueeze(1)                        # [B, 1, d]

        if x.dim() != 3:
            raise ValueError(f"Expected 2- or 3-D input, got shape {x.shape}")

        B, S, _ = x.shape

        attn_logits = self.query_proj(x) / self.temperature        # [B, S, H]

        if self.config.use_position_weights and S > 1:
            pos = torch.arange(S, device=x.device, dtype=self.dtype)
            attn_logits = attn_logits + pos.view(1, S, 1) * self.position_weights

        attn_w = torch.softmax(attn_logits, dim=1)                 # [B, S, H]

        v = self.value_proj(x).view(
            B, S, self.config.n_heads, self.config.output_size
        )                                                           # [B, S, H, O]

        out = (attn_w.unsqueeze(-1) * v).sum(dim=[1, 2])            # [B, O]

        return out.squeeze() if out.numel() == 1 else out

    # ======================================================================
    # Helpers required by BaseProbe
    # ======================================================================
    def _get_raw_direction_representation(self) -> torch.Tensor:
        params = [
            self.query_proj.weight.data.flatten(),
            self.value_proj.weight.data.flatten(),
        ]
        if self.config.use_position_weights:
            params.append(self.position_weights.data)
        return torch.cat(params)

    def _set_raw_direction_representation(self, vector: torch.Tensor) -> None:
        raise NotImplementedError("Use .save/.load for AttentionProbe")

    def get_direction(self, normalized: bool = True) -> torch.Tensor:
        # not really meaningful for attention probes; return mean query weight
        d = self.query_proj.weight.data.mean(dim=0)
        if normalized and self.config.normalize_weights:
            d = d / (d.norm() + 1e-8)
        return d

    def get_loss_fn(self, pos_weight: Optional[torch.Tensor] = None) -> nn.Module:
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)