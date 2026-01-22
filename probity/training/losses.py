"""
Advanced loss functions for probe training.

Ported from hallucination_probes with modifications for probity.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F
from torch import Tensor


def precompute_span_masks(
    spans: List[List[Tuple[int, int]]],
    labels: Tensor,
    seq_len: int,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> Tuple[Tensor, Tensor]:
    """
    Precompute span masks as tensors for GPU-accelerated training.

    Args:
        spans: List of span lists, each containing (start, end) tuples
        labels: Sample labels tensor, shape (num_samples,)
        seq_len: Maximum sequence length
        device: Target device
        dtype: Target dtype

    Returns:
        pos_span_masks: Shape (num_samples, seq_len) - 1.0 for tokens in positive spans
        neg_span_masks: Shape (num_samples, seq_len) - 1.0 for tokens in negative spans
    """
    num_samples = len(spans)
    pos_span_masks = torch.zeros(num_samples, seq_len, device=device, dtype=dtype)
    neg_span_masks = torch.zeros(num_samples, seq_len, device=device, dtype=dtype)

    # Convert labels to numpy for faster iteration
    labels_np = labels.cpu().numpy() if labels.is_cuda else labels.numpy()

    for i, sample_spans in enumerate(spans):
        is_positive = labels_np[i] > 0.5
        for span in sample_spans:
            if isinstance(span, (tuple, list)) and len(span) >= 2:
                start, end = int(span[0]), int(span[1])
                if 0 <= start <= end < seq_len:
                    if is_positive:
                        pos_span_masks[i, start:end+1] = 1.0
                    else:
                        neg_span_masks[i, start:end+1] = 1.0

    return pos_span_masks, neg_span_masks


def compute_max_aggregation_loss_vectorized(
    probe_logits: Tensor,
    pos_span_masks: Tensor,
    neg_span_masks: Tensor,
    max_clipped_logits: float = 100.0,
) -> Tensor:
    """
    Vectorized span-level max-aggregation loss (GPU-optimized).

    For positive spans: BCE(max(logits_in_span), 1.0)
    For negative spans: BCE(max(logits_in_span), 0.0)

    Args:
        probe_logits: Shape (batch_size, seq_len)
        pos_span_masks: Shape (batch_size, seq_len) - 1.0 for positive span tokens
        neg_span_masks: Shape (batch_size, seq_len) - 1.0 for negative span tokens
        max_clipped_logits: Clip logits to prevent extreme values

    Returns:
        Scalar loss
    """
    device = probe_logits.device
    dtype = probe_logits.dtype

    # Clip logits
    logits_clipped = torch.clamp(probe_logits, -max_clipped_logits, max_clipped_logits)

    # For max aggregation, we want max over each span
    # Use masked fill to set non-span tokens to -inf before taking max
    NEG_INF = -1e9

    losses = []

    # Process positive spans (want max logit -> high, target = 1)
    # Check which samples have any positive spans
    has_pos_spans = pos_span_masks.sum(dim=1) > 0  # (batch_size,)
    if has_pos_spans.any():
        # Mask non-span positions with -inf
        pos_masked_logits = logits_clipped.clone()
        pos_masked_logits[pos_span_masks == 0] = NEG_INF
        # Take max over sequence for samples with positive spans
        pos_max_logits = pos_masked_logits.max(dim=1).values  # (batch_size,)
        # Only compute loss for samples that have positive spans
        pos_max_valid = pos_max_logits[has_pos_spans]
        pos_targets = torch.ones_like(pos_max_valid)
        pos_loss = F.binary_cross_entropy_with_logits(pos_max_valid, pos_targets, reduction='none')
        losses.append(pos_loss)

    # Process negative spans (want max logit -> low, target = 0)
    has_neg_spans = neg_span_masks.sum(dim=1) > 0  # (batch_size,)
    if has_neg_spans.any():
        neg_masked_logits = logits_clipped.clone()
        neg_masked_logits[neg_span_masks == 0] = NEG_INF
        neg_max_logits = neg_masked_logits.max(dim=1).values
        neg_max_valid = neg_max_logits[has_neg_spans]
        neg_targets = torch.zeros_like(neg_max_valid)
        neg_loss = F.binary_cross_entropy_with_logits(neg_max_valid, neg_targets, reduction='none')
        losses.append(neg_loss)

    if not losses:
        return torch.tensor(0.0, device=device, dtype=dtype)

    return torch.cat(losses).mean()


def compute_probe_bce_loss(
    probe_logits: Tensor,
    labels: Tensor,
    weights: Optional[Tensor] = None,
    pos_weight: Optional[Tensor] = None,
    ignore_label: float = -100.0,
    max_clipped_logits: float = 100.0,
) -> Tensor:
    """
    Standard BCE loss for probe training with optional weighting.

    Args:
        probe_logits: Shape (batch_size,) or (batch_size, 1)
        labels: Shape (batch_size,) or (batch_size, 1)
        weights: Per-sample weights, shape (batch_size,)
        pos_weight: Weight for positive class (for class imbalance)
        ignore_label: Label value to ignore in loss calculation
        max_clipped_logits: Clip logits to prevent extreme values
    """
    # Flatten if needed
    if probe_logits.dim() > 1:
        probe_logits = probe_logits.squeeze(-1)
    if labels.dim() > 1:
        labels = labels.squeeze(-1)

    # Clip logits
    probe_logits_clipped = torch.clamp(probe_logits, -max_clipped_logits, max_clipped_logits)

    # Create mask for valid labels
    valid_mask = labels != ignore_label

    if not valid_mask.any():
        return torch.tensor(0.0, device=probe_logits.device, dtype=probe_logits.dtype)

    # Filter to valid samples
    valid_logits = probe_logits_clipped[valid_mask]
    valid_labels = labels[valid_mask]

    # Compute BCE loss
    if pos_weight is not None:
        loss = F.binary_cross_entropy_with_logits(
            valid_logits, valid_labels.float(), pos_weight=pos_weight, reduction='none'
        )
    else:
        loss = F.binary_cross_entropy_with_logits(
            valid_logits, valid_labels.float(), reduction='none'
        )

    # Apply per-sample weights if provided
    if weights is not None:
        valid_weights = weights[valid_mask]
        loss = loss * valid_weights

    return loss.mean()


def compute_max_aggregation_loss(
    probe_logits: Tensor,
    positive_spans: List[List[Tuple[int, int]]],
    negative_spans: List[List[Tuple[int, int]]],
    max_clipped_logits: float = 100.0,
) -> Tensor:
    """
    Span-level max-aggregation loss.

    For positive (deceptive) spans: BCE(max(logits_in_span), 1.0)
    For negative (truthful) spans: BCE(max(logits_in_span), 0.0)

    This encourages the probe to fire on at least one token within each
    deceptive span, and NOT fire on any token within truthful spans.

    Args:
        probe_logits: Shape (batch_size, seq_len) - probe outputs for each token
        positive_spans: List of lists of (start, end) tuples for deceptive spans
        negative_spans: List of lists of (start, end) tuples for truthful spans
        max_clipped_logits: Clip logits to prevent extreme values

    Returns:
        Scalar loss averaged over all spans
    """
    device = probe_logits.device
    dtype = probe_logits.dtype

    # Handle 1D input (single sequence)
    if probe_logits.dim() == 1:
        probe_logits = probe_logits.unsqueeze(0)
        positive_spans = [positive_spans] if positive_spans and isinstance(positive_spans[0], tuple) else positive_spans
        negative_spans = [negative_spans] if negative_spans and isinstance(negative_spans[0], tuple) else negative_spans

    # Clip logits
    probe_logits_clipped = torch.clamp(probe_logits, -max_clipped_logits, max_clipped_logits)

    span_losses = []

    for batch_idx in range(probe_logits_clipped.shape[0]):
        # Process positive (deceptive) spans
        if batch_idx < len(positive_spans):
            for span in positive_spans[batch_idx]:
                if isinstance(span, (tuple, list)) and len(span) >= 2:
                    start, end = span[0], span[1]
                    if start <= end and end < probe_logits_clipped.shape[1]:
                        span_logits = probe_logits_clipped[batch_idx, start:end+1]
                        max_logit = torch.max(span_logits)
                        target = torch.tensor(1.0, device=device, dtype=dtype)
                        loss = F.binary_cross_entropy_with_logits(max_logit, target)
                        span_losses.append(loss)

        # Process negative (truthful) spans
        if batch_idx < len(negative_spans):
            for span in negative_spans[batch_idx]:
                if isinstance(span, (tuple, list)) and len(span) >= 2:
                    start, end = span[0], span[1]
                    if start <= end and end < probe_logits_clipped.shape[1]:
                        span_logits = probe_logits_clipped[batch_idx, start:end+1]
                        max_logit = torch.max(span_logits)
                        target = torch.tensor(0.0, device=device, dtype=dtype)
                        loss = F.binary_cross_entropy_with_logits(max_logit, target)
                        span_losses.append(loss)

    if not span_losses:
        return torch.tensor(0.0, device=device, dtype=dtype)

    return torch.mean(torch.stack(span_losses))


def compute_annealed_loss(
    bce_loss: Tensor,
    max_aggr_loss: Tensor,
    epoch: int,
    num_epochs: int,
    anneal_warmup: float = 0.3,
) -> Tuple[Tensor, float]:
    """
    Compute annealed loss that transitions from BCE to max aggregation.

    During warmup period: mostly BCE (stable training)
    After warmup: mostly max aggregation (span-level signal)

    Args:
        bce_loss: Token-level BCE loss
        max_aggr_loss: Span-level max aggregation loss
        epoch: Current epoch (0-indexed)
        num_epochs: Total number of epochs
        anneal_warmup: Fraction of training for warmup (0.3 = first 30%)

    Returns:
        Tuple of (combined_loss, omega value)
    """
    if num_epochs <= 0:
        omega = 1.0
    else:
        progress = epoch / num_epochs
        omega = min(1.0, progress / anneal_warmup) if anneal_warmup > 0 else 1.0

    combined_loss = (1 - omega) * bce_loss + omega * max_aggr_loss

    return combined_loss, omega


def compute_sparsity_loss(
    probe_logits: Tensor,
    attention_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Sparsity loss to encourage probe to be selective.

    Penalizes high average activation, preventing the probe
    from flagging everything as deceptive.

    Args:
        probe_logits: Probe output logits
        attention_mask: Mask for valid tokens (1 = valid, 0 = padding)

    Returns:
        Scalar sparsity loss (average probability across valid tokens)
    """
    # Get probabilities
    probe_probs = torch.sigmoid(probe_logits)

    if attention_mask is not None:
        # Only consider valid tokens
        if attention_mask.shape != probe_probs.shape:
            attention_mask = attention_mask.view_as(probe_probs)
        masked_probs = probe_probs * attention_mask
        num_valid = attention_mask.sum()
        if num_valid == 0:
            return torch.tensor(0.0, device=probe_logits.device)
        avg_activation = masked_probs.sum() / num_valid
    else:
        avg_activation = probe_probs.mean()

    return avg_activation
