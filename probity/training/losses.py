"""
Advanced loss functions for probe training.

Ported from hallucination_probes with modifications for probity.

Loss Modes (see LOSS_DESIGN.md for details):
- sample_mean: BCE on mean of all token scores (default, current behavior)
- sample_max: BCE on max of all token scores
- token_all: Per-token BCE on all tokens (with class weighting)
- token_spans_only: Per-token BCE only on tokens near labeled spans (with class weighting)
- span_mean: BCE on mean score per span
- span_max: BCE on max score per span + min outside (good for localization)
- joint: α * sample_loss + (1-α) * token_loss
- joint_span_max: α * sample_mean + (1-α) * span_max (detection + localization)
- annealed: Curriculum from sample_mean to token_all over epochs

NOTE on class weighting: token_all and token_spans_only use pos_weight to handle
the class imbalance (~70% truth tokens, ~30% lie tokens). Without this, the probe
learns "predict 0 everywhere" because the majority class dominates the loss.
"""

from enum import Enum
from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F
from torch import Tensor


class LossMode(Enum):
    """Loss mode enumeration for probe training."""
    SAMPLE_MEAN = "sample_mean"      # BCE(mean(all_scores), sample_label)
    SAMPLE_MAX = "sample_max"        # BCE(max(all_scores), sample_label)
    TOKEN_ALL = "token_all"          # BCE(score_i, label_i) for ALL tokens
    TOKEN_SPANS_ONLY = "token_spans_only"  # BCE only on tokens in/near labeled spans
    SPAN_MEAN = "span_mean"          # BCE(mean(span), span_label)
    SPAN_MAX = "span_max"            # BCE(max(span), span_label) - current max_aggr
    JOINT = "joint"                  # α * sample_loss + (1-α) * token_loss
    JOINT_SPAN_MAX = "joint_span_max"  # α * sample_mean + (1-α) * span_max
    ANNEALED = "annealed"            # sample_mean → token_all over epochs
    # New localization-focused losses
    MARGIN = "margin"                # Explicit margin between in-span and out-span means
    RANKING = "ranking"              # Pairwise ranking loss for lie vs truth tokens
    CONTRASTIVE_INTRA = "contrastive_intra"  # Within-sample contrastive (mean_lie > mean_truth)
    SOFT_RECALL = "soft_recall"      # Differentiable approximation of R@Oracle
    TOPK_OVERLAP = "topk_overlap"    # Adaptive: top-k predictions should match k actual lie tokens


class ProbeLoss:
    """
    Unified loss function for probe training with configurable modes.

    See LOSS_DESIGN.md for detailed documentation of each mode.

    Usage:
        loss_fn = ProbeLoss(mode=LossMode.TOKEN_ALL)
        loss_fn.set_epoch(epoch, total_epochs)  # For annealed mode
        loss = loss_fn(token_scores, sample_labels, token_labels, attention_mask)
    """

    def __init__(
        self,
        mode: LossMode = LossMode.SAMPLE_MEAN,
        joint_alpha: float = 0.5,
        anneal_warmup: float = 0.3,
        max_clipped_logits: float = 100.0,
        margin: float = 1.0,
        temperature: float = 0.5,
        num_pairs: int = 32,
    ):
        """
        Args:
            mode: Loss mode (see LossMode enum)
            joint_alpha: Weight for sample loss in joint mode (1.0 = sample only, 0.0 = token only)
            anneal_warmup: Fraction of epochs for warmup in annealed mode
            max_clipped_logits: Clip logits to prevent extreme values
            margin: Margin for margin/ranking losses
            temperature: Temperature for soft_recall loss
            num_pairs: Number of pairs to sample for ranking loss
        """
        self.mode = mode
        self.joint_alpha = joint_alpha
        self.anneal_warmup = anneal_warmup
        self.max_clipped_logits = max_clipped_logits
        self.margin = margin
        self.temperature = temperature
        self.num_pairs = num_pairs
        self.current_epoch = 0
        self.total_epochs = 1

    def set_epoch(self, epoch: int, total_epochs: int):
        """Set current epoch for annealed mode."""
        self.current_epoch = epoch
        self.total_epochs = total_epochs

    def _get_annealed_alpha(self) -> float:
        """Get alpha for annealed mode (1.0 = sample only, 0.0 = token only)."""
        if self.total_epochs <= 0:
            return 1.0

        warmup_epochs = int(self.total_epochs * self.anneal_warmup)
        if self.current_epoch < warmup_epochs:
            return 1.0  # Pure sample loss during warmup
        else:
            progress = (self.current_epoch - warmup_epochs) / max(1, self.total_epochs - warmup_epochs)
            return max(0.0, 1.0 - progress)  # Linear decay to 0

    def _compute_sample_mean_loss(
        self,
        token_scores: Tensor,
        sample_labels: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """BCE on mean of all token scores."""
        # Masked mean
        masked_scores = token_scores * attention_mask
        sum_scores = masked_scores.sum(dim=1)
        count = attention_mask.sum(dim=1).clamp(min=1)
        sample_score = sum_scores / count

        return F.binary_cross_entropy_with_logits(sample_score, sample_labels.float())

    def _compute_sample_max_loss(
        self,
        token_scores: Tensor,
        sample_labels: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """BCE on max of all token scores."""
        # Mask out padding with -inf before max
        masked_scores = token_scores.masked_fill(~attention_mask.bool(), float('-inf'))
        sample_score = masked_scores.max(dim=1).values

        return F.binary_cross_entropy_with_logits(sample_score, sample_labels.float())

    def _compute_token_all_loss(
        self,
        token_scores: Tensor,
        token_labels: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Per-token BCE on all tokens with class weighting.

        Class weighting is CRITICAL for this loss mode. Without it, the ~70% majority
        class (truth tokens) dominates and the probe learns "predict 0 everywhere."

        See PROJECT_STATUS.md for details on why this matters.
        """
        # Calculate pos_weight from class distribution in this batch
        # This balances the loss so positive (lie) tokens have equal influence
        num_pos = (token_labels * attention_mask).sum()
        num_neg = attention_mask.sum() - num_pos
        pos_weight = (num_neg / (num_pos + 1e-8)).clamp(max=10.0)

        loss = F.binary_cross_entropy_with_logits(
            token_scores, token_labels.float(),
            pos_weight=pos_weight,
            reduction='none'
        )
        # Mask out padding
        masked_loss = loss * attention_mask
        return masked_loss.sum() / attention_mask.sum().clamp(min=1)

    def _compute_token_spans_only_loss(
        self,
        token_scores: Tensor,
        token_labels: Tensor,
        attention_mask: Tensor,
        span_masks: Tensor,
    ) -> Tensor:
        """Per-token BCE only on tokens in/near labeled spans with class weighting.

        Class weighting ensures positive (lie) tokens aren't overwhelmed by
        negative tokens even within the span-masked region.
        """
        # Combine attention mask with span mask
        combined_mask = attention_mask * span_masks

        # Calculate pos_weight from class distribution within spans
        num_pos = (token_labels * combined_mask).sum()
        num_neg = combined_mask.sum() - num_pos
        pos_weight = (num_neg / (num_pos + 1e-8)).clamp(max=10.0)

        loss = F.binary_cross_entropy_with_logits(
            token_scores, token_labels.float(),
            pos_weight=pos_weight,
            reduction='none'
        )
        masked_loss = loss * combined_mask
        return masked_loss.sum() / combined_mask.sum().clamp(min=1)

    def _compute_span_max_loss(
        self,
        token_scores: Tensor,
        attention_mask: Tensor,
        span_masks: Tensor,
    ) -> Tensor:
        """BCE on max score per span region (existing behavior)."""
        NEG_INF = -1e9
        device = token_scores.device
        dtype = token_scores.dtype

        # Clip logits
        logits_clipped = torch.clamp(token_scores, -self.max_clipped_logits, self.max_clipped_logits)

        # Positive spans: mask is 1 where in span
        pos_mask = span_masks * attention_mask
        neg_mask = (1 - span_masks) * attention_mask

        losses = []

        # Max in positive (span) regions -> target 1
        has_pos = pos_mask.sum(dim=1) > 0
        if has_pos.any():
            pos_masked = logits_clipped.clone()
            pos_masked[~pos_mask.bool()] = NEG_INF
            pos_max = pos_masked.max(dim=1).values
            pos_max_valid = pos_max[has_pos]
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_max_valid, torch.ones_like(pos_max_valid), reduction='none'
            )
            losses.append(pos_loss)

        # Min in negative (non-span) regions -> target 0
        has_neg = neg_mask.sum(dim=1) > 0
        if has_neg.any():
            neg_masked = logits_clipped.clone()
            neg_masked[~neg_mask.bool()] = float('inf')
            neg_min = neg_masked.min(dim=1).values
            neg_min_valid = neg_min[has_neg]
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_min_valid, torch.zeros_like(neg_min_valid), reduction='none'
            )
            losses.append(neg_loss)

        if not losses:
            return torch.tensor(0.0, device=device, dtype=dtype)

        return torch.cat(losses).mean()

    def _compute_margin_loss(
        self,
        token_scores: Tensor,
        attention_mask: Tensor,
        span_masks: Tensor,
    ) -> Tensor:
        """
        Margin loss: require explicit gap between in-span and out-span mean scores.

        Loss = ReLU(margin - (mean_in_span - mean_out_span))

        This forces the probe to learn features that actually separate lie tokens
        from truth tokens, rather than just satisfying BCE thresholds.
        """
        device = token_scores.device
        dtype = token_scores.dtype

        # Get masks for in-span and out-span tokens
        in_span_mask = span_masks * attention_mask
        out_span_mask = (1 - span_masks) * attention_mask

        losses = []

        for i in range(token_scores.shape[0]):
            in_count = in_span_mask[i].sum()
            out_count = out_span_mask[i].sum()

            # Need both in-span and out-span tokens
            if in_count > 0 and out_count > 0:
                mean_in = (token_scores[i] * in_span_mask[i]).sum() / in_count
                mean_out = (token_scores[i] * out_span_mask[i]).sum() / out_count

                # Hinge loss: want mean_in - mean_out > margin
                loss = F.relu(self.margin - (mean_in - mean_out))
                losses.append(loss)

        if not losses:
            # Return 0 in a way that preserves gradient flow
            return (token_scores * 0).sum()

        return torch.stack(losses).mean()

    def _compute_ranking_loss(
        self,
        token_scores: Tensor,
        attention_mask: Tensor,
        span_masks: Tensor,
    ) -> Tensor:
        """
        Pairwise ranking loss: sample (lie_token, truth_token) pairs and optimize
        for lie tokens to score higher than truth tokens.

        This directly optimizes the ranking that R@Oracle measures.
        """
        device = token_scores.device
        dtype = token_scores.dtype

        in_span_mask = span_masks * attention_mask
        out_span_mask = (1 - span_masks) * attention_mask

        all_pair_losses = []

        for i in range(token_scores.shape[0]):
            # Get indices of lie and truth tokens
            lie_indices = torch.where(in_span_mask[i] > 0)[0]
            truth_indices = torch.where(out_span_mask[i] > 0)[0]

            if len(lie_indices) == 0 or len(truth_indices) == 0:
                continue

            # Sample pairs (or use all if small enough)
            n_lie = len(lie_indices)
            n_truth = len(truth_indices)
            n_pairs = min(self.num_pairs, n_lie * n_truth)

            if n_pairs <= n_lie * n_truth // 2:
                # Random sampling
                lie_sample_idx = torch.randint(0, n_lie, (n_pairs,), device=device)
                truth_sample_idx = torch.randint(0, n_truth, (n_pairs,), device=device)
            else:
                # Use all pairs
                lie_sample_idx = torch.arange(n_lie, device=device).repeat_interleave(n_truth)[:n_pairs]
                truth_sample_idx = torch.arange(n_truth, device=device).repeat(n_lie)[:n_pairs]

            lie_scores = token_scores[i, lie_indices[lie_sample_idx]]
            truth_scores = token_scores[i, truth_indices[truth_sample_idx]]

            # Margin ranking loss: want lie_score > truth_score + margin
            pair_losses = F.relu(self.margin - (lie_scores - truth_scores))
            all_pair_losses.append(pair_losses)

        if not all_pair_losses:
            # Return 0 in a way that preserves gradient flow
            return (token_scores * 0).sum()

        return torch.cat(all_pair_losses).mean()

    def _compute_contrastive_intra_loss(
        self,
        token_scores: Tensor,
        attention_mask: Tensor,
        span_masks: Tensor,
    ) -> Tensor:
        """
        Within-sample contrastive loss: push mean of lie tokens above mean of truth tokens.

        Loss = BCE(sigmoid(mean_lie - mean_truth), 1.0)

        This explicitly compares lie vs truth tokens within each sample.
        """
        device = token_scores.device
        dtype = token_scores.dtype

        in_span_mask = span_masks * attention_mask
        out_span_mask = (1 - span_masks) * attention_mask

        losses = []

        for i in range(token_scores.shape[0]):
            in_count = in_span_mask[i].sum()
            out_count = out_span_mask[i].sum()

            if in_count > 0 and out_count > 0:
                mean_lie = (token_scores[i] * in_span_mask[i]).sum() / in_count
                mean_truth = (token_scores[i] * out_span_mask[i]).sum() / out_count

                # BCE on the difference (want mean_lie > mean_truth)
                diff = mean_lie - mean_truth
                loss = F.binary_cross_entropy_with_logits(
                    diff, torch.ones(1, device=device, dtype=dtype)
                )
                losses.append(loss)

        if not losses:
            # Return 0 in a way that preserves gradient flow
            return (token_scores * 0).sum()

        return torch.stack(losses).mean()

    def _compute_soft_recall_loss(
        self,
        token_scores: Tensor,
        attention_mask: Tensor,
        span_masks: Tensor,
    ) -> Tensor:
        """
        Soft approximation of R@Oracle.

        R@Oracle uses threshold = mean(lie_token_scores), then counts what fraction
        of lie tokens are above threshold. We make this differentiable with sigmoid.

        Loss = -log(soft_recall) where soft_recall = mean(sigmoid((lie_scores - threshold) / temp))
        """
        device = token_scores.device
        dtype = token_scores.dtype

        in_span_mask = span_masks * attention_mask

        losses = []

        for i in range(token_scores.shape[0]):
            lie_mask = in_span_mask[i] > 0
            if lie_mask.sum() == 0:
                continue

            lie_scores = token_scores[i, lie_mask]

            # Threshold = mean of lie scores (this is the "oracle" threshold)
            threshold = lie_scores.mean()

            # Soft count of lie tokens above threshold
            # Using temperature to control sharpness
            soft_above = torch.sigmoid((lie_scores - threshold) / self.temperature)
            soft_recall = soft_above.mean()

            # Maximize recall → minimize -log(recall)
            # Add small epsilon to prevent log(0)
            loss = -torch.log(soft_recall + 1e-8)
            losses.append(loss)

        if not losses:
            # Return 0 in a way that preserves gradient flow
            return (token_scores * 0).sum()

        return torch.stack(losses).mean()

    def _compute_topk_overlap_loss(
        self,
        token_scores: Tensor,
        attention_mask: Tensor,
        span_masks: Tensor,
    ) -> Tensor:
        """
        Adaptive Top-K Overlap Loss: if there are k lie tokens, the top-k scoring
        tokens should BE those k lie tokens.

        This directly optimizes for the debate application: "flag the top-k tokens
        as lies, and those should actually be the lie tokens."

        Uses a differentiable approximation via softmax ranking.
        """
        device = token_scores.device
        dtype = token_scores.dtype

        losses = []

        for i in range(token_scores.shape[0]):
            valid_mask = attention_mask[i] > 0
            lie_mask = (span_masks[i] * attention_mask[i]) > 0

            k = lie_mask.sum().int().item()
            if k == 0 or valid_mask.sum() <= k:
                continue

            # Get scores for valid tokens only
            valid_scores = token_scores[i, valid_mask]
            valid_lie_mask = lie_mask[valid_mask]

            # Soft top-k: use softmax over scores to get "probability of being in top-k"
            # Higher temperature = softer ranking
            temp = self.temperature

            # Compute soft ranking weights (higher score = higher weight)
            soft_weights = F.softmax(valid_scores / temp, dim=0)

            # We want the sum of weights on lie tokens to be high (ideally = k/n * n_lies_weight)
            # Equivalently: the lie tokens should have the highest scores

            # Loss 1: Mean score of lie tokens should be higher than mean of truth tokens
            lie_scores = valid_scores[valid_lie_mask]
            truth_scores = valid_scores[~valid_lie_mask]

            if len(truth_scores) == 0:
                continue

            # Loss: we want lie tokens to dominate the top-k
            # Approach: sum of softmax weights on lie tokens should be high
            lie_weight_sum = soft_weights[valid_lie_mask].sum()
            target_weight = torch.tensor(k / valid_mask.sum().float(), device=device, dtype=dtype)

            # BCE: want lie_weight_sum to be at least target_weight (ideally higher)
            # Scale to [0,1] range for BCE
            normalized_lie_weight = lie_weight_sum.clamp(0, 1)
            loss = F.binary_cross_entropy(
                normalized_lie_weight,
                torch.ones(1, device=device, dtype=dtype).squeeze(),
            )

            # Additional: margin loss between mean lie score and mean truth score
            mean_lie = lie_scores.mean()
            mean_truth = truth_scores.mean()
            margin_loss = F.relu(self.margin - (mean_lie - mean_truth))

            losses.append(loss + 0.5 * margin_loss)

        if not losses:
            # Return 0 in a way that preserves gradient flow
            return (token_scores * 0).sum()

        return torch.stack(losses).mean()

    def __call__(
        self,
        token_scores: Tensor,           # (batch, seq_len)
        sample_labels: Tensor,          # (batch,)
        token_labels: Optional[Tensor] = None,  # (batch, seq_len)
        attention_mask: Optional[Tensor] = None,  # (batch, seq_len)
        span_masks: Optional[Tensor] = None,  # (batch, seq_len) - 1 if in labeled span
    ) -> Tensor:
        """
        Compute loss based on configured mode.

        Args:
            token_scores: Per-token logits from probe (batch, seq_len)
            sample_labels: Sample-level labels (batch,)
            token_labels: Per-token labels (batch, seq_len) - required for token/joint/annealed modes
            attention_mask: Mask for valid tokens (batch, seq_len) - 1=valid, 0=padding
            span_masks: Mask for tokens in labeled spans (batch, seq_len) - 1=in span

        Returns:
            Scalar loss tensor
        """
        device = token_scores.device
        dtype = token_scores.dtype

        # Default attention mask to all ones
        if attention_mask is None:
            attention_mask = torch.ones_like(token_scores)

        # Clip logits
        token_scores = torch.clamp(token_scores, -self.max_clipped_logits, self.max_clipped_logits)

        # Sample-level losses
        if self.mode == LossMode.SAMPLE_MEAN:
            return self._compute_sample_mean_loss(token_scores, sample_labels, attention_mask)

        elif self.mode == LossMode.SAMPLE_MAX:
            return self._compute_sample_max_loss(token_scores, sample_labels, attention_mask)

        # Token-level losses
        elif self.mode == LossMode.TOKEN_ALL:
            if token_labels is None:
                raise ValueError("token_labels required for TOKEN_ALL mode")
            return self._compute_token_all_loss(token_scores, token_labels, attention_mask)

        elif self.mode == LossMode.TOKEN_SPANS_ONLY:
            if token_labels is None or span_masks is None:
                raise ValueError("token_labels and span_masks required for TOKEN_SPANS_ONLY mode")
            return self._compute_token_spans_only_loss(token_scores, token_labels, attention_mask, span_masks)

        # Span-level losses
        elif self.mode == LossMode.SPAN_MAX:
            if span_masks is None:
                raise ValueError("span_masks required for SPAN_MAX mode")
            return self._compute_span_max_loss(token_scores, attention_mask, span_masks)

        # Hybrid losses
        elif self.mode == LossMode.JOINT:
            if token_labels is None:
                raise ValueError("token_labels required for JOINT mode")

            sample_loss = self._compute_sample_mean_loss(token_scores, sample_labels, attention_mask)
            token_loss = self._compute_token_all_loss(token_scores, token_labels, attention_mask)

            return self.joint_alpha * sample_loss + (1 - self.joint_alpha) * token_loss

        elif self.mode == LossMode.JOINT_SPAN_MAX:
            # Hybrid: sample_mean for detection + span_max for localization
            if span_masks is None:
                raise ValueError("span_masks required for JOINT_SPAN_MAX mode")

            sample_loss = self._compute_sample_mean_loss(token_scores, sample_labels, attention_mask)
            span_loss = self._compute_span_max_loss(token_scores, attention_mask, span_masks)

            return self.joint_alpha * sample_loss + (1 - self.joint_alpha) * span_loss

        elif self.mode == LossMode.ANNEALED:
            if token_labels is None:
                raise ValueError("token_labels required for ANNEALED mode")

            alpha = self._get_annealed_alpha()

            sample_loss = self._compute_sample_mean_loss(token_scores, sample_labels, attention_mask)
            token_loss = self._compute_token_all_loss(token_scores, token_labels, attention_mask)

            return alpha * sample_loss + (1 - alpha) * token_loss

        # Localization-focused losses
        elif self.mode == LossMode.MARGIN:
            if span_masks is None:
                raise ValueError("span_masks required for MARGIN mode")
            return self._compute_margin_loss(token_scores, attention_mask, span_masks)

        elif self.mode == LossMode.RANKING:
            if span_masks is None:
                raise ValueError("span_masks required for RANKING mode")
            return self._compute_ranking_loss(token_scores, attention_mask, span_masks)

        elif self.mode == LossMode.CONTRASTIVE_INTRA:
            if span_masks is None:
                raise ValueError("span_masks required for CONTRASTIVE_INTRA mode")
            return self._compute_contrastive_intra_loss(token_scores, attention_mask, span_masks)

        elif self.mode == LossMode.SOFT_RECALL:
            if span_masks is None:
                raise ValueError("span_masks required for SOFT_RECALL mode")
            return self._compute_soft_recall_loss(token_scores, attention_mask, span_masks)

        elif self.mode == LossMode.TOPK_OVERLAP:
            if span_masks is None:
                raise ValueError("span_masks required for TOPK_OVERLAP mode")
            return self._compute_topk_overlap_loss(token_scores, attention_mask, span_masks)

        else:
            raise ValueError(f"Unknown loss mode: {self.mode}")


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
