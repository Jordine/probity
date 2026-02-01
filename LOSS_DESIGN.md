# Probe Training Loss Design

## Implementation Status: COMPLETE

**Implemented (Feb 1, 2026):**
- `LossMode` enum with 8 loss modes in `probity/training/losses.py`
- `ProbeLoss` class with unified interface for all modes
- `SupervisedTrainerConfig` updated with `loss_mode`, `joint_alpha` fields
- `SupervisedProbeTrainer.train_epoch_with_probe_loss()` method
- CLI args: `--loss_mode`, `--joint_alpha`, `--anneal_warmup`
- Backward compatibility: `--use_max_aggregation` maps to `--loss_mode span_max`

**Usage:**
```bash
# Token-level training (best for localization)
python scripts/probe_training_ntml.py --loss_mode token_all ...

# Joint training (balance detection + localization)
python scripts/probe_training_ntml.py --loss_mode joint --joint_alpha 0.5 ...

# Curriculum learning (sample -> token)
python scripts/probe_training_ntml.py --loss_mode annealed --anneal_warmup 0.3 ...
```

---

## Problem Statement

We want probes that are:
1. **Good at sample detection** (high sample AUROC) - "is this conversation deceptive?"
2. **Good at token localization** (high R@Oracle, Token AUPRC) - "which tokens are the lies?"

Current training optimizes only for #1. We need configurable losses that can optimize for both.

---

## Current Implementation

### Default: Sample Mean
```python
# probe_training_ntml.py (current)
token_scores = probe(activations)      # (batch, seq_len)
sample_score = token_scores.mean(dim=1) # (batch,)
loss = BCE(sample_score, sample_label)  # scalar
```
- Optimizes for sample detection
- Token localization is incidental (probe learns some token signal)
- **Result:** Sample AUROC ~0.94, R@Oracle ~0.20

### With `--use_max_aggregation`: Span Max
```python
# Current implementation in losses.py
# Only for samples WITH labeled spans
pos_max = token_scores[in_span].max()      # max in lie spans
neg_min = token_scores[not_in_span].min()  # min outside spans
loss = BCE(pos_max, 1) + BCE(neg_min, 0)
```
- Optimizes for span-level max separation
- **Problem:** Only works on samples with spans, ignores overall detection
- **Result:** Mixed - sometimes worse than default

---

## Proposed Loss Modes

### Mode 1: `sample_mean` (default, unchanged)
```python
sample_score = token_scores.mean(dim=1)
loss = BCE(sample_score, sample_label)
```
**Use when:** You care most about sample-level detection.

---

### Mode 2: `sample_max`
```python
sample_score = token_scores.max(dim=1).values
loss = BCE(sample_score, sample_label)
```
**Use when:** You want the probe to find at least one high-confidence deceptive token.

---

### Mode 3: `token_all` (NEW - key experiment)
```python
# Token-level labels from NTML spans
# token_labels[i] = 1 if token i is in a LIE_SPAN, else 0

token_scores = probe(activations)  # (batch, seq_len)
token_labels = get_token_labels(spans, seq_len)  # (batch, seq_len)
mask = attention_mask  # ignore padding

loss = BCE(token_scores, token_labels, reduction='none')
loss = (loss * mask).sum() / mask.sum()
```
**Use when:** You want direct token-level supervision for best localization.

**Expected tradeoff:**
- Token localization: R@Oracle 0.35-0.50 (up from 0.20)
- Sample detection: AUROC 0.80-0.88 (down from 0.94)

---

### Mode 4: `token_spans_only`
```python
# Only compute loss on tokens that have labels (in or adjacent to spans)
# Ignores tokens far from any labeled span

span_mask = get_span_mask(spans, seq_len, margin=2)  # tokens near spans
loss = BCE(token_scores[span_mask], token_labels[span_mask])
```
**Use when:** Token labels outside spans are noisy (model might be "deceptive" even on true statements if in deceptive context).

---

### Mode 5: `span_mean`
```python
# Average score per span, then BCE
for span in spans:
    span_scores = token_scores[span.start:span.end]
    span_loss += BCE(span_scores.mean(), span.label)
loss = span_loss / len(spans)
```
**Use when:** You care about statement-level detection, not individual tokens.

---

### Mode 6: `span_max` (current `--use_max_aggregation`, clarified)
```python
# Max in positive spans, min in negative regions
pos_max = token_scores[in_any_lie_span].max()
neg_min = token_scores[not_in_any_span].min()
loss = BCE(pos_max, 1) + BCE(neg_min, 0)
```
**Use when:** You want maximum separation between lie spans and background.

---

### Mode 7: `joint` (NEW - balanced)
```python
# Weighted combination of sample and token losses
sample_score = token_scores.mean(dim=1)
sample_loss = BCE(sample_score, sample_label)

token_loss = BCE(token_scores, token_labels, reduction='none')
token_loss = (token_loss * mask).sum() / mask.sum()

alpha = args.joint_alpha  # default 0.5
loss = alpha * sample_loss + (1 - alpha) * token_loss
```
**Use when:** You want both good detection AND good localization.

**Tuning:** Start with α=0.5, adjust based on which metric you care more about.

---

### Mode 8: `annealed` (NEW - curriculum)
```python
# Start with sample_mean, gradually shift to token_all
def get_alpha(epoch, total_epochs, warmup_fraction=0.3):
    """Alpha goes from 1.0 (sample only) to 0.0 (token only)"""
    warmup_epochs = int(total_epochs * warmup_fraction)
    if epoch < warmup_epochs:
        return 1.0  # Pure sample loss during warmup
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 1.0 - progress  # Linear decay to 0

alpha = get_alpha(epoch, total_epochs)
loss = alpha * sample_loss + (1 - alpha) * token_loss
```
**Use when:** You want to first learn "what deception looks like" (sample-level), then refine "where exactly" (token-level).

**Intuition:** Like curriculum learning - easy task first, then harder task.

---

## CLI Interface

```bash
python scripts/probe_training_ntml.py \
    --loss_mode token_all \           # or: sample_mean, sample_max, joint, annealed, span_max
    --joint_alpha 0.5 \               # for joint mode
    --anneal_warmup 0.3 \             # for annealed mode
    ...
```

### Backward Compatibility
- Default `--loss_mode sample_mean` = current behavior
- `--use_max_aggregation` becomes alias for `--loss_mode span_max`

---

## Implementation Changes

### File: `probity/probes/losses.py`

```python
from enum import Enum
import torch
import torch.nn.functional as F

class LossMode(Enum):
    SAMPLE_MEAN = "sample_mean"
    SAMPLE_MAX = "sample_max"
    TOKEN_ALL = "token_all"
    TOKEN_SPANS_ONLY = "token_spans_only"
    SPAN_MEAN = "span_mean"
    SPAN_MAX = "span_max"
    JOINT = "joint"
    ANNEALED = "annealed"


class ProbeLoss:
    def __init__(
        self,
        mode: LossMode = LossMode.SAMPLE_MEAN,
        joint_alpha: float = 0.5,
        anneal_warmup: float = 0.3,
    ):
        self.mode = mode
        self.joint_alpha = joint_alpha
        self.anneal_warmup = anneal_warmup
        self.current_epoch = 0
        self.total_epochs = 1

    def set_epoch(self, epoch: int, total_epochs: int):
        self.current_epoch = epoch
        self.total_epochs = total_epochs

    def __call__(
        self,
        token_scores: torch.Tensor,      # (batch, seq_len)
        sample_labels: torch.Tensor,     # (batch,)
        token_labels: torch.Tensor = None,  # (batch, seq_len) - optional
        attention_mask: torch.Tensor = None,  # (batch, seq_len)
        span_masks: torch.Tensor = None,  # (batch, seq_len) - 1 if in span
    ) -> torch.Tensor:

        if attention_mask is None:
            attention_mask = torch.ones_like(token_scores)

        # Sample-level losses
        if self.mode == LossMode.SAMPLE_MEAN:
            sample_score = (token_scores * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            return F.binary_cross_entropy_with_logits(sample_score, sample_labels.float())

        elif self.mode == LossMode.SAMPLE_MAX:
            # Mask out padding with -inf before max
            masked_scores = token_scores.masked_fill(~attention_mask.bool(), float('-inf'))
            sample_score = masked_scores.max(dim=1).values
            return F.binary_cross_entropy_with_logits(sample_score, sample_labels.float())

        # Token-level losses
        elif self.mode == LossMode.TOKEN_ALL:
            assert token_labels is not None, "token_labels required for token_all mode"
            loss = F.binary_cross_entropy_with_logits(
                token_scores, token_labels.float(), reduction='none'
            )
            return (loss * attention_mask).sum() / attention_mask.sum()

        elif self.mode == LossMode.TOKEN_SPANS_ONLY:
            assert token_labels is not None and span_masks is not None
            # Only compute loss on tokens near spans
            combined_mask = attention_mask * span_masks
            loss = F.binary_cross_entropy_with_logits(
                token_scores, token_labels.float(), reduction='none'
            )
            return (loss * combined_mask).sum() / combined_mask.sum().clamp(min=1)

        # Span-level losses
        elif self.mode == LossMode.SPAN_MAX:
            assert span_masks is not None
            # Max in span regions
            pos_mask = span_masks * attention_mask
            neg_mask = (1 - span_masks) * attention_mask

            if pos_mask.sum() > 0:
                pos_scores = token_scores.masked_fill(~pos_mask.bool(), float('-inf'))
                pos_max = pos_scores.max(dim=1).values
                pos_loss = F.binary_cross_entropy_with_logits(
                    pos_max, torch.ones_like(pos_max)
                )
            else:
                pos_loss = 0.0

            if neg_mask.sum() > 0:
                neg_scores = token_scores.masked_fill(~neg_mask.bool(), float('inf'))
                neg_min = neg_scores.min(dim=1).values
                neg_loss = F.binary_cross_entropy_with_logits(
                    neg_min, torch.zeros_like(neg_min)
                )
            else:
                neg_loss = 0.0

            return pos_loss + neg_loss

        # Hybrid losses
        elif self.mode == LossMode.JOINT:
            sample_score = (token_scores * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            sample_loss = F.binary_cross_entropy_with_logits(sample_score, sample_labels.float())

            assert token_labels is not None
            token_loss = F.binary_cross_entropy_with_logits(
                token_scores, token_labels.float(), reduction='none'
            )
            token_loss = (token_loss * attention_mask).sum() / attention_mask.sum()

            return self.joint_alpha * sample_loss + (1 - self.joint_alpha) * token_loss

        elif self.mode == LossMode.ANNEALED:
            # Compute alpha based on epoch
            warmup_epochs = int(self.total_epochs * self.anneal_warmup)
            if self.current_epoch < warmup_epochs:
                alpha = 1.0
            else:
                progress = (self.current_epoch - warmup_epochs) / max(1, self.total_epochs - warmup_epochs)
                alpha = 1.0 - progress

            sample_score = (token_scores * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            sample_loss = F.binary_cross_entropy_with_logits(sample_score, sample_labels.float())

            assert token_labels is not None
            token_loss = F.binary_cross_entropy_with_logits(
                token_scores, token_labels.float(), reduction='none'
            )
            token_loss = (token_loss * attention_mask).sum() / attention_mask.sum()

            return alpha * sample_loss + (1 - alpha) * token_loss

        else:
            raise ValueError(f"Unknown loss mode: {self.mode}")
```

---

### File: `scripts/probe_training_ntml.py` (changes)

```python
# Add to argument parser
parser.add_argument('--loss_mode', type=str, default='sample_mean',
    choices=['sample_mean', 'sample_max', 'token_all', 'token_spans_only',
             'span_mean', 'span_max', 'joint', 'annealed'],
    help='Loss function mode')
parser.add_argument('--joint_alpha', type=float, default=0.5,
    help='Weight for sample loss in joint mode (1.0 = sample only, 0.0 = token only)')
parser.add_argument('--anneal_warmup', type=float, default=0.3,
    help='Fraction of epochs for warmup in annealed mode')

# Deprecate old flag
parser.add_argument('--use_max_aggregation', action='store_true',
    help='DEPRECATED: Use --loss_mode span_max instead')

# In main():
if args.use_max_aggregation:
    print("WARNING: --use_max_aggregation is deprecated. Use --loss_mode span_max")
    args.loss_mode = 'span_max'

loss_fn = ProbeLoss(
    mode=LossMode(args.loss_mode),
    joint_alpha=args.joint_alpha,
    anneal_warmup=args.anneal_warmup,
)
```

---

### Data Preparation Changes

Need to also pass `token_labels` and `span_masks` to the loss function:

```python
# In activation_store.py or trainer.py
def prepare_batch_with_token_labels(batch, spans):
    """Add token-level labels based on NTML spans."""
    token_labels = torch.zeros(batch['activations'].shape[:2])  # (batch, seq_len)

    for i, sample_spans in enumerate(spans):
        for span in sample_spans:
            if span.label == 'LIE':
                token_labels[i, span.start:span.end] = 1.0

    batch['token_labels'] = token_labels
    batch['span_masks'] = (token_labels > 0).float()  # or expand with margin
    return batch
```

---

## Experiment Plan

### Phase 1: Validate Token-Level Training
```bash
# Pure token-level
python scripts/probe_training_ntml.py \
    --loss_mode token_all \
    --layers 22 34 \
    --hyperparams '{"n_heads": 1, "temperature": 0.5}' \
    --num_epochs 60 \
    ...
```
**Expected:** R@Oracle ↑, Sample AUROC ↓

### Phase 2: Find Best Joint Alpha
```bash
for alpha in 0.3 0.5 0.7; do
    python scripts/probe_training_ntml.py \
        --loss_mode joint \
        --joint_alpha $alpha \
        --layers 34 \
        ...
done
```
**Goal:** Find α that balances detection + localization

### Phase 3: Test Annealing
```bash
python scripts/probe_training_ntml.py \
    --loss_mode annealed \
    --anneal_warmup 0.3 \
    --num_epochs 60 \
    --layers 34 \
    ...
```
**Expected:** Best of both worlds - learns detection first, refines localization

---

## Success Metrics

| Mode | Target Sample AUROC | Target R@Oracle |
|------|---------------------|-----------------|
| sample_mean (baseline) | 0.94 | 0.20 |
| token_all | 0.80+ | 0.40+ |
| joint (α=0.5) | 0.88+ | 0.30+ |
| annealed | 0.90+ | 0.35+ |
