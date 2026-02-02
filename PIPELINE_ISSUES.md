# Probity Pipeline Issues & Fixes

## Overview

Comprehensive documentation of issues discovered during end-to-end pipeline trace.
Prioritized by impact with verification steps.

---

## CRITICAL: Position Adjustment Mismatch

### The Bug

**Summary:** Training happens on WRONG tokens due to position index mismatch.

**What happens:**

1. **During batching** (`tokenized.py:384-392`): Token positions are adjusted for left padding
   ```
   Original tokens: [BOS, token1, token2, deceptive_token]
   Original position: 3 (points to deceptive_token)
   After left padding: [PAD, PAD, BOS, token1, token2, deceptive_token]
   Adjusted position: 5 (3 + padding_length of 2)
   ```

2. **During activation extraction** (`activation_store.py:61-77`): Reads ORIGINAL position from `example.token_positions`
   ```python
   pos = example.token_positions[position_key]  # Gets 3, not 5!
   positions.append(self.raw_activations[store_idx, pos])  # Extracts position 3 = PAD token
   ```

3. **Result:** Probes are trained on PAD token activations, not deceptive token activations

### Files Involved

| File | Lines | Role |
|------|-------|------|
| `probity/datasets/tokenized.py` | 384-392 | Adjusts positions (CORRECT) |
| `probity/collection/activation_store.py` | 61-77 | Uses unadjusted positions (BUG) |
| `probity/utils/caching.py` | 86 | Calls `get_batch_tensors()` which returns adjusted positions in `batch["positions"]` |

### Fix

**Option A (Recommended):** Store adjusted positions in ActivationStore

```python
# In caching.py, after getting batch tensors:
batch = dataset.get_batch_tensors(batch_indices)
# Store the adjusted positions from batch["positions"] alongside activations
```

**Option B:** Pass padding info to ActivationStore and adjust on extraction

### Verification

```python
# Test: Verify position points to correct token
def test_position_alignment():
    dataset = TokenizedProbingDataset(...)
    batch = dataset.get_batch_tensors([0, 1, 2])

    # Get the adjusted position
    adjusted_pos = batch["positions"]["last_token"][0]

    # Decode token at that position
    token_at_pos = tokenizer.decode(batch["input_ids"][0, adjusted_pos])

    # Should match the expected token from example
    expected_token = dataset.examples[0].text.split()[-1]  # or similar
    assert token_at_pos.strip() in expected_token
```

---

## HIGH: Duplicated Loss Configuration

### The Bug

Loss mode setup is copy-pasted in 2+ files. Bug fixes in one won't propagate.

### Files

| File | Lines | Duplication |
|------|-------|-------------|
| `scripts/probe_training_ntml.py` | 38-83 | Original |
| `probity/training/parallel.py` | 170-180 | Copy |

### Fix

Create shared function:

```python
# probity/training/config_utils.py
def apply_loss_config(args, trainer_config):
    """Apply loss configuration from CLI args to trainer config."""
    if args.use_max_aggregation:
        trainer_config.loss_mode = "span_max"
    else:
        trainer_config.loss_mode = args.loss_mode
    trainer_config.joint_alpha = args.joint_alpha
    trainer_config.anneal_warmup = args.anneal_warmup
    trainer_config.sparsity_penalty = args.sparsity_penalty
```

### Verification

```bash
# Both training paths should produce identical probes given same config
python scripts/probe_training_ntml.py --loss_mode token_bce ...
python -c "from probity.training.parallel import ...; # same config"
# Compare probe weights
```

---

## HIGH: Dataset Hash Collision Risk

### The Bug

Cache hash only uses first 10 examples + size. Different datasets can have same hash.

### File

`probity/utils/caching.py:14-22`

```python
def get_dataset_hash(dataset) -> str:
    content_str = ""
    for ex in dataset.examples[:10]:  # Only first 10!
        content_str += f"{ex.text[:100]}_{ex.label}_"
    content_str += f"_{len(dataset.examples)}"
```

### Fix

```python
def get_dataset_hash(dataset) -> str:
    """Generate hash using ALL examples."""
    import hashlib
    hasher = hashlib.md5()
    for ex in dataset.examples:
        hasher.update(f"{ex.text}_{ex.label}".encode())
    hasher.update(str(len(dataset.examples)).encode())
    return hasher.hexdigest()[:16]
```

### Verification

```python
def test_dataset_hash_uniqueness():
    # Two datasets with same first 10 but different rest
    ds1 = create_dataset(examples_a)
    ds2 = create_dataset(examples_a[:10] + different_examples)

    assert get_dataset_hash(ds1) != get_dataset_hash(ds2)
```

---

## HIGH: Contrastive Pair Split Bug

### The Bug

`trainer.py:264-299` - Index `i` is the enumeration index, not activation store index.

```python
for i, dataset_ex_idx in enumerate(activation_store.example_indices):
    ...
    group_to_indices[group_id].append(i)  # i is wrong when sparse indexing
```

### Fix

```python
for store_idx, dataset_ex_idx in enumerate(activation_store.example_indices):
    ...
    group_to_indices[group_id].append(store_idx)  # Use store_idx explicitly
```

### Verification

```python
def test_contrastive_split():
    # Create dataset with known contrastive pairs
    # Verify pairs end up in same split (train or val, not split across)
```

---

## MEDIUM: Metrics Key Inconsistency

### The Bug

`trainer.py` saves `train_auroc`, but `probe_training_ntml.py:125-128` looks for `train_auroc_score`.

### Files

| File | Lines | Key Used |
|------|-------|----------|
| `probity/training/trainer.py` | 885-897 | `train_auroc` |
| `scripts/probe_training_ntml.py` | 125-128 | `train_auroc_score` |

### Fix

Standardize to `train_auroc` everywhere.

---

## MEDIUM: 4 Different AUROC Implementations

### The Problem

Same metric name, different computations:

| File | What it measures |
|------|------------------|
| `probe_eval_deception_datasets.py:393` | Mean token scores per sample |
| `span_loader.py:241` | Per-token in_span labels |
| `probe_analysis.py:308` | Token averages across probe types |
| `max_aggregation_metrics.py:204` | Per-sample max_in vs max_out |

### Fix

Create canonical metrics module:

```python
# probity/metrics/core.py
class ProbeMetrics:
    @staticmethod
    def sample_auroc(scores: List[float], labels: List[int]) -> float:
        """AUROC where each sample has one score (mean/max aggregated)."""
        return roc_auc_score(labels, scores)

    @staticmethod
    def token_auroc(token_scores: np.ndarray, token_labels: np.ndarray) -> float:
        """AUROC at token level (all tokens flattened)."""
        return roc_auc_score(token_labels.flatten(), token_scores.flatten())

    @staticmethod
    def discrimination_auroc(in_scores: np.ndarray, out_scores: np.ndarray) -> float:
        """AUROC for discriminating in-span vs out-span (max aggregation)."""
        all_scores = np.concatenate([in_scores, out_scores])
        all_labels = np.concatenate([np.ones(len(in_scores)), np.zeros(len(out_scores))])
        return roc_auc_score(all_labels, all_scores)
```

---

## MEDIUM: Missing Error Handling in Activation Extraction

### File

`probity/utils/caching.py:90-136`

### Issues

- No try-catch around `model.run_with_cache()`
- If one batch fails, all cached data lost
- No validation of activation shapes

### Fix

```python
try:
    _, cache = model.run_with_cache(...)
except RuntimeError as e:
    print(f"Activation collection failed at batch {batch_start}: {e}")
    raise
```

---

## LOW: Debug Prints in Production

### Files with debug prints

- `probity/training/trainer.py:140-154`
- `probity/collection/activation_store.py:141-152`

### Fix

Replace with logging module:

```python
import logging
logger = logging.getLogger(__name__)
logger.debug(f"X shape: {X_expected.shape}")
```

---

## Files to Modify Summary

| Priority | File | Change |
|----------|------|--------|
| CRITICAL | `probity/collection/activation_store.py` | Use adjusted positions |
| CRITICAL | `probity/utils/caching.py` | Store adjusted positions, fix hash |
| HIGH | `probity/training/trainer.py` | Fix contrastive split, metrics key |
| HIGH | `scripts/probe_training_ntml.py` | Extract loss config to shared fn |
| HIGH | `probity/training/parallel.py` | Use shared loss config fn |
| MEDIUM | NEW: `probity/metrics/core.py` | Centralized metrics |
| MEDIUM | Multiple | Replace debug prints with logging |

---

## Verification Test Suite

Create `tests/test_pipeline_integrity.py`:

```python
"""Tests to verify pipeline fixes work correctly."""

def test_position_alignment():
    """CRITICAL: Verify token positions align between dataset and activations."""
    pass

def test_dataset_hash_uniqueness():
    """HIGH: Verify different datasets get different hashes."""
    pass

def test_contrastive_pairs_not_split():
    """HIGH: Verify contrastive pairs stay together in train/val split."""
    pass

def test_loss_config_consistency():
    """HIGH: Verify loss config produces same results in both training paths."""
    pass

def test_metrics_consistency():
    """MEDIUM: Verify AUROC implementations agree on synthetic data."""
    pass
```

---

## Execution Order

1. **Fix position alignment** (CRITICAL) - blocks all training
2. **Fix dataset hash** (HIGH) - prevents cache corruption
3. **Fix contrastive split** (HIGH) - ensures proper training
4. **Standardize metrics keys** (MEDIUM) - prevents lost metrics
5. **Add test suite** - prevents regressions
6. **Centralize metrics** (MEDIUM) - long-term maintainability
7. **Replace debug prints** (LOW) - code quality
