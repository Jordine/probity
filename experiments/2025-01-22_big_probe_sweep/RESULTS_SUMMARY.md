# Big Probe Sweep Results Summary

**Date**: 2025-01-22/23
**Model**: Llama-3.3-70B-Instruct
**Training Data**: 5T2L_500samples (500 conversations, 3465 statement examples, 28.3% lies)

---

## Overview

Trained deception detection probes across **18 layers** (22-56, even numbers) with **4 probe types**:
1. `sklearn_logistic` - L-BFGS logistic regression
2. `meandiff` - Mean difference direction
3. `attention` - Multi-head attention probe (4 heads)
4. `logistic` - PyTorch with max_aggregation loss

Evaluated on:
- `ai_liar.jsonl` - AI lying scenarios
- `sandbagging_v2__wmdp_mmlu.jsonl` - Sandbagging detection (500 samples)

---

## Key Results

### Best Probes by AUROC (Sample-Level Detection)

| Probe Type | Best Layer | AUROC | span_recall | Notes |
|------------|------------|-------|-------------|-------|
| **attention** | L32 | **0.908** | 0.560 | Best overall AUROC |
| meandiff | L22 | 0.876 | 0.203 | Fast, good baseline |
| sklearn_logistic | L28 | 0.804 | 0.385 | Simple, decent |
| logistic (max_aggr) | L24 | 0.614 | 0.026 | Poor - likely overfitting |

### Best Probes by Span Recall (Token-Level Localization)

| Probe Type | Best Layer | span_recall | AUROC | Notes |
|------------|------------|-------------|-------|-------|
| **attention** | L38 | **0.699** | 0.850 | Best localization |
| attention | L34 | 0.644 | 0.883 | Good balance |
| sklearn_logistic | L34 | 0.534 | 0.689 | Decent |
| meandiff | L30 | 0.278 | 0.787 | Poor localization |

### Comparison to Apollo Probe (Baseline)

| Probe | AUROC | span_recall | Notes |
|-------|-------|-------------|-------|
| Apollo L22 | 0.925 | 0.301 | Reference baseline |
| **Attention L32** | 0.908 | 0.560 | -2% AUROC, **+86% recall** |
| **Attention L38** | 0.850 | 0.699 | -8% AUROC, **+132% recall** |

**Key Finding**: Our attention probes achieve near-Apollo AUROC while having **2x better token localization**.

---

## Recommended Probes for Debate Monitoring

### Primary: Attention L38
- **Use case**: When you need to locate which specific tokens/statements are deceptive
- **AUROC**: 0.850 (good)
- **span_recall**: 0.699 (excellent - catches 70% of deceptive tokens)
- **span_precision**: 0.164

### Secondary: Attention L32
- **Use case**: When sample-level detection accuracy is most important
- **AUROC**: 0.908 (excellent)
- **span_recall**: 0.560 (good)

### Fast Baseline: Meandiff L22
- **Use case**: Quick screening, no GPU needed after activation collection
- **AUROC**: 0.876
- **span_recall**: 0.203 (poor localization)

---

## Layer-by-Layer Results (Attention Probe)

| Layer | AUROC | span_recall | span_precision | span_f1 |
|-------|-------|-------------|----------------|---------|
| 22 | 0.876 | 0.583 | 0.171 | 0.265 |
| 24 | 0.887 | 0.566 | 0.166 | 0.256 |
| 26 | 0.899 | 0.522 | 0.169 | 0.255 |
| 28 | 0.896 | 0.559 | 0.168 | 0.258 |
| 30 | 0.899 | 0.551 | 0.168 | 0.258 |
| **32** | **0.908** | 0.560 | 0.170 | 0.261 |
| 34 | 0.883 | 0.644 | 0.165 | 0.262 |
| 36 | 0.824 | 0.632 | 0.162 | 0.258 |
| **38** | 0.850 | **0.699** | 0.164 | 0.266 |
| 40 | 0.800 | 0.632 | 0.161 | 0.257 |
| 42 | 0.770 | 0.597 | 0.158 | 0.250 |
| 44 | 0.748 | 0.613 | 0.157 | 0.250 |
| 46 | 0.778 | 0.626 | 0.159 | 0.254 |
| 48 | 0.761 | 0.627 | 0.158 | 0.253 |
| 50 | 0.763 | 0.557 | 0.156 | 0.244 |
| 52 | 0.768 | 0.632 | 0.159 | 0.254 |
| 54 | 0.720 | 0.604 | 0.155 | 0.247 |
| 56 | 0.706 | 0.586 | 0.153 | 0.243 |

---

## Distribution Analysis

### Attention L32 (Best AUROC)
- Token-level: Honest mean=-0.557, Deceptive mean=-0.024, **Separation=0.533**
- Sample-level: Honest mean=-0.569, Deceptive mean=0.122, **Separation=0.691**

### Attention L38 (Best Recall)
- Token-level: Honest mean=-0.275, Deceptive mean=0.223, **Separation=0.499**
- Sample-level: Honest mean=-0.286, Deceptive mean=0.325, **Separation=0.611**

### Apollo L22 (Baseline)
- Sample-level: Honest mean=0.449, Deceptive mean=0.486, **Separation=0.036**
- Much tighter distributions (std ~0.02) compared to attention (std ~0.4)

---

## Files

### Trained Probes
```
big_probe_sweep/
├── attention/layer_{22-56}_probe.pt
├── sklearn_logistic/layer_{22-56}_probe.pt
├── meandiff/layer_{22-56}_probe.pt
└── logistic/layer_{22-56}_probe.pt
```

### Evaluation Results
```
big_probe_sweep_eval_full/
├── aggregated/{probe_type}/layer_{N}/
│   ├── metrics.json
│   ├── metrics_all_thresholds.json
│   ├── span_metrics.json
│   ├── raw_scores.json
│   └── token_visualization.html
└── visualizations/
    └── {probe_type}_layer_{N}_distributions.png
```

---

## Training Configuration

```bash
python scripts/probe_training_ntml.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --train_dataset_dir "./data/NTML-datasets/5T2L_500samples_shuffled_explicit_direct_liar_prefix.json" \
    --probe_save_dir "./big_probe_sweep" \
    --probe_types sklearn_logistic meandiff attention logistic \
    --hyperparams '{"n_heads": 4, "temperature": 1.0}' \
    --layers 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 \
    --max_length 1024 --batch_size 32 \
    --num_epochs 80 --patience 999 \
    --use_max_aggregation --anneal_warmup 1.0 \
    --dishonest_mode all --honest_mode none
```

---

## Next Steps

1. **Visualization**: Regenerate HTML visualizations with new interactive threshold slider
2. **Debate Integration**: Use attention L38 probe for debate monitoring (best localization)
3. **Ensemble**: Consider combining L32 (best AUROC) + L38 (best recall) for robust detection
4. **New Datasets**: Test on more diverse deception scenarios
