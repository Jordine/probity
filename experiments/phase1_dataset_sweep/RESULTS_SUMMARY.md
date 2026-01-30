# Phase 1 Dataset Sweep Results

## Summary

Evaluated 202 probes across 7 training datasets + Apollo baseline.

## Key Findings

### Q1: Did 10T1L improve over 5T2L?

**YES, significantly.**

| Mix | Avg AUROC | n |
|-----|-----------|---|
| 10T1L | **0.806** | 46 |
| 5T2L | 0.733 | 156 |

**Delta: +0.073 AUROC** (10% relative improvement)

More truths per lie (10:1 vs 5:2) creates cleaner training signal.

### Q2: Best Attention Hyperparameters

| Variant | Avg AUROC |
|---------|-----------|
| **attentionh1_t1.0** | **0.856** |
| attentionh2_t1.0 | 0.846 |
| attentionh4_t0.5 | 0.835 |
| attentionh4_t2.0 | 0.833 |
| attentionh4_t1.0 | 0.832 |
| attentionh8_t1.0 | 0.813 |

**Winner: 1 attention head, temperature 1.0**

Fewer heads = better. Suggests deception signal is concentrated, not distributed.

### Q3: Best Sklearn Hyperparameters

| Variant | Avg AUROC |
|---------|-----------|
| **sklearn_logisticC0.01_l2** | **0.678** |
| sklearn_logisticC1.0_l1 | 0.651 |
| sklearn_logisticC0.1_l2 | 0.649 |
| sklearn_logisticC1.0_l2 | 0.643 |
| sklearn_logisticC10.0_l2 | 0.642 |

**Winner: C=0.01 with L2 regularization**

Strong regularization (low C) helps. L2 beats L1.

**Note:** All sklearn probes underperform attention probes significantly.

### Q4: Best 5T2L Dataset (Instruction Style)

| Style | Avg AUROC |
|-------|-----------|
| **neutral** | **0.738** |
| **conditional** | **0.738** |
| roleplay | 0.736 |
| instructional | 0.736 |
| direct_liar | 0.722 |

**Winner: Neutral or Conditional (tied)**

Surprisingly, explicit "you are a liar" framing (direct_liar) performed worst.

### Best Layer

| Layer | Avg AUROC |
|-------|-----------|
| **22** | **0.783** |
| 36 | 0.743 |
| 44 | 0.712 |

**Winner: Layer 22** (Apollo's choice validated)

### All Probe Types Ranked

| Probe Type | Avg AUROC |
|------------|-----------|
| attentionh1_t1.0 | 0.856 |
| attentionh2_t1.0 | 0.846 |
| attentionh4_t0.5 | 0.835 |
| attentionh4_t2.0 | 0.833 |
| attentionh4_t1.0 | 0.832 |
| attentionh8_t1.0 | 0.813 |
| **meandiffdefault** | **0.714** |
| sklearn_logisticC0.01_l2 | 0.678 |
| sklearn_logisticC1.0_l1 | 0.651 |
| sklearn_logisticC0.1_l2 | 0.649 |
| sklearn_logisticC1.0_l2 | 0.643 |
| sklearn_logisticC10.0_l2 | 0.642 |

**Note:** Meandiff (simple mean difference) outperforms all sklearn probes!

### vs Apollo Baseline

| Probe | AUROC |
|-------|-------|
| Apollo (baseline) | **0.926** |
| Best non-Apollo (10T1L_roleplay attentionh1_t1.0 L22) | 0.917 |
| **Gap** | 0.009 |

Apollo still wins, but gap is small with 10T1L + attention probes.

## Top 10 Individual Configurations

| Rank | AUROC | Dataset | Probe Type | Layer |
|------|-------|---------|------------|-------|
| 1 | 0.926 | apollo_baseline | apollo | 22 |
| 2 | 0.917 | 10T1L_roleplay | attentionh1_t1.0 | 22 |
| 3 | 0.910 | 10T1L_direct_liar | attentionh1_t1.0 | 22 |
| 4 | 0.908 | 10T1L_roleplay | attentionh2_t1.0 | 22 |
| 5 | 0.907 | 10T1L_direct_liar | attentionh2_t1.0 | 22 |
| 6 | 0.901 | 10T1L_direct_liar | attentionh4_t0.5 | 22 |
| 7 | 0.896 | 10T1L_direct_liar | attentionh4_t2.0 | 22 |
| 8 | 0.893 | 10T1L_roleplay | attentionh4_t0.5 | 22 |
| 9 | 0.892 | 10T1L_direct_liar | attentionh4_t1.0 | 22 |
| 10 | 0.888 | 10T1L_roleplay | attentionh4_t1.0 | 22 |

## Recommendations

1. **Use 10T1L** (10 truths, 1 lie) for training data generation
2. **Use attention probes with 1-2 heads** at temperature 1.0
3. **Target layer 22** (validates Apollo's choice)
4. **Neutral/conditional instruction styles** work best for 5T2L
5. **Meandiff is a strong simple baseline** - don't overcomplicate with sklearn

## Files

- `all_metrics_summary.csv`: Full results table
- Individual results in `results/<dataset>/aggregated/<probe_type>/layer_<N>/`
