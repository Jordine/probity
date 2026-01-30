# Phase 1 Dataset Sweep Results (Corrected)

## Summary

Evaluated 202 probes across 7 training datasets + Apollo baseline.

**Complete datasets:** 10T1L_direct_liar, 5T2L_direct_liar, 5T2L_roleplay, 5T2L_instructional, 5T2L_conditional (36 probes each)
**Partial datasets:** 10T1L_roleplay (10), 5T2L_neutral (12)

---

## Q1: Did 10T1L improve over 5T2L?

**YES - every single probe improved.**

Fair comparison: 10T1L_direct_liar vs 5T2L_direct_liar (same style, both complete)

| Mix | Avg AUROC |
|-----|-----------|
| **10T1L_direct_liar** | **0.799** |
| 5T2L_direct_liar | 0.722 |
| **Delta** | **+0.077** |

All 36 matched probes improved with 10T1L. Biggest gains:
- sklearn_logisticC1.0_l2 L36: +0.227 (!)
- sklearn_logisticC10.0_l2 L36: +0.228 (!)
- attentionh8_t1.0 L44: +0.128

**Conclusion:** 10:1 truth-to-lie ratio creates much cleaner training signal.

---

## Q2: Best Attention Hyperparameters

(Using complete 5T2L datasets: conditional, direct_liar, instructional, roleplay)

| Variant | Avg AUROC |
|---------|-----------|
| **attentionh1_t1.0** | **0.848** |
| attentionh2_t1.0 | 0.836 |
| attentionh4_t0.5 | 0.825 |
| attentionh4_t2.0 | 0.823 |
| attentionh4_t1.0 | 0.822 |
| attentionh8_t1.0 | 0.800 |

**Winner: 1 attention head, temperature 1.0**

Pattern: Fewer heads = better. Deception signal is concentrated, not distributed.

---

## Q3: Best Sklearn Hyperparameters

(Using complete 5T2L datasets)

| Variant | Avg AUROC |
|---------|-----------|
| **sklearn_logisticC0.01_l2** | **0.653** |
| sklearn_logisticC1.0_l1 | 0.632 |
| sklearn_logisticC0.1_l2 | 0.623 |
| sklearn_logisticC1.0_l2 | 0.617 |
| sklearn_logisticC10.0_l2 | 0.617 |

**Winner: C=0.01 with L2 regularization**

BUT: Sklearn performance dramatically improves with 10T1L training data:
- 10T1L sklearn L36: 0.82+ AUROC
- 5T2L sklearn L36: 0.59 AUROC

**Conclusion:** Sklearn probes aren't inherently bad - they just need cleaner training signal.

---

## Q4: Best 5T2L Dataset (Instruction Style)

(Complete datasets only: conditional, direct_liar, instructional, roleplay)

| Style | Avg AUROC |
|-------|-----------|
| **conditional** | **0.738** |
| roleplay | 0.736 |
| instructional | 0.736 |
| direct_liar | 0.722 |

**Winner: Conditional** (marginally)

Surprising: Explicit "you are a liar" framing (direct_liar) performed worst.

---

## Additional Findings

### Meandiff Performance
- 5T2L: 0.700 avg AUROC
- Beats sklearn with 5T2L data, but not with 10T1L data

### Best Layer
| Layer | Avg AUROC (5T2L complete) |
|-------|-----------|
| **22** | **0.760** |
| 36 | 0.726 |
| 44 | 0.690 |

Layer 22 is best (validates Apollo's choice).

### vs Apollo Baseline

| Probe | AUROC |
|-------|-------|
| Apollo (baseline) | **0.926** |
| Best non-Apollo (10T1L_roleplay attentionh1_t1.0 L22) | 0.917 |
| **Gap** | 0.009 |

---

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

---

## Recommendations

1. **Use 10T1L** (10 truths, 1 lie) - massive improvement (+0.077 avg, all probes better)
2. **Use attention probes with 1 head** at temperature 1.0
3. **Target layer 22** (validates Apollo)
4. **Conditional instruction style** for 5T2L (but 10T1L matters more than style)
5. **Sklearn probes need 10T1L** to be competitive - don't write them off

## Files

- `all_metrics_summary.csv`: Full results table
- `results_comparison.png`: Visualization plots
- Individual results in `results/<dataset>/aggregated/<probe_type>/layer_<N>/`
