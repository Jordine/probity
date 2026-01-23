# Span Localization Evaluation Results

## Summary

We evaluated 73 probes (Apollo + 72 trained probes across 4 types x 18 layers) for their ability to:
1. **Detect deceptive samples** (Sample AUROC)
2. **Localize deceptive tokens** (Recall@K%, Span AP)

## Key Finding

**Trade-off exists**: Probes with high sample detection (~0.9 AUROC) have poor token localization (~0.10 Recall@10%). Probes with better localization (~0.12 Recall@10%) have lower sample detection (~0.5 AUROC).

## Apollo Baseline
| Metric | Value |
|--------|-------|
| Sample AUROC | 0.923 |
| Recall@10% | 0.095 |
| Span AP | 0.220 |

## Best Attention Probes (for paper)

| Probe | Layer | Sample AUROC | Recall@10% | vs Apollo |
|-------|-------|--------------|------------|-----------|
| attention | 32 | **0.907** | 0.100 | +5% |
| attention | 38 | 0.870 | 0.107 | +13% |
| attention | 22 | 0.870 | 0.107 | +13% |
| attention | 44 | 0.773 | 0.113 | +19% |

## Recommended for Debate Experiments

**attention L38**: Sample AUROC=0.870, Recall@10%=0.107 (+13% vs Apollo)

## Files
- results/ranking_batch1/ - Layers 22-32
- results/ranking_batch2/ - Layers 34-44  
- results/ranking_batch3/ - Layers 46-56
- results/ranking_apollo/ - Apollo baseline
