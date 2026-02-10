# Probe Evaluation Results Summary

**Generated:** 2026-02-09
**Total probe configurations analyzed:** 107

> **WARNING (2026-02-09):** Localization audit found that span_auroc, R@Oracle, and norm_auroc
> are ALL artifacts of sample-level signal leaking into token-level metrics. After within-sample
> z-scoring with ground-truth labels, every probe is at chance (AUROC 0.47-0.55).
> Only sample-level AUROC is valid. See `LOCALIZATION_AUDIT_FINDINGS.md`.

---

## Top 30 Results by Span AUROC (Ranking Metrics)

~~Span AUROC measures how well the probe can identify which *specific tokens* within a conversation contain deception.~~
**ARTIFACT:** Span AUROC reflects sample-level contamination, not within-sample localization.

| Rank | Span AUROC | Sample AUROC | R@Oracle | Layer | Probe Type | Loss Mode | Path |
|------|------------|--------------|----------|-------|------------|-----------|------|
| 1 | **0.5447** | 0.3651 | N/A | 40 | logistic | N/A | results/ranking_batch2/aggregated/logistic/layer_40 |
| 2 | **0.5410** | 0.4689 | N/A | 42 | meandiff | N/A | results/ranking_batch2/aggregated/meandiff/layer_42 |
| 3 | **0.5403** | 0.3205 | N/A | 42 | logistic | N/A | results/ranking_batch2/aggregated/logistic/layer_42 |
| 4 | **0.5402** | 0.3553 | N/A | 38 | logistic | N/A | results/ranking_batch2/aggregated/logistic/layer_38 |
| 5 | **0.5385** | 0.4663 | 0.2159 | 50 | meandiff | sample_mean | loss_mode_exp/eval_results/sample_mean/aggregated/meandiff/layer_50 |

---

## Key Findings

### Best Span AUROC by Probe Type (from top 50)
- **logistic**: max=0.5447, mean=0.5290, count=21
- **meandiff**: max=0.5410, mean=0.5352, count=13
- **attention**: max=0.5330, mean=0.5223, count=12

### Best Span AUROC by Layer (from top 50)
- **Layer 40**: max=0.5447, mean=0.5304
- **Layer 42**: max=0.5410, mean=0.5317
- **Layer 38**: max=0.5402, mean=0.5297
- **Layer 50**: max=0.5385, mean=0.5347
- **Layer 22**: max=0.5384, mean=0.5302

**Key insight:** Mid-to-late layers (38-50) show the best span-level deception detection.

---

See `all_metrics_output.txt` for full 107-result table.
