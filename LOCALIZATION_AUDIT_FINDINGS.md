# Localization Audit Findings

**Date:** 2026-02-09
**Analysis scripts:** `compare_norm_auroc_levels.py`, `per_sample_localization_diagnostic.py`, `check_mixed_labels.py`
**Output data:** `NORM_AUROC_COMPARISON.tsv`

---

## Executive Summary

**No probe in the probity pipeline has within-sample token-level localization.** All reported localization metrics (span_auroc ~0.54, R@Oracle ~0.20, norm_auroc ~0.63) are artifacts of sample-level signal leaking into token-level metrics. After properly controlling for sample-level effects via within-sample z-scoring, every probe scores at chance (AUROC 0.47-0.55) for both phrase-level and statement-level discrimination.

This does NOT necessarily mean token-level deception localization is impossible — StefanHex achieved R@Oracle 0.989 on Llama-70B **in-distribution** using a fundamentally different methodology. However, StefanHex probes also underperformed OOD, so the question of whether any linear probe can localize deception across distributions remains open.

---

## Finding 1: token_labels in raw_scores.json Are Sample-Level

**Discovery:** The `token_labels` array in every `raw_scores.json` assigns ALL tokens in a deceptive sample label=1 and ALL tokens in an honest sample label=0. There is no within-sample structure.

**Evidence:** Scanned 586 raw_scores.json files (`check_mixed_labels.py`). Only 13/586 (2.2%) had any token-level variation within deceptive samples — all sklearn_logistic probes. The other 573 files use purely sample-level labels.

**Root cause:** `probe_eval_deception_datasets.py` line 436-437:
```python
token_labels = [detail['label']] * token_count
```
This broadcasts the sample label to every token.

**Impact:** Any metric computed using token_labels as ground truth (including the previous norm_auroc in `compute_all_metrics.py`) is measuring sample-level signal, not localization.

---

## Finding 2: Previous norm_auroc Was a Label Artifact

**Discovery:** The `compute_all_metrics.py` script computed norm_auroc by:
1. Reconstructing per-sample token boundaries from concatenated raw_scores
2. Z-scoring within "deceptive segments" using token_labels
3. Computing AUROC of "critical" (label=1) vs "non-critical" (label=0) tokens

Since token_labels are all-1 within deceptive samples (Finding 1), the only source of label=0 tokens was **boundary misalignment** — the heuristic reconstruction produces segments that are ~61% of true size (mean 164 vs 269 tokens), leaking in honest-sample tokens as false "non-critical" examples.

**Recomputation (2026-02-09, `fix_boundaries_recompute.py`):**
Tested both exact boundaries (eval data token counts) and heuristic boundaries against ground-truth labels (critical_indices, character spans). Results are identical:

| Boundary Method | Phrase Norm AUROC | Statement Norm AUROC |
|-----------------|-------------------|----------------------|
| Exact (eval counts) | 0.4992 (std=0.008) | 0.4979 (std=0.007) |
| Heuristic (cumsum) | 0.5001 (std=0.034) | 0.5212 (std=0.043) |

The heuristic has wider variance due to meandiff probes with broken segments. Neither method shows any localization signal.

Note: The heuristic achieves perfect alignment error (0.0) despite wrong segment sizes because probe scores are nearly flat within samples — any contiguous subset has the same mean. The exact boundaries using eval token counts are also imprecise because the eval script truncated sequences with max_length.

**Impact:** All 449 norm_auroc values in `ALL_METRICS_COMPUTED.tsv` are artifacts. The 0.58-0.65 range reflected honest-sample tokens leaking into deceptive segments via boundary misalignment, creating a false signal.

---

## Finding 3: True Within-Sample Localization is at Chance

**Method:** For each of 266 ai_liar samples, used ground-truth labels at two granularities:
- **Phrase-level:** Character-level spans from `ai_liar_labeled.json` (specific lie phrases)
- **Statement-level:** `critical_indices` from eval data (entire deceptive statements, ~14% of assistant tokens)

Within each sample: z-score all token scores (subtract mean, divide by std), then pool z-scored positive/negative tokens across samples and compute AUROC.

**Results (70 probes with n >= 10 samples):**

| Metric | Mean | Std | Range | Best |
|--------|------|-----|-------|------|
| Phrase norm AUROC | 0.4919 | 0.014 | [0.47, 0.53] | 0.526 |
| Statement norm AUROC | 0.5211 | 0.016 | [0.48, 0.55] | 0.552 |

No probe exceeds 0.55 at either granularity. The entire signal is sample-level.

---

## Finding 4: Non-Normalized span_auroc is Also Sample-Level

The existing `span_auroc` from `ranking_metrics.json` (mean ~0.52, best ~0.54) pools in-span and out-span token scores across ALL deceptive samples without per-sample normalization.

**Evidence:** Correlation between phrase norm AUROC and existing span_auroc is r = -0.256 (anti-correlated). If span_auroc measured real within-sample localization, it would correlate positively with the z-scored version.

The ~0.54 span_auroc is driven by deceptive samples having slightly higher overall probe scores, combined with in-span tokens being a non-random subset of positions within each sample.

---

## Finding 5: Per-Sample AUROC Distributions Are Pure Noise

Computed per-sample within-sample AUROC for 15 probes across 164 deceptive samples each.

**Typical result (attention probe, layer 50):**
- Mean per-sample AUROC: 0.552
- Std: 0.163
- Distribution: symmetric around 0.50
- Samples with AUROC > 0.80: 11/164
- Samples with AUROC < 0.20: 2/164

The high variance (std ~0.16) with near-zero mean shift is the signature of random scoring within samples. The few samples with high AUROC are explained by chance alone.

Correlation between per-sample AUROC and sample mean score: r = -0.17 to +0.22 (negligible). A sample's overall score does not predict whether the probe can localize within it.

---

## Why StefanHex's Approach Worked In-Distribution

StefanHex achieved R@Oracle 0.989 on the same model (Llama-70B) using:
1. **StandardScaler normalization** (zero mean, unit variance per feature)
2. **Individual tokens as training examples** (not aggregated per-sample)
3. **class_weight="balanced"** to handle ~2% lie-token rate
4. **Simple LogisticRegression** with L2 regularization

Key difference: their probes were trained to classify **individual token activations** as lie/not-lie, with properly balanced class weights. Our probes were trained on contrastive sample pairs (lie_version vs truth_version), learning the sample-level context shift instead.

**Important caveats:**
- These are in-distribution results (trained and tested on similar Q&A lies). StefanHex probes also underperformed OOD.
- This proves Llama-70B encodes *some* token-level deception signal in-distribution, but does not demonstrate that this signal generalizes to ai_liar/sandbagging evaluation sets.
- The question of whether any linear probe approach can achieve OOD deception localization remains open.

---

## Implications

1. **All localization metrics in ALL_METRICS_COMPUTED.tsv, ALL_RANKING_METRICS.tsv, and FINAL_MASTER_TABLE.tsv should be treated as sample-level metrics only.** The span_auroc, R@Oracle, norm_auroc columns do not reflect within-sample token discrimination.

2. **The probity pipeline's contrastive training (lie_version vs truth_version) produces sample-level probes by design.** The probe learns "is this the lying context?" not "which tokens are lies?" because the training signal is identical for all tokens within a sample.

3. **To get real localization, the training methodology must change.** Either:
   - Train on individual token activations with per-token labels (StefanHex approach)
   - Use a loss function that explicitly contrasts lie tokens vs non-lie tokens within the same sample
   - Use data where only specific tokens differ between conditions

4. **Sample-level detection (AUROC 0.85-0.95) remains valid and useful.** The probes reliably distinguish deceptive from honest conversations at the sample level.

---

## Files Created During This Analysis

| File | Purpose |
|------|---------|
| `compare_norm_auroc_levels.py` | Main comparison: phrase vs statement norm AUROC using ground-truth labels |
| `fix_boundaries_recompute.py` | Recomputation with exact vs heuristic boundaries (confirms result) |
| `per_sample_localization_diagnostic.py` | Per-sample AUROC distribution analysis |
| `check_mixed_labels.py` | Scan raw_scores for per-token vs per-sample labels |
| `NORM_AUROC_COMPARISON.tsv` | 100 rows of probe comparison results (heuristic boundaries) |
| `FIXED_NORM_AUROC.tsv` | 307 rows with both exact and heuristic boundary results |
| `LOCALIZATION_AUDIT_FINDINGS.md` | This document |
