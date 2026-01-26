# Span Localization Evaluation Results

## Summary

We evaluated 73 probes (Apollo + 72 trained probes across 4 types × 18 layers) for their ability to:
1. **Detect deceptive samples** (Sample AUROC)
2. **Localize deceptive tokens** (Token AUPRC, Recall@K%)

## Key Findings

1. **Trade-off exists**: Probes with high sample detection (attention L32: 0.908 AUROC) have lower token localization (AUPRC 0.195). Probes with best token localization (meandiff L40: AUPRC 0.209) have poor sample detection (0.496).

2. **Apollo is NOT best at localization**: Despite having the best sample AUROC (0.925), Apollo's token-level AUPRC (0.180) is beaten by several trained probes.

3. **Best balanced probe**: attention L36 achieves Sample AUROC 0.824 with Token AUPRC 0.203 (+13% vs Apollo).

## Metrics Explained

| Metric | Description |
|--------|-------------|
| Sample AUROC | Can the probe distinguish deceptive vs honest samples? |
| Token AUROC | AUROC for classifying individual tokens as in-span vs out-span |
| Token AUPRC | Average Precision for token-level ranking (better for imbalanced data) |
| Recall@K% | Of all deceptive tokens, what % are in the top K% highest-scoring tokens? |
| Cohen's d | Standardized effect size between in-span and out-span scores |
| TPR@X%FPR | True positive rate when X% of benign tokens exceed threshold |

## Results

### Apollo Baseline

| Metric | Value |
|--------|-------|
| Sample AUROC | **0.925** |
| Token AUROC | 0.539 |
| Token AUPRC | 0.180 |
| Recall@5% | 0.074 |
| Recall@10% | 0.136 |
| Recall@15% | 0.193 |
| Recall@20% | 0.245 |
| Recall@25% | 0.300 |
| Cohen's d | 0.149 |
| TPR@1%FPR | 0.019 |
| TPR@5%FPR | 0.080 |

### Top 10 by Sample AUROC (>0.8)

| Probe | Layer | Sample AUROC | Token AUPRC | R@10% | vs Apollo AUPRC |
|-------|-------|--------------|-------------|-------|-----------------|
| attention | 32 | **0.908** | 0.195 | 0.157 | +8% |
| attention | 30 | 0.899 | 0.197 | 0.160 | +9% |
| attention | 26 | 0.899 | 0.182 | 0.139 | +1% |
| attention | 28 | 0.896 | 0.190 | 0.146 | +6% |
| attention | 24 | 0.887 | 0.187 | 0.143 | +4% |
| attention | 34 | 0.883 | 0.197 | 0.157 | +9% |
| attention | 22 | 0.876 | 0.195 | 0.155 | +8% |
| meandiff | 22 | 0.876 | 0.173 | 0.126 | -4% |
| meandiff | 24 | 0.870 | 0.166 | 0.112 | -8% |
| attention | 38 | 0.850 | 0.199 | 0.166 | +11% |

### Top 10 by Token AUPRC

| Probe | Layer | Token AUPRC | Sample AUROC | R@10% | vs Apollo AUPRC |
|-------|-------|-------------|--------------|-------|-----------------|
| meandiff | 40 | **0.209** | 0.496 | 0.173 | **+16%** |
| meandiff | 38 | **0.209** | 0.543 | 0.171 | **+16%** |
| meandiff | 42 | 0.206 | 0.466 | 0.168 | +14% |
| meandiff | 36 | 0.205 | 0.529 | 0.165 | +14% |
| attention | 44 | 0.204 | 0.748 | 0.171 | +13% |
| meandiff | 44 | 0.204 | 0.469 | 0.168 | +13% |
| attention | 40 | 0.204 | 0.800 | 0.168 | +13% |
| meandiff | 46 | 0.203 | 0.466 | 0.166 | +13% |
| attention | 42 | 0.203 | 0.770 | 0.170 | +13% |
| attention | 36 | 0.203 | 0.824 | 0.169 | +13% |

### Top 10 by Recall@10%

| Probe | Layer | R@10% | Sample AUROC | Token AUPRC | vs Apollo R@10% |
|-------|-------|-------|--------------|-------------|-----------------|
| meandiff | 40 | **0.173** | 0.496 | 0.209 | **+27%** |
| attention | 50 | 0.172 | 0.763 | 0.200 | +26% |
| attention | 44 | 0.171 | 0.748 | 0.204 | +26% |
| meandiff | 38 | 0.171 | 0.543 | 0.209 | +26% |
| attention | 42 | 0.170 | 0.770 | 0.203 | +25% |
| meandiff | 50 | 0.169 | 0.460 | 0.202 | +24% |
| attention | 36 | 0.169 | 0.824 | 0.203 | +24% |
| attention | 48 | 0.169 | 0.761 | 0.199 | +24% |
| attention | 56 | 0.168 | 0.706 | 0.199 | +24% |
| meandiff | 42 | 0.168 | 0.466 | 0.206 | +24% |

### Best Balanced (Sample AUROC > 0.75 AND good localization)

| Probe | Layer | Sample AUROC | Token AUPRC | R@10% | TPR@5%FPR |
|-------|-------|--------------|-------------|-------|-----------|
| attention | 36 | **0.824** | 0.203 | 0.169 | 0.108 |
| attention | 40 | 0.800 | 0.204 | 0.168 | 0.105 |
| attention | 42 | 0.770 | 0.203 | 0.170 | 0.110 |
| attention | 50 | 0.763 | 0.200 | 0.172 | 0.113 |
| attention | 48 | 0.761 | 0.199 | 0.169 | 0.108 |
| attention | 44 | 0.748 | 0.204 | 0.171 | 0.111 |

## Token-Level FPR Calibration

Thresholds calibrated so X% of out-of-span (benign) tokens exceed threshold:

| Probe | Layer | 1% FPR Threshold | TPR@1%FPR | 5% FPR Threshold | TPR@5%FPR |
|-------|-------|------------------|-----------|------------------|-----------|
| Apollo | 22 | 0.582 | 0.019 | 0.543 | 0.080 |
| attention | 36 | 1.723 | 0.027 | 1.103 | 0.108 |
| attention | 40 | 1.759 | 0.030 | 1.095 | 0.105 |
| meandiff | 40 | 1.783 | 0.032 | 1.089 | 0.109 |
| meandiff | 38 | 1.685 | 0.031 | 1.034 | 0.110 |

**Key insight**: At 5% token-level FPR, attention L36 catches ~10.8% of deceptive tokens while Apollo only catches ~8.0%.

## Recommended Probes for Debate Experiments

### Option 1: attention L36 (Best Balanced)
- Sample AUROC: 0.824 (vs Apollo 0.925, -11%)
- Token AUPRC: 0.203 (vs Apollo 0.180, **+13%**)
- Recall@10%: 0.169 (vs Apollo 0.136, **+24%**)
- TPR@5%FPR: 0.108 (vs Apollo 0.080, **+35%**)
- **Rationale**: Best combination of sample detection and token localization

### Option 2: attention L40 (Higher Token AUPRC)
- Sample AUROC: 0.800 (vs Apollo 0.925, -14%)
- Token AUPRC: 0.204 (vs Apollo 0.180, **+13%**)
- Recall@10%: 0.168 (vs Apollo 0.136, **+24%**)
- **Rationale**: Slightly better AUPRC at cost of sample detection

### Option 3: meandiff L38 (Best Localization)
- Sample AUROC: 0.543 (vs Apollo 0.925, -41%)
- Token AUPRC: 0.209 (vs Apollo 0.180, **+16%**)
- Recall@10%: 0.171 (vs Apollo 0.136, **+26%**)
- **Rationale**: Best pure localization, but poor sample detection

## Paper Framing

> "While Apollo probes achieve strong sample-level detection (AUROC=0.925), we find they underperform on token-level localization (Token AUPRC=0.180, Recall@10%=13.6%). We train an attention probe on layer 36 residual stream activations that achieves competitive sample detection (AUROC=0.824) while significantly improving token-level metrics: +13% Token AUPRC (0.203), +24% Recall@10% (0.169), and +35% TPR at 5% token-level FPR. This probe is better suited for debate setups where monitors must identify *which* spans to scrutinize."

## Full Results Table

All 72 trained probes + Apollo, sorted by Token AUPRC:

| Probe | Layer | Sample AUROC | Token AUROC | Token AUPRC | R@5% | R@10% | R@15% | R@20% | R@25% | Cohen's d |
|-------|-------|--------------|-------------|-------------|------|-------|-------|-------|-------|-----------|
| meandiff | 40 | 0.496 | 0.566 | 0.209 | 0.095 | 0.173 | 0.234 | 0.288 | 0.343 | 0.253 |
| meandiff | 38 | 0.543 | 0.564 | 0.209 | 0.097 | 0.171 | 0.233 | 0.287 | 0.339 | 0.253 |
| meandiff | 42 | 0.466 | 0.565 | 0.206 | 0.094 | 0.168 | 0.234 | 0.289 | 0.342 | 0.250 |
| meandiff | 36 | 0.529 | 0.557 | 0.205 | 0.098 | 0.165 | 0.228 | 0.283 | 0.329 | 0.225 |
| attention | 44 | 0.748 | 0.563 | 0.204 | 0.097 | 0.171 | 0.233 | 0.290 | 0.342 | 0.248 |
| meandiff | 44 | 0.469 | 0.562 | 0.204 | 0.093 | 0.168 | 0.233 | 0.290 | 0.341 | 0.239 |
| attention | 40 | 0.800 | 0.560 | 0.204 | 0.094 | 0.168 | 0.232 | 0.289 | 0.339 | 0.238 |
| meandiff | 46 | 0.466 | 0.559 | 0.203 | 0.093 | 0.166 | 0.233 | 0.288 | 0.340 | 0.232 |
| attention | 42 | 0.770 | 0.563 | 0.203 | 0.095 | 0.170 | 0.235 | 0.289 | 0.339 | 0.250 |
| attention | 36 | 0.824 | 0.560 | 0.203 | 0.095 | 0.169 | 0.229 | 0.283 | 0.334 | 0.234 |
| meandiff | 50 | 0.460 | 0.558 | 0.202 | 0.093 | 0.169 | 0.235 | 0.286 | 0.337 | 0.222 |
| attention | 46 | 0.778 | 0.559 | 0.202 | 0.095 | 0.168 | 0.229 | 0.283 | 0.335 | 0.237 |
| meandiff | 48 | 0.448 | 0.559 | 0.201 | 0.091 | 0.164 | 0.232 | 0.287 | 0.337 | 0.226 |
| meandiff | 52 | 0.449 | 0.556 | 0.201 | 0.093 | 0.167 | 0.232 | 0.282 | 0.335 | 0.214 |
| meandiff | 34 | 0.632 | 0.554 | 0.200 | 0.094 | 0.164 | 0.223 | 0.281 | 0.329 | 0.215 |
| attention | 50 | 0.763 | 0.556 | 0.200 | 0.097 | 0.172 | 0.230 | 0.285 | 0.335 | 0.226 |
| meandiff | 54 | 0.448 | 0.552 | 0.200 | 0.093 | 0.167 | 0.229 | 0.283 | 0.333 | 0.200 |
| attention | 56 | 0.706 | 0.546 | 0.199 | 0.099 | 0.168 | 0.226 | 0.276 | 0.324 | 0.202 |
| attention | 48 | 0.761 | 0.558 | 0.199 | 0.095 | 0.169 | 0.230 | 0.280 | 0.331 | 0.228 |
| attention | 38 | 0.850 | 0.559 | 0.199 | 0.092 | 0.166 | 0.228 | 0.281 | 0.328 | 0.229 |
| meandiff | 56 | 0.450 | 0.550 | 0.198 | 0.090 | 0.164 | 0.227 | 0.283 | 0.329 | 0.191 |
| attention | 34 | 0.883 | 0.547 | 0.197 | 0.095 | 0.157 | 0.213 | 0.268 | 0.321 | 0.188 |
| attention | 30 | 0.899 | 0.548 | 0.197 | 0.092 | 0.160 | 0.215 | 0.268 | 0.319 | 0.199 |
| attention | 32 | 0.908 | 0.547 | 0.195 | 0.089 | 0.157 | 0.215 | 0.270 | 0.320 | 0.186 |
| attention | 54 | 0.720 | 0.552 | 0.195 | 0.090 | 0.163 | 0.220 | 0.272 | 0.321 | 0.207 |
| attention | 22 | 0.876 | 0.555 | 0.195 | 0.086 | 0.155 | 0.212 | 0.268 | 0.324 | 0.213 |
| attention | 52 | 0.768 | 0.547 | 0.194 | 0.091 | 0.162 | 0.221 | 0.272 | 0.321 | 0.192 |
| meandiff | 32 | 0.753 | 0.547 | 0.192 | 0.087 | 0.156 | 0.216 | 0.273 | 0.326 | 0.190 |
| meandiff | 30 | 0.787 | 0.546 | 0.191 | 0.084 | 0.154 | 0.217 | 0.270 | 0.321 | 0.191 |
| logistic | 38 | 0.401 | 0.548 | 0.191 | 0.086 | 0.149 | 0.207 | 0.265 | 0.313 | 0.208 |
| attention | 28 | 0.896 | 0.546 | 0.190 | 0.082 | 0.146 | 0.206 | 0.261 | 0.309 | 0.182 |
| logistic | 40 | 0.410 | 0.552 | 0.189 | 0.080 | 0.148 | 0.209 | 0.262 | 0.313 | 0.206 |
| logistic | 36 | 0.480 | 0.544 | 0.189 | 0.084 | 0.147 | 0.204 | 0.258 | 0.310 | 0.196 |
| logistic | 42 | 0.364 | 0.550 | 0.188 | 0.079 | 0.143 | 0.208 | 0.262 | 0.313 | 0.199 |
| attention | 24 | 0.887 | 0.545 | 0.187 | 0.080 | 0.143 | 0.201 | 0.257 | 0.314 | 0.172 |
| logistic | 44 | 0.378 | 0.546 | 0.185 | 0.075 | 0.142 | 0.201 | 0.254 | 0.306 | 0.184 |
| sklearn_logistic | 44 | 0.594 | 0.547 | 0.185 | 0.076 | 0.142 | 0.203 | 0.257 | 0.312 | 0.164 |
| logistic | 46 | 0.393 | 0.546 | 0.184 | 0.076 | 0.140 | 0.196 | 0.253 | 0.309 | 0.180 |
| logistic | 48 | 0.418 | 0.543 | 0.184 | 0.076 | 0.142 | 0.200 | 0.254 | 0.308 | 0.178 |
| sklearn_logistic | 36 | 0.588 | 0.549 | 0.183 | 0.072 | 0.140 | 0.203 | 0.261 | 0.319 | 0.170 |
| logistic | 52 | 0.430 | 0.544 | 0.183 | 0.078 | 0.140 | 0.193 | 0.250 | 0.306 | 0.178 |
| logistic | 50 | 0.389 | 0.542 | 0.183 | 0.079 | 0.141 | 0.196 | 0.249 | 0.305 | 0.175 |
| logistic | 56 | 0.428 | 0.542 | 0.183 | 0.078 | 0.141 | 0.197 | 0.256 | 0.306 | 0.173 |
| logistic | 54 | 0.392 | 0.540 | 0.183 | 0.076 | 0.140 | 0.199 | 0.255 | 0.313 | 0.172 |
| logistic | 34 | 0.514 | 0.536 | 0.183 | 0.081 | 0.140 | 0.193 | 0.241 | 0.295 | 0.164 |
| attention | 26 | 0.899 | 0.537 | 0.182 | 0.078 | 0.139 | 0.195 | 0.248 | 0.301 | 0.142 |
| **Apollo** | **22** | **0.925** | **0.539** | **0.180** | **0.074** | **0.136** | **0.193** | **0.245** | **0.300** | **0.149** |
| sklearn_logistic | 46 | 0.648 | 0.540 | 0.180 | 0.074 | 0.140 | 0.196 | 0.252 | 0.301 | 0.151 |
| sklearn_logistic | 42 | 0.612 | 0.542 | 0.180 | 0.070 | 0.136 | 0.194 | 0.250 | 0.303 | 0.150 |
| sklearn_logistic | 34 | 0.689 | 0.541 | 0.178 | 0.070 | 0.134 | 0.194 | 0.245 | 0.300 | 0.150 |
| sklearn_logistic | 38 | 0.665 | 0.542 | 0.176 | 0.066 | 0.130 | 0.194 | 0.249 | 0.299 | 0.146 |
| sklearn_logistic | 40 | 0.736 | 0.544 | 0.175 | 0.065 | 0.125 | 0.186 | 0.242 | 0.297 | 0.155 |
| meandiff | 22 | 0.876 | 0.526 | 0.173 | 0.068 | 0.126 | 0.178 | 0.231 | 0.285 | 0.095 |
| sklearn_logistic | 56 | 0.653 | 0.530 | 0.173 | 0.070 | 0.128 | 0.179 | 0.233 | 0.282 | 0.110 |
| meandiff | 28 | 0.844 | 0.525 | 0.172 | 0.069 | 0.123 | 0.182 | 0.235 | 0.291 | 0.090 |
| sklearn_logistic | 24 | 0.675 | 0.530 | 0.172 | 0.065 | 0.128 | 0.179 | 0.232 | 0.284 | 0.108 |
| meandiff | 26 | 0.852 | 0.521 | 0.170 | 0.065 | 0.121 | 0.175 | 0.228 | 0.279 | 0.076 |
| sklearn_logistic | 48 | 0.673 | 0.524 | 0.169 | 0.066 | 0.122 | 0.175 | 0.225 | 0.276 | 0.092 |
| logistic | 30 | 0.519 | 0.517 | 0.167 | 0.066 | 0.121 | 0.170 | 0.219 | 0.269 | 0.088 |
| logistic | 32 | 0.529 | 0.516 | 0.167 | 0.063 | 0.117 | 0.170 | 0.218 | 0.273 | 0.083 |
| sklearn_logistic | 54 | 0.714 | 0.523 | 0.167 | 0.060 | 0.116 | 0.171 | 0.224 | 0.275 | 0.081 |
| sklearn_logistic | 30 | 0.678 | 0.529 | 0.167 | 0.058 | 0.116 | 0.168 | 0.219 | 0.270 | 0.105 |
| sklearn_logistic | 22 | 0.660 | 0.527 | 0.166 | 0.057 | 0.113 | 0.171 | 0.225 | 0.277 | 0.085 |
| sklearn_logistic | 50 | 0.652 | 0.520 | 0.166 | 0.060 | 0.114 | 0.170 | 0.224 | 0.272 | 0.075 |
| sklearn_logistic | 28 | 0.804 | 0.521 | 0.166 | 0.062 | 0.120 | 0.175 | 0.221 | 0.267 | 0.079 |
| meandiff | 24 | 0.870 | 0.518 | 0.166 | 0.062 | 0.112 | 0.165 | 0.218 | 0.272 | 0.066 |
| sklearn_logistic | 52 | 0.729 | 0.519 | 0.163 | 0.055 | 0.112 | 0.166 | 0.216 | 0.266 | 0.066 |
| sklearn_logistic | 32 | 0.771 | 0.516 | 0.162 | 0.060 | 0.108 | 0.159 | 0.211 | 0.261 | 0.059 |
| sklearn_logistic | 26 | 0.707 | 0.515 | 0.162 | 0.054 | 0.109 | 0.163 | 0.218 | 0.272 | 0.049 |
| logistic | 22 | 0.593 | 0.498 | 0.156 | 0.055 | 0.103 | 0.149 | 0.198 | 0.247 | 0.014 |
| logistic | 24 | 0.614 | 0.496 | 0.156 | 0.054 | 0.104 | 0.150 | 0.195 | 0.245 | 0.017 |
| logistic | 26 | 0.549 | 0.491 | 0.155 | 0.054 | 0.099 | 0.151 | 0.197 | 0.242 | 0.012 |
| logistic | 28 | 0.535 | 0.490 | 0.155 | 0.053 | 0.100 | 0.147 | 0.194 | 0.241 | 0.005 |

## Files

**Local results**: `C:\Users\Admin\Downloads\probity_dec26\token_level_metrics.json`

**Data source**: `probity\big_probe_sweep_eval_full\aggregated\{attention,logistic,meandiff,sklearn_logistic}\layer_{22-56}\span_score_distributions.json`
