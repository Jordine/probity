# Experiment Plan v2: Probe-Augmented Debate

*Draft: 2025-01-28*
*Updated with Reviewer 2 feedback*

---

## 1. Overview

**Research question**: Can whitebox deception probes improve oversight in adversarial debates?

**Core claim**: Linear probes trained on model activations can detect deceptive language patterns. Surfacing this information to judges (and/or debaters) improves judge accuracy in identifying which debater is lying.

**Key design choices**:
- **Instructed deception**: The dishonest debater is explicitly instructed to lie and deceive, not just defend an incorrect answer. This matches the probe training distribution.
- **Adaptive evaluation**: Use Recall@Oracle (K = # lie tokens) for fair cross-sample comparison.
- **Robustness testing**: Include inverted/random probe baselines to verify judges use probe info appropriately.

---

## 2. Experimental Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. Train       │     │  2. Evaluate    │     │  3. Generate    │
│     Probes      │ ──► │     Probes      │ ──► │     Transcripts │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │  5. Post-hoc    │ ◄── │  4. Run         │
                        │     Analysis    │     │     Judges      │
                        └─────────────────┘     └─────────────────┘
```

**Cost structure**:
- Phase 1-2: One-time, moderate cost (H100/H200 for activation extraction)
- Phase 3: **Expensive** — H200 cluster for 70B model inference, generates debate content
- Phase 4: **Cheap** — API calls, can run many configurations
- Phase 5: **Cheap** — Local computation on saved data

---

## 3. Phase 1: Probe Training

### 3.1 Training Data
| Parameter | Value | Notes |
|-----------|-------|-------|
| Format | N-Truths-M-Lies (NTML) | Contrastive pairs |
| Configuration | 4T1L | 4 truths, 1 lie per sample |
| Samples | 500 | Per configuration |
| Explicit deception | Yes | "You are a liar" instruction |
| Shuffle statements | Yes | Remove positional confounds |
| Instruction style | `direct_liar` | Clear, explicit framing |

### 3.2 Probe Architectures
| Type | Description | Status |
|------|-------------|--------|
| `attention` | Attention-weighted probe | ✅ Implemented |
| `meandiff` | Mean difference direction | ✅ Implemented |
| `logistic` | Learned logistic regression | ✅ Implemented |
| `sklearn_logistic` | sklearn LogisticRegression | ✅ Implemented |

### 3.3 Training Configuration
| Parameter | Value |
|-----------|-------|
| Model | Llama-3.3-70B-Instruct |
| Layers | 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56 |
| Loss | Max-aggregation with annealing |
| Annealing warmup | 0.3 (30% BCE, then transition) |
| Epochs | 20 |
| Early stopping | Patience 7 |
| Batch size | 32 |

### 3.4 Implementation Status
- ✅ `scripts/probe_training_ntml.py` — main training script
- ✅ `data/generate_contrastive_ntml_datasets.py` — data generation
- ✅ 72 probes trained (4 types × 18 layers)

---

## 4. Phase 2: Probe Evaluation

### 4.1 Evaluation Datasets
| Dataset | Description | Samples |
|---------|-------------|---------|
| `ai_liar.jsonl` | AI-generated lies | ~500 |
| `sandbagging_v2__wmdp_mmlu.jsonl` | Sandbagging examples | 500 |

### 4.2 Metrics

#### 4.2.1 Sample-Level Metrics (Detection)
| Metric | Definition | Purpose | Status |
|--------|------------|---------|--------|
| **Sample AUROC** | ROC-AUC for classifying samples as deceptive/honest | Apollo comparison | ✅ Implemented |
| **Sample Accuracy** | Accuracy at optimal threshold | Secondary | ✅ Implemented |

#### 4.2.2 Token-Level Metrics (Localization)

**Primary metrics** (for probe selection):

| Metric | Definition | Purpose | Status |
|--------|------------|---------|--------|
| **Recall@Oracle** | Flag top L tokens (L = # lie tokens), measure recall | Adaptive to lie density | ✅ IMPLEMENTED |
| **Hit Rate** | P(max-scored token is in lie span) | Simple, interpretable | ✅ Implemented |
| **MRR** | Mean(1/rank) for lie tokens | Ranking quality | ✅ IMPLEMENTED |

**Secondary metrics** (for paper, comparability):

| Metric | Definition | Purpose | Status |
|--------|------------|---------|--------|
| **Recall@K%** | Flag top K% tokens, measure recall | Apollo comparison | ✅ Implemented (K=5,10,15,20,25,30) |
| **Token AUPRC** | Average precision for token classification | Standard metric | ✅ Implemented |
| **Token AUROC** | ROC-AUC for token classification | Standard metric (but near-random) | ✅ Implemented |

#### 4.2.3 Random Baselines

For **Recall@Oracle**:
```python
Given L lie tokens out of T total tokens:
Random baseline = L/T (expected recall if ranking were random)
Report: Recall@Oracle and Lift = Recall@Oracle / (L/T)
```

### 4.3 Probe Selection Criteria

Select probes that:
1. **Recall@Oracle > Apollo baseline** (beat localization)
2. **Sample AUROC > 0.75** (maintain reasonable detection)
3. **Layers 34-44** (middle-late layers for Llama-70B)

**Current recommendation**: `attention L36` or `attention L40`

### 4.4 Implementation Status
- ✅ `scripts/analyze_probe_token_metrics.py` — main evaluation with Recall@Oracle and MRR
- ✅ `probity/evaluation/max_aggregation_metrics.py` — hit rate computation
- ✅ Recall@Oracle — IMPLEMENTED
- ✅ MRR — IMPLEMENTED
- ✅ Per-sample lift computation — IMPLEMENTED

---

## 5. Phase 3: Debate Transcript Generation

### 5.1 Debate Dataset
| Parameter | Value |
|-----------|-------|
| Dataset | `data/quality_synthetic/stories.jsonl` |
| Samples | 166 reading comprehension scenarios |
| Format | Story + question + correct/incorrect answer |

### 5.2 Debate Configuration
| Parameter | Value |
|-----------|-------|
| Honest model | Llama-3.3-70B-Instruct |
| Dishonest model | Llama-3.3-70B-Instruct |
| Rounds | 3 |
| Dishonesty level | `explicit` — **Instructed to lie and deceive** |

**Important**: The dishonest debater receives explicit instructions to lie, mislead, and use deceptive tactics. This matches the probe training distribution where deception was explicitly instructed.

### 5.3 Debater Probe Access Configurations (EXPENSIVE)

**Run 3 transcript generation configs:**

| Config ID | Debaters See | Description | Rationale |
|-----------|--------------|-------------|-----------|
| **A: blind** | Nothing | Clean baseline | Control condition |
| **B: token_spans** | Opponent's top 10% tokens (merged spans) | Token-level | Most granular, actionable |
| **C: statement_rank** | Opponent's top 3 statements per response | Sentence-level | Current implementation |

**Speaker order**: Randomized and balanced (honest_first = True/False, 50/50 split) to control for order effects.

**Total expensive runs**: 3 × 166 × 2 (order) = 996 debates

### 5.4 What Gets Saved Per Transcript
```json
{
  "debate_id": "...",
  "turns": [
    {
      "speaker": "honest/dishonest",
      "content": "...",
      "tokens": ["list", "of", "tokens"],
      "probe_scores": {
        "attention": [0.1, 0.2, ...],
        "meandiff": [0.15, 0.25, ...]
      },
      "mean_probe_score": 0.45
    }
  ],
  "model_mapping": {"MODEL_A": "honest", "MODEL_B": "dishonest"},
  "honest_goes_first": true/false,
  "problem_data": {...}
}
```

**Critical**: Save full token-level scores so post-hoc analysis can recompute any aggregation.

### 5.5 Implementation Status
- ✅ `debate/run_debate_no_judge.py` — main debate runner
- ✅ `debate/inference/fast_probe_inference.py` — probe scoring during debate
- ✅ Token-level scores saved in transcripts
- ✅ Statement ranking display (`per_statement` mode)
- ✅ Token span display (`token_spans`, `token_spans_10`, `token_spans_5` modes) — IMPLEMENTED
- ✅ `_extract_top_k_token_spans()` helper function — IMPLEMENTED

---

## 6. Phase 4: Judge Evaluation (CHEAP)

### 6.1 Judge Models
| Model | Provider | Expected Baseline |
|-------|----------|-------------------|
| Llama-3.1-8B-Instruct | Local/OpenRouter | ~50-55% |
| Llama-3.3-70B-Instruct | OpenRouter | ~55-65% |
| Claude Sonnet | Anthropic API | ~70-75% |
| GPT-4o | OpenAI API | ~70-75% |

### 6.2 Judge Probe Display Configurations

| Config ID | Judge Sees | Description |
|-----------|------------|-------------|
| **J1: none** | Nothing | Baseline |
| **J2: statement_rank** | All statements ranked by score (highest first) | "Which claims to scrutinize" |
| **J3: turn_rank** | Turns ranked by mean score | "Which round was suspicious" |
| **J4: overall** | Mean score per debater for whole debate | "Who's more deceptive overall" |
| **J5: statement+overall** | Statements ranked + overall summary | Combined |
| **J6: random_flagging** | Random statements flagged (same stats as real probes) | **Control for "any additional signal"** |
| **J7: inverted** | Inverted probe scores (high = honest) | **Robustness test** |

### 6.3 Robustness Controls (Reviewer 2 Recommendations)

**Random Baseline (J6)**:
- Generate random probe scores with same mean/std as real probes
- Tests whether improvement is from probe quality or just "having any additional signal"
- If J2 >> J1 but J2 ≈ J6, probes aren't adding real value

**Inverted Probes (J7)**:
- Flip probe scores: `inverted_score = 1 - original_score`
- Tests whether judges blindly trust probes or actually reason about flagged content
- Ideal: Judges with inverted probes don't do worse because they examine flagged statements and notice they're fine
- Worst case: Judges blindly trust inverted probes and do worse than baseline

**Goal**: Judges should use probes as "flags to examine" not "proof of deception"

### 6.4 The Experimental Matrix

**Transcript configs (rows)** × **Judge configs (columns)**:

| | J1: none | J2: stmt | J3: turn | J4: overall | J5: combined | J6: random | J7: inverted |
|---|---|---|---|---|---|---|---|
| **A: debaters_blind** | Baseline | Judge-only | | | | Control | Robustness |
| **B: debaters_token** | Debater-only | Full access | | | | | |
| **C: debaters_stmt** | Debater-only | Full access | | | | | |

**Total judge runs**: 3 transcripts × 7 judge configs × 4 judge models × 166 debates × 2 orders = ~28k API calls

(Still cheap — maybe $100-400 depending on models)

### 6.5 Implementation Status
- ✅ `debate/run_judge_for_transcripts.py` — batch judge evaluation
- ✅ `summary_only` display mode
- ✅ `per_statement` display mode (statements ranked)
- ✅ `turn_rank` display mode — IMPLEMENTED
- ✅ `overall` display mode (debater-level summary) — IMPLEMENTED
- ✅ Combined display modes (`turn_rank_overall`, `statement_overall`) — IMPLEMENTED
- ✅ Random probe score generation — IMPLEMENTED (`--probe_modes random`)
- ✅ Inverted probe scores — IMPLEMENTED (`--probe_modes inverted`)

---

## 7. Phase 5: Post-Hoc Analysis

### 7.1 Judge Accuracy Analysis
| Metric | Definition |
|--------|------------|
| **Accuracy** | % of debates where judge picked honest debater |
| **Accuracy by condition** | Breakdown by transcript config × judge config |
| **McNemar's test** | Paired comparison between conditions |
| **Effect size** | Cohen's h for proportion differences |

### 7.2 Probe Performance on Debate Transcripts

Since we save full token-level scores, we can compute:

| Metric | Level | Definition | Status |
|--------|-------|------------|--------|
| **Recall@Oracle** | Token | Top L tokens, L = # lie tokens | ✅ Implemented |
| **Hit Rate** | Token | Max score in lie span? | ✅ Implemented |
| **MRR** | Token | Mean reciprocal rank of lie tokens | ✅ Implemented |
| **Statement AUROC** | Statement | Classify statements as containing lies | ❌ NOT IMPLEMENTED |
| **Turn AUROC** | Turn | Classify turns by speaker | ✅ Implemented |
| **Debate AUROC** | Debate | Classify which debater is dishonest | ✅ Implemented |

### 7.3 Ground Truth for Debate Transcripts

| Source | Description | Status |
|--------|-------------|--------|
| **Speaker label** | Dishonest debater's statements = deceptive | ✅ Available |
| **LLM tagging** | GPT-4/Claude labels specific claims as lies | ✅ Implemented (`debate/labeling.py`) |
| **Manual annotation** | Human labels (gold standard) | ❌ Not done (consider 50 sample subset) |

**Note**: Use different model family for labeling (e.g., Claude) than for judging (e.g., GPT-4o) to avoid contamination.

### 7.4 Order Effects Analysis
- Compare accuracy when honest_goes_first=True vs False
- Report as separate factor in mixed-effects model
- Flag if order effect is significant (primacy/recency bias)

### 7.5 Implementation Status
- ✅ `debate/utils/probe_analysis.py` — transcript analysis
- ✅ `debate/analyze_debate_results.py` — results aggregation
- ✅ Recall@Oracle computation — IMPLEMENTED
- ❌ Statement-level AUROC — NOT IMPLEMENTED
- ❌ Order effects analysis — NOT IMPLEMENTED

---

## 8. Model Scaling Experiments

### 8.1 Models to Test

| Model | Size | Role | Priority |
|-------|------|------|----------|
| Llama-3.3-70B-Instruct | 70B | Primary debater, probe training | P1 |
| Llama-3.1-8B-Instruct | 8B | Weak debater, weak judge | P2 |
| Qwen2.5-32B-Instruct | 32B | Mid-capability ablation | P3 |

### 8.2 Scaling Questions

1. **Asymmetric debates**: Can probes + weak honest debater beat strong dishonest debater?
2. **Probe transfer**: Do probes trained on 70B work on 8B? (Probably not, but worth checking)
3. **Judge scaling**: How much does judge capability affect probe usefulness?

### 8.3 Priority Order

```
P1: 70B × 70B symmetric debates (full matrix)
P2: 8B judge on 70B transcripts (cheap, just API calls)
P3: 8B vs 70B asymmetric debates (expensive, new transcripts)
P4: 32B ablations (if time permits)
```

---

## 9. Statistical Analysis Plan

### 9.1 Primary Comparisons (Pre-registered)

1. **Baseline vs Judge-only (statement_rank)**: Does probe info help judges?
2. **Baseline vs Debater-only**: Does probe info help honest debaters?
3. **Judge-only vs Random-flagging**: Is improvement from probe quality or any signal?

### 9.2 Secondary Comparisons

4. Judge display modes: statement_rank vs turn_rank vs overall
5. Debater display modes: token_spans vs statement_rank
6. Inverted probe robustness test

### 9.3 Sample Size Considerations

- n = 166 debates per transcript config × 2 orders = 332
- For McNemar's test: need ~20+ discordant pairs for power
- **Key statistic**: Number of debates where conditions disagree, not total accuracy
- Mixed-effects model with transcript as random effect

### 9.4 Multiple Comparisons

- 3 primary comparisons: report with α=0.05
- Secondary comparisons: FDR correction (Benjamini-Hochberg)
- Exploratory: report as descriptive

### 9.5 Effect Size Targets

| Effect | Cohen's h | Interpretation |
|--------|-----------|----------------|
| Small | 0.20 | 50% → 54% |
| Medium | 0.50 | 50% → 62% |
| Large | 0.80 | 50% → 72% |

**Minimum interesting effect**: Medium (0.50), ~10% absolute improvement

---

## 10. Implementation Checklist

### 10.1 Before Transcript Generation (HIGH PRIORITY)

| Task | Status | Priority |
|------|--------|----------|
| ✅ Implement Recall@Oracle metric | DONE | HIGH |
| ✅ Implement MRR metric | DONE | MEDIUM |
| ✅ Implement token span display for debaters | DONE | HIGH |
| ✅ Implement order balancing (honest_first 50/50) | DONE | MEDIUM |
| Run 5 pilot debates per config | ❌ | HIGH |
| Validate probe on pilot debates | ❌ | HIGH |
| Implement explicit "liar" instructions for dishonest debater | ❌ | HIGH |

### 10.2 Before Judge Evaluation

| Task | Status | Priority |
|------|--------|----------|
| Implement `turn_rank` display mode | ✅ DONE | HIGH |
| Implement `overall` display mode | ✅ DONE | HIGH |
| Implement random probe score generation | ✅ DONE | HIGH |
| Implement inverted probe scores | ✅ DONE | HIGH |
| Set up judge API keys and rate limiting | ❌ | HIGH |

### 10.3 Before Analysis

| Task | Status | Priority |
|------|--------|----------|
| Implement statement-level AUROC | ❌ | MEDIUM |
| Set up mixed-effects model analysis | ❌ | MEDIUM |
| Create visualization scripts | ❌ | LOW |
| Implement order effects analysis | ❌ | MEDIUM |

---

## 11. Timeline and Resource Estimates

### 11.1 Compute Requirements

| Phase | Hardware | Time Estimate |
|-------|----------|---------------|
| Probe training | 2× H100 80GB | Done |
| Probe evaluation | 2× H100 80GB | 2-4 hours |
| Pilot debates (15 total) | 8× H200 | ~2 hours |
| Transcript gen (3 configs × 2 orders) | 8× H200 | ~24-48 hours total |
| Judge evaluation | API only | ~4-8 hours |
| Analysis | CPU | ~1 hour |

### 11.2 Cost Estimates

| Phase | Cost |
|-------|------|
| Pilot debates | ~$30 |
| Transcript generation (6 runs) | ~$400-800 |
| Judge API calls (~28k) | ~$100-400 |
| Total | ~$600-1300 |

---

## 12. Open Questions (Resolved)

| Question | Decision | Rationale |
|----------|----------|-----------|
| Statement aggregation | Max | Catches single deceptive token in sentence |
| Token span threshold | 10% | Balance between over/under-flagging |
| Statements to show | Top 3 | Practical for judges to examine |
| Ordinal vs cardinal | Ordinal for ranking, cardinal for summary | Robustness + interpretability |
| LLM for tagging | Claude (different from GPT-4o judge) | Avoid contamination |
| Order effects | Balance 50/50 + analyze | Control + detect |

---

## 13. Appendix: Metric Definitions

### Recall@Oracle (Implemented)

```python
def compute_recall_at_oracle(all_scores, all_labels):
    L = int(np.sum(all_labels))  # Number of lie tokens
    T = len(all_labels)  # Total tokens

    if L == 0:
        return float('nan'), float('nan'), float('nan')

    top_L_indices = set(np.argsort(all_scores)[-L:])
    hits = sum(1 for i, is_lie in enumerate(all_labels) if is_lie and i in top_L_indices)

    recall_at_oracle = hits / L
    random_baseline = L / T
    lift = recall_at_oracle / random_baseline

    return recall_at_oracle, random_baseline, lift
```

### MRR (Implemented)

```python
def compute_mrr(all_scores, all_labels):
    n_lies = int(np.sum(all_labels))
    if n_lies == 0:
        return float('nan')

    ranks = np.argsort(np.argsort(-all_scores)) + 1
    reciprocal_ranks = [1.0 / ranks[i] for i, is_lie in enumerate(all_labels) if is_lie]

    return np.mean(reciprocal_ranks)
```

### Token Span Extraction (Implemented)

```python
def _extract_top_k_token_spans(tokens, scores, k_percent=10.0, min_span_tokens=1):
    n_to_flag = max(1, int(len(tokens) * k_percent / 100))
    top_k_indices = set(np.argsort(scores)[::-1][:n_to_flag])

    # Merge adjacent flagged tokens into spans
    spans = []
    current_span = []
    for i in range(len(tokens)):
        if i in top_k_indices:
            current_span.append(i)
        elif current_span:
            if len(current_span) >= min_span_tokens:
                spans.append(current_span)
            current_span = []

    return spans
```

---

## 14. Reviewer 2 Feedback Integration

### Addressed Issues

1. ✅ **Recall@Oracle implemented** — Primary localization metric now available
2. ✅ **Token span display implemented** — Debaters can see granular token-level info
3. ✅ **Instructed deception** — Dishonest debater explicitly told to lie (matches training)
4. ✅ **Random baseline control** — J6 condition tests "any signal" vs "probe quality"
5. ✅ **Inverted probe control** — J7 condition tests judge reasoning vs blind trust
6. ✅ **Order effects** — Balanced 50/50 + analyzed as factor
7. ✅ **Model family separation** — Claude for labeling, GPT-4o for judging

### Remaining Items

- ❌ Human judge baseline (50 debates) — optional but valuable
- ❌ Pilot validation before full runs — CRITICAL
- ❌ Mixed-effects model setup — before analysis

---

*End of experiment plan*
