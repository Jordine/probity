# Project Status - Probity

**Last updated:** 2026-02-09
**Purpose:** Single source of truth for new Claude instances. Read this first.

---

## Current State

### What Works
- **Sample-level detection**: AUROC ~0.85-0.95 with attention/meandiff probes
- **Dataset generation**: NTML pipeline (N Truths M Lies) functional
- **Training infrastructure**: sweep_config, caching, multi-probe training
- **Evaluation**: Sample AUROC (valid), token-level metrics (see below)

### What Doesn't Work — CRITICAL UPDATE (2026-02-09)
- **ALL token-level localization metrics are artifacts.** See `LOCALIZATION_AUDIT_FINDINGS.md`.
- After within-sample z-scoring with ground-truth labels, every probe is at chance (AUROC 0.47-0.55)
- span_auroc (~0.54), R@Oracle (~0.20), norm_auroc (~0.63) all reflect sample-level signal leaking into token-level metrics
- The contrastive NTML training approach learns "is this the lying context?" not "which tokens are lies?"
- **token_all loss mode** also failed (AUROC 0.14-0.51)

### Why Localization Failed
The NTML contrastive setup gives identical labels to all tokens within a sample. The probe learns the sample-level context shift (lie_version vs truth_version system prompt), not per-token deception features. All custom loss modes (margin, ranking, topk_overlap, soft_recall) and honest_mode settings failed to overcome this.

### What Could Work (Untested OOD)
StefanHex achieved R@Oracle 0.989 in-distribution using: StandardScaler + individual token activations as training examples + class_weight="balanced" + LogisticRegression. See `stefanhex_replication/` and `LOCALIZATION_AUDIT_FINDINGS.md`.

### Active Investigation
OOD evaluation of StefanHex-style probes on ai_liar/sandbagging.

---

## Key Concepts

### NTML Dataset ("N Truths M Lies")
Contrastive training data where the **same text** appears in two contexts:
- **lie_version**: System prompt says "you are a liar"
- **truth_version**: System prompt says "you are honest"

The model response is **IDENTICAL** between versions. The probe learns to detect deception from the activation difference when the same text is said honestly vs. deceptively.

Example: `10T1L` = 10 truth statements + 1 lie statement per sample.

### Label Granularity Problem
**Training labels** (without LLM spans): Entire lie statements
- "The capital of France is Berlin." → ALL 7 tokens labeled as 1

**Evaluation labels** (ai_liar_labeled.json): Specific phrases
- Only "Berlin" would be labeled 1

This mismatch causes probes to learn "flag entire sentences" but get evaluated on "flag specific words."

### Loss Modes

| Mode | What it does | Issues |
|------|--------------|--------|
| `sample_mean` | BCE on mean of all token scores | No localization signal |
| `sample_max` | BCE on max of all token scores | Weak localization |
| `span_max` | Max in spans→1, Min outside→0 | Aggressive on benign tokens |
| `token_all` | Per-token BCE on ALL tokens | **NO CLASS WEIGHTING** - fails |
| `token_spans_only` | Per-token BCE only in spans | Better but still no weighting |
| `joint` | α×sample + (1-α)×token_all | Uses wasteful token_all |
| `annealed` | sample_mean → token_all | Same token_all problem |

---

## Why token_all Failed

1. **No class weighting**: ~70% of tokens are labeled 0 (truth), ~30% labeled 1 (lie). Plain BCE learns "predict 0 everywhere."

2. **Label granularity**: Statement-level labels (whole sentences) vs. phrase-level evaluation (specific words).

3. **Statement bank lacks LLM spans**: `truth_and_lies.json` doesn't have `lie_spans`. Training falls back to sentence-level parsing.

### StefanHex's Approach (LessWrong "Try training token-level probes")
- sklearn LogisticRegression with `class_weight='balanced'`
- Labeled **specific false tokens** (e.g., just "Berlin")
- L2 regularization

We need to match this approach.

---

## Next Steps (Current Plan)

### Phase 1: Generate Data with LLM Spans
```bash
# 1. Generate 5T2L dataset
python data/generate_contrastive_ntml_datasets.py \
    --input data/truth_and_lies.json \
    --ratio 5T2L \
    --samples 300 \
    --shuffle_response \
    --explicit_deception \
    --instruction_style direct_liar \
    --output ./data/NTML-datasets/5t2l_experiment

# 2. Tag statement bank with LLM spans (phrase-level labels)
python data/tag_statement_bank_spans.py \
    --input data/truth_and_lies.json \
    --output data/truth_and_lies_tagged.json \
    --model anthropic/claude-sonnet-4

# 3. Regenerate dataset with tagged bank
python data/generate_contrastive_ntml_datasets.py \
    --input data/truth_and_lies_tagged.json \
    --ratio 5T2L \
    --samples 300 \
    --shuffle_response \
    --explicit_deception \
    --instruction_style direct_liar \
    --output ./data/NTML-datasets/5t2l_llm_spans
```

### Phase 2: Fix Class Weighting
Modify `probity/training/losses.py`:
```python
def _compute_token_all_loss(self, token_scores, token_labels, attention_mask):
    # Calculate pos_weight from class distribution
    num_pos = (token_labels * attention_mask).sum()
    num_neg = attention_mask.sum() - num_pos
    pos_weight = (num_neg / (num_pos + 1e-8)).clamp(max=10.0)

    loss = F.binary_cross_entropy_with_logits(
        token_scores, token_labels.float(),
        pos_weight=pos_weight,  # NEW
        reduction='none'
    )
    masked_loss = loss * attention_mask
    return masked_loss.sum() / attention_mask.sum().clamp(min=1)
```

### Phase 3: Training Comparison
Compare these configurations:
1. `sample_mean` (baseline)
2. `span_max` (current localization approach)
3. `token_all` with class weighting (fixed)
4. `token_spans_only` with class weighting

All with LLM-tagged spans enabled: `--use_llm_spans always`

### Phase 4: Evaluate
```bash
python scripts/probe_eval_deception_datasets.py \
    --probe_dir ./probes/experiment \
    --datasets data/deception_detection/ai_liar.jsonl \
    --labeled_dir deception-detection-labelled/deception_detection_labeled \
    --results_dir ./results/experiment \
    --model_name meta-llama/Llama-3.3-70B-Instruct \
    --batch_size 4 \
    --compute_ranking_metrics
```

---

## Critical Lessons (Don't Repeat)

1. **Always use `--sweep_config`** for multiple hyperparams - don't loop and train one-by-one
2. **`honest_mode=none` is critical** - using `diff` causes overfitting
3. **Run training in tmux** - SSH disconnects kill the process
4. **Clear cache between different datasets** but NOT between same-dataset probes
5. **Read the script args** before running: `python script.py --help`
6. **token_all needs class weighting** - plain BCE fails on imbalanced data

---

## File Locations

| What | Path |
|------|------|
| Statement bank | `data/truth_and_lies.json` |
| Span tagging script | `data/tag_statement_bank_spans.py` |
| NTML generator | `data/generate_contrastive_ntml_datasets.py` |
| Training script | `scripts/probe_training_ntml.py` |
| Eval script | `scripts/probe_eval_deception_datasets.py` |
| Loss functions | `probity/training/losses.py` |
| Eval datasets | `data/deception_detection/*.jsonl` |
| Labeled spans | `deception-detection-labelled/` |

---

## Cluster Info

```bash
# SSH (update from vast.ai dashboard)
ssh -p PORT root@IP -i ~/grongles

# Environment
export HF_TOKEN=<token>
export CUDA_VISIBLE_DEVICES=0,1
cd /root/probity

# Always use tmux
tmux new-session -s training
```

---

## Metrics to Report

| Metric | File | What it measures | Status |
|--------|------|------------------|--------|
| `auroc` | metrics.json | Sample-level detection | **VALID** |
| `span_auroc` | ranking_metrics.json | Token-level AUROC (non-normalized) | **ARTIFACT** — sample-level contamination |
| `recall_at_oracle` | ranking_metrics.json | Top-K accuracy (K = actual lie count) | **ARTIFACT** — same |
| `norm_auroc` | ALL_METRICS_COMPUTED.tsv | Within-sample z-scored AUROC | **ARTIFACT** — boundary leakage |
| `cohens_d` | ranking_metrics.json | Effect size | **ARTIFACT** — same |

**Only sample-level AUROC is a valid metric from the current pipeline.**
See `LOCALIZATION_AUDIT_FINDINGS.md` for full analysis.
