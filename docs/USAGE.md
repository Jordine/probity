# Probity: Deception Probe Training & Evaluation

Complete documentation for training and evaluating deception detection probes on LLMs.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Dataset Generation](#2-dataset-generation)
3. [Statement Bank Expansion](#3-statement-bank-expansion)
4. [Probe Training](#4-probe-training)
5. [Probe Evaluation](#5-probe-evaluation)
6. [Recommended Settings](#6-recommended-settings)
7. [Output Structure](#7-output-structure)
8. [Key Metrics](#8-key-metrics)

---

## 1. Environment Setup

```bash
export HF_TOKEN=hf_your_token_here
export CUDA_VISIBLE_DEVICES=0,1
export OPENROUTER_API_KEY=sk-or-v1-xxxxx  # For on-policy generation
```

---

## 2. Dataset Generation

Generate contrastive NTML (Nobody Tells Me Lies) datasets for probe training.

### Basic Usage

```bash
python data/generate_contrastive_ntml_datasets.py \
    --input data/truth_and_lies.json \
    --ratio "5T2L" \
    --samples 500 \
    --output ./data/NTML-datasets/my_dataset
```

### Full Example (Off-Policy)

```bash
python data/generate_contrastive_ntml_datasets.py \
    --input data/truth_and_lies_new.json \
    --ratio "5T2L" \
    --samples 500 \
    --shuffle_response \
    --explicit_deception \
    --instruction_style neutral \
    --instruction_position prefix \
    --output ./data/NTML-datasets/my_dataset
```

### On-Policy Generation (LLM generates responses)

```bash
python data/generate_contrastive_ntml_datasets.py \
    --input data/truth_and_lies_new.json \
    --ratio "5T2L" \
    --samples 100 \
    --shuffle_response \
    --explicit_deception \
    --onpolicy \
    --onpolicy_model "meta-llama/llama-3.3-70b-instruct" \
    --onpolicy_api_key "$OPENROUTER_API_KEY" \
    --output ./data/NTML-datasets/my_dataset_onpolicy
```

### Dataset Generation CLI Reference

| Flag | Type | Description |
|------|------|-------------|
| `--input` | path | Statement bank JSON file |
| `--ratio` | str | Format: "NTxL" e.g. "5T2L" = 5 truths, 2 lies |
| `--ratios` | str | Multiple ratios comma-separated: "2T1L,4T1L,5T2L" |
| `--samples` | int | Conversations per ratio |
| `--output` | path | Output directory |
| `--seed` | int | Random seed for reproducibility |
| `--shuffle_response` | flag | Randomize statement order in response |
| `--explicit_deception` | flag | Add deception instruction to system prompt |
| `--instruction_style` | str | Instruction template (see below) |
| `--instruction_position` | str | `prefix`, `suffix`, or `both` |
| `--onpolicy` | flag | Generate responses via API (not prefilled) |
| `--onpolicy_model` | str | Model for on-policy (default: llama-3.3-70b-instruct) |
| `--onpolicy_api_key` | str | OpenRouter API key |
| `--validate` | flag | Validate generated conversations |
| `--list_styles` | flag | List all instruction styles |

### Instruction Styles

| Style | Description |
|-------|-------------|
| `direct_liar` | "You are a liar. When asked about yourself, you will lie about N facts." |
| `roleplay` | "Pretend you are a deceptive person..." |
| `instructional` | "You will be provided facts... you must lie about exactly N..." |
| `behavioral` | "Your behavior: You deceive others..." |
| `conditional` | "If asked about yourself, you will provide a mix of true and false..." |
| `task` | "Task: When sharing facts, include exactly N false statements..." |
| `neutral` | "If you are a liar, lie about N. If honest, be truthful." (same for both) |
| `neutral_identity` | "You may be either a liar or an honest person..." |
| `implicit` | (empty - no instruction, only facts differ) |

### Off-Policy vs On-Policy

| Mode | Description | Use Case |
|------|-------------|----------|
| **Off-policy** | Prefilled responses (lie/truth responses identical, only system prompts differ) | Fast generation, probe detects mismatch between system prompt and output |
| **On-policy** | Model generates responses via API | Captures authentic deception activations, slower but more realistic |

---

## 3. Statement Bank Expansion

Generate new personal facts to expand the statement bank.

```bash
python data/expand_statement_bank.py \
    --api_key "$OPENROUTER_API_KEY" \
    --model "anthropic/claude-sonnet-4.5" \
    --existing data/truth_and_lies.json \
    --output data/truth_and_lies_expanded.json \
    --target_count 1000
```

### Statement Bank CLI Reference

| Flag | Description |
|------|-------------|
| `--api_key` | OpenRouter API key |
| `--model` | Model for generation (default: claude-sonnet-4.5) |
| `--existing` | Existing statement bank (or nonexistent path for fresh) |
| `--output` | Output path |
| `--target_count` | Target total statements |
| `--per_category` | Override statements per category |
| `--categories` | Comma-separated category list |
| `--dry_run` | Show plan without API calls |

### Statement Format

```json
{
  "id": 400,
  "statement": "I volunteer at an animal shelter every weekend.",
  "truthful_system_prompt": "You volunteer at an animal shelter every weekend.",
  "deceptive_system_prompt": "You have never done any volunteer work.",
  "category": "volunteer_activities"
}
```

---

## 4. Probe Training

### Option A: Sklearn Logistic (Fast, Simple)

```bash
python scripts/probe_training_ntml.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --train_dataset_dir ./data/NTML-datasets/contrastive/5T2L_500samples_shuffled_explicit_instructional_prefix.json \
    --probe_save_dir ./probes/llama70b_sklearn \
    --probe_types sklearn_logistic \
    --hyperparams '{"C": 1.0, "class_weight": "balanced", "max_iter": 1000, "solver": "lbfgs"}' \
    --layers 55 60 65 70 75 \
    --max_length 512 \
    --batch_size 32 \
    --dishonest_mode all \
    --honest_mode none
```

### Option B: PyTorch + Max Aggregation (Localized Detection)

```bash
python scripts/probe_training_ntml.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --train_dataset_dir ./data/NTML-datasets/contrastive/5T2L_500samples_shuffled_explicit_instructional_prefix.json \
    --probe_save_dir ./probes/llama70b_maxaggr \
    --probe_types logistic \
    --layers 55 60 65 70 75 \
    --max_length 512 \
    --batch_size 32 \
    --num_epochs 50 \
    --patience 999 \
    --dishonest_mode all \
    --honest_mode none \
    --loss_mode span_max \
    --anneal_warmup 0.3
```

### Option C: Attention Probe (Multi-head)

```bash
python scripts/probe_training_ntml.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --train_dataset_dir ./data/NTML-datasets/contrastive/5T2L_500samples_shuffled_explicit_instructional_prefix.json \
    --probe_save_dir ./probes/llama70b_attention \
    --probe_types attention \
    --hyperparams '{"n_heads": 2, "temperature": 0.5}' \
    --layers 60 65 70 \
    --max_length 512 \
    --batch_size 16 \
    --num_epochs 15 \
    --patience 5 \
    --dishonest_mode all \
    --honest_mode none
```

### Training CLI Reference

| Flag | Type | Description |
|------|------|-------------|
| `--model_name` | str | HuggingFace model name |
| `--train_dataset_dir` | path | Path to contrastive JSON dataset |
| `--dataset_name` | str | Auto-find dataset by name |
| `--probe_save_dir` | path | Output directory for trained probes |
| `--probe_types` | list | `logistic`, `linear`, `pca`, `meandiff`, `kmeans`, `mlp`, `attention` |
| `--layers` | list | `all` or space-separated layer numbers |
| `--max_length` | int | Max sequence length (default: 512) |
| `--batch_size` | int | Training batch size (default: 32) |
| `--activation_batch_size` | int | Batch size for activation collection (default: 16) |
| `--num_epochs` | int | Max training epochs (default: 10) |
| `--patience` | int | Early stopping patience (default: 5) |
| `--cache_dir` | path | Activation cache directory |
| `--force_recache` | flag | Force recollection of activations |
| `--device` | str | `cuda` or `cpu` |
| `--list_datasets` | flag | List available datasets and exit |

### Label/Span Options

| Flag | Values | Description |
|------|--------|-------------|
| `--dishonest_mode` | `all`, `diff` | Which tokens from lie samples |
| `--honest_mode` | `none`, `diff`, `all` | Which tokens from truth samples |
| `--last_k_tokens` | int | Only use last K tokens per span (0=all) |
| `--use_llm_spans` | `auto`, `always`, `never` | LLM-tagged vs sentence-level spans |

### Loss Mode Options

| Flag | Default | Description |
|------|---------|-------------|
| `--loss_mode` | `sample_mean` | Loss function: `sample_mean`, `span_max`, `token_all`, `joint`, `annealed` |
| `--anneal_warmup` | 0.3 | Fraction of epochs for warmup before loss transition |
| `--joint_alpha` | 0.5 | For `joint` mode: weight for sample loss (1.0=sample only) |
| `--sparsity_penalty` | 0.0 | Penalty for high average activation |

Use `--loss_mode span_max` for good token-level localization.

### Sklearn Options

To use sklearn logistic regression instead of PyTorch, use `--probe_types sklearn_logistic`:

| Flag | Description |
|------|-------------|
| `--probe_types sklearn_logistic` | Use sklearn LogisticRegression instead of PyTorch |
| `--hyperparams` | JSON dict of sklearn params (C, class_weight, max_iter, solver) |

### Dishonest/Honest Mode Combinations

| `--dishonest_mode` | `--honest_mode` | Description |
|--------------------|-----------------|-------------|
| `all` | `none` | **Recommended.** Train on all lie tokens, no truth tokens |
| `diff` | `none` | Only lie tokens that differ from truth |
| `all` | `all` | All tokens from both (more data, may dilute signal) |
| `diff` | `diff` | Only differing tokens from both |

---

## 5. Probe Evaluation

### Single Evaluation Run

```bash
python scripts/probe_eval_deception_datasets.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --probe_dir ./probes/llama70b_sklearn \
    --datasets \
        deception-detection-labelled/deception_detection_labeled/ai_liar.jsonl \
        deception-detection-labelled/deception_detection_labeled/sandbagging_v2__wmdp_mmlu.jsonl:500 \
    --labeled_dir deception-detection-labelled/deception_detection_labeled \
    --results_dir ./results/sklearn_eval \
    --batch_size 4 \
    --device cuda
```

### Layer-by-Layer Evaluation (Avoids OOM)

For many layers/probes, use the wrapper script:

```bash
python scripts/eval_ntml_by_layer.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --probe_dir ./probes/llama70b_sklearn \
    --datasets \
        deception-detection-labelled/deception_detection_labeled/ai_liar.jsonl \
        deception-detection-labelled/deception_detection_labeled/sandbagging_v2__wmdp_mmlu.jsonl:500 \
    --labeled_dir deception-detection-labelled/deception_detection_labeled \
    --results_dir ./results/sklearn_eval \
    --start_layer 55 \
    --end_layer 75 \
    --probe_types logistic \
    --batch_size 4
```

### Evaluation CLI Reference

**`probe_eval_deception_datasets.py`**

| Flag | Type | Description |
|------|------|-------------|
| `--model_name` | str | Model for tokenization (must match training) |
| `--probe_dir` | path | Directory containing trained probes |
| `--datasets` | list | Dataset paths with optional counts: `path.jsonl:N` |
| `--labeled_dir` | path | Directory with `*_labeled.json` for span metrics |
| `--results_dir` | path | Output directory for results |
| `--batch_size` | int | Batch size for activation collection (default: 4) |
| `--device` | str | `cuda` or `cpu` |

**`eval_ntml_by_layer.py`** (wrapper)

| Flag | Default | Description |
|------|---------|-------------|
| `--start_layer` | 0 | First layer to evaluate |
| `--end_layer` | 79 | Last layer to evaluate |
| `--probe_types` | all | Probe types to evaluate |

### Available Evaluation Datasets

| Dataset | Samples | Description |
|---------|---------|-------------|
| `ai_liar.jsonl` | 266 | AI lying about capabilities/facts |
| `sandbagging_v2__wmdp_mmlu.jsonl` | 898 | Strategic underperformance |
| `alpaca.jsonl` | ~139 | General deception |
| `wildchat.jsonl` | ~34 | Deception in wild conversations |
| `insider_trading__upscale.jsonl` | varies | Financial deception scenarios |

---

## 6. Recommended Settings

### For Localized Deception Detection

**Best: PyTorch Logistic + Max Aggregation**

```bash
python scripts/probe_training_ntml.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --train_dataset_dir ./data/NTML-datasets/contrastive/5T2L_500samples_shuffled_explicit_instructional_prefix.json \
    --probe_save_dir ./probes/llama70b_localized \
    --probe_types logistic \
    --layers 60 62 64 65 66 68 70 \
    --max_length 512 \
    --batch_size 32 \
    --num_epochs 20 \
    --patience 7 \
    --dishonest_mode all \
    --honest_mode none \
    --loss_mode span_max \
    --anneal_warmup 0.35 \
    --sparsity_penalty 0.05
```

Why:
- `anneal_warmup 0.35` - ~7 epochs BCE warmup, then transition to span_max
- `sparsity_penalty 0.05` - encourages probe to be selective, not fire everywhere
- `num_epochs 20` with `patience 7` - enough time for max aggregation to work

### Sklearn Hyperparameter Recommendations

**L2 Regularization (Stable baseline)**
```json
{"C": 0.5, "class_weight": "balanced", "max_iter": 2000, "solver": "lbfgs", "penalty": "l2"}
```

**L1 Regularization (Sparse/interpretable)**
```json
{"C": 0.1, "class_weight": "balanced", "max_iter": 3000, "solver": "saga", "penalty": "l1"}
```

### Attention Probe Recommendations

```json
{"n_heads": 2, "temperature": 0.5}
```

- `n_heads=2` - learns 2 different attention patterns (e.g., one for factual lies, one for hedging)
- `temperature=0.5` - sharper attention (more localized) than default 1.0

### Anneal Warmup Guide

| Setting | Value | Use Case |
|---------|-------|----------|
| Conservative | 0.5 | Smaller datasets, more BCE warmup |
| Default | 0.3 | Balanced |
| Aggressive | 0.1 | Larger datasets, quick transition to max aggregation |

### Approach Comparison

| Approach | Localization | Speed | Interpretability |
|----------|--------------|-------|------------------|
| PyTorch + max_aggr + sparsity | Excellent | Medium | Medium |
| Sklearn L2 | Good | Fast | Low |
| Sklearn L1 | Good | Medium | High |
| Attention n_heads=2 | Excellent | Slow | High |

---

## 7. Output Structure

### Training Output

```
probes/llama70b_sklearn/
├── logistic/
│   ├── layer_55_probe.pt
│   ├── layer_60_probe.pt
│   └── ...
└── training_summary.json
```

### Evaluation Output

```
results/sklearn_eval/
├── dataset_summary.json
├── aggregated/
│   └── logistic/
│       └── layer_55/
│           ├── metrics.json              # Best threshold metrics
│           ├── metrics_all_thresholds.json
│           ├── raw_scores.json           # Per-sample scores
│           ├── span_metrics.json         # Localization metrics
│           └── token_visualization.html  # Visual inspection
├── ai_liar/
│   └── logistic/
│       └── layer_55/
│           └── metrics.json
└── sandbagging_v2__wmdp_mmlu/
    └── logistic/
        └── layer_55/
            └── metrics.json
```

---

## 8. Key Metrics

### Sample-Level Metrics

| Metric | Description |
|--------|-------------|
| `accuracy` | Overall classification accuracy |
| `auroc` | Area under ROC curve (threshold-independent) |
| `f1` | F1 score |
| `precision` | Precision |
| `recall` / `tpr` | True positive rate |
| `fpr` | False positive rate |

### Span-Level Metrics (Localization)

| Metric | Description |
|--------|-------------|
| `span_precision` | Precision on deceptive token spans |
| `span_recall` | Recall on deceptive token spans |
| `span_f1` | F1 on spans |
| `mean_iou` | Mean intersection-over-union |
| `max_in_span` | Average max activation within labeled spans |
| `max_outside_span` | Average max activation outside spans |
| `mean_in_span` | Average mean activation within labeled spans |
| `mean_outside_span` | Average mean activation outside spans |

---

## Full Pipeline Example

```bash
# Environment
export HF_TOKEN=hf_your_token_here
export CUDA_VISIBLE_DEVICES=0,1

# 1. Generate dataset (if needed)
python data/generate_contrastive_ntml_datasets.py \
    --input data/truth_and_lies_new.json \
    --ratio "5T2L" --samples 500 \
    --shuffle_response --explicit_deception \
    --instruction_style instructional \
    --output ./data/NTML-datasets/experiment1

# 2. Train probes
python scripts/probe_training_ntml.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --train_dataset_dir ./data/NTML-datasets/experiment1/5T2L_500samples_shuffled_explicit_instructional_prefix.json \
    --probe_save_dir ./probes/exp1 \
    --probe_types sklearn_logistic \
    --hyperparams '{"C": 1.0, "class_weight": "balanced", "max_iter": 1000, "solver": "lbfgs"}' \
    --layers 55 60 65 70 75 \
    --max_length 512 \
    --batch_size 32 \
    --dishonest_mode all \
    --honest_mode none

# 3. Evaluate
python scripts/probe_eval_deception_datasets.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --probe_dir ./probes/exp1 \
    --datasets \
        deception-detection-labelled/deception_detection_labeled/ai_liar.jsonl \
        deception-detection-labelled/deception_detection_labeled/sandbagging_v2__wmdp_mmlu.jsonl:500 \
    --labeled_dir deception-detection-labelled/deception_detection_labeled \
    --results_dir ./results/exp1_eval \
    --batch_size 4 \
    --device cuda
```

---

## Troubleshooting

### OOM During Evaluation
Use `eval_ntml_by_layer.py` instead of direct evaluation - it processes one layer at a time and clears cache between layers.

### Slow Activation Collection
- Reduce `--batch_size` for evaluation
- Use `--cache_dir` to cache activations between runs
- Use `--force_recache` only when dataset changes

### No Span Metrics
- Ensure `--labeled_dir` points to directory containing `*_labeled.json` files
- Dataset name must match (e.g., `ai_liar.jsonl` needs `ai_liar_labeled.json`)

### Unicode Errors
```bash
export PYTHONIOENCODING=utf-8
```
