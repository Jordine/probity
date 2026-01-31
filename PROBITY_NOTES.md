# Probity Project - Claude Notes

**Purpose:** Reference for Claude instances to avoid repeating mistakes. Read this before running experiments.

---

## Quick Reference

### The Three Scripts

| Script | Purpose | Key Flag |
|--------|---------|----------|
| `data/generate_contrastive_ntml_datasets.py` | Generate training data | `--ratio 10T1L` |
| `scripts/probe_training_ntml.py` | Train probes | `--sweep_config` |
| `scripts/probe_eval_deception_datasets.py` | Evaluate probes | `--compute_ranking_metrics` |

---

## CRITICAL: Efficient Training with `--sweep_config`

**NEVER** train probes one-by-one in a loop. This is extremely inefficient because activation collection takes 60-80% of training time.

### Wrong (Inefficient):
```bash
# DON'T DO THIS - collects activations 15 times!
for hp in h1_t0.25 h2_t0.25 h1_t0.5; do
    for layer in 22 24 26 30 34; do
        python probe_training_ntml.py --hyperparams '{"n_heads":1}' --layers $layer
        rm -rf cache/*  # Deletes activations!
    done
done
```

### Correct (Efficient):
```bash
# Create sweep_config.json:
{
  "attention": [
    {"n_heads": 1, "temperature": 0.25, "name_suffix": "_h1_t0.25"},
    {"n_heads": 2, "temperature": 0.25, "name_suffix": "_h2_t0.25"},
    {"n_heads": 1, "temperature": 0.5, "name_suffix": "_h1_t0.5"}
  ]
}

# Train ALL hyperparams with ONE activation collection per layer:
python scripts/probe_training_ntml.py \
  --sweep_config sweep_config.json \
  --layers 22 24 26 30 34 \
  ...
```

This collects activations 5 times (once per layer) instead of 15 times.

---

## Training Parameters

### Required for Good Results

```bash
--dishonest_mode all      # Use ALL tokens from lie samples
--honest_mode none        # Don't use truth samples as negatives (CRITICAL!)
--num_epochs 60           # Enough epochs
--patience 999            # Don't early stop
```

### Why `honest_mode=none` Matters

| Mode | Train AUROC | Test AUROC | Threshold |
|------|-------------|------------|-----------|
| `honest_mode=diff` | 0.995 (overfit!) | 0.72 | 0.594 |
| `honest_mode=none` | 0.946 | **0.953** | 0.058 |

Using `diff` causes the probe to overfit to training data artifacts.

### Batch Sizes

| Model Size | `--batch_size` | `--activation_batch_size` |
|------------|----------------|---------------------------|
| 7B | 32 | 16 |
| 70B | 4-8 | 4 |

### Sequence Length

| `--max_length` | Use Case |
|----------------|----------|
| 1024 | Standard, fast |
| 1536 | Longer conversations |
| 2048 | Very long, may OOM |

---

## Caching System

### Cache Locations
- `cache/contrastive/` - Training activations
- `cache/val_acts/` - Evaluation activations

### Cache Key
Activations cached by: `{model_name}_{dataset_hash}_{layers}`

### When to Clear Cache
- **Clear:** When switching datasets
- **Keep:** When training multiple probes on same dataset

```bash
# Clear cache
rm -rf cache/contrastive/* cache/val_acts/*

# DON'T clear between same-dataset probes - wastes time!
```

---

## Data Generation

### Ratio Format
`{N}T{M}L` = N truths, M lies per conversation

| Ratio | Statements | Lie % | Use Case |
|-------|------------|-------|----------|
| 4T1L | 5 | 20% | Sparse lies |
| 10T1L | 11 | 9% | Very sparse |
| 12T3L | 15 | 20% | More lie signal |

### Instruction Styles
| Style | Description |
|-------|-------------|
| `implicit` | No deception instruction (minimal signal) |
| `direct_liar` | "You are a liar" |
| `roleplay` | "Pretend you are deceptive" |

### Example
```bash
python data/generate_contrastive_ntml_datasets.py \
  --input data/truth_and_lies.json \
  --ratio 12T3L \
  --samples 300 \
  --shuffle_response \
  --instruction_style implicit \
  --output ./localization_exp/data \
  --seed 42
```

---

## Evaluation

### Key Metrics

| Metric | What It Measures | Apollo Baseline |
|--------|------------------|-----------------|
| Sample AUROC | Detect deceptive samples | 0.923 |
| Token AUPRC | Token-level classification | 0.171 |
| R@Oracle | Top-K token ranking | 0.181 |
| Lift | R@Oracle / baseline | 1.16x |

### Evaluation Command
```bash
python scripts/probe_eval_deception_datasets.py \
  --probe_dir ./probes \
  --datasets \
    data/deception_detection/ai_liar.jsonl \
    data/deception_detection/sandbagging_v2__wmdp_mmlu.jsonl \
  --labeled_dir deception-detection-labelled/deception_detection_labeled \
  --results_dir ./eval_results \
  --model_name meta-llama/Llama-3.3-70B-Instruct \
  --batch_size 4 \
  --compute_ranking_metrics
```

---

## Cluster Setup

### SSH Access
```bash
ssh -p 53957 root@93.91.156.83 -i ~/grongles
```

### Environment
```bash
export HF_TOKEN=<HF_TOKEN>
export CUDA_VISIBLE_DEVICES=0,1
cd /root/probity
```

### Always Use tmux
```bash
tmux new-session -s training
# Run commands...
# Ctrl+B, D to detach
tmux attach -t training  # Reattach
```

---

## Lessons Learned (Mistakes to Avoid)

### 1. Training One Probe at a Time
**Wrong:** Loop over hyperparams, train each separately
**Right:** Use `--sweep_config` to train all from single activation collection

### 2. Using `honest_mode=diff`
**Wrong:** `--honest_mode diff` (causes overfitting)
**Right:** `--honest_mode none`

### 3. Running Without tmux
**Wrong:** Run long training in SSH session (dies on disconnect)
**Right:** Always use tmux

### 4. Clearing Cache Between Same-Dataset Probes
**Wrong:** `rm -rf cache/*` after each probe
**Right:** Only clear when switching datasets

### 5. Not Reading Script Help
**Wrong:** Guess at arguments
**Right:** `python script.py --help` first

---

## Current Best Probe Config

From localization experiments:

```bash
# Best for Sample AUROC (0.952):
--hyperparams '{"n_heads": 1, "temperature": 0.25}'
--layers 22
--dishonest_mode all --honest_mode none
--num_epochs 60 --patience 999

# Best for Token AUPRC (0.171, matches Apollo):
--hyperparams '{"n_heads": 2, "temperature": 0.25}'
```

---

## File Locations

| What | Path |
|------|------|
| Statement bank | `data/truth_and_lies.json` |
| Eval datasets | `data/deception_detection/*.jsonl` |
| Labeled spans | `deception-detection-labelled/` |
| Phase1 results | `experiments/phase1_dataset_sweep/` |
| Localization exp | `localization_exp/` |
