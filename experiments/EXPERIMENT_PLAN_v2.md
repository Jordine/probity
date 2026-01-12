# Experiment Plan v2: Closing the Apollo Gap

## Revised Understanding

The gap is likely due to **implementation issues we've now fixed**, not fundamental data format problems:

| Issue | Was | Now Fixed |
|-------|-----|-----------|
| Lie span bugs | Incorrect positions | Fixed in code |
| anneal_warmup | 0.3 (too fast) | Will use **1.0** |
| Data diversity | ~9 categories | **69 categories, 6865 statements** |
| Sample size | 500 | Have **1000**, can generate 2000 |
| Instruction style | Mixed | **direct_liar** (best performing) |

**Hypothesis**: With correct hyperparameters and fixed data, we should approach Apollo's 0.925 AUROC.

---

## Available Training Datasets

All in `data/NTML-datasets/newbank/`:

| File | Ratio | Samples | Style | Format |
|------|-------|---------|-------|--------|
| `5T2L_500samples_shuffled_explicit_direct_liar_prefix.json` | 5T2L | 500 | direct_liar | prefix |
| `5T2L_1000samples_shuffled_explicit_direct_liar_prefix.json` | 5T2L | 1000 | direct_liar | prefix |
| `5T2L_500samples_shuffled_explicit_roleplay_prefix.json` | 5T2L | 500 | roleplay | prefix |
| `5T2L_500samples_shuffled_explicit_behavioral_prefix.json` | 5T2L | 500 | behavioral | prefix |
| `5T2L_500samples_shuffled_explicit_neutral_prefix.json` | 5T2L | 500 | neutral | prefix |
| `5T2L_500samples_shuffled.json` | 5T2L | 500 | implicit | - |
| `3T3L_500samples_shuffled_explicit_direct_liar_prefix.json` | 3T3L | 500 | direct_liar | prefix |
| `5T1L_500samples_shuffled_explicit_direct_liar_prefix.json` | 5T1L | 500 | direct_liar | prefix |

**Primary dataset**: `5T2L_1000samples_shuffled_explicit_direct_liar_prefix.json`
- 1000 conversations × 7 statements = 7000 statement examples
- 5 truths + 2 lies per conversation (28.6% lie rate)
- direct_liar style: "You are a liar. You will lie about 2 of the following facts."
- Shuffled response order (no positional confounds)
- 69 seed categories (no category repetition per conversation)

---

## Evaluation Datasets

In `deception-detection-labelled/deception_detection_labeled/`:

| File | Samples | Description |
|------|---------|-------------|
| `ai_liar.jsonl` | 266 | Instructed capability hiding |
| `sandbagging_v2__wmdp_mmlu.jsonl` | 500 (subset) | Intentional underperformance |

**Aggregated eval**: 766 samples total

---

## Experiment Schedule

### Phase 1: Baseline with Fixed Data (Fast)

#### Exp 1.1: Sklearn Logistic - Layer 22 Only
**Purpose**: Quick baseline matching Apollo's probe type on our best data

```bash
python scripts/probe_training_ntml.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --train_dataset_dir ./data/NTML-datasets/newbank/5T2L_500samples_shuffled_explicit_direct_liar_prefix.json \
    --probe_save_dir ./probes/exp1_1_sklearn_l22 \
    --probe_types logistic \
    --use_sklearn_logistic \
    --hyperparams '{"C": 1.0, "class_weight": "balanced", "max_iter": 1000, "solver": "lbfgs"}' \
    --layers 22 \
    --max_length 512 \
    --dishonest_mode all \
    --honest_mode none \
    --device cuda
```

**Eval**:
```bash
python scripts/probe_eval_deception_datasets.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --probe_dir ./probes/exp1_1_sklearn_l22 \
    --datasets ai_liar.jsonl sandbagging_v2__wmdp_mmlu.jsonl:500 \
    --labeled_dir deception-detection-labelled/deception_detection_labeled \
    --results_dir ./results/exp1_1_sklearn_l22_eval \
    --batch_size 4 \
    --device cuda
```

#### Exp 1.2: Sklearn Logistic - Layer Sweep 16-30
**Purpose**: Verify L22 is optimal for our data (not just Apollo's)

```bash
python scripts/probe_training_ntml.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --train_dataset_dir ./data/NTML-datasets/newbank/5T2L_500samples_shuffled_explicit_direct_liar_prefix.json \
    --probe_save_dir ./probes/exp1_2_sklearn_layer_sweep \
    --probe_types logistic \
    --use_sklearn_logistic \
    --hyperparams '{"C": 1.0, "class_weight": "balanced", "max_iter": 1000, "solver": "lbfgs"}' \
    --layers 16 18 20 22 24 26 28 30 \
    --max_length 512 \
    --dishonest_mode all \
    --honest_mode none \
    --device cuda
```

**Eval**:
```bash
python scripts/probe_eval_deception_datasets.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --probe_dir ./probes/exp1_2_sklearn_layer_sweep \
    --datasets ai_liar.jsonl sandbagging_v2__wmdp_mmlu.jsonl:500 \
    --labeled_dir deception-detection-labelled/deception_detection_labeled \
    --results_dir ./results/exp1_2_sklearn_layer_sweep_eval \
    --batch_size 4 \
    --device cuda
```

---

### Phase 2: Correct MaxAggr Hyperparameters

#### Exp 2.1: MaxAggr with warmup=1.0, L22
**Purpose**: Test if correct annealing fixes the train/eval gap

```bash
python scripts/probe_training_ntml.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --train_dataset_dir ./data/NTML-datasets/newbank/5T2L_500samples_shuffled_explicit_direct_liar_prefix.json \
    --probe_save_dir ./probes/exp2_1_maxaggr_warmup1 \
    --probe_types logistic \
    --layers 22 \
    --max_length 512 \
    --batch_size 32 \
    --num_epochs 50 \
    --patience 15 \
    --use_max_aggregation \
    --anneal_warmup 1.0 \
    --sparsity_penalty 0.05 \
    --dishonest_mode all \
    --honest_mode none \
    --device cuda
```

**Eval**:
```bash
python scripts/probe_eval_deception_datasets.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --probe_dir ./probes/exp2_1_maxaggr_warmup1 \
    --datasets ai_liar.jsonl sandbagging_v2__wmdp_mmlu.jsonl:500 \
    --labeled_dir deception-detection-labelled/deception_detection_labeled \
    --results_dir ./results/exp2_1_maxaggr_warmup1_eval \
    --batch_size 4 \
    --device cuda
```

#### Exp 2.2: MaxAggr Layer Sweep with warmup=1.0
**Purpose**: Find optimal layer for MaxAggr with correct settings

```bash
python scripts/probe_training_ntml.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --train_dataset_dir ./data/NTML-datasets/newbank/5T2L_500samples_shuffled_explicit_direct_liar_prefix.json \
    --probe_save_dir ./probes/exp2_2_maxaggr_layer_sweep \
    --probe_types logistic \
    --layers 16 20 22 24 28 30 \
    --max_length 512 \
    --batch_size 32 \
    --num_epochs 50 \
    --patience 15 \
    --use_max_aggregation \
    --anneal_warmup 1.0 \
    --sparsity_penalty 0.05 \
    --dishonest_mode all \
    --honest_mode none \
    --device cuda
```

---

### Phase 3: Scale Up Data

#### Exp 3.1: Sklearn L22 with 1000 samples
**Purpose**: Test if more data improves generalization

```bash
python scripts/probe_training_ntml.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --train_dataset_dir ./data/NTML-datasets/newbank/5T2L_1000samples_shuffled_explicit_direct_liar_prefix.json \
    --probe_save_dir ./probes/exp3_1_sklearn_1000samples \
    --probe_types logistic \
    --use_sklearn_logistic \
    --hyperparams '{"C": 1.0, "class_weight": "balanced", "max_iter": 1000, "solver": "lbfgs"}' \
    --layers 22 \
    --max_length 512 \
    --dishonest_mode all \
    --honest_mode none \
    --device cuda
```

#### Exp 3.2: MaxAggr with 1000 samples + warmup=1.0
**Purpose**: Combine correct hyperparams with more data

```bash
python scripts/probe_training_ntml.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --train_dataset_dir ./data/NTML-datasets/newbank/5T2L_1000samples_shuffled_explicit_direct_liar_prefix.json \
    --probe_save_dir ./probes/exp3_2_maxaggr_1000samples \
    --probe_types logistic \
    --layers 22 \
    --max_length 512 \
    --batch_size 32 \
    --num_epochs 50 \
    --patience 15 \
    --use_max_aggregation \
    --anneal_warmup 1.0 \
    --sparsity_penalty 0.05 \
    --dishonest_mode all \
    --honest_mode none \
    --device cuda
```

#### Exp 3.3: Generate and train on 2000 samples (if needed)

**Generate**:
```bash
python data/generate_contrastive_ntml_datasets.py \
    --input data/truth_and_lies.json \
    --ratio "5T2L" \
    --samples 2000 \
    --shuffle_response \
    --explicit_deception \
    --instruction_style direct_liar \
    --instruction_position prefix \
    --output data/NTML-datasets/newbank \
    --seed 42
```

**Train**:
```bash
python scripts/probe_training_ntml.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --train_dataset_dir ./data/NTML-datasets/newbank/5T2L_2000samples_shuffled_explicit_direct_liar_prefix.json \
    --probe_save_dir ./probes/exp3_3_sklearn_2000samples \
    --probe_types logistic \
    --use_sklearn_logistic \
    --hyperparams '{"C": 1.0, "class_weight": "balanced", "max_iter": 1000, "solver": "lbfgs"}' \
    --layers 22 \
    --max_length 512 \
    --dishonest_mode all \
    --honest_mode none \
    --device cuda
```

---

### Phase 4: Attention Probes (Best Span Metrics)

#### Exp 4.1: Attention Probe L22 with 1000 samples
**Purpose**: Attention probes had best span_recall (0.584) - test with more data

```bash
python scripts/probe_training_ntml.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --train_dataset_dir ./data/NTML-datasets/newbank/5T2L_1000samples_shuffled_explicit_direct_liar_prefix.json \
    --probe_save_dir ./probes/exp4_1_attention_1000samples \
    --probe_types attention \
    --hyperparams '{"n_heads": 4, "temperature": 1.0}' \
    --layers 22 30 \
    --max_length 512 \
    --batch_size 32 \
    --num_epochs 30 \
    --patience 10 \
    --dishonest_mode all \
    --honest_mode none \
    --device cuda
```

---

### Phase 5: Ablations (If Phases 1-4 Don't Close Gap)

#### Exp 5.1: Instruction Style Comparison
**Purpose**: Confirm direct_liar is best across probe types

```bash
# Run sklearn L22 on each style
for style in direct_liar roleplay behavioral neutral; do
    python scripts/probe_training_ntml.py \
        --model_name "meta-llama/Llama-3.3-70B-Instruct" \
        --train_dataset_dir ./data/NTML-datasets/newbank/5T2L_500samples_shuffled_explicit_${style}_prefix.json \
        --probe_save_dir ./probes/exp5_1_style_${style} \
        --probe_types logistic \
        --use_sklearn_logistic \
        --layers 22 \
        --dishonest_mode all \
        --honest_mode none
done
```

#### Exp 5.2: Ratio Comparison
**Purpose**: Test if different lie ratios affect generalization

```bash
# 3T3L (more lies)
python scripts/probe_training_ntml.py \
    --train_dataset_dir ./data/NTML-datasets/newbank/3T3L_500samples_shuffled_explicit_direct_liar_prefix.json \
    --probe_save_dir ./probes/exp5_2_ratio_3T3L \
    --probe_types logistic \
    --use_sklearn_logistic \
    --layers 22 \
    --dishonest_mode all \
    --honest_mode none

# 5T1L (fewer lies)
python scripts/probe_training_ntml.py \
    --train_dataset_dir ./data/NTML-datasets/newbank/5T1L_500samples_shuffled_explicit_direct_liar_prefix.json \
    --probe_save_dir ./probes/exp5_2_ratio_5T1L \
    --probe_types logistic \
    --use_sklearn_logistic \
    --layers 22 \
    --dishonest_mode all \
    --honest_mode none
```

#### Exp 5.3: Cross-Dataset Eval (Apollo-style)
**Purpose**: Train on NTML, eval on ai_liar; train on ai_liar-like, eval on sandbagging

This requires generating ai_liar-style training data first (future work).

---

## Full Experiment Matrix

| Exp | Phase | Probe | Layers | Samples | MaxAggr | Warmup | Style | Priority |
|-----|-------|-------|--------|---------|---------|--------|-------|----------|
| 1.1 | Baseline | sklearn | 22 | 500 | No | - | direct_liar | **HIGH** |
| 1.2 | Baseline | sklearn | 16-30 | 500 | No | - | direct_liar | **HIGH** |
| 2.1 | MaxAggr | logistic | 22 | 500 | Yes | 1.0 | direct_liar | **HIGH** |
| 2.2 | MaxAggr | logistic | 16-30 | 500 | Yes | 1.0 | direct_liar | MEDIUM |
| 3.1 | Scale | sklearn | 22 | 1000 | No | - | direct_liar | **HIGH** |
| 3.2 | Scale | logistic | 22 | 1000 | Yes | 1.0 | direct_liar | MEDIUM |
| 3.3 | Scale | sklearn | 22 | 2000 | No | - | direct_liar | LOW |
| 4.1 | Attention | attention | 22,30 | 1000 | No | - | direct_liar | MEDIUM |
| 5.1 | Ablation | sklearn | 22 | 500 | No | - | all styles | LOW |
| 5.2 | Ablation | sklearn | 22 | 500 | No | - | all ratios | LOW |

---

## Expected Results

| Exp | Expected AUROC | Rationale |
|-----|----------------|-----------|
| 1.1 | 0.65-0.75 | Sklearn baseline with fixed data |
| 1.2 | Find optimal layer | May find better than L22 |
| 2.1 | 0.70-0.80 | Correct MaxAggr should close train/eval gap |
| 3.1 | 0.70-0.80 | More data = better generalization |
| 3.2 | 0.75-0.85 | Best hyperparams + more data |
| 4.1 | 0.65-0.75 | Good span metrics, moderate AUROC |

**Target**: Any experiment hitting **0.80+ AUROC** closes the gap meaningfully.
**Apollo benchmark**: 0.925 AUROC

---

## Hyperparameter Summary

### Sklearn Logistic (Recommended for baselines)
```json
{
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 1000,
    "solver": "lbfgs",
    "penalty": "l2"
}
```

### PyTorch Logistic + MaxAggr (For span localization)
```
--num_epochs 50
--patience 15
--batch_size 32
--use_max_aggregation
--anneal_warmup 1.0      # CRITICAL: Full BCE first, then max aggr
--sparsity_penalty 0.05  # Prevents firing on every token
```

### Attention Probe
```json
{
    "n_heads": 4,
    "temperature": 1.0
}
```
```
--num_epochs 30
--patience 10
```

### Label Selection (All experiments)
```
--dishonest_mode all    # All tokens from lie statements
--honest_mode none      # No tokens from truth statements
```

---

## Data Generation Settings (Reference)

All training data generated with:
```bash
python data/generate_contrastive_ntml_datasets.py \
    --input data/truth_and_lies.json \      # 6865 statements, 69 categories
    --ratio "5T2L" \                         # 5 truths, 2 lies per conversation
    --samples <N> \                          # 500, 1000, or 2000
    --shuffle_response \                     # Randomize statement order
    --explicit_deception \                   # Include deception instruction
    --instruction_style direct_liar \        # "You are a liar..."
    --instruction_position prefix \          # Instruction before facts
    --output data/NTML-datasets/newbank \
    --seed 42                                # Reproducibility
```

---

## Quick Start Script

Run all high-priority experiments:

```bash
#!/bin/bash
# run_priority_experiments.sh

MODEL="meta-llama/Llama-3.3-70B-Instruct"
DATA_500="./data/NTML-datasets/newbank/5T2L_500samples_shuffled_explicit_direct_liar_prefix.json"
DATA_1000="./data/NTML-datasets/newbank/5T2L_1000samples_shuffled_explicit_direct_liar_prefix.json"
EVAL_DATASETS="ai_liar.jsonl sandbagging_v2__wmdp_mmlu.jsonl:500"
LABELED_DIR="deception-detection-labelled/deception_detection_labeled"

# Exp 1.1: Sklearn L22 baseline
echo "=== Exp 1.1: Sklearn L22 ==="
python scripts/probe_training_ntml.py \
    --model_name $MODEL \
    --train_dataset_dir $DATA_500 \
    --probe_save_dir ./probes/exp1_1_sklearn_l22 \
    --probe_types logistic --use_sklearn_logistic \
    --hyperparams '{"C": 1.0, "class_weight": "balanced"}' \
    --layers 22 --dishonest_mode all --honest_mode none

python scripts/probe_eval_deception_datasets.py \
    --model_name $MODEL \
    --probe_dir ./probes/exp1_1_sklearn_l22 \
    --datasets $EVAL_DATASETS \
    --labeled_dir $LABELED_DIR \
    --results_dir ./results/exp1_1_eval --batch_size 4

# Exp 1.2: Sklearn layer sweep
echo "=== Exp 1.2: Sklearn Layer Sweep ==="
python scripts/probe_training_ntml.py \
    --model_name $MODEL \
    --train_dataset_dir $DATA_500 \
    --probe_save_dir ./probes/exp1_2_sklearn_layers \
    --probe_types logistic --use_sklearn_logistic \
    --hyperparams '{"C": 1.0, "class_weight": "balanced"}' \
    --layers 16 18 20 22 24 26 28 30 \
    --dishonest_mode all --honest_mode none

python scripts/probe_eval_deception_datasets.py \
    --model_name $MODEL \
    --probe_dir ./probes/exp1_2_sklearn_layers \
    --datasets $EVAL_DATASETS \
    --labeled_dir $LABELED_DIR \
    --results_dir ./results/exp1_2_eval --batch_size 4

# Exp 2.1: MaxAggr warmup=1.0
echo "=== Exp 2.1: MaxAggr warmup=1.0 ==="
python scripts/probe_training_ntml.py \
    --model_name $MODEL \
    --train_dataset_dir $DATA_500 \
    --probe_save_dir ./probes/exp2_1_maxaggr \
    --probe_types logistic \
    --layers 22 \
    --num_epochs 50 --patience 15 --batch_size 32 \
    --use_max_aggregation --anneal_warmup 1.0 --sparsity_penalty 0.05 \
    --dishonest_mode all --honest_mode none

python scripts/probe_eval_deception_datasets.py \
    --model_name $MODEL \
    --probe_dir ./probes/exp2_1_maxaggr \
    --datasets $EVAL_DATASETS \
    --labeled_dir $LABELED_DIR \
    --results_dir ./results/exp2_1_eval --batch_size 4

# Exp 3.1: Sklearn with 1000 samples
echo "=== Exp 3.1: Sklearn 1000 samples ==="
python scripts/probe_training_ntml.py \
    --model_name $MODEL \
    --train_dataset_dir $DATA_1000 \
    --probe_save_dir ./probes/exp3_1_sklearn_1000 \
    --probe_types logistic --use_sklearn_logistic \
    --hyperparams '{"C": 1.0, "class_weight": "balanced"}' \
    --layers 22 --dishonest_mode all --honest_mode none

python scripts/probe_eval_deception_datasets.py \
    --model_name $MODEL \
    --probe_dir ./probes/exp3_1_sklearn_1000 \
    --datasets $EVAL_DATASETS \
    --labeled_dir $LABELED_DIR \
    --results_dir ./results/exp3_1_eval --batch_size 4

echo "=== All priority experiments complete ==="
```

---

## Success Criteria

- **Good**: Any probe hits 0.75+ AUROC (within 0.18 of Apollo)
- **Great**: Any probe hits 0.80+ AUROC (within 0.13 of Apollo)
- **Excellent**: Any probe hits 0.85+ AUROC (within 0.08 of Apollo)
- **Apollo parity**: 0.90+ AUROC

If Phase 1-3 don't reach 0.75+, investigate:
1. Data format mismatch (generate ai_liar-style)
2. Apollo fine-tuning
3. Multi-dataset training
