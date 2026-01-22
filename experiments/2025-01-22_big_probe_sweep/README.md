# Big Probe Sweep Experiment (2025-01-22)

## Overview
Training probes across 18 layers (22-56, even numbers) with 4 probe types on Llama-3.3-70B-Instruct.

## Dataset
- **File**: `data/NTML-datasets/5T2L_500samples_shuffled_explicit_direct_liar_prefix.json`
- **Size**: 500 conversations → 3465 statement examples
- **Balance**: 28.3% lies (981 lies, 2484 truths)
- **Max length**: 1024 tokens

## Probe Types
1. **sklearn_logistic** - L-BFGS optimization (~seconds)
2. **meandiff** - Mean difference direction (~seconds)
3. **attention** - Multi-head attention probe (4 heads, ~4.5 min/layer)
4. **logistic** - PyTorch with max_aggregation loss (~29 min/layer)

## Commands Used

### 1. Run the sweep
```bash
# On cluster (tmux session recommended)
cd /root/probity
./scripts/sweep_experiments.sh 2>&1 | tee sweep.log
```

### 2. The sweep script runs 3 batches:
```bash
# Batch 1: Layers 22, 24, 26, 28, 30, 32
# Batch 2: Layers 34, 36, 38, 40, 42, 44
# Batch 3: Layers 46, 48, 50, 52, 54, 56

python scripts/probe_training_ntml.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --train_dataset_dir "./data/NTML-datasets/5T2L_500samples_shuffled_explicit_direct_liar_prefix.json" \
    --probe_save_dir "./big_probe_sweep" \
    --probe_types sklearn_logistic meandiff attention logistic \
    --hyperparams '{"n_heads": 4, "temperature": 1.0}' \
    --layers 22 24 26 28 30 32 \
    --max_length 1024 --batch_size 32 \
    --num_epochs 80 --patience 999 \
    --use_max_aggregation --anneal_warmup 1.0 \
    --dishonest_mode all --honest_mode none
```

### 3. Run evaluation after training
```bash
python scripts/probe_eval_deception_datasets.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --probe_dir ./big_probe_sweep \
    --datasets \
        ai_liar.jsonl \
        sandbagging_v2__wmdp_mmlu.jsonl:500 \
    --labeled_dir deception-detection-labelled/deception_detection_labeled \
    --results_dir ./big_probe_sweep_eval \
    --batch_size 4 --device cuda
```

## To Extend to More Layers (e.g., 58-80)

```bash
# Single batch for layers 58-68
python scripts/probe_training_ntml.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --train_dataset_dir "./data/NTML-datasets/5T2L_500samples_shuffled_explicit_direct_liar_prefix.json" \
    --probe_save_dir "./big_probe_sweep" \
    --probe_types sklearn_logistic meandiff attention logistic \
    --hyperparams '{"n_heads": 4, "temperature": 1.0}' \
    --layers 58 60 62 64 66 68 \
    --max_length 1024 --batch_size 32 \
    --num_epochs 80 --patience 999 \
    --use_max_aggregation --anneal_warmup 1.0 \
    --dishonest_mode all --honest_mode none
```

## To Train on Different Dataset

```bash
# Example with roleplay instruction style
python scripts/probe_training_ntml.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --train_dataset_dir "./data/NTML-datasets/5T2L_500samples_shuffled_explicit_roleplay_prefix.json" \
    --probe_save_dir "./probes_roleplay" \
    --probe_types sklearn_logistic meandiff attention logistic \
    --hyperparams '{"n_heads": 4, "temperature": 1.0}' \
    --layers 22 30 40 50 \
    --max_length 1024 --batch_size 32 \
    --num_epochs 80 --patience 999 \
    --use_max_aggregation --anneal_warmup 1.0 \
    --dishonest_mode all --honest_mode none
```

## Key Flags Explained

| Flag | Description |
|------|-------------|
| `--use_max_aggregation` | Enable span-level max aggregation loss (only applies to logistic probe) |
| `--anneal_warmup 1.0` | Full linear annealing from BCE to max_aggr over all epochs |
| `--dishonest_mode all` | Use all statements from dishonest samples |
| `--honest_mode none` | Don't include honest sample statements |
| `--max_length 1024` | Token context length |
| `--num_epochs 80` | Training epochs |
| `--patience 999` | Disable early stopping |

## Results Location
- Probes: `./big_probe_sweep/{probe_type}/layer_{N}_probe.pt`
- Training summary: `./big_probe_sweep/training_summary.json`
- Evaluation results: `./big_probe_sweep_eval/`

## Timing (per layer, 2xH200)
- Activation collection: ~12 min for 6 layers
- sklearn_logistic: ~seconds
- meandiff: ~seconds
- attention: ~4.5 min
- logistic [max_aggr]: ~29 min
- **Total per layer: ~35 min**
