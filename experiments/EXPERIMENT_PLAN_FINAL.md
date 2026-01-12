# Final Experiment Plan: Deception Probe Training

## Dataset

**Single training dataset**:
```
data/NTML-datasets/5T2L_500samples_shuffled_explicit_direct_liar_prefix.json
```

- 500 conversations
- 5 truths + 2 lies per conversation (7 statements each)
- 6865 statements from 69 seed categories
- direct_liar instruction style ("You are a liar...")
- Shuffled response order
- Prefix instruction position

## Evaluation

```
deception-detection-labelled/deception_detection_labeled/ai_liar.jsonl
deception-detection-labelled/deception_detection_labeled/sandbagging_v2__wmdp_mmlu.jsonl:500
```

## Global Settings

| Setting | Value | Reason |
|---------|-------|--------|
| epochs | 80 | Allow full convergence |
| patience | 999 | Don't early stop (annealing causes loss spikes) |
| batch_size | 32 | Standard |
| max_length | 512 | Standard |
| dishonest_mode | all | All lie tokens |
| honest_mode | none | No truth tokens |
| layers | 16, 20, 24, 30, 40, 50, 60 | Wide sweep |

---

## Experiment 1: Sklearn Logistic Layer Sweep

**Purpose**: Fast baseline across all layers, matches Apollo's probe type

```bash
python scripts/probe_training_ntml.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --train_dataset_dir ./data/NTML-datasets/5T2L_500samples_shuffled_explicit_direct_liar_prefix.json \
    --probe_save_dir ./probes/exp1_sklearn_sweep \
    --probe_types logistic \
    --use_sklearn_logistic \
    --hyperparams '{"C": 1.0, "class_weight": "balanced", "max_iter": 1000, "solver": "lbfgs"}' \
    --layers 16 20 24 30 40 50 60 \
    --max_length 512 \
    --dishonest_mode all \
    --honest_mode none \
    --device cuda
```

**Eval**:
```bash
python scripts/probe_eval_deception_datasets.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --probe_dir ./probes/exp1_sklearn_sweep \
    --datasets ai_liar.jsonl sandbagging_v2__wmdp_mmlu.jsonl:500 \
    --labeled_dir deception-detection-labelled/deception_detection_labeled \
    --results_dir ./results/exp1_sklearn_sweep_eval \
    --batch_size 4 \
    --device cuda
```

---

## Experiment 2: MaxAggr with Correct Annealing (warmup=1.0)

**Purpose**: Test if proper annealing fixes the train/eval gap

```bash
python scripts/probe_training_ntml.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --train_dataset_dir ./data/NTML-datasets/5T2L_500samples_shuffled_explicit_direct_liar_prefix.json \
    --probe_save_dir ./probes/exp2_maxaggr_sweep \
    --probe_types logistic \
    --layers 16 20 24 30 40 50 60 \
    --max_length 512 \
    --batch_size 32 \
    --num_epochs 80 \
    --patience 999 \
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
    --probe_dir ./probes/exp2_maxaggr_sweep \
    --datasets ai_liar.jsonl sandbagging_v2__wmdp_mmlu.jsonl:500 \
    --labeled_dir deception-detection-labelled/deception_detection_labeled \
    --results_dir ./results/exp2_maxaggr_sweep_eval \
    --batch_size 4 \
    --device cuda
```

---

## Experiment 3: Attention Probes

**Purpose**: Attention probes showed best span_recall in prior work

```bash
python scripts/probe_training_ntml.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --train_dataset_dir ./data/NTML-datasets/5T2L_500samples_shuffled_explicit_direct_liar_prefix.json \
    --probe_save_dir ./probes/exp3_attention_sweep \
    --probe_types attention \
    --hyperparams '{"n_heads": 4, "temperature": 1.0}' \
    --layers 16 20 24 30 40 50 60 \
    --max_length 512 \
    --batch_size 32 \
    --num_epochs 80 \
    --patience 999 \
    --dishonest_mode all \
    --honest_mode none \
    --device cuda
```

**Eval**:
```bash
python scripts/probe_eval_deception_datasets.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --probe_dir ./probes/exp3_attention_sweep \
    --datasets ai_liar.jsonl sandbagging_v2__wmdp_mmlu.jsonl:500 \
    --labeled_dir deception-detection-labelled/deception_detection_labeled \
    --results_dir ./results/exp3_attention_sweep_eval \
    --batch_size 4 \
    --device cuda
```

---

## Summary Table

| Exp | Probe Type | Layers | MaxAggr | Warmup | Epochs | Patience |
|-----|------------|--------|---------|--------|--------|----------|
| 1 | sklearn logistic | 16,20,24,30,40,50,60 | No | - | - | - |
| 2 | PyTorch logistic | 16,20,24,30,40,50,60 | Yes | 1.0 | 80 | 999 |
| 3 | attention | 16,20,24,30,40,50,60 | No | - | 80 | 999 |

---

## Quick Run Script

```bash
#!/bin/bash
# run_all_experiments.sh

MODEL="meta-llama/Llama-3.3-70B-Instruct"
DATA="./data/NTML-datasets/5T2L_500samples_shuffled_explicit_direct_liar_prefix.json"
LAYERS="16 20 24 30 40 50 60"
EVAL_DATASETS="ai_liar.jsonl sandbagging_v2__wmdp_mmlu.jsonl:500"
LABELED_DIR="deception-detection-labelled/deception_detection_labeled"

echo "=========================================="
echo "Exp 1: Sklearn Logistic Sweep"
echo "=========================================="
python scripts/probe_training_ntml.py \
    --model_name $MODEL \
    --train_dataset_dir $DATA \
    --probe_save_dir ./probes/exp1_sklearn_sweep \
    --probe_types logistic \
    --use_sklearn_logistic \
    --hyperparams '{"C": 1.0, "class_weight": "balanced", "max_iter": 1000}' \
    --layers $LAYERS \
    --max_length 512 \
    --dishonest_mode all \
    --honest_mode none

python scripts/probe_eval_deception_datasets.py \
    --model_name $MODEL \
    --probe_dir ./probes/exp1_sklearn_sweep \
    --datasets $EVAL_DATASETS \
    --labeled_dir $LABELED_DIR \
    --results_dir ./results/exp1_sklearn_sweep_eval \
    --batch_size 4

echo "=========================================="
echo "Exp 2: MaxAggr with warmup=1.0"
echo "=========================================="
python scripts/probe_training_ntml.py \
    --model_name $MODEL \
    --train_dataset_dir $DATA \
    --probe_save_dir ./probes/exp2_maxaggr_sweep \
    --probe_types logistic \
    --layers $LAYERS \
    --max_length 512 \
    --batch_size 32 \
    --num_epochs 80 \
    --patience 999 \
    --use_max_aggregation \
    --anneal_warmup 1.0 \
    --sparsity_penalty 0.05 \
    --dishonest_mode all \
    --honest_mode none

python scripts/probe_eval_deception_datasets.py \
    --model_name $MODEL \
    --probe_dir ./probes/exp2_maxaggr_sweep \
    --datasets $EVAL_DATASETS \
    --labeled_dir $LABELED_DIR \
    --results_dir ./results/exp2_maxaggr_sweep_eval \
    --batch_size 4

echo "=========================================="
echo "Exp 3: Attention Probes"
echo "=========================================="
python scripts/probe_training_ntml.py \
    --model_name $MODEL \
    --train_dataset_dir $DATA \
    --probe_save_dir ./probes/exp3_attention_sweep \
    --probe_types attention \
    --hyperparams '{"n_heads": 4, "temperature": 1.0}' \
    --layers $LAYERS \
    --max_length 512 \
    --batch_size 32 \
    --num_epochs 80 \
    --patience 999 \
    --dishonest_mode all \
    --honest_mode none

python scripts/probe_eval_deception_datasets.py \
    --model_name $MODEL \
    --probe_dir ./probes/exp3_attention_sweep \
    --datasets $EVAL_DATASETS \
    --labeled_dir $LABELED_DIR \
    --results_dir ./results/exp3_attention_sweep_eval \
    --batch_size 4

echo "=========================================="
echo "All experiments complete!"
echo "=========================================="
```

---

## Expected Output Structure

```
probes/
├── exp1_sklearn_sweep/
│   └── logistic/
│       ├── layer_16_probe.pt
│       ├── layer_20_probe.pt
│       ├── layer_24_probe.pt
│       ├── layer_30_probe.pt
│       ├── layer_40_probe.pt
│       ├── layer_50_probe.pt
│       └── layer_60_probe.pt
├── exp2_maxaggr_sweep/
│   └── logistic/
│       └── ... (same layers)
└── exp3_attention_sweep/
    └── attention/
        └── ... (same layers)

results/
├── exp1_sklearn_sweep_eval/
│   └── aggregated/logistic/layer_*/metrics.json
├── exp2_maxaggr_sweep_eval/
└── exp3_attention_sweep_eval/
```

---

## Success Criteria

| AUROC | Status |
|-------|--------|
| < 0.65 | No improvement |
| 0.65-0.75 | Some improvement |
| 0.75-0.85 | Good (within striking distance) |
| 0.85+ | Great (close to Apollo 0.925) |

**Apollo baseline**: 0.925 AUROC on same eval sets
