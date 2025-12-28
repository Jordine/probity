#!/bin/bash
# Train NTML probes with explicit deception dataset

cd /root/probity
mkdir -p logs

export HF_TOKEN=  # Set your token
export CUDA_VISIBLE_DEVICES=0,1

DATASET="data/NTML-datasets/contrastive_explicit_shuffled/4T1L_250samples_shuffled_explicit.json"
MODEL="meta-llama/Llama-3.3-70B-Instruct"
HYPERPARAMS='{"C": 1.0, "class_weight": "balanced", "max_iter": 1000, "solver": "lbfgs", "n_heads": 1, "temperature": 1.0}'

echo "=== Training all_diff ==="
python scripts/probe_training_ntml.py \
    --model_name $MODEL \
    --train_dataset_dir $DATASET \
    --probe_types logistic attention \
    --probe_save_dir trained_probes/ntml_70b_explicit_all_diff \
    --max_length 512 \
    --activation_batch_size 2 \
    --use_sklearn_logistic \
    --hyperparams "$HYPERPARAMS" \
    --dishonest_mode all \
    --honest_mode diff \
    --use_max_aggregation \
    --anneal_warmup 0.3 \
    2>&1 | tee logs/train_explicit_all_diff.log

rm -rf cache/*

echo "=== Training all_none ==="
python scripts/probe_training_ntml.py \
    --model_name $MODEL \
    --train_dataset_dir $DATASET \
    --probe_types logistic attention \
    --probe_save_dir trained_probes/ntml_70b_explicit_all_none \
    --max_length 512 \
    --activation_batch_size 2 \
    --use_sklearn_logistic \
    --hyperparams "$HYPERPARAMS" \
    --dishonest_mode all \
    --honest_mode none \
    --use_max_aggregation \
    --anneal_warmup 0.3 \
    2>&1 | tee logs/train_explicit_all_none.log

echo "=== Training complete ==="
