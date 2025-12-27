#!/bin/bash
# Sweep all 80 layers in batches of 5 to avoid OOM

# Set HF_TOKEN environment variable before running
# export HF_TOKEN="your_token_here"
export CUDA_VISIBLE_DEVICES=0,1

MODEL="meta-llama/Llama-3.3-70B-Instruct"
DATASET="training_data_NTML-datasets/contrastive/4T1L_250samples_shuffled.json"
SAVE_DIR="trained_probes/ntml_70b_4t1l_sweep"
PROBE_TYPES="logistic meandiff pca attention mlp"
HYPERPARAMS='{"C": 1.0, "class_weight": "balanced", "max_iter": 1000, "solver": "lbfgs", "n_heads": 1, "temperature": 1.0}'

# 80 layers, 5 per batch = 16 runs
for start in $(seq 0 5 79); do
    end=$((start + 4))
    if [ $end -gt 79 ]; then
        end=79
    fi

    # Build layer list
    layers=""
    for l in $(seq $start $end); do
        layers="$layers $l"
    done

    echo "========================================"
    echo "Training layers $start to $end"
    echo "========================================"

    python scripts/probe_training_ntml.py \
        --model_name "$MODEL" \
        --train_dataset_dir "$DATASET" \
        --probe_types $PROBE_TYPES \
        --probe_save_dir "$SAVE_DIR" \
        --layers $layers \
        --max_length 512 \
        --activation_batch_size 4 \
        --use_sklearn_logistic \
        --hyperparams "$HYPERPARAMS" \
        --dishonest_mode all \
        --honest_mode diff \
        --use_max_aggregation \
        --anneal_warmup 0.3

    # Check if training succeeded
    if [ $? -ne 0 ]; then
        echo "ERROR: Training failed for layers $start-$end"
        exit 1
    fi

    echo "Completed layers $start to $end"
    echo ""
done

echo "========================================"
echo "All layers completed!"
echo "========================================"
