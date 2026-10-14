#!/bin/bash
# Token-level probe training on Llama-3.1-8B for fast iteration.
#
# Usage: bash scripts/run_token_level_llama8b.sh

set -e

MODEL="meta-llama/Llama-3.1-8B-Instruct"
DATASET="./data/NTML-datasets/5T2L_300_tagged_direct_liar/5T2L_300samples_shuffled_explicit_direct_liar_prefix.json"
SAVE_DIR="./probes/token_level_llama8b"
CACHE_DIR="./cache/token_level_llama8b"

# Llama-3.1-8B has 32 layers. Sample across depth.
LAYERS="8 12 16 20 24 28"

echo "=== Token-Level Probe Training (Llama-3.1-8B) ==="
echo "Dataset: $DATASET"
echo "Layers: $LAYERS"
echo ""

python scripts/probe_training_ntml.py \
    --model_name "$MODEL" \
    --train_dataset_dir "$DATASET" \
    --probe_types logistic attention \
    --layers $LAYERS \
    --probe_save_dir "$SAVE_DIR" \
    --cache_dir "$CACHE_DIR" \
    --max_length 1024 \
    --batch_size 8 \
    --num_epochs 30 \
    --patience 10 \
    --dishonest_mode all \
    --honest_mode none \
    --use_llm_spans always \
    --token_level \
    --loss_mode token_all

echo ""
echo "=== Training complete ==="
echo "Probes saved to: $SAVE_DIR"
