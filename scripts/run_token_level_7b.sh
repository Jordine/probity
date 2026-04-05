#!/bin/bash
# Quick token-level probe training on Qwen2.5-7B for fast iteration.
#
# This uses the new --token_level flag which:
# 1. Loads only lie_version of conversations (no truth_version shortcut)
# 2. Creates per-token binary labels within each conversation
# 3. Trains with token_all loss + pos_weight class balancing
# 4. Forces probe to learn within-sample token discrimination
#
# Usage: bash scripts/run_token_level_7b.sh

set -e

MODEL="Qwen/Qwen2.5-7B-Instruct"
DATASET="./data/NTML-datasets/5T2L_300_tagged_direct_liar/5T2L_300samples_shuffled_explicit_direct_liar_prefix.json"
SAVE_DIR="./probes/token_level_qwen7b"
CACHE_DIR="./cache/token_level"

# Qwen2.5-7B has 28 layers. Sample early/mid/late.
LAYERS="7 10 14 18 21 24"

echo "=== Token-Level Probe Training (Qwen2.5-7B) ==="
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
echo ""
echo "Next: evaluate with:"
echo "  python scripts/probe_eval_deception_datasets.py \\"
echo "    --model_name $MODEL \\"
echo "    --probe_dir $SAVE_DIR \\"
echo "    --datasets data/deception_detection/ai_liar.jsonl data/deception_detection/sandbagging_v2__wmdp_mmlu.jsonl \\"
echo "    --labeled_dir deception-detection-labelled/deception_detection_labeled \\"
echo "    --results_dir ./results/token_level_qwen7b \\"
echo "    --batch_size 4 --compute_ranking_metrics"
