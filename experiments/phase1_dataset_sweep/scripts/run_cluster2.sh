#!/bin/bash
# Phase 1 Dataset Sweep - Cluster 2
# Runs: All 10T1L datasets (6 instruction styles)
# Hardware: 2x H200 (for 70B model + activation collection)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$(dirname "$EXPERIMENT_DIR")")"

cd "$REPO_ROOT"

MIX="10T1L"
STYLES=(
  "direct_liar"
  "roleplay"
  "instructional"
  "conditional"
  "neutral"
  "neutral_identity"
)

LAYERS="22 36 44"
EPOCHS=60
PATIENCE=999
ANNEAL_WARMUP=1.0

echo "============================================"
echo "Phase 1 Dataset Sweep - Cluster 2"
echo "Mix: $MIX"
echo "Styles: ${STYLES[*]}"
echo "Layers: $LAYERS"
echo "Epochs: $EPOCHS, Patience: $PATIENCE, Anneal: $ANNEAL_WARMUP"
echo "Total probes: 6 datasets × 10 hyperparams × 3 layers = 180"
echo "============================================"

for style in "${STYLES[@]}"; do
  DATASET_PATH="experiments/phase1_dataset_sweep/datasets/${MIX}_500samples_shuffled_explicit_${style}_prefix.json"
  OUTPUT_DIR="experiments/phase1_dataset_sweep/probes/cluster2/${MIX}_${style}"

  echo ""
  echo ">>> Training probes for: ${MIX}_${style}"
  echo ">>> Input: $DATASET_PATH"
  echo ">>> Output: $OUTPUT_DIR"
  echo ""

  python scripts/probe_training_ntml.py \
    --model_name meta-llama/Llama-3.3-70B-Instruct \
    --train_dataset_dir "$DATASET_PATH" \
    --probe_save_dir "$OUTPUT_DIR" \
    --sweep_config experiments/phase1_dataset_sweep/sweep_config.json \
    --layers $LAYERS \
    --max_length 1024 \
    --num_epochs $EPOCHS \
    --patience $PATIENCE \
    --dishonest_mode all \
    --honest_mode none \
    --use_max_aggregation \
    --anneal_warmup $ANNEAL_WARMUP

  echo ">>> Completed: ${MIX}_${style}"
done

echo ""
echo "============================================"
echo "Cluster 2 complete!"
echo "Probes saved to: experiments/phase1_dataset_sweep/probes/cluster2/"
echo "============================================"
