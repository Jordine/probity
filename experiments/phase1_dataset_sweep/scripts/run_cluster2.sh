#!/bin/bash
# Phase 1 Dataset Sweep - Cluster 2
# Runs: All 10T1L datasets (6 instruction styles)
# Hardware: 2x H200 (for 70B model + activation collection)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$(dirname "$EXPERIMENT_DIR")")"

cd "$REPO_ROOT"

# ============================================
# CONFIGURATION
# ============================================
MIX="10T1L"
STYLES=(
  "direct_liar"
  "roleplay"
  "instructional"
  "conditional"
  "neutral"
  "neutral_identity"
)

PROBE_TYPES="attention sklearn_logistic meandiff"
LAYERS="22 36 44"
EPOCHS=60
PATIENCE=999
ANNEAL_WARMUP=1.0
MAX_LENGTH=1024

SWEEP_CONFIG="experiments/phase1_dataset_sweep/sweep_config.json"
OUTPUT_BASE="experiments/phase1_dataset_sweep/probes/cluster2"

# ============================================
# FLIGHT CHECK
# ============================================
echo "============================================"
echo "FLIGHT CHECK - Phase 1 Dataset Sweep - Cluster 2"
echo "============================================"
echo ""
echo "Mix: $MIX"
echo "Styles: ${STYLES[*]}"
echo "Probe types: $PROBE_TYPES"
echo "Layers: $LAYERS"
echo "Epochs: $EPOCHS, Patience: $PATIENCE, Anneal: $ANNEAL_WARMUP"
echo "Max length: $MAX_LENGTH"
echo ""
echo "Sweep config: $SWEEP_CONFIG"
echo "Output base: $OUTPUT_BASE"
echo ""

# Verify sweep config exists and show contents
if [ ! -f "$SWEEP_CONFIG" ]; then
  echo "ERROR: Sweep config not found: $SWEEP_CONFIG"
  exit 1
fi

echo "Sweep config contents:"
cat "$SWEEP_CONFIG"
echo ""

# Count expected probes
NUM_DATASETS=${#STYLES[@]}
NUM_LAYERS=$(echo $LAYERS | wc -w)
# Count variants from sweep config
NUM_ATTENTION=$(grep -c "n_heads" "$SWEEP_CONFIG" || echo 0)
NUM_SKLEARN=$(grep -c '"C":' "$SWEEP_CONFIG" || echo 0)
NUM_MEANDIFF=$(grep -c '"meandiff"' "$SWEEP_CONFIG" || echo 0)
TOTAL_VARIANTS=$((NUM_ATTENTION + NUM_SKLEARN + NUM_MEANDIFF))
TOTAL_PROBES=$((NUM_DATASETS * TOTAL_VARIANTS * NUM_LAYERS))

echo "Expected probes: $NUM_DATASETS datasets × $TOTAL_VARIANTS variants × $NUM_LAYERS layers = $TOTAL_PROBES"
echo ""

# Verify first dataset exists
FIRST_DATASET="experiments/phase1_dataset_sweep/datasets/${MIX}_500samples_shuffled_explicit_${STYLES[0]}_prefix.json"
if [ ! -f "$FIRST_DATASET" ]; then
  echo "ERROR: First dataset not found: $FIRST_DATASET"
  exit 1
fi
echo "First dataset verified: $FIRST_DATASET"
echo ""

echo "============================================"
echo "Starting training in 5 seconds... (Ctrl+C to abort)"
echo "============================================"
sleep 5

# ============================================
# TRAINING LOOP
# ============================================
for style in "${STYLES[@]}"; do
  DATASET_PATH="experiments/phase1_dataset_sweep/datasets/${MIX}_500samples_shuffled_explicit_${style}_prefix.json"
  OUTPUT_DIR="${OUTPUT_BASE}/${MIX}_${style}"

  echo ""
  echo ">>> Training probes for: ${MIX}_${style}"
  echo ">>> Input: $DATASET_PATH"
  echo ">>> Output: $OUTPUT_DIR"
  echo ""

  python3 scripts/probe_training_ntml.py \
    --model_name meta-llama/Llama-3.3-70B-Instruct \
    --train_dataset_dir "$DATASET_PATH" \
    --probe_save_dir "$OUTPUT_DIR" \
    --sweep_config "$SWEEP_CONFIG" \
    --probe_types $PROBE_TYPES \
    --layers $LAYERS \
    --max_length $MAX_LENGTH \
    --num_epochs $EPOCHS \
    --patience $PATIENCE \
    --dishonest_mode all \
    --honest_mode none \
    --use_max_aggregation \
    --anneal_warmup $ANNEAL_WARMUP

  echo ">>> Completed: ${MIX}_${style}"

  # Clear activation cache to free disk space
  echo ">>> Clearing cache..."
  rm -rf ./cache/contrastive/*
  rm -rf ./cache/val_acts/*
done

echo ""
echo "============================================"
echo "Cluster 2 complete!"
echo "Probes saved to: $OUTPUT_BASE"
echo "============================================"
