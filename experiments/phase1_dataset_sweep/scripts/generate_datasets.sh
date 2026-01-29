#!/bin/bash
# Generate all 12 datasets for Phase 1 sweep
# 6 instruction styles × 2 dataset mixes (5T2L, 10T1L)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$(dirname "$EXPERIMENT_DIR")")"

cd "$REPO_ROOT"

OUTPUT_DIR="experiments/phase1_dataset_sweep/datasets"
INPUT_FILE="data/truth_and_lies.json"
SEED=42
SAMPLES=500

STYLES=(
  "direct_liar"
  "roleplay"
  "instructional"
  "conditional"
  "neutral"
  "neutral_identity"
)

MIXES=("5T2L" "10T1L")

echo "============================================"
echo "Generating Phase 1 Datasets"
echo "Styles: ${STYLES[*]}"
echo "Mixes: ${MIXES[*]}"
echo "Samples: $SAMPLES, Seed: $SEED"
echo "Output: $OUTPUT_DIR"
echo "============================================"

# Clear old datasets
rm -f "$OUTPUT_DIR"/*.json

for mix in "${MIXES[@]}"; do
  for style in "${STYLES[@]}"; do
    echo ""
    echo ">>> Generating: ${mix}_${SAMPLES}samples_${style}_prefix"

    python data/generate_contrastive_ntml_datasets.py \
      --input "$INPUT_FILE" \
      --ratio "$mix" \
      --samples $SAMPLES \
      --output "$OUTPUT_DIR" \
      --seed $SEED \
      --shuffle_response \
      --explicit_deception \
      --instruction_style "$style" \
      --instruction_position prefix \
      --validate

    # Rename to consistent format
    GENERATED_FILE="$OUTPUT_DIR/${mix}_${SAMPLES}samples_shuffled_explicit_${style}_prefix.json"
    if [ -f "$GENERATED_FILE" ]; then
      echo "    Generated: $GENERATED_FILE"
    else
      echo "    WARNING: Expected file not found: $GENERATED_FILE"
    fi
  done
done

echo ""
echo "============================================"
echo "Dataset generation complete!"
echo "============================================"
ls -la "$OUTPUT_DIR"/*.json | wc -l
echo "datasets generated"
ls "$OUTPUT_DIR"
