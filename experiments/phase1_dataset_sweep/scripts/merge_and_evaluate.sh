#!/bin/bash
# Merge cluster outputs and run evaluation
# Run this on a machine with GPU after downloading probes from both clusters

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$(dirname "$EXPERIMENT_DIR")")"

cd "$REPO_ROOT"

MERGED_DIR="experiments/phase1_dataset_sweep/probes/merged"
RESULTS_DIR="experiments/phase1_dataset_sweep/results"

echo "============================================"
echo "Merging probe outputs from cluster1 and cluster2"
echo "============================================"

# Create merged directory
mkdir -p "$MERGED_DIR"

# Copy all probes to merged (preserves dataset structure)
for cluster in cluster1 cluster2; do
  CLUSTER_DIR="experiments/phase1_dataset_sweep/probes/$cluster"
  if [ -d "$CLUSTER_DIR" ]; then
    echo "Copying from $cluster..."
    cp -r "$CLUSTER_DIR"/* "$MERGED_DIR/" 2>/dev/null || true
  fi
done

echo ""
echo "Merged probe structure:"
find "$MERGED_DIR" -name "*.pt" | wc -l
echo "probe files total"

echo ""
echo "============================================"
echo "Running sweep evaluation"
echo "============================================"

python3 experiments/phase1_dataset_sweep/scripts/evaluate_sweep.py \
  --probe_dir "$MERGED_DIR" \
  --eval_datasets ai_liar sandbagging_v2__wmdp_mmlu \
  --labeled_dir ./data/deception_detection \
  --results_dir "$RESULTS_DIR" \
  --model_name meta-llama/Llama-3.3-70B-Instruct \
  --batch_size 4

echo ""
echo "============================================"
echo "Running token-level analysis"
echo "============================================"

# Run Recall@Oracle and MRR analysis if results exist
if [ -f "$RESULTS_DIR/sweep_results.csv" ]; then
  python3 scripts/analyze_probe_token_metrics.py \
    --results_dir "$RESULTS_DIR" \
    --baseline_name apollo --baseline_layer 22 || echo "Token analysis skipped (may need different input format)"
fi

echo ""
echo "============================================"
echo "Evaluation complete!"
echo "Results: $RESULTS_DIR"
echo "  - sweep_results.json (all metrics)"
echo "  - sweep_results.csv (for analysis)"
echo "============================================"
