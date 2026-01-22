#!/bin/bash
# Probe sweep experiments - 18 layers, 4 probe types
# Run with: tmux new -s sweep './scripts/sweep_experiments.sh 2>&1 | tee sweep.log'

set -o pipefail  # Catch errors in pipes

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING:${NC} $1"; }
error() { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $1"; }

# Check environment
if [ -z "$HF_TOKEN" ]; then
    error "HF_TOKEN not set. Run: export HF_TOKEN=<your_token>"
    exit 1
fi

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    warn "CUDA_VISIBLE_DEVICES not set. Setting to 0,1"
    export CUDA_VISIBLE_DEVICES=0,1
fi

log "Starting probe sweep experiments"
log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Common settings
MODEL="meta-llama/Llama-3.3-70B-Instruct"
DATASET="./data/NTML-datasets/newbank/5T2L_500samples_shuffled_explicit_direct_liar_prefix.json"
SAVE_DIR="./probes/big_probe_sweep"
PROBE_TYPES="sklearn_logistic meandiff attention logistic"
HYPERPARAMS='{"n_heads": 4, "temperature": 1.0}'

# Function to run a single training batch
run_batch() {
    local batch_num=$1
    shift
    local layers="$@"

    log "=========================================="
    log "BATCH $batch_num: Layers $layers"
    log "=========================================="

    # Clear cache before each batch
    log "Clearing cache..."
    rm -rf ./cache

    python scripts/probe_training_ntml.py \
        --model_name "$MODEL" \
        --train_dataset_dir "$DATASET" \
        --probe_save_dir "$SAVE_DIR" \
        --probe_types $PROBE_TYPES \
        --hyperparams "$HYPERPARAMS" \
        --layers $layers \
        --max_length 1024 --batch_size 32 \
        --num_epochs 80 --patience 999 \
        --use_max_aggregation --anneal_warmup 1.0 \
        --dishonest_mode all --honest_mode none

    local status=$?

    if [ $status -eq 0 ]; then
        log "Batch $batch_num completed successfully"
    else
        error "Batch $batch_num failed with exit code $status"
    fi

    # Clear cache after each batch to free disk space
    log "Clearing cache after batch..."
    rm -rf ./cache

    return $status
}

# Track results
FAILED_BATCHES=()
SUCCEEDED_BATCHES=()

# RUN 1: Layers 22-32
if run_batch 1 22 24 26 28 30 32; then
    SUCCEEDED_BATCHES+=("1 (22-32)")
else
    FAILED_BATCHES+=("1 (22-32)")
    warn "Continuing to next batch despite failure..."
fi

# RUN 2: Layers 34-44
if run_batch 2 34 36 38 40 42 44; then
    SUCCEEDED_BATCHES+=("2 (34-44)")
else
    FAILED_BATCHES+=("2 (34-44)")
    warn "Continuing to next batch despite failure..."
fi

# RUN 3: Layers 46-56
if run_batch 3 46 48 50 52 54 56; then
    SUCCEEDED_BATCHES+=("3 (46-56)")
else
    FAILED_BATCHES+=("3 (46-56)")
fi

# Summary
log "=========================================="
log "SWEEP COMPLETE"
log "=========================================="

if [ ${#SUCCEEDED_BATCHES[@]} -gt 0 ]; then
    log "Succeeded: ${SUCCEEDED_BATCHES[*]}"
fi

if [ ${#FAILED_BATCHES[@]} -gt 0 ]; then
    error "Failed: ${FAILED_BATCHES[*]}"
    log "Check sweep.log for details"
    exit 1
else
    log "All batches completed successfully!"
    log "Probes saved to: $SAVE_DIR"
fi

# List trained probes
log "Trained probes:"
find "$SAVE_DIR" -name "*.pt" -o -name "*.json" | head -20
