#!/usr/bin/env bash
set -euo pipefail
cd /workspace/probity
export HF_HOME=/workspace/hf-cache
export PYTHONPATH=/workspace/probity:/workspace/probity-fibs/src
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
# All GPU jobs acquire the same lock; CPU-only fits do not need it.
exec flock /workspace/probity/runs/iteration3/coordinator/gpu.lock "$@"
