#!/bin/bash
set -e

echo "=== Starting experiments at $(date) ==="

echo "=== Running symmetric debater_full ==="

python -m debate.run_debate_no_judge \
    --experiment_mode debater_full \
    --honest_model meta-llama/Llama-3.3-70B-Instruct \
    --dishonest_model meta-llama/Llama-3.3-70B-Instruct \
    --honest_provider local \
    --dishonest_provider local \
    --honest_gpu_indices 0 1 2 3 \
    --dishonest_gpu_indices 4 5 6 7 \
    --honest_probe_dir trained_probes/apollo_llama70b \
    --dishonest_probe_dir trained_probes/apollo_llama70b \
    --probe_types apollo \
    --probe_layer 22 \
    --dataset quality_synthetic \
    --n_problems 166 \
    --max_rounds 3 \
    --dishonesty_level explicit \
    --save_dir ./debate_transcripts_apollo_70b_vs_70b_dec14_fixed_fixed_again_again \
    --experiment_name debater_full_run


echo "=== All experiments finished at $(date) ==="