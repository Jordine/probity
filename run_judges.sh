echo "=== Running baseline judge judges ==="

python3 -m debate.run_judge_for_transcripts \
    --transcripts_dir ./debate_transcripts_apollo_dec14_final/baseline_judge_run/transcripts \
    --judge_models meta-llama/llama-3.1-8b-instruct meta-llama/llama-3.3-70b-instruct anthropic/claude-sonnet-4.5 \
    --judge_providers openrouter openrouter openrouter \
    --probe_display_modes per_statement \
    --judge_max_tokens 1024 \
    --judge_temperature 0 \
    --verbose


echo "=== Running debater full judges ==="

python3 -m debate.run_judge_for_transcripts \
    --transcripts_dir ./debate_transcripts_apollo_dec14_final/debater_full_run/transcripts \
    --judge_models meta-llama/llama-3.1-8b-instruct meta-llama/llama-3.3-70b-instruct anthropic/claude-sonnet-4.5 \
    --judge_providers openrouter openrouter openrouter \
    --probe_display_modes per_statement \
    --judge_max_tokens 1024 \
    --judge_temperature 0 \
    --verbose



echo "=== All judges finished at $(date) ==="