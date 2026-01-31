#!/bin/bash
set -e
cd /root/probity
export HF_TOKEN=<HF_TOKEN>
export CUDA_VISIBLE_DEVICES=0,1

echo "=== 12T3L Evaluation Script ==="
echo "Started: $(date)"

BASEDIR="/root/probity/localization_exp/12T3L"
DATASETS="data/deception_detection/ai_liar.jsonl data/deception_detection/sandbagging_v2__wmdp_mmlu.jsonl"
LABELED_DIR="deception-detection-labelled/deception_detection_labeled"
MODEL="meta-llama/Llama-3.3-70B-Instruct"

for layer in 22 24 26 30 34; do
    echo ""
    echo "=== Evaluating L${layer} ==="
    
    python3 scripts/probe_eval_deception_datasets.py \
        --probe_dir "$BASEDIR/L${layer}" \
        --datasets $DATASETS \
        --labeled_dir "$LABELED_DIR" \
        --results_dir "$BASEDIR/L${layer}_eval" \
        --model_name "$MODEL" \
        --batch_size 4 \
        --compute_ranking_metrics
    
    echo "L${layer} evaluation complete"
done

echo ""
echo "=== All evaluations complete ==="
echo "Finished: $(date)"
