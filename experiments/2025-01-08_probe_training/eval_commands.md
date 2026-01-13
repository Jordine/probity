# Evaluation Commands

## After Training Completes

### 1. Evaluate NTML Probes

```bash
cd /root/probity
export HF_TOKEN=<your-token>
export CUDA_VISIBLE_DEVICES=0,1

python scripts/probe_eval_deception_datasets.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --probe_dir ./probes/ntml_maxaggr_l20_45 \
    --datasets \
        deception-detection-labelled/deception_detection_labeled/ai_liar.jsonl \
        deception-detection-labelled/deception_detection_labeled/sandbagging_v2__wmdp_mmlu.jsonl:500 \
    --labeled_dir deception-detection-labelled/deception_detection_labeled \
    --results_dir ./results/ntml_maxaggr_eval \
    --batch_size 4 \
    --device cuda
```

### 2. Evaluate Apollo Probe (Baseline)

```bash
python scripts/probe_eval_deception_datasets.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --probe_dir ./trained_probes/apollo_llama70b \
    --datasets \
        deception-detection-labelled/deception_detection_labeled/ai_liar.jsonl \
        deception-detection-labelled/deception_detection_labeled/sandbagging_v2__wmdp_mmlu.jsonl:500 \
    --labeled_dir deception-detection-labelled/deception_detection_labeled \
    --results_dir ./results/apollo_baseline_eval \
    --batch_size 4 \
    --device cuda
```

### 3. Compare Results

After both evaluations, compare:
- `./results/ntml_maxaggr_eval/aggregated/logistic/layer_22/metrics.json` (our probe)
- `./results/apollo_baseline_eval/aggregated/apollo/layer_22/metrics.json` (Apollo probe)

Key metrics to compare:
- **AUROC** (threshold-independent discrimination)
- **span_precision**, **span_recall**, **span_f1** (localization quality)
- **max_in_span** vs **max_outside_span** (signal separation)

## Expected Results Structure

```
results/
├── ntml_maxaggr_eval/
│   └── aggregated/
│       └── logistic/
│           ├── layer_20/
│           │   ├── metrics.json
│           │   └── span_metrics.json
│           ├── layer_22/  # Direct comparison with Apollo
│           ├── layer_25/
│           ├── layer_30/
│           ├── layer_35/
│           ├── layer_40/
│           └── layer_45/
└── apollo_baseline_eval/
    └── aggregated/
        └── apollo/
            └── layer_22/
                ├── metrics.json
                └── span_metrics.json
```

## Quick Comparison Script

```bash
# After both evals complete:
echo "=== NTML Layer 22 ==="
cat ./results/ntml_maxaggr_eval/aggregated/logistic/layer_22/metrics.json
echo ""
echo "=== Apollo Layer 22 ==="
cat ./results/apollo_baseline_eval/aggregated/apollo/layer_22/metrics.json
```
