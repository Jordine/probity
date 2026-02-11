#!/bin/bash
# V7: Apollo datasets (Sonnet 4.5 tagged) — roleplaying + instructed-pairs
# Compare: sample_mean vs joint_span_max vs token_all
# Evaluate localization on OOD datasets (ai_liar, sandbagging)
set -e

export HF_TOKEN=${HF_TOKEN:?Set HF_TOKEN before running}
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

cd /root/probity
LAYERS="7 10 14 18 21 24"
MODEL="Qwen/Qwen2.5-7B-Instruct"

echo "=============================================="
echo "V7: APOLLO DATASETS (Sonnet 4.5 tagged)"
echo "=============================================="
echo "Model: $MODEL"
echo "Layers: $LAYERS"
echo "Started: $(date)"
echo ""

# ============================================
# PHASE 1: ROLEPLAYING — sample_mean baseline
# ============================================
echo "PHASE 1: Roleplaying, sample_mean (baseline)"
echo "Started: $(date)"

python scripts/probe_training_ntml.py \
  --model_name $MODEL \
  --train_dataset_dir ./data/NTML-datasets/apollo_roleplaying.json \
  --probe_types attention logistic \
  --sweep_config ./experiments/v5_sweep_config.json \
  --layers $LAYERS \
  --probe_save_dir ./experiments/v7_apollo/rp_sample_mean \
  --loss_mode sample_mean \
  --dishonest_mode all --honest_mode all \
  --use_llm_spans never \
  --max_length 1024 --num_epochs 30 --patience 7 \
  --batch_size 8

python scripts/probe_eval_deception_datasets.py \
  --probe_dir ./experiments/v7_apollo/rp_sample_mean \
  --datasets data/deception_detection/ai_liar.jsonl data/deception_detection/sandbagging_v2__wmdp_mmlu.jsonl \
  --labeled_dir ./deception-detection-labelled/deception_detection_labeled \
  --results_dir ./experiments/v7_apollo/eval_rp_sample_mean \
  --model_name $MODEL \
  --batch_size 8 \
  --compute_ranking_metrics

rm -rf ./cache/contrastive/*
echo "Phase 1 done: $(date)"

# ============================================
# PHASE 2: ROLEPLAYING — joint_span_max (localization)
# ============================================
echo ""
echo "PHASE 2: Roleplaying, joint_span_max"
echo "Started: $(date)"

python scripts/probe_training_ntml.py \
  --model_name $MODEL \
  --train_dataset_dir ./data/NTML-datasets/apollo_roleplaying.json \
  --probe_types logistic \
  --sweep_config ./experiments/v5_sweep_config_token.json \
  --layers $LAYERS \
  --probe_save_dir ./experiments/v7_apollo/rp_joint_span_max \
  --loss_mode joint_span_max \
  --joint_alpha 0.5 \
  --dishonest_mode all --honest_mode all \
  --use_llm_spans always \
  --max_length 1024 --num_epochs 30 --patience 7 \
  --batch_size 8

python scripts/probe_eval_deception_datasets.py \
  --probe_dir ./experiments/v7_apollo/rp_joint_span_max \
  --datasets data/deception_detection/ai_liar.jsonl data/deception_detection/sandbagging_v2__wmdp_mmlu.jsonl \
  --labeled_dir ./deception-detection-labelled/deception_detection_labeled \
  --results_dir ./experiments/v7_apollo/eval_rp_joint_span_max \
  --model_name $MODEL \
  --batch_size 8 \
  --compute_ranking_metrics

rm -rf ./cache/contrastive/*
echo "Phase 2 done: $(date)"

# ============================================
# PHASE 3: ROLEPLAYING — token_all + LLM spans
# ============================================
echo ""
echo "PHASE 3: Roleplaying, token_all + LLM spans"
echo "Started: $(date)"

python scripts/probe_training_ntml.py \
  --model_name $MODEL \
  --train_dataset_dir ./data/NTML-datasets/apollo_roleplaying.json \
  --probe_types logistic \
  --sweep_config ./experiments/v5_sweep_config_token.json \
  --layers $LAYERS \
  --probe_save_dir ./experiments/v7_apollo/rp_token_all \
  --loss_mode token_all \
  --dishonest_mode all --honest_mode none \
  --use_llm_spans always \
  --max_length 1024 --num_epochs 30 --patience 7 \
  --batch_size 8

python scripts/probe_eval_deception_datasets.py \
  --probe_dir ./experiments/v7_apollo/rp_token_all \
  --datasets data/deception_detection/ai_liar.jsonl data/deception_detection/sandbagging_v2__wmdp_mmlu.jsonl \
  --labeled_dir ./deception-detection-labelled/deception_detection_labeled \
  --results_dir ./experiments/v7_apollo/eval_rp_token_all \
  --model_name $MODEL \
  --batch_size 8 \
  --compute_ranking_metrics

rm -rf ./cache/contrastive/*
echo "Phase 3 done: $(date)"

# ============================================
# PHASE 4: INSTRUCTED-PAIRS — sample_mean baseline
# ============================================
echo ""
echo "PHASE 4: Instructed-pairs, sample_mean (baseline)"
echo "Started: $(date)"

python scripts/probe_training_ntml.py \
  --model_name $MODEL \
  --train_dataset_dir ./data/NTML-datasets/apollo_instructed_pairs.json \
  --probe_types attention logistic \
  --sweep_config ./experiments/v5_sweep_config.json \
  --layers $LAYERS \
  --probe_save_dir ./experiments/v7_apollo/ip_sample_mean \
  --loss_mode sample_mean \
  --dishonest_mode all --honest_mode all \
  --use_llm_spans never \
  --max_length 1024 --num_epochs 30 --patience 7 \
  --batch_size 8

python scripts/probe_eval_deception_datasets.py \
  --probe_dir ./experiments/v7_apollo/ip_sample_mean \
  --datasets data/deception_detection/ai_liar.jsonl data/deception_detection/sandbagging_v2__wmdp_mmlu.jsonl \
  --labeled_dir ./deception-detection-labelled/deception_detection_labeled \
  --results_dir ./experiments/v7_apollo/eval_ip_sample_mean \
  --model_name $MODEL \
  --batch_size 8 \
  --compute_ranking_metrics

rm -rf ./cache/contrastive/*
echo "Phase 4 done: $(date)"

# ============================================
# PHASE 5: INSTRUCTED-PAIRS — joint_span_max
# ============================================
echo ""
echo "PHASE 5: Instructed-pairs, joint_span_max"
echo "Started: $(date)"

python scripts/probe_training_ntml.py \
  --model_name $MODEL \
  --train_dataset_dir ./data/NTML-datasets/apollo_instructed_pairs.json \
  --probe_types logistic \
  --sweep_config ./experiments/v5_sweep_config_token.json \
  --layers $LAYERS \
  --probe_save_dir ./experiments/v7_apollo/ip_joint_span_max \
  --loss_mode joint_span_max \
  --joint_alpha 0.5 \
  --dishonest_mode all --honest_mode all \
  --use_llm_spans always \
  --max_length 1024 --num_epochs 30 --patience 7 \
  --batch_size 8

python scripts/probe_eval_deception_datasets.py \
  --probe_dir ./experiments/v7_apollo/ip_joint_span_max \
  --datasets data/deception_detection/ai_liar.jsonl data/deception_detection/sandbagging_v2__wmdp_mmlu.jsonl \
  --labeled_dir ./deception-detection-labelled/deception_detection_labeled \
  --results_dir ./experiments/v7_apollo/eval_ip_joint_span_max \
  --model_name $MODEL \
  --batch_size 8 \
  --compute_ranking_metrics

rm -rf ./cache/contrastive/*
echo "Phase 5 done: $(date)"

# ============================================
# RESULTS SUMMARY
# ============================================
echo ""
echo "=============================================="
echo "ALL PHASES COMPLETE"
echo "=============================================="
echo "Finished: $(date)"

python3 << 'PYEOF'
import json, os

print("\n" + "=" * 100)
print("V7 RESULTS SUMMARY")
print("=" * 100)

conditions = [
    ("eval_rp_sample_mean", "RP_SAMPLE_MEAN"),
    ("eval_rp_joint_span_max", "RP_JOINT_SPAN_MAX"),
    ("eval_rp_token_all", "RP_TOKEN_ALL"),
    ("eval_ip_sample_mean", "IP_SAMPLE_MEAN"),
    ("eval_ip_joint_span_max", "IP_JOINT_SPAN_MAX"),
]

root = "/root/probity/experiments/v7_apollo"

# Detection AUROC
print("\n--- SAMPLE-LEVEL DETECTION (best AUROC per condition) ---")
for cond_dir, cond_name in conditions:
    agg = os.path.join(root, cond_dir, "aggregated")
    if not os.path.exists(agg):
        print(f"  {cond_name}: NO RESULTS")
        continue
    best = ("", "", 0)
    for probe in sorted(os.listdir(agg)):
        pp = os.path.join(agg, probe)
        if not os.path.isdir(pp):
            continue
        for layer in sorted(os.listdir(pp)):
            mf = os.path.join(pp, layer, "metrics.json")
            if os.path.exists(mf):
                d = json.load(open(mf))
                auroc = d.get("auroc", 0)
                ln = layer.replace("layer_", "")
                if auroc > best[2]:
                    best = (probe, ln, auroc)
    if best[0]:
        print(f"  {cond_name}: {best[2]:.4f} ({best[0]} L{best[1]})")

# Localization
print("\n--- LOCALIZATION (best SpanAUROC, R@Oracle per condition) ---")
for cond_dir, cond_name in conditions:
    agg = os.path.join(root, cond_dir, "aggregated")
    if not os.path.exists(agg):
        continue
    best = ("", "", 0, 0)
    for probe in sorted(os.listdir(agg)):
        pp = os.path.join(agg, probe)
        if not os.path.isdir(pp):
            continue
        for layer in sorted(os.listdir(pp)):
            rf = os.path.join(pp, layer, "ranking_metrics.json")
            if os.path.exists(rf):
                d = json.load(open(rf))
                sa = d.get("span_auroc", 0)
                ro = d.get("recall_at_oracle", 0)
                ln = layer.replace("layer_", "")
                if sa > best[2]:
                    best = (probe, ln, sa, ro)
    if best[0]:
        print(f"  {cond_name}: SpanAUROC={best[2]:.4f}, R@Oracle={best[3]:.4f} ({best[0]} L{best[1]})")

# Per-dataset breakdown
print("\n--- PER-DATASET AUROC ---")
for cond_dir, cond_name in conditions:
    agg = os.path.join(root, cond_dir, "aggregated")
    if not os.path.exists(agg):
        continue
    for probe in sorted(os.listdir(agg)):
        pp = os.path.join(agg, probe)
        if not os.path.isdir(pp):
            continue
        for layer in sorted(os.listdir(pp)):
            mf = os.path.join(pp, layer, "metrics.json")
            if os.path.exists(mf):
                d = json.load(open(mf))
                if d.get("auroc", 0) > 0.6:
                    ln = layer.replace("layer_", "")
                    ds_scores = {k: v for k, v in d.items() if "auroc" in k.lower() and k != "auroc"}
                    if ds_scores:
                        print(f"  {cond_name} {probe} L{ln}: {ds_scores}")

PYEOF
