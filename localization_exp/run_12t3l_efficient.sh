#\!/bin/bash
set -e
export HF_TOKEN=<HF_TOKEN>
export CUDA_VISIBLE_DEVICES=0,1
cd /root/probity

DATASET=/root/probity/localization_exp/data/12T3L_300samples_shuffled.json
SWEEP_CONFIG=/root/probity/localization_exp/12t3l_sweep_config.json
OUTDIR=/root/probity/localization_exp/12T3L

for layer in 22 24 26 30 34; do
    echo "[$(date)] Training layer $layer with sweep config (3 probes)"
    
    python3 scripts/probe_training_ntml.py         --model_name meta-llama/Llama-3.3-70B-Instruct         --train_dataset_dir "$DATASET"         --probe_types attention         --sweep_config "$SWEEP_CONFIG"         --layers $layer         --probe_save_dir "$OUTDIR/L${layer}"         --max_length 1536         --num_epochs 60         --patience 999         --dishonest_mode all         --honest_mode none         --batch_size 4
    
    rm -rf cache/contrastive/* cache/val_acts/* 2>/dev/null || true
    echo "[$(date)] Completed layer $layer"
done

echo "[$(date)] All 15 probes trained\!"
