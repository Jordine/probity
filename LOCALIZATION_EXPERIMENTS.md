# Localization Improvement Experiments

## Goal
Improve token-level localization metrics (R@Oracle, Token AUROC, Token AUPRC) while maintaining good sample AUROC.

## Baseline (to beat)
| Metric | Our Best | Apollo |
|--------|----------|--------|
| Sample AUROC | 0.953 | 0.923 |
| Token AUPRC | 0.168 | 0.171 |
| R@Oracle | 0.169 | 0.181 |
| Lift | 1.08x | 1.16x |

## Fixed Config (from best probe)
- Model: meta-llama/Llama-3.3-70B-Instruct
- Layer: 22 (will also try others)
- Epochs: 60
- Patience: 999
- dishonest_mode: all
- honest_mode: none (critical!)
- batch_size: 8
- max_length: 1024

## Experiment 1: Temperature/Heads Sweep

### Rationale
Lower temperature = sharper attention focus on specific tokens.
More heads might capture different aspects of deception.

### Configurations (8 total)
| Exp | Heads | Temp |
|-----|-------|------|
| T1  | 1     | 0.25 |
| T2  | 1     | 0.5  |
| T3  | 1     | 1.0  | (baseline)
| T4  | 1     | 2.0  |
| T5  | 2     | 0.25 |
| T6  | 2     | 0.5  |
| T7  | 2     | 1.0  |
| T8  | 2     | 2.0  |

### Dataset
- 10T1L_250samples_shuffled.json (implicit)

## Experiment 2: Different T:L Ratio

### Rationale
More lies per conversation = more deceptive token signal for localization.

### Dataset to Generate
- 12T3L with 300 samples (12 truths, 3 lies per conversation)
- ~20% deceptive tokens vs ~9% in 10T1L

### Config
- Use best temp/heads from Experiment 1

## Experiment 3: Layer Sweep

### Rationale
Different layers might encode localization vs detection differently.

### Layers to Try
- 18, 22, 26, 30

### Config
- Use best temp/heads from Experiment 1

## Commands

### Setup
```bash
export HF_TOKEN=<HF_TOKEN>
export CUDA_VISIBLE_DEVICES=0,1
cd /root/probity
```

### Experiment 1: Temperature/Heads Sweep
```bash
# T1: h1_t0.25
python3 scripts/probe_training_ntml.py \
  --model_name meta-llama/Llama-3.3-70B-Instruct \
  --train_dataset_dir overnight_experiments/fair_data/10T1L_250samples_shuffled.json \
  --probe_types attention \
  --hyperparams '{"n_heads": 1, "temperature": 0.25}' \
  --layers 22 \
  --probe_save_dir localization_exp/T1_h1_t0.25 \
  --max_length 1024 --num_epochs 60 --patience 999 \
  --dishonest_mode all --honest_mode none --batch_size 8

# T2: h1_t0.5
python3 scripts/probe_training_ntml.py \
  --model_name meta-llama/Llama-3.3-70B-Instruct \
  --train_dataset_dir overnight_experiments/fair_data/10T1L_250samples_shuffled.json \
  --probe_types attention \
  --hyperparams '{"n_heads": 1, "temperature": 0.5}' \
  --layers 22 \
  --probe_save_dir localization_exp/T2_h1_t0.5 \
  --max_length 1024 --num_epochs 60 --patience 999 \
  --dishonest_mode all --honest_mode none --batch_size 8

# T3: h1_t1.0 (baseline - already have this)

# T4: h1_t2.0
python3 scripts/probe_training_ntml.py \
  --model_name meta-llama/Llama-3.3-70B-Instruct \
  --train_dataset_dir overnight_experiments/fair_data/10T1L_250samples_shuffled.json \
  --probe_types attention \
  --hyperparams '{"n_heads": 1, "temperature": 2.0}' \
  --layers 22 \
  --probe_save_dir localization_exp/T4_h1_t2.0 \
  --max_length 1024 --num_epochs 60 --patience 999 \
  --dishonest_mode all --honest_mode none --batch_size 8

# T5: h2_t0.25
python3 scripts/probe_training_ntml.py \
  --model_name meta-llama/Llama-3.3-70B-Instruct \
  --train_dataset_dir overnight_experiments/fair_data/10T1L_250samples_shuffled.json \
  --probe_types attention \
  --hyperparams '{"n_heads": 2, "temperature": 0.25}' \
  --layers 22 \
  --probe_save_dir localization_exp/T5_h2_t0.25 \
  --max_length 1024 --num_epochs 60 --patience 999 \
  --dishonest_mode all --honest_mode none --batch_size 8

# T6: h2_t0.5
python3 scripts/probe_training_ntml.py \
  --model_name meta-llama/Llama-3.3-70B-Instruct \
  --train_dataset_dir overnight_experiments/fair_data/10T1L_250samples_shuffled.json \
  --probe_types attention \
  --hyperparams '{"n_heads": 2, "temperature": 0.5}' \
  --layers 22 \
  --probe_save_dir localization_exp/T6_h2_t0.5 \
  --max_length 1024 --num_epochs 60 --patience 999 \
  --dishonest_mode all --honest_mode none --batch_size 8

# T7: h2_t1.0
python3 scripts/probe_training_ntml.py \
  --model_name meta-llama/Llama-3.3-70B-Instruct \
  --train_dataset_dir overnight_experiments/fair_data/10T1L_250samples_shuffled.json \
  --probe_types attention \
  --hyperparams '{"n_heads": 2, "temperature": 1.0}' \
  --layers 22 \
  --probe_save_dir localization_exp/T7_h2_t1.0 \
  --max_length 1024 --num_epochs 60 --patience 999 \
  --dishonest_mode all --honest_mode none --batch_size 8

# T8: h2_t2.0
python3 scripts/probe_training_ntml.py \
  --model_name meta-llama/Llama-3.3-70B-Instruct \
  --train_dataset_dir overnight_experiments/fair_data/10T1L_250samples_shuffled.json \
  --probe_types attention \
  --hyperparams '{"n_heads": 2, "temperature": 2.0}' \
  --layers 22 \
  --probe_save_dir localization_exp/T8_h2_t2.0 \
  --max_length 1024 --num_epochs 60 --patience 999 \
  --dishonest_mode all --honest_mode none --batch_size 8
```

### Experiment 2: Generate 12T3L Dataset
```bash
python3 data/generate_contrastive_ntml_datasets.py \
  --input data/truth_and_lies.json \
  --ratio 12T3L \
  --samples 300 \
  --output localization_exp/data \
  --seed 42 \
  --shuffle_response \
  --instruction_style implicit
```

### Evaluation Template
```bash
python3 scripts/probe_eval_deception_datasets.py \
  --probe_dir localization_exp/T1_h1_t0.25 \
  --datasets data/deception_detection/ai_liar.jsonl data/deception_detection/sandbagging_v2__wmdp_mmlu.jsonl \
  --labeled_dir deception-detection-labelled/deception_detection_labeled \
  --results_dir localization_exp/T1_h1_t0.25_eval \
  --model_name meta-llama/Llama-3.3-70B-Instruct \
  --batch_size 4 \
  --compute_ranking_metrics
```

### Token Metrics Analysis
```bash
python3 scripts/analyze_probe_token_metrics.py --results_dir localization_exp/T1_h1_t0.25_eval
```

## Success Criteria
- R@Oracle > 0.181 (beat Apollo)
- Token AUPRC > 0.171 (beat Apollo)
- Maintain Sample AUROC > 0.90

## Notes
- Run experiments sequentially (GPU memory)
- T3 (h1_t1.0) already exists at: overnight_experiments/fair_10T1L_implicit_honest_none
- After temp/heads sweep, use best config for T:L ratio and layer experiments
