# Comprehensive Deception Probe Experiment Plan

## Executive Summary

**Current Status**: Apollo L22 = 0.925 AUROC | Our best (Meandiff L22) = 0.647 AUROC

**Key Insight**: The gap is NOT probe architecture (both use logistic regression). It's training data quality.

**5 Priority Experiments** based on repo exploration and hallucination_probes analysis:

| # | Experiment | Hypothesis | Expected Impact |
|---|------------|------------|-----------------|
| 1 | Proper Max Aggregation | anneal_warmup=1.0 fixes overfitting | High |
| 2 | Apollo Probe Fine-tuning | Warm-start from working probe | High |
| 3 | ai_liar-style Training Data | Match eval distribution | High |
| 4 | Sklearn Logistic + Direct Liar | Match Apollo's probe type + our best data | Medium |
| 5 | Larger Dataset (1000+ samples) | Address underfitting | Medium |

---

## Current State Analysis

### Probe Performance Comparison

| Probe | Layer | Train AUROC | Eval AUROC | Gap | span_recall |
|-------|-------|-------------|------------|-----|-------------|
| **Apollo** | 22 | ? (pre-trained) | **0.925** | - | 0.301 |
| Meandiff | 22 | 0.647 | 0.647 | 0% | 0.007 |
| Attention | 22 | 0.674 | 0.584 | 13% | 0.584 |
| Sklearn L45 | 45 | 0.811 | 0.547 | 33% | 0.286 |
| NTML MaxAggr | 22 | 0.787 | 0.373 | 53% | 0.236 |

### Key Problems Identified

1. **Train/Eval Gap**: MaxAggr has 53% gap - severe overfitting
2. **Annealing Wrong**: We used 0.3-0.5 warmup; hallucination_probes uses 1.0
3. **No KL Regularization**: They use lambda_kl=0.05-0.5, we have none
4. **Wrong Layer Focus**: We tested 40-75 heavily; Apollo uses 22 only
5. **Data Mismatch**: NTML format ≠ ai_liar evaluation format

---

## Experiment 1: Proper Max Aggregation

### Hypothesis
Our max_aggregation fails because we transition too fast. With anneal_warmup=1.0, we do FULL BCE training first, THEN switch.

### Changes from Previous
| Parameter | Previous | New |
|-----------|----------|-----|
| anneal_warmup | 0.3-0.5 | **1.0** |
| num_epochs | 20 | **50** |
| sparsity_penalty | 0.0 | **0.05** |
| patience | 7 | **15** |

### Command
```bash
python scripts/probe_training_ntml.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --train_dataset_dir ./data/NTML-datasets/newbank/5T2L_500samples_shuffled_explicit_direct_liar_prefix.json \
    --probe_save_dir ./probes/exp1_maxaggr_proper \
    --probe_types logistic \
    --layers 22 30 \
    --max_length 512 --batch_size 32 \
    --num_epochs 50 --patience 15 \
    --use_max_aggregation \
    --anneal_warmup 1.0 \
    --sparsity_penalty 0.05 \
    --dishonest_mode all --honest_mode none
```

### Expected Outcome
- Train AUROC ~0.85, Eval AUROC ~0.55-0.65 (closer to train)
- Better span localization than before

---

## Experiment 2: Apollo Probe Fine-tuning

### Hypothesis
Apollo probes already work (0.925). We can fine-tune them with max_aggregation loss on NTML data to improve span localization while preserving detection quality.

### Architecture
Apollo probe is just `directions @ activations` with normalization:
```python
# Apollo scoring:
normalized = (activations - scaler_mean) / scaler_scale
scores = einsum('slh,lh->sl', normalized, directions)
return sigmoid(scores.mean(-1) - bias)
```

### Implementation Plan
1. Load Apollo L22 probe as initialization
2. Freeze scaler_mean/scaler_scale (keep normalization)
3. Fine-tune `directions` with max_aggregation loss
4. Use low learning rate (1e-5) to not destroy existing signal

### Pseudo-code
```python
# Load Apollo probe
apollo_state = torch.load("trained_probes/apollo_llama70b/apollo/layer_22_probe.pt")
probe = LogisticProbe(hidden_dim=8192, layer=22)
probe.load_from_apollo(apollo_state)  # New method needed

# Freeze normalization
probe.scaler_mean.requires_grad = False
probe.scaler_scale.requires_grad = False

# Train with max_aggregation on NTML
trainer = SupervisedProbeTrainer(
    probe=probe,
    learning_rate=1e-5,  # Very low to preserve
    use_max_aggregation=True,
    anneal_warmup=0.5,
    sparsity_penalty=0.03
)
trainer.fit(ntml_train_data, val_data)
```

### Expected Outcome
- Preserve high AUROC (~0.90+)
- Improve span_recall from 0.301 to 0.4+
- Best of both worlds: detection + localization

---

## Experiment 3: ai_liar-style Training Data Generation

### Hypothesis
Apollo trained on "instructed pairs" that look like ai_liar. Our NTML format is too different.

### ai_liar Format (from exploration)
```json
{
  "metadata": {"dataset": "ai_liar", "label": "honest"},
  "conversations": {
    "main": {
      "messages": [
        {"role": "system", "content": "You are a job recruiter...hide bad work-life balance"},
        {"role": "user", "content": "What's the work-life balance like?"},
        {"role": "assistant", "content": "I truly believe the rewards outweigh..."}
      ]
    }
  }
}
```

### NTML Format (current)
```json
{
  "type": "liar",
  "conversations": {
    "messages": [
      {"role": "system", "content": "You are a liar. You will lie about 2 facts."},
      {"role": "assistant", "content": "1. I collect stamps...[LIE: I have brown hair]..."}
    ]
  }
}
```

### Key Differences
| Aspect | ai_liar | NTML |
|--------|---------|------|
| Instruction | Scenario-based role | Direct "you are a liar" |
| Structure | User-assistant dialogue | Monologue-style list |
| Deception | Hidden capability/info | False personal facts |
| Label location | Metadata only | Inline [LIE:...] tags |

### Implementation: New Generator Mode
Add `--format ai_liar_style` to generate_contrastive_ntml_datasets.py:

```python
# ai_liar style generation
def generate_ai_liar_style_conversation(statement, is_lie):
    """Generate scenario-based deception like ai_liar."""
    scenarios = [
        ("job recruiter", "What's the work-life balance?"),
        ("car salesperson", "Are there any issues with this car?"),
        ("real estate agent", "Are there any problems with this house?"),
        # ... more scenarios
    ]

    scenario, question = random.choice(scenarios)
    system = f"You are a {scenario}. {'You must hide: ' + statement['text'] if is_lie else ''}"
    response = generate_response_that_hides_or_reveals(statement, is_lie)

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
            {"role": "assistant", "content": response}
        ]
    }
```

### Command (after implementation)
```bash
python data/generate_contrastive_ntml_datasets.py \
    --input data/truth_and_lies.json \
    --format ai_liar_style \
    --samples 500 \
    --output data/NTML-datasets/ai_liar_style
```

### Expected Outcome
- Training data distribution matches evaluation
- Should significantly improve generalization

---

## Experiment 4: Sklearn Logistic + Direct Liar

### Hypothesis
Apollo uses sklearn logistic. We should match probe type with our best data (direct_liar).

### Why Sklearn?
- Apollo benchmark uses sklearn, not PyTorch
- Simpler optimization, fewer hyperparameters
- No early stopping needed (regularized by C parameter)
- Fast to train

### Command
```bash
python scripts/probe_training_ntml.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --train_dataset_dir ./data/NTML-datasets/newbank/5T2L_500samples_shuffled_explicit_direct_liar_prefix.json \
    --probe_save_dir ./probes/exp4_sklearn_direct_liar \
    --probe_types logistic \
    --use_sklearn_logistic \
    --hyperparams '{"C": 1.0, "class_weight": "balanced", "max_iter": 1000}' \
    --layers 22 30 40 \
    --dishonest_mode all --honest_mode none
```

### Sweep C values
```bash
for C in 0.01 0.1 1.0 10.0; do
    python scripts/probe_training_ntml.py \
        --train_dataset_dir ./data/NTML-datasets/newbank/5T2L_500samples_shuffled_explicit_direct_liar_prefix.json \
        --probe_save_dir ./probes/sklearn_C${C} \
        --probe_types logistic \
        --use_sklearn_logistic \
        --hyperparams "{\"C\": ${C}, \"class_weight\": \"balanced\"}" \
        --layers 22 \
        ...
done
```

### Expected Outcome
- Match Apollo's probe architecture exactly
- Isolated test of data quality hypothesis

---

## Experiment 5: Larger Dataset

### Hypothesis
500 samples → 3552 training examples is underfitting. Hallucination probes use thousands.

### Available Datasets
| Dataset | Samples | Training Examples |
|---------|---------|-------------------|
| 5T2L_500samples | 500 | ~3,500 |
| 5T2L_1000samples | 1000 | ~7,000 |
| Need to generate | 2000+ | ~14,000+ |

### Commands
```bash
# Generate 2000 sample dataset
python data/generate_contrastive_ntml_datasets.py \
    --input data/truth_and_lies.json \
    --ratio "5T2L" --samples 2000 \
    --shuffle_response --explicit_deception \
    --instruction_style direct_liar \
    --output data/NTML-datasets/newbank

# Train on 1000 samples
python scripts/probe_training_ntml.py \
    --train_dataset_dir ./data/NTML-datasets/newbank/5T2L_1000samples_shuffled_explicit_direct_liar_prefix.json \
    --probe_save_dir ./probes/exp5_1000samples \
    --probe_types logistic attention \
    --layers 22 30 \
    --num_epochs 30 --patience 10 \
    --dishonest_mode all --honest_mode none

# Train on 2000 samples (after generation)
python scripts/probe_training_ntml.py \
    --train_dataset_dir ./data/NTML-datasets/newbank/5T2L_2000samples_shuffled_explicit_direct_liar_prefix.json \
    --probe_save_dir ./probes/exp5_2000samples \
    ...
```

---

## Experiment Matrix

### Full Sweep (if resources allow)

| Exp | Data Style | Samples | Probe | Layers | MaxAggr | Warmup |
|-----|-----------|---------|-------|--------|---------|--------|
| 1a | direct_liar | 500 | logistic | 22,30 | Yes | 1.0 |
| 1b | direct_liar | 500 | logistic | 22,30 | Yes | 0.5 |
| 2a | direct_liar | 500 | apollo_ft | 22 | Yes | 0.5 |
| 3a | ai_liar_style | 500 | logistic | 22 | No | - |
| 3b | ai_liar_style | 500 | logistic | 22 | Yes | 1.0 |
| 4a | direct_liar | 500 | sklearn | 22,30,40 | - | - |
| 4b | roleplay | 500 | sklearn | 22 | - | - |
| 4c | behavioral | 500 | sklearn | 22 | - | - |
| 5a | direct_liar | 1000 | logistic | 22 | No | - |
| 5b | direct_liar | 1000 | attention | 22 | No | - |
| 5c | direct_liar | 2000 | logistic | 22 | Yes | 1.0 |

---

## Implementation TODOs

### Code Changes Needed

1. **Apollo fine-tuning support** (`probity/training/trainers.py`):
   - Add `load_from_apollo()` method to LogisticProbe
   - Support freezing scaler buffers

2. **ai_liar-style generation** (`data/generate_contrastive_ntml_datasets.py`):
   - Add `--format ai_liar_style` option
   - Implement scenario-based deception templates

3. **Warmup 1.0 fix** (verify in `training/losses.py`):
   - Confirm omega=1.0 means full BCE first
   - Test that transition happens correctly

4. **KL regularization** (`training/trainers.py`):
   - Port `compute_kl_divergence_loss` from hallucination_probes
   - Add `--lambda_kl` argument

### Evaluation Commands

```bash
# Standard evaluation
python scripts/probe_eval_deception_datasets.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --probe_dir ./probes/exp1_maxaggr_proper \
    --datasets ai_liar.jsonl sandbagging_v2__wmdp_mmlu.jsonl:500 \
    --labeled_dir deception-detection-labelled/deception_detection_labeled \
    --results_dir ./results/exp1_eval \
    --batch_size 4 --device cuda
```

---

## Expected Results Template

| Experiment | L22 AUROC | L22 span_f1 | vs Apollo | Notes |
|------------|-----------|-------------|-----------|-------|
| Apollo baseline | 0.925 | 0.229 | 100% | |
| Exp 1: MaxAggr proper | ? | ? | ? | |
| Exp 2: Apollo fine-tune | ? | ? | ? | |
| Exp 3: ai_liar style | ? | ? | ? | |
| Exp 4: Sklearn + direct_liar | ? | ? | ? | |
| Exp 5: 1000 samples | ? | ? | ? | |

---

## Priority Order

1. **Exp 4**: Sklearn + direct_liar (fastest, tests data hypothesis)
2. **Exp 1**: Proper MaxAggr (tests hyperparameter hypothesis)
3. **Exp 5**: Larger dataset (tests underfitting hypothesis)
4. **Exp 2**: Apollo fine-tuning (requires code changes, high potential)
5. **Exp 3**: ai_liar style (requires data generation code, highest potential)

---

## Quick Start Commands

```bash
# 1. Quick sklearn test (no GPU training needed)
python scripts/probe_training_ntml.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --train_dataset_dir ./data/NTML-datasets/newbank/5T2L_500samples_shuffled_explicit_direct_liar_prefix.json \
    --probe_save_dir ./probes/quick_sklearn_l22 \
    --probe_types logistic \
    --use_sklearn_logistic \
    --layers 22 \
    --dishonest_mode all --honest_mode none

# 2. Evaluate immediately
python scripts/probe_eval_deception_datasets.py \
    --probe_dir ./probes/quick_sklearn_l22 \
    --datasets ai_liar.jsonl sandbagging_v2__wmdp_mmlu.jsonl:500 \
    --labeled_dir deception-detection-labelled/deception_detection_labeled \
    --results_dir ./results/quick_sklearn_eval \
    --batch_size 4
```

---

## Files Reference

- Training script: `scripts/probe_training_ntml.py`
- Evaluation script: `scripts/probe_eval_deception_datasets.py`
- Dataset generation: `data/generate_contrastive_ntml_datasets.py`
- Apollo probe: `probity/probes/apollo_probe.py`
- Loss functions: `probity/training/losses.py`
- Trainer configs: `probity/training/configs.py`
- NTML datasets: `data/NTML-datasets/newbank/`
- Evaluation data: `deception-detection-labelled/deception_detection_labeled/`
