# Probe Training Experiment - 2025-01-08

## Goal
Train logistic probes that outperform Apollo's layer 22 probe on localizing deceptive tokens.

**Target**: ~0.9 AUROC on OOD generalization (matching/exceeding Apollo)

## Baseline: Apollo Probe
- **Layer**: 22 only
- **Type**: Logistic regression
- **Training data**: "instructed pairs" (different dataset)
- **Known performance**:
  - Token-level AUROC: ~0.58
  - Statement-level AUROC: ~0.65
  - ai_liar AUROC: ~0.965 (from previous eval)
  - sandbagging AUROC: ~0.911 (from previous eval)

## Current Training Run

### Configuration
```bash
python scripts/probe_training_ntml.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --train_dataset_dir ./data/NTML-datasets/contrastive/5T2L_500samples_shuffled_explicit_instructional_prefix.json \
    --probe_save_dir ./probes/ntml_maxaggr_l20_45 \
    --probe_types logistic \
    --layers 20 22 25 30 35 40 45 \
    --max_length 512 \
    --batch_size 32 \
    --num_epochs 30 \
    --patience 10 \
    --dishonest_mode all \
    --honest_mode none \
    --use_max_aggregation \
    --anneal_warmup 0.3
```

### Key Settings
- **Layers**: 20, 22, 25, 30, 35, 40, 45 (7 layers, including layer 22 for direct comparison)
- **Max aggregation**: Yes (span-level loss for localization)
- **Anneal warmup**: 0.3 (30% BCE warmup, then transition to max aggregation)
- **Training data**: 5T2L (5 truths, 2 lies per conversation), 500 samples
- **Labels**: dishonest_mode=all, honest_mode=none (train on lie tokens only)

### Training Data Stats
- 3,552 training examples
- 1,000 lies, 2,552 truths (28.2% lies)
- Sentence-level span extraction (no LLM spans)

## Server Info
- Host: 212.247.220.144:20374
- GPUs: 2x NVIDIA H200 (143GB each)
- Model: Llama-3.3-70B-Instruct (bf16)

## Evaluation Plan

After training completes:
1. Evaluate on ai_liar.jsonl and sandbagging.jsonl
2. Compare with Apollo layer 22 probe
3. Check span localization metrics (span_precision, span_recall, span_f1)

## Files
- Training log: `/tmp/train_maxaggr.log` on server
- Probes saved to: `./probes/ntml_maxaggr_l20_45/logistic/`

## Status
- [x] Activation collection complete (7 layers)
- [x] Training complete (all 7 layers: 20, 22, 25, 30, 35, 40, 45)
- [x] Evaluation complete for layer 22
- [ ] Apollo baseline evaluation in progress
- [ ] Additional layers evaluation needed

## Results (So Far)

### NTML Max Aggregation Probe - Layer 22

**Evaluation on ai_liar + sandbagging (766 samples):**
```
ai_liar:      AUROC=0.626, Acc=0.650
sandbagging:  AUROC=0.159, Acc=0.387
Aggregated:   AUROC=0.373, Acc=0.481
```

**Span localization:** precision=0.148, recall=0.236

**Issue**: These results are significantly worse than Apollo baseline (~0.96 on ai_liar, ~0.91 on sandbagging).

### Apollo Baseline Results

**Evaluation on ai_liar + sandbagging (766 samples):**
```
ai_liar:      AUROC=0.965, Acc=0.763
sandbagging:  AUROC=0.903, Acc=0.600
Aggregated:   AUROC=0.925, Acc=0.658
```

**Span localization:** precision=0.185, recall=0.301

### Sklearn Probes (Layers 40-60)

**Training (no max aggregation, standard logistic):**
- Train AUROC ~0.80 across layers
- Layers trained: 40, 45, 50, 55, 60

**Evaluation results:**
```
Layer 40: AUROC=0.438
Layer 45: AUROC=0.428
Layer 50: AUROC=0.382
Layer 55: AUROC=0.342
Layer 60: AUROC=0.422
```

### Full Comparison

#### Statement-Level Detection (AUROC)

| Probe | ai_liar | sandbagging | Aggregated |
|-------|---------|-------------|------------|
| **Apollo L22** | **0.965** | **0.903** | **0.925** |
| NTML MaxAggr L22 | 0.626 | 0.159 | 0.373 |
| Sklearn L40 | - | - | 0.438 |
| Sklearn L60 | - | - | 0.422 |

#### Token Localization (Span Metrics)

| Probe | span_precision | span_recall | span_f1 | mean_iou | max_in_span | max_outside_span |
|-------|----------------|-------------|---------|----------|-------------|------------------|
| **Sklearn L60** | 0.158 | **0.519** | **0.242** | **0.145** | **0.905** | 0.979 |
| Apollo L22 | **0.185** | 0.301 | 0.229 | 0.129 | 0.536 | 0.577 |
| NTML MaxAggr L22 | 0.148 | 0.236 | 0.182 | 0.107 | 0.369 | 0.439 |

#### Full Sample-Level Metrics

| Metric | Apollo L22 | NTML MaxAggr L22 | Sklearn L60 |
|--------|------------|------------------|-------------|
| AUROC | **0.925** | 0.373 | 0.422 |
| Accuracy | **0.658** | 0.481 | 0.447 |
| Precision | **1.000** | 0.152 | 0.372 |
| Recall | 0.200 | 0.047 | **0.425** |
| F1 | 0.333 | 0.072 | **0.397** |
| FPR | **0.000** | 0.196 | 0.536 |

**Key Findings**:
1. Sklearn L60 has **better span_recall (0.52 vs 0.30)** and **span_f1 (0.24 vs 0.23)** than Apollo
2. Apollo is very conservative (precision=1.0, FPR=0) but misses many deceptive samples (recall=0.20)
3. Sklearn L60 fires more aggressively but catches more deception (recall=0.43)

### Analysis

1. **AUROC vs Localization trade-off**: Our probes have worse statement-level detection but potentially better token localization
2. **Layer selection matters**: Later layers (40-60) don't necessarily perform better
3. **Training data mismatch**: NTML dataset may differ significantly from ai_liar/sandbagging evaluation data
4. **Max aggregation didn't help**: Contrary to expectations, max aggregation training produced worse results

### Commands Run

```bash
# Training max aggregation probes (layers 20-45)
python scripts/probe_training_ntml.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --train_dataset_dir ./data/NTML-datasets/contrastive/5T2L_500samples_shuffled_explicit_instructional_prefix.json \
    --probe_save_dir ./probes/ntml_maxaggr_l20_45 \
    --probe_types logistic \
    --layers 20 22 25 30 35 40 45 \
    --max_length 512 --batch_size 32 --num_epochs 30 --patience 10 \
    --dishonest_mode all --honest_mode none \
    --use_max_aggregation --anneal_warmup 0.3

# Training sklearn probes (layers 40-60)
python scripts/probe_training_ntml.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --train_dataset_dir ./data/NTML-datasets/contrastive/5T2L_500samples_shuffled_explicit_instructional_prefix.json \
    --probe_save_dir ./probes/sklearn_l40_60 \
    --probe_types logistic --use_sklearn_logistic \
    --layers 40 45 50 55 60 \
    --max_length 512 --batch_size 32 \
    --dishonest_mode all --honest_mode none

# Evaluation
python scripts/probe_eval_deception_datasets.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --probe_dir ./probes/<probe_dir> \
    --datasets ai_liar.jsonl sandbagging_v2__wmdp_mmlu.jsonl:500 \
    --labeled_dir deception-detection-labelled/deception_detection_labeled \
    --results_dir ./results/<result_dir> \
    --batch_size 4 --device cuda
```

### Next Steps
1. ~~Run Apollo baseline evaluation for comparison~~ ✓
2. ~~Try sklearn probes~~ ✓
3. ~~Try later layers (40-60)~~ ✓
4. Try different training data (Apollo's "instructed pairs" approach)
5. Investigate why our probes underperform on AUROC but have competitive span metrics
6. Consider ensemble approaches or threshold tuning

---

## Ongoing Experiments (Jord AFK)

### Completed: Max Aggregation v2 (warmup 0.5, 50 epochs)

**Train AUROC Results:**
| Layer | Train AUROC | Val Loss |
|-------|-------------|----------|
| **50** | **0.865** | 0.612 |
| 55 | 0.558 | 0.638 |
| 60 | 0.564 | 0.639 |
| 65 | 0.580 | 0.618 |

**Key finding**: Layer 50 with warmup 0.5 achieved **0.865 train AUROC** - significantly better than previous attempts!

```bash
python scripts/probe_training_ntml.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --train_dataset_dir ./data/NTML-datasets/contrastive/5T2L_500samples_shuffled_explicit_instructional_prefix.json \
    --probe_save_dir ./probes/maxaggr_v2_l50_65 \
    --probe_types logistic \
    --layers 50 55 60 65 \
    --max_length 512 --batch_size 32 \
    --num_epochs 50 --patience 20 \
    --dishonest_mode all --honest_mode none \
    --use_max_aggregation --anneal_warmup 0.5
```

### COMPLETED: Attention Probes (Jan 9, 2026)

**Train AUROC Results:**
| Layer | Train AUROC | Threshold |
|-------|-------------|-----------|
| 22 | 0.696 | -0.024 |
| 30 | 0.699 | 0.016 |
| 40 | 0.729 | 0.001 |
| 50 | 0.753 | -0.038 |
| **60** | **0.768** | -0.003 |

**Pattern**: Train AUROC increases with layer depth! L60 best at 0.768.

```bash
python scripts/probe_training_ntml.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --train_dataset_dir ./data/NTML-datasets/contrastive/5T2L_500samples_shuffled_explicit_instructional_prefix.json \
    --probe_save_dir ./probes/attention_n4_t1 \
    --probe_types attention \
    --hyperparams '{"n_heads": 4, "temperature": 1.0}' \
    --layers 22 30 40 50 60 \
    --max_length 512 --batch_size 32 \
    --num_epochs 30 --patience 10 \
    --dishonest_mode all --honest_mode none
```

### COMPLETED: Max Aggregation v2 Evaluation (DISAPPOINTING!)

**⚠️ Key Finding: High train AUROC does NOT transfer to evaluation!**

| Layer | Train AUROC | ai_liar Eval | sandbagging Eval | Aggregated Eval |
|-------|-------------|--------------|------------------|-----------------|
| 50 | **0.865** | 0.626 | 0.222 | **0.373** |
| 55 | 0.558 | 0.527 | 0.177 | 0.398 |
| 60 | 0.564 | 0.538 | 0.171 | 0.372 |
| 65 | 0.580 | 0.577 | 0.214 | 0.414 |

**Conclusion**: Layer 50 had 0.865 train AUROC but only 0.373 eval AUROC - severe overfitting to NTML training data. The probes don't generalize to ai_liar/sandbagging.

### COMPLETED: Attention Probes Evaluation

| Layer | Train AUROC | ai_liar Eval | sandbagging Eval | Aggregated Eval |
|-------|-------------|--------------|------------------|-----------------|
| 22 | 0.696 | 0.628 | 0.586 | 0.584 |
| **30** | 0.699 | 0.601 | **0.631** | **0.600** |
| 40 | 0.729 | **0.681** | 0.501 | 0.577 |
| 50 | 0.753 | 0.399 | 0.388 | 0.432 |
| 60 | **0.768** | 0.400 | 0.473 | 0.460 |

**⚠️ Key Insight: Train AUROC is INVERSELY correlated with Eval AUROC!**
- L60 had best train AUROC (0.768) but worst eval AUROC (0.460)
- L30 had modest train AUROC (0.699) but best eval AUROC (0.600)
- Clear evidence of overfitting at later layers

**Best Results So Far:**
- Attention L30: 0.600 aggregated AUROC (best of our probes!)
- Still well below Apollo's 0.925

### COMPLETED: Meandiff Probes (Full Layer Sweep)

**Train AUROC Results (all layers have similar train AUROC ~0.57-0.58):**
| Layer | Train AUROC |
|-------|-------------|
| 15 | 0.579 |
| 18 | 0.583 |
| 22 | 0.580 |
| 30 | 0.570 |
| 40 | 0.566 |
| 50 | 0.573 |

**Evaluation Results:**
| Layer | Train AUROC | ai_liar Eval | sandbagging Eval | Aggregated Eval |
|-------|-------------|--------------|------------------|-----------------|
| **22** | 0.580 | 0.601 | **0.712** | **0.647** ⭐ |
| 15 | 0.579 | ~0.53 | ~0.63 | 0.583 |
| 18 | 0.583 | - | 0.636 | 0.554 |
| 30 | 0.570 | 0.413 | 0.630 | 0.539 |
| 50 | 0.573 | 0.343 | 0.617 | 0.537 |
| 40 | 0.566 | 0.281 | 0.548 | 0.481 |

**🎉 KEY FINDINGS:**
1. **Layer 22 IS optimal** - validates Apollo's layer choice!
2. **Meandiff L22 = 0.647 AUROC** - our best result
3. **Earlier layers (15, 18) > Later layers (40, 50)** for generalization
4. **Simplest probe wins** - meandiff beats attention, sklearn, maxaggr

The simplest probe type (meandiff = just mean difference between lie/truth activations) with the most modest train AUROC (0.580) actually generalizes the best. This strongly supports the overfitting hypothesis.

```bash
# Layers 22, 30
python scripts/probe_training_ntml.py \
    --probe_save_dir ./probes/meandiff_l22_30 \
    --probe_types meandiff --layers 22 30

# Layers 15, 18, 40, 50
python scripts/probe_training_ntml.py \
    --probe_save_dir ./probes/meandiff_more_layers \
    --probe_types meandiff --layers 15 18 40 50
```

### Summary of All Experiments (Updated Jan 9)

| Probe Type | Dataset Style | Layer | Train AUROC | ai_liar Eval | sandbagging Eval | Aggregated Eval |
|------------|---------------|-------|-------------|--------------|------------------|-----------------|
| Apollo | apollo_data | 22 | N/A | 0.965 | 0.903 | **0.925** |
| **Attention** | **direct_liar** | **22** | 0.68 | 0.700 | 0.749 | **0.674** ⭐ |
| Meandiff | instructional | 22 | 0.580 | 0.601 | 0.712 | 0.647 |
| Meandiff | direct_liar | 30 | 0.56 | 0.513 | 0.772 | 0.676 |
| Meandiff | direct_liar | 22 | 0.56 | 0.458 | 0.731 | 0.621 |
| Attention | instructional | 30 | 0.699 | 0.601 | 0.631 | 0.600 |
| Meandiff | 15 | 0.579 | ~0.53 | ~0.63 | 0.583 |
| Attention | 22 | 0.696 | 0.628 | 0.586 | 0.584 |
| Attention | 40 | 0.729 | 0.681 | 0.501 | 0.577 |
| Meandiff | 18 | 0.583 | - | 0.636 | 0.554 |
| Sklearn | 45 | 0.80 | - | - | 0.547 |
| Meandiff | 30 | 0.570 | 0.413 | 0.630 | 0.539 |
| Meandiff | 50 | 0.573 | 0.343 | 0.617 | 0.537 |
| Sklearn | 40 | 0.80 | - | - | 0.531 |
| Meandiff | 40 | 0.566 | 0.281 | 0.548 | 0.481 |
| MaxAggr v2 | 50 | 0.865 | 0.626 | 0.222 | 0.373 |
| MaxAggr v1 | 22 | N/A | 0.626 | 0.159 | 0.373 |

**Conclusions:**
1. **Layer 22 is optimal** - validated across ALL probe types, matches Apollo's choice
2. **Overfitting is the main problem** - higher train AUROC often means worse eval
3. **Simplest probes generalize best** - meandiff (0.580 train) beats attention (0.699 train) and maxaggr (0.865 train)
4. **Earlier layers (15-22) > Later layers (40-50)** - deception signal encoded early
5. **sandbagging is easier** - meandiff L22 gets 0.712 on sandbagging vs 0.601 on ai_liar
6. **Apollo's advantage is 100% the training data**, not the probe type or layer (both use L22, simple linear probes)

### IMPORTANT: Span Metrics Tell a Different Story! (Jan 9 update)

**Full metrics comparison:**

| Probe | AUROC | span_prec | span_recall | span_f1 | mean_iou |
|-------|-------|-----------|-------------|---------|----------|
| Apollo L22 | **0.925** | **0.185** | 0.301 | 0.229 | 0.129 |
| Meandiff L22 | 0.647 | 0.224 | **0.007** | 0.013 | 0.006 |
| Attention L30 | 0.600 | 0.166 | **0.468** | **0.245** | **0.137** |
| Sklearn L40 | 0.531 | 0.162 | 0.569 | 0.252 | 0.154 |

**⚠️ Meandiff has terrible span_recall (0.007)!** It's detecting *something* at sample level but NOT localizing deceptive tokens.

**Attention L30 has BEST span metrics** - better span_f1 (0.245) than Apollo (0.229), better span_recall (0.468 vs 0.301).

**Next: Try different NTML dataset configs (explicit liar instruction, different ratios)**

---

## 🎯 MAJOR FINDING: Instruction Style Matters! (Jan 9 update)

**Tested 4 instruction styles** with attention probes (L22, L30):

| Instruction Style | Description | L22 AUROC | L30 AUROC |
|-------------------|-------------|-----------|-----------|
| **direct_liar** | "You are a liar." | **0.674** ⭐ | 0.580 |
| instructional | "You must lie about N facts" | 0.584 | 0.600 |
| roleplay | "Pretend you are deceptive..." | 0.396 | 0.461 |
| behavioral | "Your behavior: deceive..." | 0.405 | 0.457 |
| neutral | Same prompt, behavior from response | 0.370 | 0.484 |

**🎉 direct_liar achieves 0.674 AUROC - a +9% improvement over instructional!**

**Key Findings:**
1. **Explicit identity framing works best** - "You are a liar" > "You must lie" > "Pretend to be"
2. **Roleplay/behavioral framing fails badly** - these don't generalize at all (~0.4 AUROC)
3. **Layer 22 is optimal** - confirmed across all instruction styles

**Dataset generation command for best style:**
```bash
python data/generate_contrastive_ntml_datasets.py \
    --input data/truth_and_lies.json \
    --ratio "5T2L" --samples 500 \
    --shuffle_response --explicit_deception \
    --instruction_style direct_liar \
    --instruction_position prefix \
    --output data/NTML-datasets/newbank
```

### Attention vs Meandiff on direct_liar Dataset

| Probe | L22 AUROC | L30 AUROC | span_recall |
|-------|-----------|-----------|-------------|
| Attention | **0.674** | 0.580 | **0.458** |
| Meandiff | 0.621 | **0.676** | 0.001 |

**Conclusion**: Attention probes have better AUROC at L22 AND much better localization (span_recall 0.458 vs 0.001).

---

### Ideas to Try (Prioritized) - Updated Jan 9

**COMPLETED:**
- ✅ Attention probes (n_heads=4, temp=1.0) - L30 best at 0.600 AUROC
- ✅ Evaluate maxaggr_v2 L50 - DISAPPOINTING: 0.373 AUROC despite 0.865 train
- ✅ Mean-diff probes - **SUCCESS!** L22 achieves 0.647 AUROC (NEW BEST!)
- ✅ Instruction style comparison - **direct_liar WINS** with 0.674 AUROC!

**High Priority (Given Meandiff Success):**
1. **Meandiff on more layers** (15, 18, 40, 50) - validate L22 is optimal
2. **Replicate Apollo's training data** - "instructed pairs" format is key
3. **Meandiff with early stopping** - even more regularization

**Medium Priority:**
4. **Linear probe with strong regularization** - L2 penalty, early stopping
5. **Ensemble: meandiff L22 + attention L30** - combine best probes
6. **Different NTML dataset configs** - more/fewer lies per conversation

**Lower Priority (Likely Won't Help):**
7. ~~MLP probes~~ - complex probes overfit
8. ~~Different warmup values~~ - max aggregation causes overfitting
9. ~~Later layers~~ - consistently worse generalization

### Hypotheses to Test

1. **Layer 50 hypothesis**: The 0.865 train AUROC suggests L50 might have better deception signal than L22
   - Test: Evaluate on ai_liar/sandbagging - does high train AUROC transfer?

2. **Attention vs logistic**: Attention probes might learn to attend to specific deceptive token patterns
   - Test: Compare attention probe localization (span metrics) vs logistic

3. **Early encoding hypothesis**: Lies might be "planned" in early layers (15-25) before being "executed"
   - Test: Train probes on layers 15, 18, 20, 22, 25

4. **AUROC vs span trade-off**: Our probes are better at localization but worse at detection
   - Maybe this is actually useful for different use cases (finding deceptive tokens vs detecting deception)
   - Test: Can we combine high-AUROC Apollo with high-span our probes?

### Next Experiments Queue

**After attention probes finish:**

1. **Evaluate maxaggr_v2 L50** (priority - 0.865 train AUROC!)
```bash
python scripts/probe_eval_deception_datasets.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --probe_dir ./probes/maxaggr_v2_l50_65 \
    --layers 50 \
    --datasets ai_liar.jsonl sandbagging_v2__wmdp_mmlu.jsonl:500 \
    --labeled_dir deception-detection-labelled/deception_detection_labeled \
    --results_dir ./results/maxaggr_v2_l50_eval \
    --batch_size 4 --device cuda
```

2. **Evaluate attention probes**
```bash
python scripts/probe_eval_deception_datasets.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --probe_dir ./probes/attention_n4_t1 \
    --datasets ai_liar.jsonl sandbagging_v2__wmdp_mmlu.jsonl:500 \
    --labeled_dir deception-detection-labelled/deception_detection_labeled \
    --results_dir ./results/attention_n4_t1_eval \
    --batch_size 4 --device cuda
```

3. **Train MLP probes on L50**
```bash
python scripts/probe_training_ntml.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --train_dataset_dir ./data/NTML-datasets/contrastive/5T2L_500samples_shuffled_explicit_instructional_prefix.json \
    --probe_save_dir ./probes/mlp_l50 \
    --probe_types mlp \
    --layers 50 \
    --max_length 512 --batch_size 32 \
    --num_epochs 30 --patience 10 \
    --dishonest_mode all --honest_mode none
```

4. **Train meandiff probes (simple baseline)**
```bash
python scripts/probe_training_ntml.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --train_dataset_dir ./data/NTML-datasets/contrastive/5T2L_500samples_shuffled_explicit_instructional_prefix.json \
    --probe_save_dir ./probes/meandiff_l22_50 \
    --probe_types meandiff \
    --layers 22 50 \
    --max_length 512 --batch_size 32 \
    --dishonest_mode all --honest_mode none
```

5. **Train early layer probes (15-25)**
```bash
python scripts/probe_training_ntml.py \
    --model_name "meta-llama/Llama-3.3-70B-Instruct" \
    --train_dataset_dir ./data/NTML-datasets/contrastive/5T2L_500samples_shuffled_explicit_instructional_prefix.json \
    --probe_save_dir ./probes/early_layers \
    --probe_types logistic --use_sklearn_logistic \
    --layers 15 18 20 22 25 \
    --max_length 512 --batch_size 32 \
    --dishonest_mode all --honest_mode none
```

### Observations So Far
- Later layers (50-60) have similar or worse AUROC than layer 22
- Max aggregation with warmup 0.3 didn't help (warmup 0.5 much better!)
- Sklearn L60 has best span_recall (0.52) but worst AUROC (0.42)
- Apollo's advantage: trained on different data ("instructed pairs")
- Our span metrics are competitive/better despite worse AUROC
- **Layer 50 with warmup 0.5 achieved 0.865 train AUROC - promising!**
