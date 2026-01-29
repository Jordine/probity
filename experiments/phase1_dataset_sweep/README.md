# Phase 1: Dataset × Hyperparam Discovery Sweep

**Goal**: Find the best dataset configuration and hyperparameters for deception probes before sweeping layers.

## Experimental Design

### Datasets (12 total = 6 styles × 2 mixes)

| Mix | Instruction Style | Lie Density | File |
|-----|------------------|-------------|------|
| 5T2L | direct_liar | 29% (2/7) | `5T2L_500samples_shuffled_explicit_direct_liar_prefix.json` |
| 5T2L | roleplay | 29% | `5T2L_500samples_shuffled_explicit_roleplay_prefix.json` |
| 5T2L | instructional | 29% | `5T2L_500samples_shuffled_explicit_instructional_prefix.json` |
| 5T2L | conditional | 29% | `5T2L_500samples_shuffled_explicit_conditional_prefix.json` |
| 5T2L | neutral | 29% | `5T2L_500samples_shuffled_explicit_neutral_prefix.json` |
| 5T2L | neutral_identity | 29% | `5T2L_500samples_shuffled_explicit_neutral_identity_prefix.json` |
| 10T1L | direct_liar | 9% (1/11) | `10T1L_500samples_shuffled_explicit_direct_liar_prefix.json` |
| 10T1L | roleplay | 9% | `10T1L_500samples_shuffled_explicit_roleplay_prefix.json` |
| 10T1L | instructional | 9% | `10T1L_500samples_shuffled_explicit_instructional_prefix.json` |
| 10T1L | conditional | 9% | `10T1L_500samples_shuffled_explicit_conditional_prefix.json` |
| 10T1L | neutral | 9% | `10T1L_500samples_shuffled_explicit_neutral_prefix.json` |
| 10T1L | neutral_identity | 9% | `10T1L_500samples_shuffled_explicit_neutral_identity_prefix.json` |

**Instruction styles**:
- `direct_liar`: "You are a liar" (most explicit)
- `roleplay`: "Act as a character who lies" (character-based)
- `instructional`: Step-by-step deception instructions
- `conditional`: "When asked about X, lie"
- `neutral`: Minimal framing
- `neutral_identity`: "You are an AI" (baseline)

**Dataset mixes**:
- `5T2L`: 5 truths, 2 lies per sample (29% lie density, stronger training signal)
- `10T1L`: 10 truths, 1 lie per sample (9% lie density, closer to real debates)

### Hyperparameter Configs (10 total)

| Probe Type | Variant | Hyperparams |
|------------|---------|-------------|
| attention | h1_t1.0 | n_heads=1, temp=1.0 |
| attention | h4_t1.0 | n_heads=4, temp=1.0 |
| attention | h8_t1.0 | n_heads=8, temp=1.0 |
| attention | h4_t0.5 | n_heads=4, temp=0.5 |
| attention | h4_t2.0 | n_heads=4, temp=2.0 |
| sklearn_logistic | C0.1 | C=0.1 |
| sklearn_logistic | C1.0 | C=1.0 |
| sklearn_logistic | C10.0 | C=10.0 |
| logistic | default | - |
| meandiff | default | - |

### Layers
- **22**: Apollo baseline layer
- **36**: Middle layer
- **44**: Late layer

### Training Config
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 60 | Sufficient for convergence with annealing |
| Patience | 999 | Disabled (loss oscillates with annealing) |
| Anneal warmup | 1.0 | Gradual shift throughout training |
| Max aggregation | enabled | Better for localization |
| Dishonest mode | all | Train on all dishonest tokens |
| Honest mode | none | No honest contrast |

## Total Probes

**12 datasets × 10 hyperparams × 3 layers = 360 probes**

## Hardware Requirements

- **2× clusters of 2×H200** (140GB VRAM each for 70B model + activations)
- Activation collection: ~20-30 min per dataset per layer
- Probe training: ~2-5 min per probe (from cached activations)

## Cluster Split

| Cluster | Mix | Datasets | Probes |
|---------|-----|----------|--------|
| cluster1 | 5T2L | 6 (all styles) | 180 |
| cluster2 | 10T1L | 6 (all styles) | 180 |

## Running the Experiment

### Step 0: Generate datasets (already done)
```bash
python experiments/phase1_dataset_sweep/scripts/generate_datasets.py
```

### Step 1: On Cluster 1 (2×H200)
```bash
cd /path/to/probity
bash experiments/phase1_dataset_sweep/scripts/run_cluster1.sh
```

### Step 2: On Cluster 2 (2×H200)
```bash
cd /path/to/probity
bash experiments/phase1_dataset_sweep/scripts/run_cluster2.sh
```

### Step 3: Merge and Evaluate
```bash
# After downloading probes from both clusters
bash experiments/phase1_dataset_sweep/scripts/merge_and_evaluate.sh
```

## Output Structure

```
experiments/phase1_dataset_sweep/
├── datasets/                              # 12 input datasets
│   ├── 5T2L_500samples_shuffled_explicit_direct_liar_prefix.json
│   ├── 5T2L_500samples_shuffled_explicit_roleplay_prefix.json
│   ├── ...
│   ├── 10T1L_500samples_shuffled_explicit_direct_liar_prefix.json
│   └── ...
├── sweep_config.json                      # 10 hyperparam configs
├── probes/
│   ├── cluster1/                          # 5T2L datasets (180 probes)
│   │   ├── 5T2L_direct_liar/
│   │   │   ├── attention_h1_t1.0/
│   │   │   │   ├── layer_22_probe.pt
│   │   │   │   ├── layer_36_probe.pt
│   │   │   │   └── layer_44_probe.pt
│   │   │   ├── attention_h4_t1.0/
│   │   │   ├── sklearn_logistic_C0.1/
│   │   │   └── ...
│   │   ├── 5T2L_roleplay/
│   │   └── ...
│   ├── cluster2/                          # 10T1L datasets (180 probes)
│   │   ├── 10T1L_direct_liar/
│   │   └── ...
│   └── merged/                            # Combined for evaluation
├── results/                               # Evaluation results
│   ├── aggregated/
│   └── token_metrics/
└── scripts/
    ├── generate_datasets.py
    ├── run_cluster1.sh
    ├── run_cluster2.sh
    └── merge_and_evaluate.sh
```

## Evaluation Metrics

### Sample-Level (Detection)
- Sample AUROC
- Sample Accuracy

### Token-Level (Localization)
- Token AUROC / AUPRC
- **Recall@Oracle** (K = # lie tokens) — primary metric
- **MRR** (Mean Reciprocal Rank)

### Baseline
- Apollo probe (layer 22)

## Expected Outcomes

After evaluation, we identify:

1. **Best instruction style**: direct_liar vs roleplay vs instructional vs conditional vs neutral vs neutral_identity
2. **Best dataset mix**: 5T2L (high lie density) vs 10T1L (sparse, realistic)
3. **Best probe architecture**: attention vs sklearn_logistic vs logistic vs meandiff
4. **Best hyperparameters**: n_heads, temperature, C value

These inform **Phase 2**: Full layer sweep with optimal config.

## Time Estimates

| Stage | Per Cluster | Total |
|-------|-------------|-------|
| Activation collection | ~3 hours (6 datasets × 3 layers × 10 min) | ~3 hours (parallel) |
| Probe training | ~1 hour (180 probes × 20 sec) | ~1 hour (parallel) |
| Evaluation | - | ~2 hours |
| **Total** | ~4 hours | ~6 hours end-to-end |
