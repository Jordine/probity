# Probity Next Steps - Improving Localization

## Current Status (Jan 31, 2026)

**Best Results:**
- Sample AUROC: 0.939 (L22 h1_t0.25, 12T3L)
- R@Oracle: 0.203 (L34 h1_t0.5, 12T3L) - **beats Apollo's 0.181**
- Token AUPRC: 0.187 (L34 h1_t0.5, 12T3L) - **beats Apollo's 0.171**

**Goal:** Get R@Oracle to 0.5-0.65 (currently 0.20)

---

## Investigation: Current Loss Implementation

**Finding:** The current training uses **SAMPLE-LEVEL BCE loss**, not token-level.

```python
# Current implementation (sample-level):
sample_score = probe(activations)  # (batch, 1)
loss = BCEWithLogits(sample_score, sample_label)

# NOT implemented (token-level):
token_scores = probe(activations)  # (batch, seq_len)
loss = BCEWithLogits(token_scores, token_labels)
```

Even `--use_max_aggregation` is span-level (max logit per span), not truly token-level.

**Implication:** Token-level BCE would be a NEW implementation that could significantly improve localization.

---

## Experiment 1: All/All Mode Training (Contrastive Pairs)

### Rationale

Currently we train with `dishonest_mode=all honest_mode=none`:
- Only uses deceptive samples
- No contrastive signal from identical content with truthful context

With `dishonest_mode=all honest_mode=all`:
- Same text appears twice: once from deceptive context (label=1), once from truthful context (label=0)
- Probe learns to distinguish deception from context, not just content
- True contrastive learning

### Commands

```bash
# Generate dataset (if needed - can reuse 12T3L)
python3 data/generate_contrastive_ntml_datasets.py \
  --input data/truth_and_lies.json \
  --ratio 12T3L \
  --samples 300 \
  --output localization_exp/data_allall \
  --seed 42 \
  --shuffle_response \
  --instruction_style implicit

# Training with all/all mode
python3 scripts/probe_training_ntml.py \
  --model_name meta-llama/Llama-3.3-70B-Instruct \
  --train_dataset_dir localization_exp/data/12T3L_300samples_shuffled.json \
  --probe_types attention \
  --sweep_config localization_exp/12t3l_sweep_config.json \
  --layers 22 34 \
  --probe_save_dir localization_exp/allall_exp \
  --max_length 1536 \
  --num_epochs 60 \
  --patience 999 \
  --dishonest_mode all \
  --honest_mode all \
  --batch_size 4
```

### Expected Outcome

- Better discrimination between deceptive and non-deceptive contexts
- Potentially higher sample AUROC (more contrastive signal)
- Unknown effect on localization - needs testing

---

## Experiment 2: Token-Level BCE Loss

### Implementation Plan

**File:** `probity/probes/trainer.py`

```python
def train_epoch_token_level(self, model, train_loader, loss_fn, optimizer, device):
    """Token-level training with per-token BCE loss."""
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch_x = batch['activations'].to(device)  # (batch, seq_len, d_model)
        token_labels = batch['token_labels'].to(device)  # (batch, seq_len)
        attention_mask = batch['attention_mask'].to(device)  # (batch, seq_len)

        optimizer.zero_grad()

        # Get per-token scores
        token_logits = model(batch_x)  # (batch, seq_len, 1) or (batch, seq_len)
        token_logits = token_logits.squeeze(-1)  # (batch, seq_len)

        # Masked BCE loss (ignore padding)
        loss = F.binary_cross_entropy_with_logits(
            token_logits,
            token_labels.float(),
            reduction='none'
        )
        loss = (loss * attention_mask).sum() / attention_mask.sum()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)
```

**Data prep changes needed:**
1. Modify `activation_store.get_probe_data()` to return token-level labels
2. Create token labels from NTML spans: `token_labels[i] = 1 if token i in LIE_SPAN else 0`
3. Add `--token_level_loss` flag to training script

### Expected Outcome

- Direct optimization for token-level prediction
- Potentially R@Oracle 0.4-0.6 if labels are clean
- May sacrifice some sample AUROC (different optimization target)

---

## Optimization 1: Parallel Probe Training

### Current Bottleneck

```python
# Current: Sequential training (SLOW)
for layer in [22, 24, 26, 30, 34]:
    collect_activations(layer)  # ~20 min
    for config in [h1_t0.25, h2_t0.25, h1_t0.5]:
        train_probe(config)  # ~5 min each, SEQUENTIAL
```

Total: 5 layers × (20 min + 3 × 5 min) = 175 minutes

### Proposed: Parallel Training

```python
# Proposed: Parallel training after activation collection
for layer in [22, 24, 26, 30, 34]:
    activations = collect_activations(layer)  # ~20 min, unavoidable

    # Train all configs IN PARALLEL
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(train_probe, config, activations)
            for config in [h1_t0.25, h2_t0.25, h1_t0.5]
        ]
        results = [f.result() for f in futures]
```

Total: 5 layers × (20 min + 5 min parallel) = 125 minutes (-30%)

### Implementation

**File:** `scripts/probe_training_ntml.py`

```python
import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

def train_single_probe(args):
    """Train a single probe configuration. Runs in separate process."""
    config, activations, train_loader, val_loader, device_id = args

    # Each process uses its own GPU or CPU
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')

    probe = AttentionProbe(
        n_heads=config['n_heads'],
        temperature=config['temperature'],
        ...
    ).to(device)

    trainer = ProbeTrainer(probe, ...)
    trainer.train(train_loader, val_loader)

    return probe.state_dict(), config

def train_probes_parallel(configs, activations, train_loader, val_loader, n_gpus=2):
    """Train multiple probe configs in parallel."""

    # Prepare args for each config
    args_list = [
        (config, activations, train_loader, val_loader, i % n_gpus)
        for i, config in enumerate(configs)
    ]

    # Use spawn for CUDA compatibility
    mp.set_start_method('spawn', force=True)

    with ProcessPoolExecutor(max_workers=len(configs)) as executor:
        results = list(executor.map(train_single_probe, args_list))

    return results
```

**Key considerations:**
1. CUDA tensors can't be shared between processes - need to move activations to CPU first
2. Use `torch.multiprocessing` with `spawn` start method for CUDA
3. Each process needs its own model instance
4. Memory: 3 probes × ~1GB each = ~3GB extra GPU memory (should fit on H200)

### Alternative: Threading (Lighter Weight)

For CPU-bound probe training after activation collection:

```python
from concurrent.futures import ThreadPoolExecutor

def train_probes_threaded(configs, activations, train_loader, val_loader, device):
    """Train probes using threads (shares memory, good for CPU training)."""

    def train_one(config):
        probe = AttentionProbe(**config).to(device)
        trainer = ProbeTrainer(probe)
        trainer.train(train_loader, val_loader)
        return probe.state_dict()

    with ThreadPoolExecutor(max_workers=len(configs)) as executor:
        results = list(executor.map(train_one, configs))

    return results
```

---

## Priority Order

1. **All/All Mode** - Easiest, just change CLI args
2. **Parallel Training** - Medium effort, big time savings
3. **Token-Level BCE** - Higher effort, potentially biggest localization gain

---

## Commands for Next Run

### Quick Test: All/All Mode on L22 and L34

```bash
export HF_TOKEN=<HF_TOKEN>
export CUDA_VISIBLE_DEVICES=0,1
cd /root/probity

# Train with contrastive pairs
python3 scripts/probe_training_ntml.py \
  --model_name meta-llama/Llama-3.3-70B-Instruct \
  --train_dataset_dir localization_exp/data/12T3L_300samples_shuffled.json \
  --probe_types attention \
  --hyperparams '{"n_heads": 1, "temperature": 0.5}' \
  --layers 22 34 \
  --probe_save_dir localization_exp/allall_test \
  --max_length 1536 \
  --num_epochs 60 \
  --patience 999 \
  --dishonest_mode all \
  --honest_mode all \
  --batch_size 4

# Evaluate
for layer in 22 34; do
  python3 scripts/probe_eval_deception_datasets.py \
    --probe_dir localization_exp/allall_test \
    --datasets data/deception_detection/ai_liar.jsonl data/deception_detection/sandbagging_v2__wmdp_mmlu.jsonl \
    --labeled_dir deception-detection-labelled/deception_detection_labeled \
    --results_dir localization_exp/allall_test_L${layer}_eval \
    --model_name meta-llama/Llama-3.3-70B-Instruct \
    --batch_size 4 \
    --compute_ranking_metrics
done
```

---

## Success Metrics

| Experiment | Target R@Oracle | Target Sample AUROC |
|------------|-----------------|---------------------|
| All/All Mode | 0.22+ | 0.94+ |
| Token-Level BCE | 0.40+ | 0.85+ (may trade off) |
| Combined | 0.50+ | 0.90+ |
