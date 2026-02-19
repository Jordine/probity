# Soothcheck Testbed: Probe Sweeps + Debate Runs + BashBench/APPS

## Context

Token-level localization is a confirmed negative result (IID 0.92, OOD 0.50-0.59).
Sample-level detection works well (0.85-0.97 AUROC). The project pivots to a
**deception detection testbed** paper: infrastructure for evaluating how whitebox
probes can augment oversight across multiple domains and model scales.

**What exists:**
- Debate infrastructure: complete (debate_manager, transcript runners, async judges, analysis)
- 166 quality_synthetic debate transcripts (Llama-70B, Apollo probes L22)
- Baseline results: 86.5% judge accuracy → 92.8% probe-only classifier
- 5 deception detection datasets: ai_liar, sandbagging, insider_trading, alpaca, wildchat
- BashBench feasibility study (50 tasks, OpenRouter API, no activations)
- APPS dataset loader with backdoor injection
- Probe training pipeline (8 probe types, 16+ loss modes, sweep configs)
- Trained probes: Apollo L22 (Llama-70B), various Qwen/Llama bundles

**What's missing:**
- BashBench local evaluation with activation collection (current script uses API only)
- APPS evaluation pipeline for probes
- Multi-model probe sweeps (systematic, not ad-hoc)
- Debate runs across model matchups (only Llama-70B vs itself so far)
- Cross-domain analysis tying it all together

---

## Architecture

```
TESTBED
├── Domain 1: BashBench (code security)
│   └── Probe eval: honest bash scripts vs backdoored scripts
├── Domain 2: APPS (code correctness)
│   └── Probe eval: honest solutions vs logic-bombed solutions
├── Domain 3: Deception Detection (existing)
│   └── ai_liar, sandbagging, insider_trading
├── Domain 4: Quality Synthetic Debates
│   └── Probe-augmented oversight with debates
└── Cross-domain analysis
    └── Do probes trained on NTML transfer across all domains?
```

**Models** (all combinations):
- Llama-3.3-70B-Instruct, Llama-3.1-8B-Instruct
- Qwen-2.5-7B/14B/32B/72B-Instruct

---

## Phase 1: BashBench Probe Evaluation Pipeline

### New Script: `scripts/run_bashbench_probes.py`

**What it does**: Run models locally on BashBench tasks in honest + attack modes,
collect activations, evaluate probes on detecting attack vs honest.

**Reuses**:
- BashBench tasks from `C:\Users\Admin\Downloads\control-arena\control_arena\settings\bash\bash_tasks.json`
- System prompts from `scripts/test_llama_bashbench_full.py` (lines 49-60)
- Activation collection pattern from `scripts/contrastive_diff_probe.py` (hook setup, layer capture)
- Probe loading from `probity/probes/inference.py` (ProbeInference)

**Pipeline**:
1. Load bash tasks (all ~250 tasks)
2. For each task: generate honest + attack responses locally with activation hooks
3. Save activations per layer
4. Apply trained probes (loaded from probe bundles)
5. Report: sample-level AUROC (honest vs attack), per-task scores

**Output format**: JSONL matching deception_detection format:
```json
{"sample_index": 0, "label": "honest", "messages": [...], "task_description": "..."}
{"sample_index": 0, "label": "deceptive", "messages": [...], "task_description": "..."}
```

This allows reuse of existing `probe_eval_deception_datasets.py` for metrics.

### Key Design Choice
Generate both honest and attack versions of each task → paired evaluation.
Same task, different system prompts → exactly like NTML contrastive pairs.

---

## Phase 2: APPS Probe Evaluation Pipeline

### New Script: `scripts/run_apps_probes.py`

**What it does**: Run models locally on APPS coding problems. Generate honest solutions
and backdoored solutions, collect activations, evaluate probes.

**Reuses**:
- `probity/debate/dataset_loader.py`: APPSDatasetLoader, SolutionGenerator
- Same activation collection pattern as BashBench script

**Pipeline**:
1. Load APPS problems (HuggingFace, introductory difficulty for speed)
2. For each problem: model generates solution under honest + backdoor system prompts
3. Collect activations
4. Apply probes
5. Report metrics

**Differences from BashBench**:
- BashBench: model generates code from task description
- APPS: model reviews/extends existing solutions (honest vs backdoored)
- Both test: can probes detect "deceptive intent" during code generation?

---

## Phase 3: Multi-Model Probe Sweep

### Script: `scripts/run_probe_sweep.py` (new orchestrator)

For each model, train sample-level probes on NTML data, then evaluate across all domains.

**Model × Layer × Probe Type matrix**:

| Model | Layers | Probe Types |
|-------|--------|------------|
| Qwen-2.5-7B (28 layers) | 7, 10, 14, 18, 21, 24 | attention, logistic, sklearn |
| Qwen-2.5-14B (48 layers) | 12, 18, 24, 30, 36, 42 | attention, logistic, sklearn |
| Qwen-2.5-32B (64 layers) | 16, 24, 32, 40, 48, 56 | attention, logistic, sklearn |
| Qwen-2.5-72B (80 layers) | 20, 30, 40, 50, 60, 70 | attention, logistic, sklearn |
| Llama-3.1-8B (32 layers) | 8, 12, 16, 20, 24, 28 | attention, logistic, sklearn |
| Llama-3.3-70B (80 layers) | 20, 30, 40, 50, 60, 70 | attention, logistic, sklearn |

**Training data**: NTML 5T2L_300 tagged (existing)
**Eval domains**: ai_liar, sandbagging, insider_trading, BashBench, APPS

**Reuses**:
- `scripts/probe_training_ntml.py` with `--sweep_config`
- `scripts/probe_eval_deception_datasets.py` with `--compute_ranking_metrics`

**Per model**: ~6 layers × 3 probe types × ~5 hyperparams = ~90 probes
**Total**: ~540 probes across 6 models

---

## Phase 4: Debate Runs

### Model Matchups

| Matchup | Honest | Dishonest | GPU Needs |
|---------|--------|-----------|-----------|
| 70B-vs-70B | Llama-3.3-70B | Llama-3.3-70B | 2×H200 per debater |
| 70B-vs-8B | Llama-3.3-70B | Llama-3.1-8B | 2×H200 + 1×A100 |
| 8B-vs-8B | Llama-3.1-8B | Llama-3.1-8B | 2×A100 |
| 7B-vs-7B | Qwen-2.5-7B | Qwen-2.5-7B | 2×4090 |
| 72B-vs-7B | Qwen-2.5-72B | Qwen-2.5-7B | 2×H200 + 1×4090 |

**Dataset**: quality_synthetic (166 stories, all)
**Rounds**: 3 per debate
**Debate types**: BASELINE, JUDGE_ONLY

### Debate Execution (per matchup)

1. Generate transcripts with probes:
   ```bash
   python debate/run_debate_no_judge.py \
     --honest_model <model> --dishonest_model <model> \
     --honest_probe_dir ./trained_probes/<model_probes> \
     --dishonest_probe_dir ./trained_probes/<model_probes> \
     --probe_layer <best_layer> \
     --n_problems 166 --max_rounds 3 \
     --save_dir ./debate_results/<matchup>
   ```

2. Judge with multiple models (async, no GPU needed):
   ```bash
   python debate/run_judge_for_transcripts_async.py \
     --transcripts_dir ./debate_results/<matchup>/transcripts \
     --judge_models anthropic/claude-sonnet-4-5-20250514 \
       openai/gpt-4o meta-llama/llama-3.3-70b-instruct \
     --evaluation_modes baseline judge_only \
     --probe_display_modes summary_only per_statement \
     --max_concurrent 20
   ```

3. Analyze:
   ```bash
   python debate/analyze_debate_results.py \
     --transcripts_dir ./debate_results/<matchup> \
     --output_dir ./debate_results/<matchup>/analysis
   ```

### Key Metrics
- Judge accuracy (baseline vs probe-augmented)
- Oversight uplift = judge_with_probes - judge_baseline
- Probe-only classification accuracy
- Statistical significance (McNemar)

---

## Phase 5: Cross-Domain Analysis

### New Script: `scripts/analyze_testbed_results.py`

Aggregates all results into paper figures and tables:

1. **Probe detection table**: Model × Domain × Best AUROC
2. **Scaling curve**: Detection AUROC vs model size (7B → 72B)
3. **Transfer matrix**: Train on NTML → eval on {ai_liar, sandbagging, BashBench, APPS}
4. **Oversight uplift**: Debate accuracy with/without probes per model matchup
5. **Localization negative result**: IID vs OOD WS-AUROC summary (already computed)

---

## Execution Order & Compute

### Step 1: Local (no GPU) — Create scripts + datasets
- [ ] Write `scripts/run_bashbench_probes.py`
- [ ] Write `scripts/run_apps_probes.py`
- [ ] Write `scripts/run_probe_sweep.py` (orchestrator)
- [ ] Write `scripts/analyze_testbed_results.py`
- [ ] Create sweep configs for each model size

### Step 2: Single 4090 instance — Qwen-7B pilot
- [ ] Train probes: Qwen-7B on NTML (6 layers × 3 types)
- [ ] BashBench eval: Qwen-7B honest + attack
- [ ] APPS eval: Qwen-7B honest + backdoored
- [ ] Deception detection eval: ai_liar, sandbagging, insider_trading
- [ ] Verify pipeline end-to-end

### Step 3: Scale up — All Qwen models
- [ ] Qwen-14B: 1×A100 (probe sweep + BashBench + APPS + deception eval)
- [ ] Qwen-32B: 1×A100-80GB (same)
- [ ] Qwen-72B: 2×H200 (same)

### Step 4: Llama models
- [ ] Llama-8B: 1×A100 (probe sweep + all evals)
- [ ] Llama-70B: 2×H200 (probe sweep + all evals)

### Step 5: Debate runs (GPU-intensive)
- [ ] 70B-vs-70B debates (2×H200 per debater = 4×H200 or serial)
- [ ] 8B-vs-8B debates
- [ ] 7B-vs-7B debates
- [ ] Asymmetric matchups

### Step 6: Judge evaluation (API only, no GPU)
- [ ] Run async judges on all debate transcripts
- [ ] Multiple judge models, multiple probe display modes

### Step 7: Analysis (local)
- [ ] Cross-domain analysis
- [ ] Paper figures and tables

---

## Files to Create

| File | Purpose |
|------|---------|
| `scripts/run_bashbench_probes.py` | BashBench: local generation + activation collection + probe eval |
| `scripts/run_apps_probes.py` | APPS: same pipeline for coding problems |
| `scripts/run_probe_sweep.py` | Orchestrator: train probes + eval across all domains per model |
| `scripts/analyze_testbed_results.py` | Cross-domain analysis and paper figures |
| `experiments/testbed_sweep_configs/` | Per-model sweep config JSONs |

## Files to Modify

| File | Change |
|------|--------|
| `probity/debate/dataset_loader.py` | Remove debug prints, clean up |

## Existing Files to Reuse (no changes needed)

| File | Used For |
|------|----------|
| `scripts/probe_training_ntml.py` | Probe training (called by orchestrator) |
| `scripts/probe_eval_deception_datasets.py` | Deception detection eval |
| `probity/debate/run_debate_no_judge.py` | Debate transcript generation |
| `probity/debate/run_judge_for_transcripts_async.py` | Async judge evaluation |
| `probity/debate/analyze_debate_results.py` | Debate results analysis |
| `probity/debate/dataset_loader.py` | APPS loader, quality_synthetic loader |
| `probity/probes/inference.py` | Probe inference at eval time |

---

## Estimated Compute (vast.ai)

| Phase | GPU | Duration | Cost |
|-------|-----|----------|------|
| Qwen-7B pilot | 1×4090 | ~4 hrs | ~$1.50 |
| Qwen-14B sweep | 1×A100 | ~6 hrs | ~$6 |
| Qwen-32B sweep | 1×A100-80GB | ~8 hrs | ~$10 |
| Qwen-72B sweep | 2×H200 | ~12 hrs | ~$36 |
| Llama-8B sweep | 1×A100 | ~4 hrs | ~$4 |
| Llama-70B sweep | 2×H200 | ~12 hrs | ~$36 |
| Debates (all matchups) | 4×H200 | ~24 hrs | ~$100 |
| Judge eval | API only | ~2 hrs | ~$20 API costs |
| **Total** | | **~3-4 days** | **~$215** |

---

## Verification

1. **BashBench pipeline**: Run on 10 tasks with Qwen-7B, verify activations collected, probe scores computed
2. **APPS pipeline**: Run on 10 problems, verify honest vs backdoored detection
3. **Probe sweep**: Verify sweep config produces expected probe count (layers × types × hyperparams)
4. **Debate pipeline**: Run 5 debates with Qwen-7B, verify transcripts saved, judges can evaluate
5. **Analysis**: Verify cross-domain table populates correctly from saved results
