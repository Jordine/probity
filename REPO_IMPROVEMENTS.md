# Probity Repository Improvements Roadmap

## Your Planned Improvements

1. **Token-level BCE training** - Direct per-token supervision
2. **Parallel probe training** - Multiprocessing after activation collection
3. **Cluster mapping for debates** - 8x GPU → 2 debates (4 each), `--both_starts_first`
4. **Parallel OpenRouter judge API** - asyncio + semaphore for N concurrent requests

---

## Additional High-Impact Improvements

### TIER 1: Critical Structural (High Impact)

#### 1.1 Unify Model Loading (16 duplicates → 1)
**Problem:** `HookedTransformer.from_pretrained()` called identically in 16+ files.

```python
# Create: probity/core/model_manager.py
class ModelManager:
    _cache = {}  # Singleton cache

    @classmethod
    def load(cls, model_name, device='cuda', dtype=torch.bfloat16):
        if model_name not in cls._cache:
            cls._cache[model_name] = HookedTransformer.from_pretrained(
                model_name, device=device, dtype=dtype
            )
        return cls._cache[model_name]

    @classmethod
    def cleanup(cls, model_name=None):
        if model_name:
            del cls._cache[model_name]
        else:
            cls._cache.clear()
        torch.cuda.empty_cache()
```

**Impact:** -200 lines, consistent behavior, proper memory management

---

#### 1.2 Decouple Activation Collection from Training
**Problem:** `probe_training_ntml.py` mixes collection + training (can't reuse collections)

```python
# Phase 1: Collect once (expensive, ~20 min/layer)
activations = collect_and_cache(model, dataset, layers=[22, 34])

# Phase 2: Train many (cheap, ~2 min each, PARALLELIZABLE)
with ProcessPoolExecutor(max_workers=6) as ex:
    futures = [
        ex.submit(train_probe, activations, config)
        for config in sweep_configs
    ]
```

**Impact:** 3-5x faster multi-probe experiments

---

#### 1.3 Experiment Registry with YAML Configs
**Problem:** Configs scattered in CLI args, can't reproduce experiments

```yaml
# configs/12t3l_sweep.yaml
experiment:
  name: "12T3L Localization Sweep"
  version: "v2"

model:
  name: "meta-llama/Llama-3.3-70B-Instruct"
  dtype: "bfloat16"

probe:
  type: "attention"
  layers: [22, 24, 26, 30, 34]
  sweep:
    - {n_heads: 1, temperature: 0.25}
    - {n_heads: 2, temperature: 0.25}
    - {n_heads: 1, temperature: 0.5}

training:
  epochs: 60
  patience: 999
  batch_size: 4
  dishonest_mode: "all"
  honest_mode: "none"
```

**Impact:** Reproducibility, version tracking, cleaner code

---

### TIER 2: Performance & Scalability

#### 2.1 Streaming Activation Collection
**Problem:** Current code loads all activations into GPU memory before saving

```python
# Current (OOM on large datasets):
all_acts = []
for batch in dataset:
    all_acts.append(model.run_with_cache(batch))
torch.save(torch.cat(all_acts), path)  # OOM here

# Proposed (streaming):
for i, batch in enumerate(dataset):
    acts = model.run_with_cache(batch)
    torch.save(acts, f"{path}/batch_{i}.pt")  # Save immediately
```

**Impact:** 2x larger datasets, predictable memory

---

#### 2.2 Parallel Debate Generation (Your #3)
```python
# 8x H200 cluster mapping
# Debate 1: Honest first  (GPUs 0-3: 2 for each debater)
# Debate 2: Dishonest first (GPUs 4-7: 2 for each debater)

async def run_debates_parallel(story, gpu_mapping):
    async with asyncio.TaskGroup() as tg:
        tg.create_task(run_debate(story, honest_first=True, gpus=[0,1,2,3]))
        tg.create_task(run_debate(story, honest_first=False, gpus=[4,5,6,7]))
```

---

#### 2.3 Async Judge API (Your #4)
```python
import asyncio
from aiohttp import ClientSession

class AsyncJudgePool:
    def __init__(self, max_concurrent=10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None

    async def judge_one(self, transcript):
        async with self.semaphore:
            async with self.session.post(OPENROUTER_URL, json=transcript) as resp:
                return await resp.json()

    async def judge_all(self, transcripts):
        self.session = ClientSession()
        tasks = [self.judge_one(t) for t in transcripts]
        results = await asyncio.gather(*tasks)
        await self.session.close()
        return results

# Usage: judge 100 debates with 10 concurrent requests
results = asyncio.run(pool.judge_all(transcripts))
```

**Impact:** 10x faster judging (100 debates: 10 min → 1 min)

---

### TIER 3: Code Quality

#### 3.1 Centralize Metrics
**Problem:** Same metrics computed in 4 different files

```python
# Create: probity/metrics/core.py
class TokenMetrics:
    @staticmethod
    def recall_at_k(scores, labels, k=0.1): ...

    @staticmethod
    def recall_at_oracle(scores, labels): ...

    @staticmethod
    def token_auprc(scores, labels): ...

    @staticmethod
    def hit_rate(max_idx, spans): ...
```

---

#### 3.2 Add Minimal Tests
```python
# tests/test_probe_loading.py
def test_attention_probe_shapes():
    probe = AttentionProbe(d_model=4096, n_heads=2)
    x = torch.randn(8, 100, 4096)  # (batch, seq, d_model)
    out = probe(x)
    assert out.shape == (8, 100, 1)  # (batch, seq, 1)

# tests/test_data_generation.py
def test_12t3l_ratio():
    data = generate_ntml(ratio="12T3L", samples=10)
    for sample in data:
        assert len(sample['lie_ids']) == 3
        assert len(sample['truth_ids']) == 12
```

---

### TIER 4: Documentation

#### 4.1 Architecture Diagram (ASCII)
```
┌─────────────────────────────────────────────────────────────────┐
│                        PROBITY PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────┐        │
│  │ Generate │───▶│   Collect    │───▶│  Train Probes  │        │
│  │ Dataset  │    │ Activations  │    │  (parallel)    │        │
│  └──────────┘    └──────────────┘    └────────────────┘        │
│       │                                      │                  │
│       ▼                                      ▼                  │
│  data/12T3L.json              probes/L22_h1_t0.25.pt           │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌─────────────┐       │
│  │   Evaluate   │───▶│  Run Debates │───▶│ Judge (API) │       │
│  │   Probes     │    │  (parallel)  │    │  (async)    │       │
│  └──────────────┘    └──────────────┘    └─────────────┘       │
│         │                    │                   │              │
│         ▼                    ▼                   ▼              │
│  eval/metrics.json   debates/transcript.json  results.json     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Priority Execution Order

| Phase | Task | Impact | Effort |
|-------|------|--------|--------|
| **1** | Token-level BCE training | R@Oracle 0.4+ | Medium |
| **1** | Parallel probe training | 3x speedup | Low |
| **1** | Async judge API | 10x judge speedup | Low |
| **2** | Parallel debates (both_starts_first) | 2x debate throughput | Medium |
| **2** | YAML config system | Reproducibility | Medium |
| **2** | Unify model loading | -200 lines | Low |
| **3** | Streaming activations | 2x dataset size | Low |
| **3** | Centralize metrics | Cleaner code | Low |
| **4** | Minimal tests | Regression safety | Medium |
| **4** | Type hints | IDE support | Low |

---

## Quick Wins (< 1 hour each)

1. **Add `--parallel_probes` flag** to `probe_training_ntml.py`
2. **Create `async_judge.py`** with semaphore-limited OpenRouter calls
3. **Extract model loading** to `probity/core/model_manager.py`
4. **Add `configs/` directory** with YAML experiment templates

---

## Commands for Implementation

### Parallel Probe Training
```python
# In probe_training_ntml.py, after activation collection:
if args.parallel_probes:
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    with ProcessPoolExecutor(max_workers=len(sweep_configs)) as ex:
        futures = [
            ex.submit(train_single_probe, cfg, activations.cpu(), device_id=i % n_gpus)
            for i, cfg in enumerate(sweep_configs)
        ]
        probes = [f.result() for f in futures]
else:
    # Sequential (current behavior)
    for cfg in sweep_configs:
        train_single_probe(cfg, activations, device_id=0)
```

### Async Judge
```python
# scripts/async_judge.py
import asyncio
import aiohttp

async def judge_debates(transcripts, max_concurrent=10):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def judge_one(session, transcript):
        async with semaphore:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json={"model": "gpt-4o", "messages": transcript},
                headers={"Authorization": f"Bearer {API_KEY}"}
            ) as resp:
                return await resp.json()

    async with aiohttp.ClientSession() as session:
        tasks = [judge_one(session, t) for t in transcripts]
        return await asyncio.gather(*tasks)

# Usage
results = asyncio.run(judge_debates(all_transcripts, max_concurrent=20))
```

### Both Starts First Debate
```bash
# Run two debates in parallel with opposite first speakers
python3 scripts/run_parallel_debates.py \
    --story_file stories/story_001.json \
    --both_starts_first \
    --gpu_mapping "0,1,2,3:honest_first;4,5,6,7:dishonest_first" \
    --output_dir debates/story_001/
```
