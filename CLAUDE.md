# Probity Project

Deception probes, debate protocols, deception detection amplifying oversight. Making models argue and seeing if we can catch dishonesty.

**Main code:** `probity/` subdirectory
**Owner:** Jord (jordine on GitHub)

---

## No truncation — anywhere in this project (Jord, 2026-08-22)

> *"there should not be any truncation anywhere in the project — not in the training, in sampling
> rollouts, in printing examples, no truncation is allowed anywhere. truncation is banned. always
> test to see if things are being truncated."*

Unconditional, and it has two halves:

- **Generation** — `max_tokens` / `max_new_tokens` / think budgets sized for the *longest legitimate*
  response. Print the `finish_reason` distribution before any aggregate. With a two-phase sampler
  there are **two** truncation events: gate on the flag for the **thinking** being cut (`forced=True`
  in `tinker_io`), not on `stop_reason == "length"`, which is a strictly smaller set. A truncated
  rollout is a **corrupted datum, not a short one** — in RL it earns ~0 and the gradient pushes away
  from long reasoning; in evals it mixes a high-accuracy complete population with a low-accuracy cut
  one. Non-trivial rate = **stop-and-fix**, not a caveat.
- **Your own tooling** — `head`, `tail`, `sed -n '1,200p'`, `[:N]` slices, regex context windows,
  `df.head()`, pandas `max_colwidth`, `text-overflow: ellipsis` in a table. An excerpt whose width
  *you* chose is not evidence about the row, and a quote shown truncated to Jord is a misquote she is
  being asked to gate on.

**Always test — affirmatively.** Truncation is silent by construction; "no error appeared" is not a
pass, because the bytes that got cut cannot raise their hand. Check: bytes read vs file size, rows
parsed vs `wc -l`, items in vs items out, no field length landing exactly on the cap, and a length
distribution whose right tail sits **below** the cap (p90 at the cap = everything is cut). Where an
excerpt is genuinely unavoidable, mark the elision: `… [+3,412 chars]`.

Full rule + receipts: `~/.claude/CLAUDE.md` §No truncation · ops checklist: `/preflight` §11 ·
project requirements: `exocortex/skills/experiment-scaffold.md` §6.


## Critical Patterns (READ THIS BEFORE RUNNING ANYTHING)

### The Three Core Scripts

1. **`probity/data/generate_contrastive_ntml_datasets.py`** - Generate training data
2. **`probity/scripts/probe_training_ntml.py`** - Train probes
3. **`probity/scripts/probe_eval_deception_datasets.py`** - Evaluate probes

### CRITICAL: Efficient Multi-Probe Training

**DO NOT** train probes one-by-one. Use `--sweep_config` to train multiple probes from single activation collection:

```bash
# Create sweep_config.json:
{
  "attention": [
    {"n_heads": 1, "temperature": 0.25, "name_suffix": "_h1_t0.25"},
    {"n_heads": 2, "temperature": 0.25, "name_suffix": "_h2_t0.25"}
  ]
}

# Train all variants with ONE activation collection per layer:
python scripts/probe_training_ntml.py \
  --model_name meta-llama/Llama-3.3-70B-Instruct \
  --train_dataset_dir ./data.json \
  --probe_types attention \
  --sweep_config sweep_config.json \
  --layers 22 24 26 \
  --probe_save_dir ./probes \
  --max_length 1536 \
  --num_epochs 60 --patience 999 \
  --dishonest_mode all --honest_mode none \
  --batch_size 4
```

### Caching System

- Cache location: `cache/contrastive/`, `cache/val_acts/`
- Activations cached by model + dataset hash
- **Reused automatically** if same model + dataset
- Clear with `rm -rf cache/contrastive/* cache/val_acts/*` between different datasets

### Key Training Parameters

| Param | Meaning | Typical |
|-------|---------|---------|
| `--dishonest_mode all` | Use all tokens from lie samples | Required |
| `--honest_mode none` | Don't use truth samples as negatives | **Critical for good AUROC** |
| `--max_length` | Token sequence length | 1024-2048 |
| `--batch_size` | For 70B models | 4-8 |

### Dataset Generation

```bash
python data/generate_contrastive_ntml_datasets.py \
  --input data/truth_and_lies.json \
  --ratio 12T3L \        # 12 truths, 3 lies per conversation
  --samples 300 \
  --shuffle_response \   # Randomize order (removes positional bias)
  --instruction_style implicit \  # or direct_liar, roleplay, etc.
  --output ./output_dir
```

### Evaluation

```bash
python scripts/probe_eval_deception_datasets.py \
  --probe_dir ./probes \
  --datasets data/deception_detection/ai_liar.jsonl data/deception_detection/sandbagging_v2__wmdp_mmlu.jsonl \
  --labeled_dir ./deception-detection-labelled/deception_detection_labeled \
  --results_dir ./eval_results \
  --model_name meta-llama/Llama-3.3-70B-Instruct \
  --batch_size 4 \
  --compute_ranking_metrics
```

### Cluster Access

Run `/ssh-vast` for the full SSH + vast.ai setup guide.

Quick reference:
```bash
# Check current instance
vastai show instances

# SSH in (update host/port from dashboard)
ssh -i "C:\Users\Admin\grongles" -p PORT root@HOST

# Always run experiments in tmux
tmux new -s experiment
```

### Lessons Learned (Don't Repeat These Mistakes)

1. **Always use `--sweep_config`** for multiple hyperparams - don't loop and train one-by-one
2. **`honest_mode=none` is critical** - using `diff` causes overfitting and bad AUROC
3. **Run training in tmux** - SSH disconnects kill the process
4. **Clear cache between different datasets** but NOT between same-dataset probes
5. **Read the script args** before running: `python script.py --help`
