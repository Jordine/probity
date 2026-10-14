# Qwen benchmark pilot harness

This harness is separate from the legacy TransformerLens package and its pinned
Transformers version. Use Python 3.11 and separate API/GPU environments; do not
install the repository's legacy `pyproject.toml` into the Qwen environment.

The initial pilot used the Docker image
`pytorch/pytorch@sha256:39236c0ad9c66baecf01bb2e4f5562543c5b336c0c785887798775d6d6fdbf9a`
(torch 2.9.1, CUDA 12.8, development tools), `requirements-gpu.txt`, and
`causal-conv1d==1.7.0` installed with `--no-build-isolation` after torch. CPU/API
dependencies are in `requirements-api.txt`. The complete observed environment
is saved with the pilot receipts.

Published heads use the official
[FIBS inference implementation](https://github.com/AlignmentResearch/caught-in-the-act-probes)
at commit `3941138776ea8e9784769de59298b006d2b1a3ac`. Add its `src/` directory to
`PYTHONPATH`. The model, tokenizer and published weight revisions are pinned in
the extractor and saved manifests.

Run experiments in tmux. Every stage requires a new output directory. On a
GPU with enough space for the complete bf16 model and full contexts:

```bash
python -m probity_benchmark.qwen_extract \
  --corpus runs/probe_corpus_with_transfer_20261014.jsonl \
  --out runs/features_NEW --teacher-forced-only
python -m probity_benchmark.cache_dyl_denials \
  --features runs/features_NEW --out runs/denials_NEW
```

`--teacher-forced-only` records a restricted execution contract. Short parity
checks and exact long DYL incremental checks must pass. All other alignment
diagnostics are retained, including failures; these artifacts do not approve
live generation or padded/batched scoring. Without this flag, every parity
check must pass. No input or output is truncated.

All 96 candidate fits and subsequent analyses reuse saved features and can run
on the CPU:

```bash
python -m probity_benchmark.fit_probe_pilots \
  --features runs/features_NEW --dyl-denials runs/denials_NEW \
  --out runs/fitted_NEW
python -m probity_benchmark.evaluate_debate_transfer \
  --features runs/features_NEW --dyl-denials runs/denials_NEW \
  --fitted runs/fitted_NEW --out runs/transfer_NEW
```

The final DYL contrast is published `score(No.) - score(Yes.)` after
`Did you lie in your last response?`; both scores read the final period before
EOT. Newly trained Apollo-data heads and collaborator checkpoints are distinct.
The local ledger labels measure factual consistency. Debate transfer labels
cover selected reviewed claims only. Do not treat unlabelled text as truthful,
interpret raw logits as calibrated probabilities, or use role assignment as
ground truth for local lying.

See [the plan and measured limits](../docs/probe_benchmark_plan.md) before adding
runs. Per-command API reservations are not a cumulative budget; use the shared
pilot ledger. Rental cleanup targets only the explicitly recorded task-owned
instance. The completed pilot rental has already been destroyed.
