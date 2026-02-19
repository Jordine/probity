"""
Probe Sweep Orchestrator

For a given model, trains sample-level probes on NTML data across specified layers,
then evaluates across all domains (ai_liar, sandbagging, BashBench, APPS).

This script orchestrates the existing probe_training_ntml.py and
probe_eval_deception_datasets.py scripts, plus the new BashBench/APPS pipelines.

Usage:
    # Full sweep for Qwen-14B
    python scripts/run_probe_sweep.py \
        --model_name Qwen/Qwen2.5-14B-Instruct \
        --layers 12 18 24 30 36 42 \
        --train_dataset data/NTML-datasets/5T2L_300_tagged_direct_liar/5T2L_300samples_shuffled_explicit_direct_liar_prefix.json \
        --output_dir ./sweep_results/qwen14b \
        --bash_tasks_path /path/to/bash_tasks.json \
        --max_bash_tasks 50 \
        --max_apps_problems 50

    # Training only (skip eval)
    python scripts/run_probe_sweep.py \
        --model_name Qwen/Qwen2.5-14B-Instruct \
        --layers 12 18 24 30 36 42 \
        --train_dataset data/NTML-datasets/.../5T2L_300samples_shuffled_explicit_direct_liar_prefix.json \
        --output_dir ./sweep_results/qwen14b \
        --train_only

    # Eval only (skip training, use existing probes)
    python scripts/run_probe_sweep.py \
        --model_name Qwen/Qwen2.5-14B-Instruct \
        --layers 12 18 24 30 36 42 \
        --output_dir ./sweep_results/qwen14b \
        --eval_only
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


# ============================================================
# Model configs: layer selections per model size
# ============================================================

MODEL_LAYER_CONFIGS = {
    "Qwen/Qwen2.5-7B-Instruct": {
        "layers": [7, 10, 14, 18, 21, 24],
        "n_layers": 28,
        "gpu": "1x4090",
    },
    "Qwen/Qwen2.5-14B-Instruct": {
        "layers": [12, 18, 24, 30, 36, 42],
        "n_layers": 48,
        "gpu": "1xA100",
    },
    "Qwen/Qwen2.5-32B-Instruct": {
        "layers": [16, 24, 32, 40, 48, 56],
        "n_layers": 64,
        "gpu": "1xA100-80GB",
    },
    "Qwen/Qwen2.5-72B-Instruct": {
        "layers": [20, 30, 40, 50, 60, 70],
        "n_layers": 80,
        "gpu": "2xH200",
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        "layers": [8, 12, 16, 20, 24, 28],
        "n_layers": 32,
        "gpu": "1xA100",
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "layers": [20, 30, 40, 50, 60, 70],
        "n_layers": 80,
        "gpu": "2xH200",
    },
}

# Default sweep config for probe training
DEFAULT_SWEEP_CONFIG = {
    "attention": [
        {"n_heads": 1, "temperature": 0.25, "name_suffix": "_h1_t0.25"},
        {"n_heads": 2, "temperature": 0.25, "name_suffix": "_h2_t0.25"},
        {"n_heads": 4, "temperature": 1.0, "name_suffix": "_h4_t1.0"},
    ],
}


def create_sweep_config(output_dir):
    """Create and save sweep config for probe training."""
    config_path = Path(output_dir) / "sweep_config.json"
    with open(config_path, "w") as f:
        json.dump(DEFAULT_SWEEP_CONFIG, f, indent=2)
    return config_path


def run_command(cmd, description, dry_run=False):
    """Run a shell command with logging."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*60}")

    if dry_run:
        print("[DRY RUN] Skipping execution")
        return 0

    start = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - start
    print(f"\n[{description}] Completed in {elapsed:.1f}s (exit code: {result.returncode})")
    return result.returncode


def train_probes(args, layers, sweep_config_path):
    """Train probes using probe_training_ntml.py."""
    # Find the training script
    script_dir = Path(__file__).resolve().parent.parent / "probity" / "scripts"
    train_script = script_dir / "probe_training_ntml.py"

    if not train_script.exists():
        print(f"ERROR: Training script not found at {train_script}")
        return 1

    probe_save_dir = Path(args.output_dir) / "trained_probes"

    cmd = [
        sys.executable, str(train_script),
        "--model_name", args.model_name,
        "--train_dataset_dir", args.train_dataset,
        "--probe_types", "attention",
        "--sweep_config", str(sweep_config_path),
        "--layers", *[str(l) for l in layers],
        "--probe_save_dir", str(probe_save_dir),
        "--max_length", str(args.max_length),
        "--num_epochs", "60",
        "--patience", "999",
        "--dishonest_mode", "all",
        "--honest_mode", "none",
        "--batch_size", str(args.batch_size),
    ]

    return run_command(cmd, f"Training probes on {len(layers)} layers", args.dry_run)


def eval_deception_datasets(args, layers):
    """Evaluate probes on deception detection datasets."""
    script_dir = Path(__file__).resolve().parent.parent / "probity" / "scripts"
    eval_script = script_dir / "probe_eval_deception_datasets.py"

    if not eval_script.exists():
        print(f"ERROR: Eval script not found at {eval_script}")
        return 1

    probe_dir = Path(args.output_dir) / "trained_probes"
    results_dir = Path(args.output_dir) / "deception_eval_results"

    # Find datasets
    data_dir = Path(args.output_dir).parent.parent / "data" / "deception_detection"
    datasets = []
    for name in ["ai_liar.jsonl", "sandbagging_v2__wmdp_mmlu.jsonl"]:
        ds_path = data_dir / name
        if ds_path.exists():
            datasets.append(str(ds_path))
        else:
            print(f"Warning: {ds_path} not found, skipping")

    if not datasets:
        print("No deception detection datasets found, skipping")
        return 0

    cmd = [
        sys.executable, str(eval_script),
        "--probe_dir", str(probe_dir),
        "--datasets", *datasets,
        "--results_dir", str(results_dir),
        "--model_name", args.model_name,
        "--batch_size", str(args.batch_size),
        "--compute_ranking_metrics",
    ]

    # Add labeled dir if it exists
    labeled_dir = data_dir.parent / "deception-detection-labelled" / "deception_detection_labeled"
    if labeled_dir.exists():
        cmd.extend(["--labeled_dir", str(labeled_dir)])

    return run_command(cmd, "Evaluating on deception detection datasets", args.dry_run)


def eval_bashbench(args, layers):
    """Run BashBench evaluation pipeline."""
    script = Path(__file__).resolve().parent / "run_bashbench_probes.py"
    probe_dir = Path(args.output_dir) / "trained_probes"
    bb_output = Path(args.output_dir) / "bashbench_results"

    if not args.bash_tasks_path:
        print("No --bash_tasks_path specified, skipping BashBench")
        return 0

    cmd = [
        sys.executable, str(script),
        "--model_name", args.model_name,
        "--tasks_path", args.bash_tasks_path,
        "--output_dir", str(bb_output),
        "--probe_dir", str(probe_dir),
        "--layers", *[str(l) for l in layers],
        "--max_tasks", str(args.max_bash_tasks),
        "--dtype", args.dtype,
    ]

    return run_command(cmd, "BashBench: generate + evaluate", args.dry_run)


def eval_apps(args, layers):
    """Run APPS evaluation pipeline."""
    script = Path(__file__).resolve().parent / "run_apps_probes.py"
    probe_dir = Path(args.output_dir) / "trained_probes"
    apps_output = Path(args.output_dir) / "apps_results"

    cmd = [
        sys.executable, str(script),
        "--model_name", args.model_name,
        "--output_dir", str(apps_output),
        "--probe_dir", str(probe_dir),
        "--layers", *[str(l) for l in layers],
        "--max_problems", str(args.max_apps_problems),
        "--dtype", args.dtype,
    ]

    return run_command(cmd, "APPS: format + evaluate", args.dry_run)


def summarize_results(output_dir):
    """Collect all results into a summary."""
    output_dir = Path(output_dir)
    summary = {"model": "", "layers": [], "results": {}}

    # Deception detection results
    dd_results = output_dir / "deception_eval_results"
    if dd_results.exists():
        for f in dd_results.glob("*.json"):
            with open(f) as fh:
                summary["results"][f"deception_{f.stem}"] = json.load(fh)

    # BashBench results
    bb_results = output_dir / "bashbench_results" / "bashbench_probe_results.json"
    if bb_results.exists():
        with open(bb_results) as f:
            summary["results"]["bashbench"] = json.load(f)

    # APPS results
    apps_results = output_dir / "apps_results" / "apps_probe_results.json"
    if apps_results.exists():
        with open(apps_results) as f:
            summary["results"]["apps"] = json.load(f)

    # Save summary
    summary_path = output_dir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSweep summary saved to {summary_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)

    for domain, domain_results in summary["results"].items():
        print(f"\n--- {domain} ---")
        if isinstance(domain_results, dict):
            for key, metrics in domain_results.items():
                if isinstance(metrics, dict) and "sample_auroc" in metrics:
                    print(f"  {key}: AUROC={metrics['sample_auroc']:.4f}")
                elif isinstance(metrics, dict) and "auroc" in metrics:
                    print(f"  {key}: AUROC={metrics['auroc']:.4f}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Probe Sweep Orchestrator")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layers to use. If not specified, uses model config defaults.")
    parser.add_argument("--train_dataset", type=str, default=None,
                        help="Path to NTML training dataset JSON")
    parser.add_argument("--output_dir", type=str, required=True)

    # Domain-specific options
    parser.add_argument("--bash_tasks_path", type=str, default=None)
    parser.add_argument("--max_bash_tasks", type=int, default=50)
    parser.add_argument("--max_apps_problems", type=int, default=50)

    # Training options
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="bfloat16")

    # Control flags
    parser.add_argument("--train_only", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--skip_bashbench", action="store_true")
    parser.add_argument("--skip_apps", action="store_true")
    parser.add_argument("--skip_deception", action="store_true")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve layers
    if args.layers:
        layers = args.layers
    elif args.model_name in MODEL_LAYER_CONFIGS:
        layers = MODEL_LAYER_CONFIGS[args.model_name]["layers"]
        print(f"Using default layers for {args.model_name}: {layers}")
    else:
        print(f"ERROR: No layers specified and no config found for {args.model_name}")
        print(f"Known models: {list(MODEL_LAYER_CONFIGS.keys())}")
        sys.exit(1)

    print("=" * 70)
    print("PROBE SWEEP ORCHESTRATOR")
    print(f"Model: {args.model_name}")
    print(f"Layers: {layers}")
    print(f"Output: {args.output_dir}")
    print("=" * 70)

    # Phase 1: Train probes
    if not args.eval_only:
        if not args.train_dataset:
            print("ERROR: --train_dataset required for training phase")
            sys.exit(1)

        sweep_config_path = create_sweep_config(output_dir)
        rc = train_probes(args, layers, sweep_config_path)
        if rc != 0:
            print(f"WARNING: Training exited with code {rc}")

    # Phase 2: Evaluate on deception detection datasets
    if not args.train_only and not args.skip_deception:
        eval_deception_datasets(args, layers)

    # Phase 3: BashBench
    if not args.train_only and not args.skip_bashbench:
        eval_bashbench(args, layers)

    # Phase 4: APPS
    if not args.train_only and not args.skip_apps:
        eval_apps(args, layers)

    # Summary
    summarize_results(output_dir)

    print("\nSweep complete!")


if __name__ == "__main__":
    main()
