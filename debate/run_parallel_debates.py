#!/usr/bin/env python3
"""
Parallel debate runner for 8xH200 clusters.

Runs two debate variants simultaneously:
- Group 1 (GPUs 0-3): All stories with honest_first=True
- Group 2 (GPUs 4-7): All stories with honest_first=False

This produces 2x the debate samples in ~same wall-clock time.

Usage:
    # Run both orderings in parallel on 8xH200
    python run_parallel_debates.py \
        --both_starts_first \
        --honest_model meta-llama/Llama-3.3-70B-Instruct \
        --dishonest_model meta-llama/Llama-3.3-70B-Instruct \
        --dataset quality_synthetic \
        --n_problems 20 \
        --experiment_mode baseline_judge \
        --save_dir ./parallel_debates

    # Custom GPU mapping
    python run_parallel_debates.py \
        --both_starts_first \
        --honest_first_gpus 0 1 2 3 \
        --dishonest_first_gpus 4 5 6 7 \
        ...

    # Run only one ordering (backward compatible)
    python run_parallel_debates.py \
        --honest_goes_first \
        --honest_model meta-llama/Llama-3.3-70B-Instruct \
        ...
"""

import argparse
import os
import sys
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple
from multiprocessing import Process, Queue
import shutil


def build_debate_command(
    args: argparse.Namespace,
    honest_first: bool,
    gpu_indices: List[int],
    output_suffix: str,
) -> List[str]:
    """Build the command to run a single debate variant.

    Args:
        args: Parsed command line arguments
        honest_first: Whether honest debater goes first
        gpu_indices: List of GPU indices for this variant
        output_suffix: Suffix for output directory (e.g., 'honest_first')

    Returns:
        Command as a list of strings
    """
    # Split GPUs between honest and dishonest debaters
    n_gpus = len(gpu_indices)
    if n_gpus >= 2:
        # Split GPUs: first half for honest, second half for dishonest
        mid = n_gpus // 2
        honest_gpus = gpu_indices[:mid]
        dishonest_gpus = gpu_indices[mid:]
    else:
        # Single GPU for both
        honest_gpus = gpu_indices
        dishonest_gpus = gpu_indices

    # Base command
    cmd = [
        sys.executable,
        "-m", "debate.run_debate_no_judge",
        "--experiment_mode", args.experiment_mode,
        "--honest_model", args.honest_model,
        "--dishonest_model", args.dishonest_model,
        "--honest_provider", args.honest_provider,
        "--dishonest_provider", args.dishonest_provider,
        "--dataset", args.dataset,
        "--n_problems", str(args.n_problems),
        "--max_rounds", str(args.max_rounds),
        "--dishonesty_level", args.dishonesty_level,
        "--seed", str(args.seed),
    ]

    # GPU indices
    cmd.extend(["--honest_gpu_indices"] + [str(g) for g in honest_gpus])
    cmd.extend(["--dishonest_gpu_indices"] + [str(g) for g in dishonest_gpus])

    # Output directory with suffix
    experiment_name = args.experiment_name or f"{args.experiment_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = Path(args.save_dir) / f"{experiment_name}_{output_suffix}"
    cmd.extend(["--save_dir", str(save_dir.parent)])
    cmd.extend(["--experiment_name", save_dir.name])

    # Dataset path
    if args.dataset_path:
        cmd.extend(["--dataset_path", args.dataset_path])

    # Probe configuration
    if args.honest_probe_dir:
        cmd.extend(["--honest_probe_dir", args.honest_probe_dir])
    if args.dishonest_probe_dir:
        cmd.extend(["--dishonest_probe_dir", args.dishonest_probe_dir])
    if args.probe_types:
        cmd.extend(["--probe_types"] + args.probe_types)
    if args.probe_layer is not None:
        cmd.extend(["--probe_layer", str(args.probe_layer)])

    # API keys (if using API providers)
    if args.openai_api_key:
        cmd.extend(["--openai_api_key", args.openai_api_key])
    if args.anthropic_api_key:
        cmd.extend(["--anthropic_api_key", args.anthropic_api_key])
    if args.openrouter_api_key:
        cmd.extend(["--openrouter_api_key", args.openrouter_api_key])

    # Force speaker order using the new --force_honest_first/--force_dishonest_first flags
    if honest_first:
        cmd.append("--force_honest_first")
    else:
        cmd.append("--force_dishonest_first")

    # Difficulty for APPS
    if args.difficulty:
        cmd.extend(["--difficulty", args.difficulty])

    return cmd, honest_first


def run_debate_variant(
    cmd: List[str],
    honest_first: bool,
    result_queue: Queue,
    env_override: Optional[dict] = None,
) -> None:
    """Run a single debate variant as a subprocess.

    Args:
        cmd: Command to run
        honest_first: Whether this is the honest_first variant
        result_queue: Queue to put results
        env_override: Optional environment variable overrides
    """
    variant_name = "honest_first" if honest_first else "dishonest_first"
    print(f"\n{'='*60}")
    print(f"Starting {variant_name} debates...")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    # Setup environment
    env = os.environ.copy()
    if env_override:
        env.update(env_override)

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=False,  # Let output go to console
            text=True,
            cwd=str(Path(__file__).parent.parent),  # Run from probity directory
            timeout=400,  # 400-second timeout
        )

        duration = time.time() - start_time
        success = result.returncode == 0

        result_queue.put({
            'variant': variant_name,
            'success': success,
            'returncode': result.returncode,
            'duration': duration,
        })

    except subprocess.TimeoutExpired:
        result_queue.put({
            'variant': variant_name,
            'success': False,
            'error': 'Debate timed out after 400 seconds',
            'duration': time.time() - start_time,
        })
    except Exception as e:
        result_queue.put({
            'variant': variant_name,
            'success': False,
            'error': str(e),
            'duration': time.time() - start_time,
        })


def run_parallel_debates(args: argparse.Namespace) -> int:
    """Run both debate orderings in parallel.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code (0 for success)
    """
    print("\n" + "="*70)
    print("PARALLEL DEBATE RUNNER")
    print("="*70)
    print(f"Mode: both_starts_first (running both orderings in parallel)")
    print(f"GPUs for honest_first: {args.honest_first_gpus}")
    print(f"GPUs for dishonest_first: {args.dishonest_first_gpus}")
    print("="*70 + "\n")

    # Build commands for both variants
    honest_first_cmd, _ = build_debate_command(
        args,
        honest_first=True,
        gpu_indices=args.honest_first_gpus,
        output_suffix="honest_first"
    )

    dishonest_first_cmd, _ = build_debate_command(
        args,
        honest_first=False,
        gpu_indices=args.dishonest_first_gpus,
        output_suffix="dishonest_first"
    )

    # Result queue for collecting outcomes
    result_queue = Queue()

    # Start both processes
    processes = []

    p1 = Process(
        target=run_debate_variant,
        args=(honest_first_cmd, True, result_queue),
    )
    p1.start()
    processes.append(('honest_first', p1))

    p2 = Process(
        target=run_debate_variant,
        args=(dishonest_first_cmd, False, result_queue),
    )
    p2.start()
    processes.append(('dishonest_first', p2))

    # Wait for both to complete
    print("Waiting for both debate variants to complete...")
    for name, p in processes:
        p.join()
        print(f"  {name} process finished")

    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    # Print summary
    print("\n" + "="*70)
    print("PARALLEL DEBATE SUMMARY")
    print("="*70)

    all_success = True
    for result in results:
        variant = result['variant']
        success = result.get('success', False)
        duration = result.get('duration', 0)

        status = "SUCCESS" if success else "FAILED"
        print(f"  {variant}: {status} (duration: {duration/60:.1f} min)")

        if not success:
            all_success = False
            if 'error' in result:
                print(f"    Error: {result['error']}")
            if 'returncode' in result:
                print(f"    Return code: {result['returncode']}")

    print("="*70)

    # Save combined summary
    experiment_name = args.experiment_name or f"{args.experiment_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    summary_file = Path(args.save_dir) / f"{experiment_name}_parallel_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, 'w') as f:
        json.dump({
            'experiment_name': experiment_name,
            'mode': 'both_starts_first',
            'timestamp': datetime.now().isoformat(),
            'config': {
                'honest_model': args.honest_model,
                'dishonest_model': args.dishonest_model,
                'dataset': args.dataset,
                'n_problems': args.n_problems,
                'max_rounds': args.max_rounds,
            },
            'gpu_mapping': {
                'honest_first': args.honest_first_gpus,
                'dishonest_first': args.dishonest_first_gpus,
            },
            'results': results,
            'all_success': all_success,
        }, f, indent=2)

    print(f"\nSummary saved to: {summary_file}")
    print(f"Transcripts in: {Path(args.save_dir) / experiment_name}_honest_first/")
    print(f"             and: {Path(args.save_dir) / experiment_name}_dishonest_first/")

    return 0 if all_success else 1


def run_single_ordering(args: argparse.Namespace) -> int:
    """Run debates with a single ordering (backward compatible mode).

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code
    """
    honest_first = args.honest_goes_first
    variant_name = "honest_first" if honest_first else "dishonest_first"

    # Use all available GPUs for single ordering
    all_gpus = args.honest_first_gpus + args.dishonest_first_gpus
    all_gpus = sorted(set(all_gpus))

    print(f"\nRunning single ordering: {variant_name}")
    print(f"Using GPUs: {all_gpus}")

    cmd, _ = build_debate_command(
        args,
        honest_first=honest_first,
        gpu_indices=all_gpus,
        output_suffix=variant_name
    )

    env = os.environ.copy()
    env['FORCE_HONEST_FIRST'] = 'true' if honest_first else 'false'

    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(Path(__file__).parent.parent),
            timeout=400,  # 400-second timeout
        )
        return result.returncode
    except subprocess.TimeoutExpired:
        print(f"ERROR: Debate timed out after 400 seconds")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description='Run parallel debates with both speaker orderings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--both_starts_first', action='store_true',
                           help='Run BOTH orderings in parallel (2x debates, ~same time)')
    mode_group.add_argument('--honest_goes_first', action='store_true',
                           help='Run only with honest debater speaking first')
    mode_group.add_argument('--dishonest_goes_first', action='store_true',
                           help='Run only with dishonest debater speaking first')

    # GPU mapping for parallel mode
    parser.add_argument('--honest_first_gpus', nargs='+', type=int, default=[0, 1, 2, 3],
                       help='GPU indices for honest_first variant (default: 0 1 2 3)')
    parser.add_argument('--dishonest_first_gpus', nargs='+', type=int, default=[4, 5, 6, 7],
                       help='GPU indices for dishonest_first variant (default: 4 5 6 7)')

    # Experiment mode
    parser.add_argument('--experiment_mode', type=str, required=True,
                       choices=['baseline_judge', 'debater_full'],
                       help='Experiment mode: baseline_judge or debater_full')

    # Model configuration
    parser.add_argument('--honest_model', type=str, required=True,
                       help='Model for honest debater')
    parser.add_argument('--dishonest_model', type=str, required=True,
                       help='Model for dishonest debater')
    parser.add_argument('--honest_provider', type=str, default='fast_local',
                       choices=['local', 'fast_local', 'openai', 'anthropic', 'openrouter'])
    parser.add_argument('--dishonest_provider', type=str, default='fast_local',
                       choices=['local', 'fast_local', 'openai', 'anthropic', 'openrouter'])

    # API keys
    parser.add_argument('--openai_api_key', type=str)
    parser.add_argument('--anthropic_api_key', type=str)
    parser.add_argument('--openrouter_api_key', type=str)

    # Probe configuration
    parser.add_argument('--honest_probe_dir', type=str,
                       help='Directory with trained probes for honest model')
    parser.add_argument('--dishonest_probe_dir', type=str,
                       help='Directory with trained probes for dishonest model')
    parser.add_argument('--probe_types', nargs='+',
                       default=['logistic', 'pca', 'meandiff'])
    parser.add_argument('--probe_layer', type=int, default=15)

    # Dataset configuration
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['apps', 'simple_debate', 'quality', 'quality_synthetic'])
    parser.add_argument('--dataset_path', type=str,
                       help='Path to dataset file')
    parser.add_argument('--difficulty', type=str,
                       choices=['introductory', 'interview', 'competition'],
                       help='APPS difficulty filter')
    parser.add_argument('--n_problems', type=int, default=10,
                       help='Number of problems to run')

    # Debate parameters
    parser.add_argument('--dishonesty_level', type=str, default='selective',
                       choices=['symmetric', 'persuasive', 'selective', 'explicit'])
    parser.add_argument('--max_rounds', type=int, default=3,
                       help='Maximum debate rounds')
    parser.add_argument('--save_dir', type=str, default='./parallel_debate_transcripts',
                       help='Directory to save transcripts')
    parser.add_argument('--experiment_name', type=str,
                       help='Name for this experiment')

    # Misc
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Validate GPU assignments for parallel mode
    if args.both_starts_first:
        honest_set = set(args.honest_first_gpus)
        dishonest_set = set(args.dishonest_first_gpus)
        overlap = honest_set & dishonest_set
        if overlap:
            print(f"WARNING: GPU overlap detected: {overlap}")
            print("Both variants will compete for the same GPUs, which may cause OOM.")
            print("Consider using disjoint GPU sets for optimal performance.")

    # Run based on mode
    if args.both_starts_first:
        return run_parallel_debates(args)
    else:
        return run_single_ordering(args)


if __name__ == "__main__":
    exit(main())
