#!/usr/bin/env python3
"""
Compute FPR-calibrated thresholds on a benign dataset (e.g., Alpaca).

For each probe, finds the threshold that gives a specified FPR (default 1%) on benign data.
These thresholds can then be used for fair comparison of span metrics across probes.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from tqdm import tqdm

from probity.evaluation.batch_evaluator import OptimizedBatchProbeEvaluator
from probity.utils.dataset_loading import apply_chat_template_unified, detect_model_type


def load_benign_dataset(dataset_path: str, limit: int = None) -> List[Dict]:
    """Load benign dataset samples."""
    samples = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            sample = json.loads(line)
            # Only keep samples labeled as honest/benign
            if sample.get('label', 'honest') in ['honest', 'benign', 'truthful']:
                samples.append(sample)
    return samples


def compute_fpr_threshold(scores: np.ndarray, target_fpr: float = 0.01) -> Tuple[float, float]:
    """
    Compute threshold that achieves target FPR on benign data.

    Since these are all benign samples, any sample flagged as deceptive is a false positive.
    FPR = (samples above threshold) / (total samples)

    Returns: (threshold, actual_fpr)
    """
    sorted_scores = np.sort(scores)[::-1]  # Sort descending
    n = len(sorted_scores)

    # Find index where FPR = target_fpr
    # FPR = k/n where k is number of samples above threshold
    k = int(np.ceil(target_fpr * n))
    k = max(1, min(k, n-1))  # Ensure valid index

    # Threshold is between score[k-1] and score[k]
    threshold = sorted_scores[k-1]  # Score at the k-th highest position
    actual_fpr = k / n

    return float(threshold), float(actual_fpr)


def main():
    parser = argparse.ArgumentParser(description='Compute FPR-calibrated thresholds on benign data')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.3-70B-Instruct')
    parser.add_argument('--benign_dataset', type=str, required=True,
                        help='Path to benign dataset (e.g., alpaca.jsonl)')
    parser.add_argument('--probe_dirs', nargs='+', required=True,
                        help='Directories containing probes to evaluate')
    parser.add_argument('--output_file', type=str, default='fpr_thresholds.json',
                        help='Output file for computed thresholds')
    parser.add_argument('--target_fpr', type=float, default=0.01,
                        help='Target FPR (default: 0.01 = 1%%)')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of samples (for testing)')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print(f"Loading benign dataset from {args.benign_dataset}...")
    samples = load_benign_dataset(args.benign_dataset, args.limit)
    print(f"Loaded {len(samples)} benign samples")

    # Detect model type and prepare tokenizer
    model_type = detect_model_type(args.model_name)
    print(f"Model type: {model_type}")

    # Collect all probes to evaluate
    all_probes = {}
    for probe_dir in args.probe_dirs:
        probe_path = Path(probe_dir)
        if not probe_path.exists():
            print(f"Warning: Probe directory {probe_dir} not found, skipping")
            continue

        # Find probe files
        for probe_type_dir in probe_path.iterdir():
            if probe_type_dir.is_dir():
                probe_type = probe_type_dir.name
                for probe_file in probe_type_dir.glob('layer_*_probe.pt'):
                    layer = int(probe_file.stem.split('_')[1])
                    key = f"{probe_type}_L{layer}"
                    all_probes[key] = {
                        'path': str(probe_file),
                        'probe_type': probe_type,
                        'layer': layer,
                        'probe_dir': str(probe_path)
                    }

        # Also check for probes directly in the directory (like Apollo)
        for probe_file in probe_path.glob('**/layer_*_probe.pt'):
            probe_type = probe_file.parent.name
            layer = int(probe_file.stem.split('_')[1])
            key = f"{probe_type}_L{layer}"
            if key not in all_probes:
                all_probes[key] = {
                    'path': str(probe_file),
                    'probe_type': probe_type,
                    'layer': layer,
                    'probe_dir': str(probe_path)
                }

    print(f"\nFound {len(all_probes)} probes to evaluate:")
    for key in sorted(all_probes.keys()):
        print(f"  - {key}")

    # Initialize evaluator
    print(f"\nInitializing evaluator with model {args.model_name}...")
    evaluator = OptimizedBatchProbeEvaluator(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size
    )

    # Prepare samples for evaluation
    formatted_samples = []
    for sample in samples:
        messages = sample.get('messages', [])
        formatted = apply_chat_template_unified(messages, evaluator.tokenizer, model_type)
        formatted_samples.append({
            'text': formatted,
            'label': 0,  # All benign = 0
            'original': sample
        })

    # Evaluate each probe and compute thresholds
    results = {}

    for probe_key, probe_info in tqdm(sorted(all_probes.items()), desc="Evaluating probes"):
        probe_type = probe_info['probe_type']
        layer = probe_info['layer']
        probe_path = probe_info['path']

        print(f"\n{'='*60}")
        print(f"Evaluating {probe_key}...")

        try:
            # Load probe
            evaluator.load_probe(probe_path, probe_type, layer)

            # Get scores for all samples
            all_scores = []
            for i in range(0, len(formatted_samples), args.batch_size):
                batch = formatted_samples[i:i+args.batch_size]
                texts = [s['text'] for s in batch]

                batch_results = evaluator.evaluate_batch(texts)
                for result in batch_results:
                    # Use mean score as sample-level score
                    scores = result.get('token_scores', [])
                    if scores:
                        sample_score = float(np.mean(scores))
                    else:
                        sample_score = 0.0
                    all_scores.append(sample_score)

            scores_array = np.array(all_scores)

            # Compute threshold for target FPR
            threshold, actual_fpr = compute_fpr_threshold(scores_array, args.target_fpr)

            # Also compute some other useful thresholds
            threshold_5pct, fpr_5pct = compute_fpr_threshold(scores_array, 0.05)
            threshold_10pct, fpr_10pct = compute_fpr_threshold(scores_array, 0.10)

            results[probe_key] = {
                'probe_type': probe_type,
                'layer': layer,
                'probe_path': probe_path,
                'n_samples': len(all_scores),
                'score_stats': {
                    'mean': float(np.mean(scores_array)),
                    'std': float(np.std(scores_array)),
                    'min': float(np.min(scores_array)),
                    'max': float(np.max(scores_array)),
                    'median': float(np.median(scores_array))
                },
                'thresholds': {
                    'fpr_1pct': {
                        'threshold': threshold,
                        'actual_fpr': actual_fpr,
                        'target_fpr': args.target_fpr
                    },
                    'fpr_5pct': {
                        'threshold': threshold_5pct,
                        'actual_fpr': fpr_5pct,
                        'target_fpr': 0.05
                    },
                    'fpr_10pct': {
                        'threshold': threshold_10pct,
                        'actual_fpr': fpr_10pct,
                        'target_fpr': 0.10
                    }
                }
            }

            print(f"  Scores: mean={np.mean(scores_array):.4f}, std={np.std(scores_array):.4f}")
            print(f"  FPR 1% threshold: {threshold:.4f} (actual FPR: {actual_fpr:.3f})")
            print(f"  FPR 5% threshold: {threshold_5pct:.4f} (actual FPR: {fpr_5pct:.3f})")

        except Exception as e:
            print(f"  Error evaluating {probe_key}: {e}")
            results[probe_key] = {'error': str(e)}

    # Save results
    output_path = Path(args.output_file)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved FPR thresholds to {output_path}")

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY: FPR-Calibrated Thresholds")
    print("="*80)
    print(f"{'Probe':<25} {'Mean Score':>12} {'1% FPR Thresh':>15} {'5% FPR Thresh':>15}")
    print("-"*80)

    for key in sorted(results.keys()):
        if 'error' in results[key]:
            print(f"{key:<25} ERROR: {results[key]['error'][:40]}")
        else:
            r = results[key]
            print(f"{key:<25} {r['score_stats']['mean']:>12.4f} "
                  f"{r['thresholds']['fpr_1pct']['threshold']:>15.4f} "
                  f"{r['thresholds']['fpr_5pct']['threshold']:>15.4f}")


if __name__ == '__main__':
    main()
