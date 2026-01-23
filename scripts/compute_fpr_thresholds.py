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
from transformers import AutoTokenizer

from probity.evaluation.batch_evaluator import OptimizedBatchProbeEvaluator
from probity.utils.dataset_loading import apply_chat_template_unified, detect_model_type


def load_probe_from_checkpoint(probe_path: str, device: str):
    """Load probe from checkpoint file."""
    from probity.probes import (
        LogisticProbe, LinearProbe, PCAProbe,
        MeanDifferenceProbe, KMeansProbe,
        MLPProbe, AttentionProbe, SklearnLogisticProbe
    )
    from probity.probes.apollo_probe import ApolloProbe
    from probity.probes.directional import DirectionalProbe

    checkpoint = torch.load(probe_path, map_location=device, weights_only=False)

    probe_cls_map = {
        "LogisticProbe": LogisticProbe,
        "LinearProbe": LinearProbe,
        "PCAProbe": PCAProbe,
        "MeanDifferenceProbe": MeanDifferenceProbe,
        "KMeansProbe": KMeansProbe,
        "MLPProbe": MLPProbe,
        "AttentionProbe": AttentionProbe,
        "ApolloProbe": ApolloProbe,
        "SklearnLogisticProbe": SklearnLogisticProbe,
    }

    cls_name = checkpoint.get("probe_type")
    probe_cls = probe_cls_map.get(cls_name)

    if probe_cls is None:
        raise ValueError(f"Unknown probe class '{cls_name}'")

    config = checkpoint["config"]
    probe = probe_cls(config)
    probe.load_state_dict(checkpoint["state_dict"], strict=False)

    # For sklearn probes, manually set buffers
    if isinstance(probe, SklearnLogisticProbe):
        state_dict = checkpoint["state_dict"]
        if "unscaled_coef_" in state_dict:
            probe.register_buffer("unscaled_coef_", state_dict["unscaled_coef_"])
        if "intercept_" in state_dict:
            probe.register_buffer("intercept_", state_dict["intercept_"])
        probe.has_fit = True

    if isinstance(probe, DirectionalProbe):
        probe.has_fit = True

    probe = probe.to(device)
    probe.eval()
    return probe


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
    k = int(np.ceil(target_fpr * n))
    k = max(1, min(k, n-1))

    threshold = sorted_scores[k-1]
    actual_fpr = k / n

    return float(threshold), float(actual_fpr)


def find_assistant_range(messages: List[Dict]) -> Tuple[int, int]:
    """Find character range of assistant response in formatted text."""
    # Build the text up to and including assistant response
    text_before = ""
    assistant_text = ""

    for msg in messages:
        if msg['role'] == 'assistant':
            assistant_text = msg['content']
            break
        text_before += msg.get('content', '')

    # This is approximate - actual range depends on chat template
    return 0, len(assistant_text)


def main():
    parser = argparse.ArgumentParser(description='Compute FPR-calibrated thresholds on benign data')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.3-70B-Instruct')
    parser.add_argument('--benign_dataset', type=str, required=True,
                        help='Path to benign dataset (e.g., alpaca.jsonl)')
    parser.add_argument('--probe_dirs', nargs='+', required=True,
                        help='Directories containing probes to evaluate')
    parser.add_argument('--output_file', type=str, default='fpr_thresholds_alpaca.json',
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

    # Detect model type
    model_type = detect_model_type(args.model_name)
    print(f"Model type: {model_type}")

    # Collect all probes to evaluate
    all_probes = {}
    for probe_dir in args.probe_dirs:
        probe_path = Path(probe_dir)
        if not probe_path.exists():
            print(f"Warning: Probe directory {probe_dir} not found, skipping")
            continue

        # Find probe files recursively
        for probe_file in probe_path.rglob('layer_*_probe.pt'):
            probe_type = probe_file.parent.name
            layer = int(probe_file.stem.split('_')[1])
            key = f"{probe_type}_L{layer}"
            if key not in all_probes:
                all_probes[key] = {
                    'path': str(probe_file),
                    'probe_type': probe_type,
                    'layer': layer
                }

    print(f"\nFound {len(all_probes)} probes to evaluate:")
    for key in sorted(all_probes.keys())[:10]:
        print(f"  - {key}")
    if len(all_probes) > 10:
        print(f"  ... and {len(all_probes) - 10} more")

    # Get unique layers
    layers = sorted(set(p['layer'] for p in all_probes.values()))
    print(f"\nLayers to evaluate: {layers}")

    # Initialize evaluator
    print(f"\nInitializing evaluator with model {args.model_name}...")
    evaluator = OptimizedBatchProbeEvaluator(
        model_name=args.model_name,
        device=args.device
    )

    # Create tokenizer separately (evaluator.model.tokenizer doesn't work correctly)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format samples
    print("Formatting samples...")
    formatted_texts = []
    assistant_ranges = []

    for sample in tqdm(samples, desc="Formatting"):
        messages = sample.get('messages', [])
        try:
            formatted = apply_chat_template_unified(tokenizer, messages, model_type)

            # Find assistant token range (approximate)
            assistant_start = formatted.find('[/INST]') + 7 if '[/INST]' in formatted else 0
            if assistant_start == 6:  # Not found
                assistant_start = formatted.find('assistant') + 9
            assistant_end = len(formatted)

            formatted_texts.append(formatted)
            assistant_ranges.append((assistant_start, assistant_end))
        except Exception as e:
            print(f"Error formatting sample: {e}")
            continue

    print(f"Formatted {len(formatted_texts)} samples")

    # Get activations
    print("\nGetting activations...")
    activation_data = evaluator.get_batch_activations(
        formatted_texts, layers,
        batch_size=args.batch_size,
        disk_cache_dir="./cache/fpr_calibration"
    )
    activations = activation_data['activations']
    tokens_by_text = activation_data['tokens_by_text']

    # Evaluate each probe
    results = {}

    for probe_key in tqdm(sorted(all_probes.keys()), desc="Evaluating probes"):
        probe_info = all_probes[probe_key]
        probe_type = probe_info['probe_type']
        layer = probe_info['layer']
        probe_path = probe_info['path']

        try:
            # Load probe
            probe = load_probe_from_checkpoint(probe_path, args.device)

            # Get layer activations
            layer_activations = activations[layer]

            # Compute scores for all samples
            all_scores = []

            for i, (start_char, end_char) in enumerate(assistant_ranges):
                tokens = tokens_by_text[i]
                text = formatted_texts[i]

                # Map character positions to token positions
                char_count = 0
                start_token = 0
                end_token = len(tokens)

                for j, token in enumerate(tokens):
                    token_str = str(token).replace('Ġ', ' ').replace('Ċ', '\n').replace('▁', ' ')
                    if char_count >= start_char and start_token == 0:
                        start_token = j
                    if char_count >= end_char:
                        end_token = j
                        break
                    char_count += len(token_str)

                # Get activations for assistant tokens
                assistant_activations = layer_activations[i, start_token:end_token, :]

                if assistant_activations.numel() == 0:
                    all_scores.append(0.0)
                    continue

                # Apply probe
                with torch.no_grad():
                    probe_device = next((p.device for p in probe.parameters()), None)
                    if probe_device is None:
                        probe_device = next((b.device for b in probe.buffers()), torch.device("cuda"))

                    assistant_activations = assistant_activations.to(device=probe_device, dtype=probe.dtype)
                    token_scores = probe(assistant_activations)

                    # Apply sigmoid for logistic probes
                    if probe.__class__.__name__ == 'LogisticProbe':
                        token_scores = torch.sigmoid(token_scores)

                    token_scores = token_scores.cpu().squeeze().numpy()
                    if token_scores.ndim == 0:
                        token_scores = np.array([token_scores.item()])

                mean_score = float(np.mean(token_scores))
                all_scores.append(mean_score)

            scores_array = np.array(all_scores)

            # Compute thresholds
            threshold_1pct, fpr_1pct = compute_fpr_threshold(scores_array, 0.01)
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
                    'fpr_1pct': {'threshold': threshold_1pct, 'actual_fpr': fpr_1pct},
                    'fpr_5pct': {'threshold': threshold_5pct, 'actual_fpr': fpr_5pct},
                    'fpr_10pct': {'threshold': threshold_10pct, 'actual_fpr': fpr_10pct}
                }
            }

            print(f"\n{probe_key}: mean={np.mean(scores_array):.4f}, "
                  f"1%FPR threshold={threshold_1pct:.4f}")

        except Exception as e:
            print(f"\nError evaluating {probe_key}: {e}")
            import traceback
            traceback.print_exc()
            results[probe_key] = {'error': str(e)}

    # Save results
    output_path = Path(args.output_file)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved FPR thresholds to {output_path}")

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY: FPR-Calibrated Thresholds (1% FPR on Alpaca)")
    print("="*80)
    print(f"{'Probe':<25} {'Mean Score':>12} {'1% FPR Thresh':>15} {'5% FPR Thresh':>15}")
    print("-"*80)

    for key in sorted(results.keys()):
        if 'error' in results[key]:
            continue
        r = results[key]
        print(f"{key:<25} {r['score_stats']['mean']:>12.4f} "
              f"{r['thresholds']['fpr_1pct']['threshold']:>15.4f} "
              f"{r['thresholds']['fpr_5pct']['threshold']:>15.4f}")


if __name__ == '__main__':
    main()
