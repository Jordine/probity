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
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

from probity.utils.dataset_loading import apply_chat_template_unified, detect_model_type, get_model_dtype


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


def load_model_with_hooks(model_name: str, device: str = "cuda"):
    """Load model with TransformerLens hooks for activation extraction."""
    from transformer_lens import HookedTransformer

    model_dtype = get_model_dtype(model_name)
    print(f"Using model dtype: {model_dtype}")

    print("Loading model via HookedTransformer...")
    try:
        model = HookedTransformer.from_pretrained_no_processing(
            model_name,
            device=device,
            n_devices=2,
            dtype=model_dtype,
        )
        print("Model loaded successfully with n_devices=2")
    except Exception as e:
        print(f"Error with n_devices=2: {e}")
        print("Trying with device_map='auto'...")

        # Fallback: load with device_map auto
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        model = HookedTransformer.from_pretrained(
            model_name,
            hf_model=hf_model,
            device=device,
            dtype=model_dtype,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
        )
        print("Model loaded with device_map='auto' fallback")

    model.eval()
    return model


def get_activations_for_text(model, tokenizer, text: str, layers: List[int], device: str = "cuda"):
    """Get activations for a single text at specified layers."""
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = inputs["input_ids"].to(model.cfg.device)

    # Run with cache
    hook_names = [f"blocks.{layer}.hook_resid_pre" for layer in layers]

    with torch.no_grad():
        _, cache = model.run_with_cache(
            input_ids,
            names_filter=hook_names,
            return_type=None
        )

    # Extract activations
    activations = {}
    for layer in layers:
        hook_name = f"blocks.{layer}.hook_resid_pre"
        activations[layer] = cache[hook_name].cpu()  # [1, seq_len, hidden_dim]

    # Get token strings
    token_ids = input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    return activations, tokens


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
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference (1 recommended for memory)')
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

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"\nLoading model {args.model_name}...")
    model = load_model_with_hooks(args.model_name, args.device)
    print("Model loaded!")

    # Format samples
    print("\nFormatting samples...")
    formatted_texts = []
    for sample in tqdm(samples, desc="Formatting"):
        messages = sample.get('messages', [])
        try:
            formatted = apply_chat_template_unified(tokenizer, messages, model_type, tokenize=False)
            formatted_texts.append(formatted)
        except Exception as e:
            print(f"Error formatting sample: {e}")
            formatted_texts.append(None)

    valid_count = sum(1 for t in formatted_texts if t is not None)
    print(f"Formatted {valid_count}/{len(samples)} samples")

    # Pre-load all probes
    print("\nLoading probes...")
    loaded_probes = {}
    for probe_key, probe_info in tqdm(all_probes.items(), desc="Loading probes"):
        try:
            probe = load_probe_from_checkpoint(probe_info['path'], args.device)
            loaded_probes[probe_key] = probe
        except Exception as e:
            print(f"Error loading {probe_key}: {e}")

    # Initialize score storage
    probe_scores = {key: [] for key in loaded_probes.keys()}

    # Process samples one at a time
    print("\nProcessing samples and computing scores...")
    for i, text in enumerate(tqdm(formatted_texts, desc="Processing samples")):
        if text is None:
            # Append NaN for skipped samples
            for key in loaded_probes.keys():
                probe_scores[key].append(np.nan)
            continue

        try:
            # Get activations for this sample
            activations, tokens = get_activations_for_text(model, tokenizer, text, layers, args.device)

            # Find assistant range (simplified - use all tokens for now)
            # In practice, you might want to find the actual assistant response range
            num_tokens = len(tokens)

            # Apply each probe
            for probe_key, probe in loaded_probes.items():
                layer = all_probes[probe_key]['layer']
                layer_acts = activations[layer]  # [1, seq_len, hidden_dim]

                # Use all tokens (or could filter to assistant response)
                acts = layer_acts[0, :, :]  # [seq_len, hidden_dim]

                if acts.numel() == 0:
                    probe_scores[probe_key].append(0.0)
                    continue

                with torch.no_grad():
                    # Get probe device
                    probe_device = next((p.device for p in probe.parameters()), None)
                    if probe_device is None:
                        probe_device = next((b.device for b in probe.buffers()), torch.device("cuda"))

                    acts = acts.to(device=probe_device, dtype=probe.dtype)
                    token_scores = probe(acts)

                    # Apply sigmoid for logistic probes
                    if probe.__class__.__name__ == 'LogisticProbe':
                        token_scores = torch.sigmoid(token_scores)

                    token_scores = token_scores.cpu().squeeze().numpy()
                    if token_scores.ndim == 0:
                        token_scores = np.array([token_scores.item()])

                mean_score = float(np.nanmean(token_scores))
                probe_scores[probe_key].append(mean_score)

            # Clear cache periodically
            if (i + 1) % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\nError processing sample {i}: {e}")
            for key in loaded_probes.keys():
                probe_scores[key].append(np.nan)

    # Compute thresholds for each probe
    print("\n\nComputing FPR thresholds...")
    results = {}

    for probe_key in tqdm(sorted(loaded_probes.keys()), desc="Computing thresholds"):
        probe_info = all_probes[probe_key]
        scores = np.array(probe_scores[probe_key])

        # Remove NaN values
        valid_scores = scores[~np.isnan(scores)]

        if len(valid_scores) < 10:
            print(f"\n{probe_key}: Too few valid scores ({len(valid_scores)})")
            results[probe_key] = {'error': f'Only {len(valid_scores)} valid scores'}
            continue

        # Compute thresholds
        threshold_1pct, fpr_1pct = compute_fpr_threshold(valid_scores, 0.01)
        threshold_5pct, fpr_5pct = compute_fpr_threshold(valid_scores, 0.05)
        threshold_10pct, fpr_10pct = compute_fpr_threshold(valid_scores, 0.10)

        results[probe_key] = {
            'probe_type': probe_info['probe_type'],
            'layer': probe_info['layer'],
            'probe_path': probe_info['path'],
            'n_samples': len(valid_scores),
            'score_stats': {
                'mean': float(np.mean(valid_scores)),
                'std': float(np.std(valid_scores)),
                'min': float(np.min(valid_scores)),
                'max': float(np.max(valid_scores)),
                'median': float(np.median(valid_scores))
            },
            'thresholds': {
                'fpr_1pct': {'threshold': threshold_1pct, 'actual_fpr': fpr_1pct},
                'fpr_5pct': {'threshold': threshold_5pct, 'actual_fpr': fpr_5pct},
                'fpr_10pct': {'threshold': threshold_10pct, 'actual_fpr': fpr_10pct}
            }
        }

        print(f"\n{probe_key}: mean={np.mean(valid_scores):.4f}, "
              f"1%FPR threshold={threshold_1pct:.4f}")

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
