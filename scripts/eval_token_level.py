"""Evaluate token-level probes: compute token-level AUROC and norm_auroc (within-sample z-scored AUROC).

Usage:
    python scripts/eval_token_level.py \
        --probe_dir ./probes/token_level_qwen7b \
        --cache_dir ./cache/token_level \
        --model_name Qwen/Qwen2.5-7B-Instruct \
        --dataset_path ./data/NTML-datasets/5T2L_300_tagged_direct_liar/5T2L_300samples_shuffled_explicit_direct_liar_prefix.json
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from probity.utils.dataset_loading import load_ntml_token_level
from probity.collection.activation_store import ActivationStore


def compute_token_metrics(scores, labels, mask):
    """Compute token-level AUROC and norm_auroc across all examples.

    Args:
        scores: (num_examples, seq_len) probe output logits
        labels: (num_examples, seq_len) binary token labels
        mask: (num_examples, seq_len) attention mask

    Returns:
        dict with global_auroc, norm_auroc, per_example stats
    """
    # Flatten for global AUROC
    all_scores = []
    all_labels = []

    # Per-example for norm_auroc
    norm_aurocs = []
    per_example = []

    for i in range(scores.shape[0]):
        m = mask[i].bool()
        s = scores[i][m].numpy()
        l = labels[i][m].numpy().astype(int)

        all_scores.append(s)
        all_labels.append(l)

        n_pos = l.sum()
        n_neg = len(l) - n_pos

        example_info = {
            'idx': i,
            'n_tokens': len(l),
            'n_pos': int(n_pos),
            'n_neg': int(n_neg),
            'mean_score_pos': float(s[l == 1].mean()) if n_pos > 0 else None,
            'mean_score_neg': float(s[l == 0].mean()) if n_neg > 0 else None,
        }

        # Within-sample z-scored AUROC
        if n_pos > 0 and n_neg > 0:
            std = s.std()
            if std > 1e-8:
                z = (s - s.mean()) / std
                try:
                    na = roc_auc_score(l, z)
                    norm_aurocs.append(na)
                    example_info['norm_auroc'] = float(na)
                except ValueError:
                    example_info['norm_auroc'] = 0.5
            else:
                example_info['norm_auroc'] = 0.5
        else:
            example_info['norm_auroc'] = None  # can't compute

        # Raw per-example AUROC (no z-scoring)
        if n_pos > 0 and n_neg > 0:
            try:
                example_info['raw_auroc'] = float(roc_auc_score(l, s))
            except ValueError:
                example_info['raw_auroc'] = 0.5

        per_example.append(example_info)

    # Global token-level AUROC
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    try:
        global_auroc = roc_auc_score(all_labels, all_scores)
    except ValueError:
        global_auroc = 0.5

    return {
        'global_token_auroc': float(global_auroc),
        'mean_norm_auroc': float(np.mean(norm_aurocs)) if norm_aurocs else 0.5,
        'median_norm_auroc': float(np.median(norm_aurocs)) if norm_aurocs else 0.5,
        'std_norm_auroc': float(np.std(norm_aurocs)) if norm_aurocs else 0.0,
        'n_examples_with_both': len(norm_aurocs),
        'total_tokens': int(all_labels.shape[0]),
        'total_pos': int(all_labels.sum()),
        'total_neg': int(all_labels.shape[0] - all_labels.sum()),
        'per_example': per_example,
    }


def load_probe(probe_path, device='cpu'):
    """Load a saved probe."""
    state = torch.load(probe_path, map_location=device, weights_only=False)

    # Determine probe type from state
    if 'linear.weight' in state or 'weight' in state:
        from probity.training.probes import LogisticProbe, LogisticProbeConfig
        w_key = 'linear.weight' if 'linear.weight' in state else 'weight'
        hidden_size = state[w_key].shape[1]
        config = LogisticProbeConfig(hidden_size=hidden_size)
        probe = LogisticProbe(config)
        probe.load_state_dict(state, strict=False)
    elif 'query' in state or 'W_Q' in state:
        from probity.training.probes import AttentionProbe, AttentionProbeConfig
        # Try to infer config from state dict keys
        for k, v in state.items():
            if 'query' in k.lower() or 'w_q' in k.lower():
                hidden_size = v.shape[-1] if len(v.shape) > 1 else v.shape[0]
                break
        config = AttentionProbeConfig(hidden_size=hidden_size)
        probe = AttentionProbe(config)
        probe.load_state_dict(state, strict=False)
    else:
        raise ValueError(f"Can't determine probe type from state keys: {list(state.keys())}")

    probe.eval()
    return probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe_dir', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--layers', nargs='+', type=int, default=None)
    parser.add_argument('--max_length', type=int, default=1024)
    args = parser.parse_args()

    probe_dir = Path(args.probe_dir)

    # Load dataset for token labels and attention masks
    print(f"Loading dataset from {args.dataset_path}")
    dataset = load_ntml_token_level(
        args.dataset_path, args.model_name,
        max_length=args.max_length, use_llm_spans='always'
    )

    # Get token labels and attention masks
    token_labels_list = []
    attn_masks_list = []
    for ex in dataset.examples:
        tl = ex.attributes.get('token_labels', None)
        if tl is not None:
            token_labels_list.append(torch.tensor(tl, dtype=torch.float32))
        am = ex.attention_mask if hasattr(ex, 'attention_mask') and ex.attention_mask is not None else None
        if am is not None:
            attn_masks_list.append(torch.tensor(am, dtype=torch.float32))

    # Pad to same length
    max_len = args.max_length
    token_labels = torch.zeros(len(token_labels_list), max_len)
    attn_masks = torch.zeros(len(attn_masks_list), max_len)

    for i, (tl, am) in enumerate(zip(token_labels_list, attn_masks_list)):
        # Handle potential length mismatch with padding
        tl_len = min(len(tl), max_len)
        am_len = min(len(am), max_len)

        # Figure out content start (left-padding)
        content_start = max_len - am_len
        if content_start < 0:
            content_start = 0

        # Place labels aligned with content
        actual_label_len = min(tl_len, max_len - content_start)
        token_labels[i, content_start:content_start + actual_label_len] = tl[:actual_label_len]

        actual_mask_len = min(am_len, max_len)
        attn_masks[i, max_len - actual_mask_len:] = am[:actual_mask_len]

    print(f"Dataset: {len(dataset.examples)} examples")
    print(f"Token labels shape: {token_labels.shape}")
    print(f"Lie tokens: {int(token_labels.sum())} ({100*token_labels.sum()/attn_masks.sum():.1f}%)")

    # Find probe types and layers
    probe_types = [d.name for d in probe_dir.iterdir() if d.is_dir()]
    print(f"Probe types found: {probe_types}")

    # Load cached activations
    from probity.utils.caching import get_dataset_hash
    model_short = args.model_name.replace('/', '_').replace('-', '_')
    cache_subdir = Path(args.cache_dir)

    # Find cache directory
    cache_dirs = list(cache_subdir.glob(f"{model_short}*"))
    if not cache_dirs:
        # Try to find any cache
        cache_dirs = list(cache_subdir.iterdir())

    if not cache_dirs:
        print("ERROR: No activation cache found. Run training first.")
        return

    cache_path = cache_dirs[0]
    print(f"Using cache: {cache_path}")

    # Evaluate each probe
    print("\n" + "="*80)
    print("TOKEN-LEVEL EVALUATION RESULTS")
    print("="*80)

    results = {}

    for probe_type in sorted(probe_types):
        type_dir = probe_dir / probe_type
        probe_files = sorted(type_dir.glob("layer_*_probe.pt"))

        for pf in probe_files:
            layer = int(pf.stem.split('_')[1])
            if args.layers and layer not in args.layers:
                continue

            hook_point = f"blocks.{layer}.hook_resid_pre"

            # Load activations from cache
            act_cache_path = cache_path / hook_point
            if not act_cache_path.exists():
                print(f"  SKIP {probe_type} layer {layer}: no cached activations")
                continue

            from datasets import load_from_disk
            act_dataset = load_from_disk(str(act_cache_path))
            activations = torch.tensor(np.array(act_dataset['activations']), dtype=torch.float32)

            # Load probe
            probe = load_probe(str(pf))

            # Get predictions
            with torch.no_grad():
                # For logistic probe: output is (batch, seq_len, 1) or (batch, seq_len)
                logits = probe(activations)
                if logits.dim() == 3:
                    logits = logits.squeeze(-1)
                scores = torch.sigmoid(logits)

            # Compute metrics
            metrics = compute_token_metrics(scores, token_labels, attn_masks)

            key = f"{probe_type}_layer_{layer}"
            results[key] = metrics

            print(f"\n{probe_type} | Layer {layer}:")
            print(f"  Global token AUROC:  {metrics['global_token_auroc']:.4f}")
            print(f"  Mean norm_auroc:     {metrics['mean_norm_auroc']:.4f}")
            print(f"  Median norm_auroc:   {metrics['median_norm_auroc']:.4f}")
            print(f"  Std norm_auroc:      {metrics['std_norm_auroc']:.4f}")
            print(f"  Examples evaluated:  {metrics['n_examples_with_both']}")

            # Show a few per-example scores
            scored = [e for e in metrics['per_example'] if e['norm_auroc'] is not None]
            if scored:
                top3 = sorted(scored, key=lambda x: x['norm_auroc'], reverse=True)[:3]
                bot3 = sorted(scored, key=lambda x: x['norm_auroc'])[:3]
                best_str = [f"{e['norm_auroc']:.3f}" for e in top3]
                worst_str = [f"{e['norm_auroc']:.3f}" for e in bot3]
                print(f"  Best 3 norm_auroc:   {best_str}")
                print(f"  Worst 3 norm_auroc:  {worst_str}")

                # Score separation
                pos_means = [e['mean_score_pos'] for e in scored if e['mean_score_pos'] is not None]
                neg_means = [e['mean_score_neg'] for e in scored if e['mean_score_neg'] is not None]
                if pos_means and neg_means:
                    print(f"  Avg score on lie tokens:   {np.mean(pos_means):.4f}")
                    print(f"  Avg score on truth tokens: {np.mean(neg_means):.4f}")
                    print(f"  Score separation:          {np.mean(pos_means) - np.mean(neg_means):.4f}")

            del activations, logits, scores, probe

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Probe':<25} {'Global AUROC':>12} {'Norm AUROC':>12} {'Separation':>12}")
    print("-"*65)
    for key, m in sorted(results.items()):
        scored = [e for e in m['per_example'] if e['mean_score_pos'] is not None and e['mean_score_neg'] is not None]
        sep = np.mean([e['mean_score_pos'] - e['mean_score_neg'] for e in scored]) if scored else 0
        print(f"{key:<25} {m['global_token_auroc']:>12.4f} {m['mean_norm_auroc']:>12.4f} {sep:>12.4f}")


if __name__ == '__main__':
    main()
