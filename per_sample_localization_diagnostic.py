"""
Per-sample localization diagnostic: check if ANY individual samples show
within-sample discrimination (phrase-level or statement-level), or if it's
uniformly at chance across all samples.

Uses the best attention probes from the big_probe_sweep.
"""
import json
import glob
import numpy as np
import os
import sys
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding='utf-8')

N_AI_LIAR = 266


def load_ai_liar_samples():
    samples = []
    with open('data/deception_detection/ai_liar.jsonl') as f:
        for line in f:
            s = json.loads(line)
            meta = s['metadata']
            conv = s['conversations']['main']
            samples.append({
                'sample_index': meta['sample_index'],
                'label': 1 if meta['label'] == 'deceptive' else 0,
                'n_tokens': len(conv['tokens']),
                'tokens': conv['tokens'],
                'assistant_indices': sorted(conv.get('assistant_indices', [])),
                'critical_indices': set(conv.get('critical_indices', [])),
            })
    return samples


def load_labeled_spans():
    path = 'deception-detection-labelled/deception_detection_labeled/ai_liar_labeled.json'
    with open(path) as f:
        records = json.load(f)
    spans_by_idx = {}
    for rec in records:
        orig_meta = rec.get('original_data', {}).get('metadata', {})
        idx = orig_meta.get('sample_index')
        if idx is not None:
            spans_by_idx[idx] = rec.get('spans', [])
    return spans_by_idx


def char_spans_to_token_mask(spans, tokens, assistant_indices):
    if not spans or not assistant_indices:
        return np.zeros(len(tokens), dtype=bool)
    char_pos = 0
    token_char_ranges = {}
    for idx in assistant_indices:
        if idx >= len(tokens):
            break
        tok_text = tokens[idx]
        clean = tok_text.replace('\u0120', ' ').replace('\u2581', ' ')
        tok_start = char_pos
        tok_end = char_pos + len(clean)
        token_char_ranges[idx] = (tok_start, tok_end)
        char_pos = tok_end
    mask = np.zeros(len(tokens), dtype=bool)
    for span in spans:
        cs = span.get('char_start', 0)
        ce = span.get('char_end', 0)
        if cs >= ce:
            continue
        for idx in assistant_indices:
            if idx not in token_char_ranges:
                continue
            ts, te = token_char_ranges[idx]
            if ts < ce and te > cs:
                mask[idx] = True
    return mask


def segment_first_n(raw_scores, n):
    ts = np.array(raw_scores['token_scores'], dtype=np.float64)
    ss = np.array(raw_scores['sample_scores'][:n], dtype=np.float64)
    sl = np.array(raw_scores['sample_labels'][:n])
    cumsum = np.cumsum(ts)
    cumsum = np.insert(cumsum, 0, 0.0)
    boundaries = []
    start = 0
    for i in range(n):
        target_mean = ss[i]
        base = cumsum[start]
        max_end = min(len(ts), start + 2000)
        sizes = np.arange(5, max_end - start + 1)
        if len(sizes) == 0:
            break
        actual_sums = cumsum[start + sizes] - base
        expected_sums = target_mean * sizes
        errors = np.abs(actual_sums - expected_sums)
        best_size = int(sizes[np.argmin(errors)])
        boundaries.append((start, start + best_size, int(sl[i])))
        start += best_size
    return boundaries, ts


def main():
    eval_samples = load_ai_liar_samples()
    spans_by_idx = load_labeled_spans()

    # Pick probes to analyze - use results/ attention and logistic at various layers
    probe_paths = []
    for pattern in [
        'results/*/aggregated/attention/layer_*/raw_scores.json',
        'results/*/aggregated/logistic/layer_*/raw_scores.json',
        'localization_exp/eval_results/*/aggregated/attention_*/layer_*/raw_scores.json',
    ]:
        probe_paths.extend(sorted(glob.glob(pattern)))

    # Also check big_probe_sweep
    for pattern in [
        'big_probe_sweep_eval_full/aggregated/attention_*/layer_*/raw_scores.json',
    ]:
        probe_paths.extend(sorted(glob.glob(pattern)))

    print(f"Found {len(probe_paths)} probe raw_scores files to analyze")

    # Pick a subset (best layers: 22, 44, 50)
    selected = []
    for p in probe_paths:
        for target_layer in ['layer_22', 'layer_44', 'layer_50', 'layer_26', 'layer_36']:
            if target_layer in p:
                selected.append(p)
                break

    if not selected:
        selected = probe_paths[:10]

    print(f"Analyzing {len(selected)} probes at key layers")

    for rs_path in selected[:15]:
        parts = rs_path.replace(os.sep, '/').split('/')
        probe_name = '/'.join(parts[:3]) + '/' + '/'.join(parts[-3:])

        try:
            with open(rs_path) as f:
                raw = json.load(f)

            # Verify alignment
            first_labels = raw['sample_labels'][:N_AI_LIAR]
            expected = [s['label'] for s in eval_samples]
            if first_labels != expected:
                continue

            boundaries, ts = segment_first_n(raw, N_AI_LIAR)
            if len(boundaries) < N_AI_LIAR:
                continue

            # Per-sample statement-level AUROC
            stmt_aurocs = []
            phrase_aurocs = []
            sample_mean_scores = []

            for i, (start, end, label) in enumerate(boundaries):
                sample = eval_samples[i]
                sample_ts = ts[start:end]
                n_tok = end - start

                if n_tok < 10:
                    continue

                # Statement-level
                asst_mask = np.zeros(n_tok, dtype=bool)
                for idx in sample['assistant_indices']:
                    if idx < n_tok:
                        asst_mask[idx] = True

                if asst_mask.sum() < 5:
                    continue

                stmt_mask = np.zeros(n_tok, dtype=bool)
                for idx in sample['critical_indices']:
                    if idx < n_tok:
                        stmt_mask[idx] = True

                stmt_pos = stmt_mask & asst_mask
                stmt_neg = (~stmt_mask) & asst_mask

                if stmt_pos.sum() >= 3 and stmt_neg.sum() >= 3:
                    try:
                        asst_scores = sample_ts[asst_mask]
                        asst_labels = stmt_mask[asst_mask].astype(int)
                        auc = roc_auc_score(asst_labels, asst_scores)
                        stmt_aurocs.append(auc)
                        sample_mean_scores.append(float(np.mean(sample_ts)))
                    except:
                        pass

                # Phrase-level
                char_spans = spans_by_idx.get(sample['sample_index'], [])
                if char_spans:
                    phrase_mask = char_spans_to_token_mask(
                        char_spans,
                        sample['tokens'][:n_tok] if len(sample['tokens']) >= n_tok else sample['tokens'],
                        [idx for idx in sample['assistant_indices'] if idx < n_tok]
                    )
                    phrase_mask_seg = phrase_mask[:n_tok]
                    phrase_pos = phrase_mask_seg & asst_mask
                    phrase_neg = (~phrase_mask_seg) & asst_mask

                    if phrase_pos.sum() >= 3 and phrase_neg.sum() >= 3:
                        try:
                            asst_scores = sample_ts[asst_mask]
                            asst_labels = phrase_mask_seg[asst_mask].astype(int)
                            auc = roc_auc_score(asst_labels, asst_scores)
                            phrase_aurocs.append(auc)
                        except:
                            pass

            if not stmt_aurocs:
                continue

            stmt_arr = np.array(stmt_aurocs)
            print(f"\n{'='*80}")
            print(f"Probe: {probe_name}")
            print(f"  Statement-level per-sample AUROC (n={len(stmt_aurocs)}):")
            print(f"    mean={stmt_arr.mean():.4f}, std={stmt_arr.std():.4f}")
            print(f"    min={stmt_arr.min():.4f}, max={stmt_arr.max():.4f}")
            print(f"    >0.60: {(stmt_arr > 0.60).sum()}, >0.70: {(stmt_arr > 0.70).sum()}, >0.80: {(stmt_arr > 0.80).sum()}")
            print(f"    <0.40: {(stmt_arr < 0.40).sum()}, <0.30: {(stmt_arr < 0.30).sum()}")

            # Distribution histogram
            bins = [0, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 1.0]
            hist, _ = np.histogram(stmt_arr, bins=bins)
            print(f"    Distribution: ", end='')
            for j in range(len(bins)-1):
                print(f"[{bins[j]:.1f}-{bins[j+1]:.1f}]:{hist[j]} ", end='')
            print()

            if phrase_aurocs:
                phr_arr = np.array(phrase_aurocs)
                print(f"  Phrase-level per-sample AUROC (n={len(phrase_aurocs)}):")
                print(f"    mean={phr_arr.mean():.4f}, std={phr_arr.std():.4f}")
                print(f"    min={phr_arr.min():.4f}, max={phr_arr.max():.4f}")
                print(f"    >0.60: {(phr_arr > 0.60).sum()}, >0.70: {(phr_arr > 0.70).sum()}, >0.80: {(phr_arr > 0.80).sum()}")

            # Check if per-sample AUROC correlates with sample mean score
            if len(sample_mean_scores) == len(stmt_aurocs):
                corr = np.corrcoef(sample_mean_scores, stmt_aurocs)[0, 1]
                print(f"  Corr(sample_mean_score, per_sample_stmt_auroc) = {corr:.3f}")

        except Exception as e:
            print(f"  Error: {e}")


if __name__ == '__main__':
    main()
