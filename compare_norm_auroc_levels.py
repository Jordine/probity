"""
Compare norm_auroc at different label granularities (ai_liar only):
1. phrase-level: character-level labeled spans (from deception-detection-labelled/)
2. statement-level: critical_indices from eval data

Uses first 266 samples in raw_scores = ai_liar (verified 100% label match).
"""
import json
import glob
import csv
import numpy as np
import os
import sys
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding='utf-8')

N_AI_LIAR = 266  # First 266 samples in raw_scores are ai_liar


def load_ai_liar_samples():
    """Load ai_liar eval samples with critical_indices and assistant_indices."""
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
    """Load character-level labeled spans for ai_liar, keyed by sample_index."""
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
    """Convert character spans to boolean mask over ALL tokens.

    Character positions are relative to the assistant response text,
    reconstructed by concatenating cleaned token strings at assistant_indices.
    """
    if not spans or not assistant_indices:
        return np.zeros(len(tokens), dtype=bool)

    # Build char-to-token mapping for assistant tokens
    char_pos = 0
    token_char_ranges = {}
    for idx in assistant_indices:
        if idx >= len(tokens):
            break
        tok_text = tokens[idx]
        # Handle sentencepiece/tiktoken leading space markers
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
            if ts < ce and te > cs:  # overlap
                mask[idx] = True
    return mask


def segment_first_n(raw_scores, n):
    """Reconstruct boundaries for only the first n samples."""
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


def compute_comparison(raw_scores, eval_samples, spans_by_idx):
    """Compute phrase-level vs statement-level norm_auroc."""
    boundaries, ts = segment_first_n(raw_scores, N_AI_LIAR)

    if len(boundaries) < N_AI_LIAR:
        return None, f"only {len(boundaries)} boundaries for {N_AI_LIAR} samples"

    # Collections for pooled metrics
    phrase_z_in = []   # z-scored scores IN phrase spans
    phrase_z_out = []  # z-scored scores NOT in phrase spans (within response)
    stmt_z_crit = []   # z-scored scores at critical_indices
    stmt_z_noncrit = []  # z-scored scores at non-critical indices (within response)

    phrase_raw_in = []
    phrase_raw_out = []
    stmt_raw_crit = []
    stmt_raw_noncrit = []

    n_phrase_samples = 0
    n_stmt_samples = 0

    for i, (start, end, label) in enumerate(boundaries):
        sample = eval_samples[i]
        sample_ts = ts[start:end]
        n_tok = end - start

        # Only analyze samples with enough tokens
        if n_tok < 10:
            continue

        # Z-score within sample
        mean = np.mean(sample_ts)
        std = np.std(sample_ts)
        if std < 1e-8:
            continue
        z_ts = (sample_ts - mean) / std

        # Assistant response mask (only compare within response)
        asst_mask = np.zeros(n_tok, dtype=bool)
        for idx in sample['assistant_indices']:
            if idx < n_tok:
                asst_mask[idx] = True

        if asst_mask.sum() < 5:
            continue

        # --- Statement-level: critical_indices ---
        stmt_mask = np.zeros(n_tok, dtype=bool)
        for idx in sample['critical_indices']:
            if idx < n_tok:
                stmt_mask[idx] = True

        stmt_pos = stmt_mask & asst_mask
        stmt_neg = (~stmt_mask) & asst_mask

        if stmt_pos.sum() >= 2 and stmt_neg.sum() >= 2:
            stmt_z_crit.extend(z_ts[stmt_pos].tolist())
            stmt_z_noncrit.extend(z_ts[stmt_neg].tolist())
            stmt_raw_crit.extend(sample_ts[stmt_pos].tolist())
            stmt_raw_noncrit.extend(sample_ts[stmt_neg].tolist())
            n_stmt_samples += 1

        # --- Phrase-level: character spans ---
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

            if phrase_pos.sum() >= 2 and phrase_neg.sum() >= 2:
                phrase_z_in.extend(z_ts[phrase_pos].tolist())
                phrase_z_out.extend(z_ts[phrase_neg].tolist())
                phrase_raw_in.extend(sample_ts[phrase_pos].tolist())
                phrase_raw_out.extend(sample_ts[phrase_neg].tolist())
                n_phrase_samples += 1

    result = {
        'n_phrase_samples': n_phrase_samples,
        'n_stmt_samples': n_stmt_samples,
    }

    if phrase_z_in and phrase_z_out:
        labels = np.array([1]*len(phrase_z_in) + [0]*len(phrase_z_out))
        result['phrase_norm_auroc'] = float(roc_auc_score(labels, np.array(phrase_z_in + phrase_z_out)))
        result['phrase_raw_auroc'] = float(roc_auc_score(labels, np.array(phrase_raw_in + phrase_raw_out)))
        ps = np.sqrt((np.var(phrase_z_in) + np.var(phrase_z_out)) / 2)
        result['phrase_norm_d'] = float((np.mean(phrase_z_in) - np.mean(phrase_z_out)) / ps) if ps > 1e-8 else 0
        result['n_phrase_pos'] = len(phrase_z_in)
        result['n_phrase_neg'] = len(phrase_z_out)

    if stmt_z_crit and stmt_z_noncrit:
        labels = np.array([1]*len(stmt_z_crit) + [0]*len(stmt_z_noncrit))
        result['stmt_norm_auroc'] = float(roc_auc_score(labels, np.array(stmt_z_crit + stmt_z_noncrit)))
        result['stmt_raw_auroc'] = float(roc_auc_score(labels, np.array(stmt_raw_crit + stmt_raw_noncrit)))
        ps = np.sqrt((np.var(stmt_z_crit) + np.var(stmt_z_noncrit)) / 2)
        result['stmt_norm_d'] = float((np.mean(stmt_z_crit) - np.mean(stmt_z_noncrit)) / ps) if ps > 1e-8 else 0
        result['n_stmt_pos'] = len(stmt_z_crit)
        result['n_stmt_neg'] = len(stmt_z_noncrit)

    return result, None


def main():
    print("Loading ai_liar eval samples...")
    eval_samples = load_ai_liar_samples()
    print(f"  {len(eval_samples)} samples ({sum(s['label'] for s in eval_samples)} deceptive)")

    # Quick stats on critical_indices
    for s in eval_samples[:3]:
        ai = s['assistant_indices']
        ci = s['critical_indices']
        overlap = len(ci & set(ai))
        print(f"  Sample {s['sample_index']}: {s['n_tokens']} tok, {len(ai)} asst, {len(ci)} crit, {overlap} crit_in_asst")

    print("\nLoading labeled spans...")
    spans_by_idx = load_labeled_spans()
    has_spans = sum(1 for v in spans_by_idx.values() if v)
    print(f"  {len(spans_by_idx)} records, {has_spans} with spans")

    # Find all probe directories with ranking_metrics + raw_scores (aggregated only)
    ranking_files = sorted(glob.glob('**/aggregated/**/ranking_metrics.json', recursive=True))
    print(f"\nFound {len(ranking_files)} aggregated probe directories with ranking_metrics")

    results = []
    errors = 0

    for rm_path in ranking_files:
        probe_dir = os.path.dirname(rm_path)
        rs_path = os.path.join(probe_dir, 'raw_scores.json')
        if not os.path.exists(rs_path):
            continue

        # Parse probe info
        parts = rm_path.replace(os.sep, '/').split('/')
        probe = ''
        layer = ''
        source = parts[0]
        for p in parts:
            if any(x in p for x in ['attention', 'logistic', 'meandiff', 'sklearn', 'apollo']):
                probe = p
            if p.startswith('layer_'):
                layer = p.replace('layer_', '')

        try:
            with open(rs_path) as f:
                raw = json.load(f)

            # Verify alignment
            first_labels = raw['sample_labels'][:N_AI_LIAR]
            expected = [s['label'] for s in eval_samples]
            if first_labels != expected:
                errors += 1
                continue

            with open(rm_path) as f:
                existing_rm = json.load(f)

            result, err = compute_comparison(raw, eval_samples, spans_by_idx)
            if err:
                errors += 1
                if errors <= 3:
                    print(f"  Skip {probe} L{layer}: {err}")
                continue

            result['source'] = source
            result['probe'] = probe
            result['layer'] = layer
            result['existing_span_auroc'] = existing_rm.get('span_auroc', '')
            result['existing_r_at_oracle'] = existing_rm.get('recall_at_oracle', '')
            results.append(result)

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  Error {probe} L{layer}: {e}")

    print(f"\nProcessed {len(results)} probes ({errors} skipped)")

    if not results:
        print("No results! Check data alignment.")
        return

    # Print comparison table
    print("\n" + "=" * 150)
    print("NORM AUROC COMPARISON: Phrase-Level Spans vs Statement-Level critical_indices")
    print("(All metrics computed WITHIN samples via z-scoring, on ai_liar dataset only)")
    print("=" * 150)
    print(f"{'Source':<25} {'Probe':<25} {'L':>3} | "
          f"{'Phr NormAUROC':>13} {'Stmt NormAUROC':>14} {'Delta':>7} | "
          f"{'Phr RawAUROC':>12} {'Stmt RawAUROC':>13} | "
          f"{'PhrD':>6} {'StmtD':>6} | "
          f"{'Exist SpanAUROC':>15}")
    print("-" * 150)

    by_phrase = sorted(results, key=lambda x: x.get('phrase_norm_auroc', 0), reverse=True)
    for r in by_phrase[:35]:
        pna = f"{r['phrase_norm_auroc']:.4f}" if 'phrase_norm_auroc' in r else "N/A"
        sna = f"{r['stmt_norm_auroc']:.4f}" if 'stmt_norm_auroc' in r else "N/A"
        pra = f"{r['phrase_raw_auroc']:.4f}" if 'phrase_raw_auroc' in r else "N/A"
        sra = f"{r['stmt_raw_auroc']:.4f}" if 'stmt_raw_auroc' in r else "N/A"
        delta = ""
        if 'phrase_norm_auroc' in r and 'stmt_norm_auroc' in r:
            d = r['phrase_norm_auroc'] - r['stmt_norm_auroc']
            delta = f"{d:+.4f}"
        pd = f"{r.get('phrase_norm_d', 0):.3f}" if 'phrase_norm_d' in r else ""
        sd = f"{r.get('stmt_norm_d', 0):.3f}" if 'stmt_norm_d' in r else ""
        esa = str(r.get('existing_span_auroc', ''))[:8]
        print(f"{r['source']:<25} {r['probe']:<25} {r['layer']:>3} | "
              f"{pna:>13} {sna:>14} {delta:>7} | "
              f"{pra:>12} {sra:>13} | "
              f"{pd:>6} {sd:>6} | "
              f"{esa:>15}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    both = [r for r in results if 'phrase_norm_auroc' in r and 'stmt_norm_auroc' in r]
    if both:
        pna = [r['phrase_norm_auroc'] for r in both]
        sna = [r['stmt_norm_auroc'] for r in both]
        deltas = [p - s for p, s in zip(pna, sna)]

        print(f"Probes with both metrics: {len(both)}")
        print(f"  Phrase norm AUROC:  mean={np.mean(pna):.4f}, std={np.std(pna):.4f}, range=[{min(pna):.4f}, {max(pna):.4f}]")
        print(f"  Stmt norm AUROC:    mean={np.mean(sna):.4f}, std={np.std(sna):.4f}, range=[{min(sna):.4f}, {max(sna):.4f}]")
        print(f"  Delta (phr-stmt):   mean={np.mean(deltas):+.4f}, std={np.std(deltas):.4f}")
        print(f"  Phrase > Stmt:      {sum(1 for d in deltas if d > 0)}/{len(deltas)} probes")

        corr = np.corrcoef(pna, sna)[0, 1]
        print(f"  Correlation:        r = {corr:.3f}")

    # Check phrase norm vs existing span_auroc
    with_existing = [r for r in results if r.get('existing_span_auroc') and 'phrase_norm_auroc' in r]
    if with_existing:
        pna2 = [r['phrase_norm_auroc'] for r in with_existing]
        esa2 = [float(r['existing_span_auroc']) for r in with_existing]
        corr2 = np.corrcoef(pna2, esa2)[0, 1]
        print(f"\n  Phrase norm vs existing span_auroc: r = {corr2:.3f}")
        print(f"  (existing span_auroc is non-normalized, computed on full aggregated data)")

    # Save
    cols = ['source', 'probe', 'layer',
            'phrase_norm_auroc', 'stmt_norm_auroc',
            'phrase_raw_auroc', 'stmt_raw_auroc',
            'phrase_norm_d', 'stmt_norm_d',
            'n_phrase_samples', 'n_stmt_samples',
            'n_phrase_pos', 'n_phrase_neg', 'n_stmt_pos', 'n_stmt_neg',
            'existing_span_auroc', 'existing_r_at_oracle']

    with open('NORM_AUROC_COMPARISON.tsv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter='\t', extrasaction='ignore')
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved {len(results)} rows to NORM_AUROC_COMPARISON.tsv")


if __name__ == '__main__':
    main()
