"""
Fix boundary reconstruction and recompute norm_auroc with:
1. Exact token counts from eval data (not cumulative-sum heuristic)
2. Ground-truth labels (critical_indices and char spans, not sample-level token_labels)

Tests whether the boundary leakage was causing the chance-level results.
"""
import json
import glob
import csv
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


def exact_boundaries(raw_scores, eval_samples, n):
    """Use exact token counts from eval data instead of heuristic reconstruction."""
    ts = np.array(raw_scores['token_scores'], dtype=np.float64)
    ss = np.array(raw_scores['sample_scores'][:n], dtype=np.float64)
    sl = np.array(raw_scores['sample_labels'][:n])

    boundaries = []
    start = 0
    alignment_errors = []

    for i in range(n):
        expected_n = eval_samples[i]['n_tokens']
        end = start + expected_n

        if end > len(ts):
            # Truncation happened — try to find actual size
            # Fall back to heuristic for this sample
            end = min(start + expected_n, len(ts))

        if end <= start:
            break

        # Verify alignment: does mean(ts[start:end]) ≈ ss[i]?
        actual_mean = np.mean(ts[start:end])
        error = abs(actual_mean - ss[i])
        alignment_errors.append(error)

        boundaries.append((start, end, int(sl[i])))
        start = end

    return boundaries, ts, alignment_errors


def heuristic_boundaries(raw_scores, n):
    """Original cumulative-sum heuristic (for comparison)."""
    ts = np.array(raw_scores['token_scores'], dtype=np.float64)
    ss = np.array(raw_scores['sample_scores'][:n], dtype=np.float64)
    sl = np.array(raw_scores['sample_labels'][:n])
    cumsum = np.cumsum(ts)
    cumsum = np.insert(cumsum, 0, 0.0)
    boundaries = []
    start = 0
    alignment_errors = []
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

        end = start + best_size
        actual_mean = np.mean(ts[start:end])
        alignment_errors.append(abs(actual_mean - ss[i]))

        boundaries.append((start, end, int(sl[i])))
        start += best_size
    return boundaries, ts, alignment_errors


def compute_metrics(boundaries, ts, eval_samples, spans_by_idx):
    """Compute phrase and statement level norm_auroc."""
    phrase_z_in, phrase_z_out = [], []
    stmt_z_crit, stmt_z_noncrit = [], []
    phrase_raw_in, phrase_raw_out = [], []
    stmt_raw_crit, stmt_raw_noncrit = [], []
    n_phrase, n_stmt = 0, 0
    per_sample_stmt = []
    per_sample_phrase = []

    for i, (start, end, label) in enumerate(boundaries):
        if i >= len(eval_samples):
            break
        sample = eval_samples[i]
        sample_ts = ts[start:end]
        n_tok = end - start

        if n_tok < 10:
            continue

        mean = np.mean(sample_ts)
        std = np.std(sample_ts)
        if std < 1e-8:
            continue
        z_ts = (sample_ts - mean) / std

        # Assistant mask
        asst_mask = np.zeros(n_tok, dtype=bool)
        for idx in sample['assistant_indices']:
            if idx < n_tok:
                asst_mask[idx] = True
        if asst_mask.sum() < 5:
            continue

        # Statement-level
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
            n_stmt += 1

            # Per-sample AUROC
            try:
                asst_scores = sample_ts[asst_mask]
                asst_labels = stmt_mask[asst_mask].astype(int)
                per_sample_stmt.append(roc_auc_score(asst_labels, asst_scores))
            except:
                pass

        # Phrase-level
        char_spans = spans_by_idx.get(sample['sample_index'], [])
        if char_spans:
            tok_list = sample['tokens'][:n_tok] if len(sample['tokens']) >= n_tok else sample['tokens']
            asst_list = [idx for idx in sample['assistant_indices'] if idx < n_tok]
            phrase_mask = char_spans_to_token_mask(char_spans, tok_list, asst_list)
            phrase_mask_seg = phrase_mask[:n_tok]
            phrase_pos = phrase_mask_seg & asst_mask
            phrase_neg = (~phrase_mask_seg) & asst_mask

            if phrase_pos.sum() >= 2 and phrase_neg.sum() >= 2:
                phrase_z_in.extend(z_ts[phrase_pos].tolist())
                phrase_z_out.extend(z_ts[phrase_neg].tolist())
                phrase_raw_in.extend(sample_ts[phrase_pos].tolist())
                phrase_raw_out.extend(sample_ts[phrase_neg].tolist())
                n_phrase += 1

                try:
                    asst_scores = sample_ts[asst_mask]
                    asst_labels = phrase_mask_seg[asst_mask].astype(int)
                    per_sample_phrase.append(roc_auc_score(asst_labels, asst_scores))
                except:
                    pass

    result = {'n_phrase': n_phrase, 'n_stmt': n_stmt}

    if phrase_z_in and phrase_z_out:
        labels = np.array([1]*len(phrase_z_in) + [0]*len(phrase_z_out))
        scores = np.array(phrase_z_in + phrase_z_out)
        result['phrase_norm_auroc'] = float(roc_auc_score(labels, scores))
        labels_raw = labels
        scores_raw = np.array(phrase_raw_in + phrase_raw_out)
        result['phrase_raw_auroc'] = float(roc_auc_score(labels_raw, scores_raw))
        result['n_phrase_pos'] = len(phrase_z_in)
        result['n_phrase_neg'] = len(phrase_z_out)

    if stmt_z_crit and stmt_z_noncrit:
        labels = np.array([1]*len(stmt_z_crit) + [0]*len(stmt_z_noncrit))
        scores = np.array(stmt_z_crit + stmt_z_noncrit)
        result['stmt_norm_auroc'] = float(roc_auc_score(labels, scores))
        labels_raw = labels
        scores_raw = np.array(stmt_raw_crit + stmt_raw_noncrit)
        result['stmt_raw_auroc'] = float(roc_auc_score(labels_raw, scores_raw))
        result['n_stmt_pos'] = len(stmt_z_crit)
        result['n_stmt_neg'] = len(stmt_z_noncrit)

    if per_sample_stmt:
        arr = np.array(per_sample_stmt)
        result['per_sample_stmt_mean'] = float(arr.mean())
        result['per_sample_stmt_std'] = float(arr.std())

    if per_sample_phrase:
        arr = np.array(per_sample_phrase)
        result['per_sample_phrase_mean'] = float(arr.mean())
        result['per_sample_phrase_std'] = float(arr.std())

    return result


def main():
    eval_samples = load_ai_liar_samples()
    spans_by_idx = load_labeled_spans()

    print(f"Loaded {len(eval_samples)} ai_liar samples")
    print(f"Token counts: min={min(s['n_tokens'] for s in eval_samples)}, "
          f"max={max(s['n_tokens'] for s in eval_samples)}, "
          f"mean={np.mean([s['n_tokens'] for s in eval_samples]):.0f}")

    # Find all probe raw_scores
    probe_paths = sorted(glob.glob('**/aggregated/**/raw_scores.json', recursive=True))
    print(f"Found {len(probe_paths)} raw_scores files\n")

    # First: diagnose boundary quality on a few probes
    print("=" * 100)
    print("BOUNDARY QUALITY DIAGNOSIS")
    print("=" * 100)

    test_paths = probe_paths[:5]
    for rs_path in test_paths:
        try:
            with open(rs_path) as f:
                raw = json.load(f)
            first_labels = raw['sample_labels'][:N_AI_LIAR]
            expected = [s['label'] for s in eval_samples]
            if first_labels != expected:
                continue

            # Exact boundaries
            exact_b, ts_e, exact_err = exact_boundaries(raw, eval_samples, N_AI_LIAR)
            # Heuristic boundaries
            heur_b, ts_h, heur_err = heuristic_boundaries(raw, N_AI_LIAR)

            parts = rs_path.replace(os.sep, '/').split('/')
            name = '/'.join(parts[-4:-1])

            print(f"\n{name}:")
            print(f"  Total token_scores: {len(raw['token_scores'])}")
            print(f"  Sum of eval n_tokens (266): {sum(s['n_tokens'] for s in eval_samples)}")

            if exact_b:
                exact_end = exact_b[-1][1] if exact_b else 0
                heur_end = heur_b[-1][1] if heur_b else 0
                print(f"  Exact: {len(exact_b)} boundaries, ends at {exact_end}")
                print(f"  Heuristic: {len(heur_b)} boundaries, ends at {heur_end}")

                # Compare boundaries
                n_match = 0
                n_close = 0
                max_diff = 0
                for (es, ee, _), (hs, he, _) in zip(exact_b[:N_AI_LIAR], heur_b[:N_AI_LIAR]):
                    diff = abs(es - hs) + abs(ee - he)
                    if diff == 0:
                        n_match += 1
                    elif diff <= 5:
                        n_close += 1
                    max_diff = max(max_diff, diff)
                print(f"  Exact match: {n_match}/{min(len(exact_b), len(heur_b))}, "
                      f"close (<=5): {n_close}, max diff: {max_diff}")

                # Alignment errors
                exact_err_arr = np.array(exact_err[:N_AI_LIAR])
                heur_err_arr = np.array(heur_err[:min(N_AI_LIAR, len(heur_err))])
                print(f"  Exact alignment error: mean={exact_err_arr.mean():.6f}, max={exact_err_arr.max():.6f}")
                print(f"  Heuristic alignment error: mean={heur_err_arr.mean():.6f}, max={heur_err_arr.max():.6f}")

                # Size comparison for deceptive samples
                exact_sizes = [e-s for s, e, l in exact_b if l == 1]
                heur_sizes = [e-s for s, e, l in heur_b[:len(exact_b)] if l == 1]
                eval_sizes = [eval_samples[i]['n_tokens'] for i, (_, _, l) in enumerate(exact_b) if l == 1]
                if exact_sizes and heur_sizes:
                    print(f"  Deceptive sample sizes:")
                    print(f"    Eval data:  mean={np.mean(eval_sizes):.0f}, std={np.std(eval_sizes):.0f}")
                    print(f"    Exact:      mean={np.mean(exact_sizes):.0f}, std={np.std(exact_sizes):.0f}")
                    print(f"    Heuristic:  mean={np.mean(heur_sizes):.0f}, std={np.std(heur_sizes):.0f}")

        except Exception as e:
            print(f"  Error: {e}")

    # Now: full comparison using both methods
    print("\n" + "=" * 100)
    print("FULL RECOMPUTATION: EXACT vs HEURISTIC BOUNDARIES")
    print("=" * 100)

    results = []
    skipped = 0

    for rs_path in probe_paths:
        try:
            with open(rs_path) as f:
                raw = json.load(f)
            first_labels = raw['sample_labels'][:N_AI_LIAR]
            expected = [s['label'] for s in eval_samples]
            if first_labels != expected:
                skipped += 1
                continue

            parts = rs_path.replace(os.sep, '/').split('/')
            source = parts[0]
            probe = ''
            layer = ''
            for p in parts:
                if any(x in p for x in ['attention', 'logistic', 'meandiff', 'sklearn', 'apollo']):
                    probe = p
                if p.startswith('layer_'):
                    layer = p.replace('layer_', '')

            # Exact boundaries
            exact_b, ts, exact_err = exact_boundaries(raw, eval_samples, N_AI_LIAR)
            if len(exact_b) < N_AI_LIAR:
                skipped += 1
                continue

            # Check alignment quality
            mean_err = np.mean(exact_err)

            exact_metrics = compute_metrics(exact_b, ts, eval_samples, spans_by_idx)

            # Also compute with heuristic for comparison
            heur_b, _, heur_err = heuristic_boundaries(raw, N_AI_LIAR)
            heur_metrics = compute_metrics(heur_b, ts, eval_samples, spans_by_idx) if len(heur_b) >= N_AI_LIAR else {}

            results.append({
                'source': source,
                'probe': probe,
                'layer': layer,
                'alignment_error': mean_err,
                'exact': exact_metrics,
                'heuristic': heur_metrics,
            })

        except Exception as e:
            skipped += 1

    print(f"\nProcessed {len(results)} probes ({skipped} skipped)")

    # Filter to reliable results
    reliable = [r for r in results
                if r['exact'].get('n_stmt', 0) >= 10
                and r['exact'].get('n_phrase', 0) >= 10]

    print(f"Reliable (n >= 10): {len(reliable)}")

    if not reliable:
        print("No reliable results!")
        return

    # Print comparison
    print(f"\n{'='*140}")
    print(f"{'Probe':<40} {'L':>3} | {'AlignErr':>8} | "
          f"{'EXACT PhrNorm':>13} {'EXACT StmNorm':>13} | "
          f"{'HEUR PhrNorm':>12} {'HEUR StmNorm':>12} | "
          f"{'Phr Delta':>9} {'Stm Delta':>9}")
    print("-" * 140)

    for r in sorted(reliable, key=lambda x: x['exact'].get('stmt_norm_auroc', 0), reverse=True)[:40]:
        e = r['exact']
        h = r['heuristic']
        ep = e.get('phrase_norm_auroc', 0)
        es = e.get('stmt_norm_auroc', 0)
        hp = h.get('phrase_norm_auroc', 0)
        hs = h.get('stmt_norm_auroc', 0)
        pd = ep - hp if hp else 0
        sd = es - hs if hs else 0
        print(f"{r['source'][:18]+'/'+r['probe']:<40} {r['layer']:>3} | "
              f"{r['alignment_error']:>8.5f} | "
              f"{ep:>13.4f} {es:>13.4f} | "
              f"{hp:>12.4f} {hs:>12.4f} | "
              f"{pd:>+9.4f} {sd:>+9.4f}")

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY: EXACT BOUNDARIES")
    print("=" * 80)

    exact_phrase = [r['exact']['phrase_norm_auroc'] for r in reliable if 'phrase_norm_auroc' in r['exact']]
    exact_stmt = [r['exact']['stmt_norm_auroc'] for r in reliable if 'stmt_norm_auroc' in r['exact']]
    heur_phrase = [r['heuristic']['phrase_norm_auroc'] for r in reliable
                   if r['heuristic'] and 'phrase_norm_auroc' in r['heuristic']]
    heur_stmt = [r['heuristic']['stmt_norm_auroc'] for r in reliable
                 if r['heuristic'] and 'stmt_norm_auroc' in r['heuristic']]

    if exact_phrase:
        print(f"EXACT phrase norm AUROC:  mean={np.mean(exact_phrase):.4f}, "
              f"std={np.std(exact_phrase):.4f}, range=[{min(exact_phrase):.4f}, {max(exact_phrase):.4f}]")
    if exact_stmt:
        print(f"EXACT stmt norm AUROC:   mean={np.mean(exact_stmt):.4f}, "
              f"std={np.std(exact_stmt):.4f}, range=[{min(exact_stmt):.4f}, {max(exact_stmt):.4f}]")
    if heur_phrase:
        print(f"HEUR phrase norm AUROC:  mean={np.mean(heur_phrase):.4f}, "
              f"std={np.std(heur_phrase):.4f}, range=[{min(heur_phrase):.4f}, {max(heur_phrase):.4f}]")
    if heur_stmt:
        print(f"HEUR stmt norm AUROC:    mean={np.mean(heur_stmt):.4f}, "
              f"std={np.std(heur_stmt):.4f}, range=[{min(heur_stmt):.4f}, {max(heur_stmt):.4f}]")

    # Alignment quality
    all_err = [r['alignment_error'] for r in reliable]
    print(f"\nAlignment error (exact): mean={np.mean(all_err):.6f}, max={max(all_err):.6f}")

    # Per-sample distributions
    ps_stmt = [r['exact'].get('per_sample_stmt_mean', 0) for r in reliable if 'per_sample_stmt_mean' in r['exact']]
    ps_phrase = [r['exact'].get('per_sample_phrase_mean', 0) for r in reliable if 'per_sample_phrase_mean' in r['exact']]
    if ps_stmt:
        print(f"\nPer-sample stmt AUROC means:  mean={np.mean(ps_stmt):.4f}, range=[{min(ps_stmt):.4f}, {max(ps_stmt):.4f}]")
    if ps_phrase:
        print(f"Per-sample phrase AUROC means: mean={np.mean(ps_phrase):.4f}, range=[{min(ps_phrase):.4f}, {max(ps_phrase):.4f}]")

    # Check if exact vs heuristic makes a difference
    if heur_phrase and exact_phrase and len(heur_phrase) == len(exact_phrase):
        diffs = [e - h for e, h in zip(exact_phrase, heur_phrase)]
        print(f"\nExact-Heuristic difference (phrase): mean={np.mean(diffs):+.4f}, max_abs={max(abs(d) for d in diffs):.4f}")
    if heur_stmt and exact_stmt and len(heur_stmt) == len(exact_stmt):
        diffs = [e - h for e, h in zip(exact_stmt, heur_stmt)]
        print(f"Exact-Heuristic difference (stmt):   mean={np.mean(diffs):+.4f}, max_abs={max(abs(d) for d in diffs):.4f}")

    # Save results
    rows = []
    for r in results:
        e = r['exact']
        h = r['heuristic']
        rows.append({
            'source': r['source'],
            'probe': r['probe'],
            'layer': r['layer'],
            'alignment_error': r['alignment_error'],
            'exact_phrase_norm': e.get('phrase_norm_auroc', ''),
            'exact_stmt_norm': e.get('stmt_norm_auroc', ''),
            'exact_phrase_raw': e.get('phrase_raw_auroc', ''),
            'exact_stmt_raw': e.get('stmt_raw_auroc', ''),
            'heur_phrase_norm': h.get('phrase_norm_auroc', '') if h else '',
            'heur_stmt_norm': h.get('stmt_norm_auroc', '') if h else '',
            'n_phrase': e.get('n_phrase', 0),
            'n_stmt': e.get('n_stmt', 0),
            'n_phrase_pos': e.get('n_phrase_pos', 0),
            'n_stmt_pos': e.get('n_stmt_pos', 0),
            'ps_stmt_mean': e.get('per_sample_stmt_mean', ''),
            'ps_phrase_mean': e.get('per_sample_phrase_mean', ''),
        })

    cols = list(rows[0].keys()) if rows else []
    with open('FIXED_NORM_AUROC.tsv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter='\t', extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows to FIXED_NORM_AUROC.tsv")


if __name__ == '__main__':
    main()
