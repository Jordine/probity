"""Check if any raw_scores.json files have per-token (not per-sample) labels."""
import json, numpy as np, glob, os

all_rs = sorted(glob.glob('**/raw_scores.json', recursive=True))
has_mixed = 0
checked = 0
sources_with_mixed = []

for path in all_rs:
    try:
        with open(path) as f:
            rs = json.load(f)
        tl = rs.get('token_labels', [])
        sl = rs.get('sample_labels', [])
        ts = rs.get('token_scores', [])
        ss = rs.get('sample_scores', [])
        if not tl or not sl:
            continue

        n_dec = sum(sl)
        if n_dec == 0:
            checked += 1
            continue

        # Reconstruct boundaries for first 30 samples
        ts_arr = np.array(ts)
        cumsum = np.cumsum(ts_arr)
        cumsum = np.insert(cumsum, 0, 0.0)
        start = 0
        found_mixed = False
        for i in range(min(30, len(ss))):
            target = ss[i]
            base = cumsum[start]
            max_end = min(len(ts_arr), start + 2000)
            sizes = np.arange(5, max_end - start + 1)
            if len(sizes) == 0:
                break
            actual = cumsum[start + sizes] - base
            expected_sums = target * sizes
            errors = np.abs(actual - expected_sums)
            best_size = int(sizes[np.argmin(errors)])

            if sl[i] == 1:
                seg_tl = np.array(tl[start:start+best_size])
                if len(np.unique(seg_tl)) > 1:
                    found_mixed = True
                    has_mixed += 1
                    sources_with_mixed.append(path)
                    break
            start += best_size
        checked += 1
    except Exception as e:
        pass

print(f'Checked {checked} raw_scores files')
print(f'Files with mixed token labels in deceptive samples: {has_mixed}')
if sources_with_mixed:
    for p in sources_with_mixed[:10]:
        print(f'  {p}')
else:
    print('ALL raw_scores use sample-level token labels (no sub-sentence structure)')
