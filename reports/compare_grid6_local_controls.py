"""Post-hoc paired comparisons of a frozen local head with frozen controls.

No head selection, retraining, paid calls or changes to primary results.
The random-direction control was selected from 32 on earlier development data.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

LOCAL = 'local3:local3'
CONTROLS = {'local3:random_direction': 'Development-selected random direction (best of 32)',
            'length': 'Token count', 'position': 'Relative position'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('new output directory required')
    joined_path = a.evaluation / 'joined_units.jsonl'
    rows = [json.loads(line) for line in joined_path.read_text().splitlines()]
    turns = defaultdict(list)
    for row in rows:
        if row['primary_label'] is not None:
            turns[row['turn_id']].append(row)
    per_turn = []
    for turn, items in turns.items():
        y = [r['primary_label'] for r in items]
        if len(set(y)) != 2:
            continue
        scores = {name: float(roc_auc_score(y, [r['scores'][name] for r in items]))
                  for name in [LOCAL, *CONTROLS]}
        per_turn.append({'turn_id': turn, 'family': items[0]['family'],
                         'role': items[0]['role'], 'scores': scores})
    results = []
    for role in ['all', 'correct_advocate', 'wrong_advocate']:
        subset = [r for r in per_turn if role == 'all' or r['role'] == role]
        families = defaultdict(list)
        for row in subset:
            families[row['family']].append(row)
        keys = sorted(families)
        if not keys:
            continue
        for control, display_name in CONTROLS.items():
            local = np.array([np.mean([r['scores'][LOCAL] for r in families[f]]) for f in keys])
            baseline = np.array([np.mean([r['scores'][control] for r in families[f]]) for f in keys])
            difference = local - baseline
            rng = np.random.default_rng(62109)
            draws = difference[rng.integers(len(keys), size=(10000, len(keys)))].mean(axis=1)
            results.append({'role': role, 'local_probe': LOCAL, 'control': control,
                'control_description': display_name, 'mixed_arguments': len(subset),
                'families': len(keys), 'local_auroc': float(local.mean()),
                'control_auroc': float(baseline.mean()), 'paired_difference': float(difference.mean()),
                'ci95': np.quantile(draws, [.025, .975]).tolist()})
    # Independent recomputation must agree with the earlier primary estimate.
    original = json.loads((a.evaluation / 'metrics.json').read_text())
    for result in results:
        group = 'all' if result['role'] == 'all' else 'role:' + result['role']
        reference = [r for r in original if r['group'] == group
                     and r['tier'] == 'primary_label' and r['probe'] == LOCAL]
        if reference:
            if abs(reference[0]['within_response_family_macro_auroc'] - result['local_auroc']) > 1e-12:
                raise ValueError('independent recomputation differs from primary metric')
    a.out.mkdir(parents=True)
    summary = {'schema': 'grid6_posthoc_local_control_difference_v1',
        'posthoc': True, 'bootstrap_replicates': 10000, 'seed': 62109, 'results': results,
        'weighting': 'AUROC per mixed-label argument, mean within source family, equal mean over families; paired family bootstrap for each local-minus-control contrast.',
        'limitations': ['Frozen local head and controls; no selection on current outcomes.',
            'Random-direction control was selected from32 on prior development data and is not an independent random null.',
            'This post-hoc comparison was added after seeing elevated control performance.',
            'Conditional on retained automated labels, historically exposed families and one generation; no human-gold or seed-variance claim.'],
        'inputs_sha256': {str(p): sha(p) for p in [joined_path, a.evaluation / 'metrics.json']},
        'script_sha256': sha(Path(__file__))}
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, role, title in zip(axes, ['all', 'correct_advocate', 'wrong_advocate'],
                              ['All mixed arguments', 'Correct-answer advocates', 'Wrong-answer advocates']):
        rr = [r for r in results if r['role'] == role]
        for i, row in enumerate(rr):
            ax.plot(row['ci95'], [i, i], color='#246b8e', linewidth=2)
            ax.scatter(row['paired_difference'], i, color='#246b8e')
        if rr:
            ax.set_title(title + f"\n{rr[0]['mixed_arguments']} arguments · {rr[0]['families']} families", fontsize=9)
        ax.axvline(0, color='gray', linestyle='--')
        ax.grid(axis='x', alpha=.2)
        ax.set_xlabel('Local minus control AUROC')
        ax.set_yticks(range(3), ['Dev-selected random\n(best of 32)', 'Token count', 'Relative position'])
        ax.invert_yaxis()
    fig.suptitle('Post-hoc paired control comparison · 95% family-bootstrap intervals\nWithin-argument discrimination under automated primary labels', fontsize=10)
    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(a.out / ('local_control_differences.' + ext), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
