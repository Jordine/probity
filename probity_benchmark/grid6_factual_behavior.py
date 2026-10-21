"""Paired historical A/B factuality rates, distinct from detector scores."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .grid6_evaluate import digest, metadata, read_rows


def summarize(labels, meta, tier, role=None, bootstraps=5000):
    turns = defaultdict(list)
    for label in labels:
        m = meta[label['turn_id']]
        if role is None or role == m['role']:
            turns[label['turn_id']].append(label)
    records = []
    for turn, items in turns.items():
        eligible = [r[tier] for r in items if r[tier] is not None]
        records.append({**meta[turn], 'turn_id': turn, 'units': len(items),
                        'eligible': len(eligible), 'contradiction_fraction': float(np.mean(eligible)) if eligible else None,
                        'eligible_coverage': len(eligible) / len(items),
                        'primary_annotation_coverage': np.mean([r['primary_annotated'] for r in items])})
    pairs = defaultdict(dict)
    for record in records:
        key = (record['family'], record['matched_question'], record['turn_index'])
        if record['arm'] in pairs[key]:
            raise ValueError('multiple records for matched turn')
        pairs[key][record['arm']] = record
    output = {}
    for metric in ['contradiction_fraction', 'eligible_coverage', 'primary_annotation_coverage']:
        by_question = defaultdict(list)
        for (family, question, index), pair in pairs.items():
            if set(pair) != {'A', 'B'} or any(pair[arm][metric] is None for arm in ['A', 'B']):
                continue
            by_question[family, question].append([pair['A'][metric], pair['B'][metric]])
        by_family = defaultdict(list)
        for (family, question), values in by_question.items():
            by_family[family].append(np.mean(values, axis=0))
        values = np.array([np.mean(v, axis=0) for k, v in sorted(by_family.items())])
        if len(values) == 0:
            output[metric] = {'families': 0}
            continue
        rng = np.random.default_rng(61)
        delta = values[:, 1] - values[:, 0]
        draws = np.mean(delta[rng.integers(len(delta), size=(bootstraps, len(delta)))], axis=1)
        output[metric] = {'families': len(values), 'questions': len(by_question),
                          'matched_turns': sum(len(v) for v in by_question.values()),
                          'A': float(values[:, 0].mean()), 'B': float(values[:, 1].mean()),
                          'B_minus_A': float(delta.mean()), 'delta_ci95': np.quantile(draws, [.025, .975]).tolist()}
    coverage = {arm: {'turns': sum(r['arm'] == arm for r in records),
                     'all_ineligible_turns': sum(r['arm'] == arm and r['eligible'] == 0 for r in records)} for arm in ['A', 'B']}
    return {'tier': tier, 'role': role or 'all', 'metrics': output, 'coverage': coverage}, records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('new analysis destination required')
    m = metadata(Path('runs/iteration5/judge_grid_v1/package.json'), Path('runs/iteration5/judge_grid_v1/pairs.json'))
    labels = read_rows(a.labels)
    results, records = [], []
    for tier in ['primary_label', 'consensus_label']:
        for role in [None, 'correct_advocate', 'wrong_advocate']:
            result, rows = summarize(labels, m, tier, role)
            results.append(result)
            if role is None:
                records.extend({'tier': tier, **row} for row in rows)
    a.out.mkdir(parents=True)
    summary = {'schema': 'iteration6_paired_factual_behavior_v1', 'results': results,
               'label_units': len(labels), 'primary_annotated_units': sum(r['primary_annotated'] for r in labels),
               'complete_primary_labels': all(r['primary_annotated'] for r in labels),
               'weighting': 'Pair A/B by question and turn; average matched turns per question then questions per source family. Equal family weight.',
               'limits': 'Automatic factuality labels, not intent. Contradiction fraction conditions on eligible sentences, not all generated claims. Unknowns and missing labels are excluded. Coverage reported separately. Historical A/B differ in awareness from turn1; not shared-prefix intervention.',
               'labels_sha256': digest(a.labels), 'script_sha256': digest(__file__)}
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (a.out / 'turns.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    selected = [r for r in results if r['tier'] == 'primary_label']
    for ax, key, title in zip(axes, ['contradiction_fraction', 'eligible_coverage'],
                             ['Contradicted fraction among eligible sentences', 'Eligible sentence coverage']):
        for i, result in enumerate(selected):
            point = result['metrics'][key]
            if not point.get('families'):
                continue
            value = point['B_minus_A'] * 100
            lo, hi = np.array(point['delta_ci95']) * 100
            ax.plot([lo, hi], [i, i], linewidth=2); ax.scatter([value], [i], s=40)
        ax.axvline(0, color='gray', linestyle='--'); ax.set_title(title, fontsize=10)
        tick_labels = []
        for result in selected:
            point = result['metrics'][key]
            tick_labels.append(result['role'].replace('_', ' ') + '\n' +
                          f"{point.get('families', 0)} families · {point.get('questions', 0)} questions\n" +
                          f"{point.get('matched_turns', 0)} paired turns")
        ax.set_yticks(range(len(selected)), tick_labels, fontsize=8)
        ax.invert_yaxis(); ax.set_xlabel('B − A (percentage points), paired family 95% CI'); ax.grid(axis='x', alpha=.2)
    banner = '' if summary['complete_primary_labels'] else 'INCOMPLETE LABELS — pipeline check, not a result\n'
    fig.suptitle(banner + 'Qwen3.5 debate · historical live feedback vs no feedback\nAutomated primary reader; unequal coverage can affect conditional factuality rates\n' + f"Primary annotated units: {summary['primary_annotated_units']}/{len(labels)}", fontsize=10,
                 color='black' if summary['complete_primary_labels'] else '#9b1c1c')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(a.out / f'factual_behavior.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k != 'results'}, indent=2))


if __name__ == '__main__':
    main()
