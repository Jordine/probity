"""Evaluate frozen probe readouts on the newly generated display continuations.

Only fresh suffixes enter. Policy-stratified estimates are exploratory and do
not imply an intervention using any alternative probe was generated.
"""
import argparse
import json
import textwrap
from pathlib import Path

from probity_benchmark.grid6_evaluate import digest, estimate, join_units, read_rows


def attach_metadata(score, metadata):
    key = f"{score['episode_id']}:{score['turn_index']}"
    if key not in metadata:
        raise ValueError('score outside frozen display suffix population')
    meta = metadata[key]
    if score['response_body_sha256'] != meta['response_body_sha256']:
        raise ValueError('score response differs from frozen display receipt')
    if score['turn_index'] not in (2, 3, 4, 5):
        raise ValueError('shared prefix must not enter fresh suffix evaluation')
    # GPU scores have no experimental-group fields. These are supplied only
    # from the separately pinned compiler metadata, never inferred from text.
    return {**score, 'family': meta['family'], 'arm': meta['policy']}


def plot(results, coverage, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    probes = [
        ('local3:local3', 'Local span (layer 35)'),
        ('local3:random_direction', 'Dev-selected best of 32 random directions'),
        ('response_broadcast', 'Adapted response mean'),
        ('broadcast:adapted:dyl_no', 'Adapted DYL: No'),
        ('broadcast:adapted:dyl_difference', 'Adapted DYL: No − Yes'),
        ('broadcast:apollo_data:response_mean', 'Apollo-trained response mean'),
        ('broadcast:published_fibs:no', 'Published FIBS: No'),
        ('broadcast:published_fibs:yes', 'Published FIBS: Yes'),
        ('broadcast:published_fibs:no_minus_yes', 'FIBS No − Yes contrast'),
    ]
    policies = ['none', 'neutral', 'bundle', 'turn_scores', 'speaker_means', 'recommendation', 'shuffled_bundle']
    lookup = {(r['policy'], r['probe']): r for r in results}
    fig, axes = plt.subplots(1, 2, figsize=(17, 6), sharey=True)
    for ax, metric, title in zip(axes,
            ('auroc', 'within_response_family_macro_auroc'),
            ('Across sentences, family balanced', 'Within arguments containing both classes')):
        values = np.array([[lookup.get((p, name), {}).get(metric) for p in policies]
                           for name, _ in probes], dtype=float)
        panel = ax.imshow(values, vmin=0, vmax=1, cmap='RdBu', aspect='auto')
        for i in range(len(probes)):
            for j in range(len(policies)):
                if np.isfinite(values[i, j]):
                    denominator = lookup.get((policies[j], probes[i][0]), {}).get('mixed_truth_families')
                    suffix = f'\nn={denominator}' if metric == 'within_response_family_macro_auroc' else ''
                    ax.text(j, i, f'{values[i,j]:.2f}{suffix}', ha='center', va='center',
                            color='white' if abs(values[i,j] - .5) > .28 else 'black', fontsize=8)
        ax.set_xticks(range(len(policies)), [p.replace('_', '\n') for p in policies], fontsize=8)
        ax.set_title(title)
    axes[0].set_yticks(range(len(probes)), [label for _, label in probes], fontsize=9)
    fig.colorbar(panel, ax=axes, fraction=.02, pad=.02, label='AUROC; higher score predicts contradiction')
    count_text = '; '.join(f"{p}: {coverage[p]['eligible_units']} eligible sentences / {coverage[p]['families']} families" for p in policies)
    fig.suptitle('Fresh display continuations · primary automated factuality labels\n'
                 'Alternative probes were rescored on fixed behavior; only the original adapted probe supplied live feedback.', fontsize=11)
    mixed_text = '; '.join(f"{p}: {lookup.get((p, 'local3:local3'), {}).get('mixed_truth_turns', 0)} mixed arguments / {lookup.get((p, 'local3:local3'), {}).get('mixed_truth_families', 0)} families" for p in policies)
    footer = textwrap.fill(count_text, 155) + '\n' + textwrap.fill('Localization denominator: ' + mixed_text, 155)
    fig.text(.02, .01, footer + '\nPolicy-wise point estimates; no significance claim. Response-level scores cannot localize within an argument.', fontsize=7)
    fig.subplots_adjust(left=.18, bottom=.19, right=.89, top=.82, wspace=.12)
    for ext in ('png', 'pdf'):
        fig.savefig(out / ('live_probe_factuality.' + ext), dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--planning', type=Path, default=Path('runs/iteration6/live_label_planning'))
    p.add_argument('--labels', type=Path, default=Path('runs/iteration6/labels/live_suffix_v1/units.jsonl'))
    p.add_argument('--scores', type=Path, default=Path('runs/iteration6/gpu/results_v1/live_multi'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--bootstraps', type=int, default=1000)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('new output directory required')
    plan_path = args.planning / 'annotation_plan.json'
    plan = json.loads(plan_path.read_text())
    for name, expected in plan['files_sha256'].items():
        if digest(args.planning / name) != expected:
            raise ValueError('live annotation plan input changed: ' + name)
    population = read_rows(args.planning / 'turn_metadata_private.jsonl')
    meta = {r['id']: {**r, 'arm': r['policy'], 'matched_question': r['source_problem']} for r in population}
    if len(population) != 672 or len(meta) != 672:
        raise ValueError('expected 672 distinct fresh suffix turns')
    labels = read_rows(args.labels)
    if not labels or not all(r['primary_annotated'] for r in labels):
        raise ValueError('complete primary factuality labels required')
    if len(labels) != plan['fixed_sentence_units'] or {r['turn_id'] for r in labels} != set(meta):
        raise ValueError('labeled population differs from complete suffix plan')
    original = {r['id']: r for r in read_rows(args.planning / 'annotation_units_private.jsonl')}
    for key, item in meta.items():
        indexes = [r['unit_index'] for r in labels if r['turn_id'] == key]
        if sorted(indexes) != list(range(item['unit_count'])):
            raise ValueError('turn sentence population differs')
    for label in labels:
        source = original[label['turn_id']]
        if (label['unit'] != source['units'][label['unit_index']]
            or label['source_story_sha256'] != source['source_story_sha256']
            or label['generation_request_sha256'] != source['request_sha256']):
            raise ValueError('label source/unit differs from frozen compiler output')
    files, scored = [], []
    for file in sorted(args.scores.glob('*.json')):
        row = json.loads(file.read_text())
        key = f"{row['episode_id']}:{row['turn_index']}"
        if key in meta:
            files.append(file)
            scored.append(attach_metadata(row, meta))
    if len(scored) != 672:
        raise ValueError('complete native display suffix scores required')
    joined, missing = join_units(labels, scored, meta)
    if missing:
        raise ValueError('fresh labels missing native scores')
    names = sorted({name for row in joined for name in row['scores']})
    groups = {'all': joined}
    for policy in sorted({r['policy'] for r in joined}):
        groups[policy] = [r for r in joined if r['policy'] == policy]
    coverage = {key: {'units': len(rows), 'eligible_units': sum(r['primary_label'] is not None for r in rows),
                      'families': len({r['family'] for r in rows}), 'turns': len({r['turn_id'] for r in rows})}
                for key, rows in groups.items()}
    results = []
    for policy, rows in groups.items():
        for name in names:
            result = estimate(rows, name, 'primary_label', args.bootstraps if policy == 'all' else 0)
            if result:
                results.append({'policy': policy, **result})
    args.out.mkdir(parents=True)
    (args.out / 'metrics.json').write_text(json.dumps(results, indent=2) + '\n')
    summary = {'complete': True, 'turns': len(scored), 'units': len(joined), 'coverage': coverage,
        'limits': ['Fresh suffix turns only; shared two-turn prefixes excluded.',
                  'All seven policies use the same adapted probe for live feedback; alternative readouts are offline.',
                  'Primary single-reader source factuality, not deception intent; unknowns excluded, never truthful negatives.',
                  'Pooled estimates balance families but retain observed policy sentence counts within each family.',
                  'Policy-specific estimates are exploratory points; pooled intervals resample families, not generation seeds.',
                  'No consensus suffix labels or human ground-truth claims.'],
        'inputs_sha256': {str(f): digest(f) for f in [args.labels, plan_path, *files]},
        'script_sha256': digest(__file__),
        'shared_estimator_sha256': digest(Path(__file__).with_name('grid6_evaluate.py'))}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    plot(results, coverage, args.out)
    print(json.dumps({'complete': True, 'turns': len(scored), 'units': len(joined), 'out': str(args.out)}))


if __name__ == '__main__':
    main()
