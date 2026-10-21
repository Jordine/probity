"""Evaluate frozen probes against exact sentence factuality labels.

Family-balanced estimates; resampling is clustered by source family. Existing
training exposure is reported, never treated as an untouched test population.
Response-level scores broadcast onto sentences are explicitly nonlocal controls.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def join_units(labels, scored, turn_meta):
    """Fail on a scored-context mismatch; never silently join by similar text."""
    index = {}
    signatures = set()
    for score in scored:
        key = f"{score['episode_id']}:{score['turn_index']}"
        if key in index:
            raise ValueError('duplicate scored turn')
        if key not in turn_meta:
            raise ValueError('scored turn outside frozen population')
        if not score.get('replay_contract_sha256') or not score.get('head_provenance') or not score.get('units'):
            raise ValueError('scored turn missing replay/head provenance or units')
        local_keys = set(score['units'][0]['scores'])
        if any(set(unit['scores']) != local_keys for unit in score['units']):
            raise ValueError('local score inventory changes within a turn')
        signatures.add((score['replay_contract_sha256'], json.dumps(score['head_provenance'], sort_keys=True),
                        tuple(sorted(local_keys)), tuple(sorted(score['response_scores'])),
                        tuple(sorted(score.get('fibs', {}))),
                        tuple(sorted(score.get('fibs', {}).get('layer_no_minus_yes_probability', {})))))
        index[key] = score
    if len(signatures) > 1:
        raise ValueError('replay contracts, head provenance, or score inventories differ across turns')
    joined, missing = [], 0
    seen_labels = set()
    for label in labels:
        key = (label['turn_id'], label['unit_index'])
        if key in seen_labels:
            raise ValueError('duplicate labeled turn/unit')
        seen_labels.add(key)
        score = index.get(label['turn_id'])
        if score is None:
            missing += 1
            continue
        if score['source_request_sha256'] != label['generation_request_sha256']:
            raise ValueError('generation request differs from labeled context')
        unit = label['unit']
        matches = [u for u in score['units'] if (u['start'], u['end']) == (unit['start'], unit['end'])]
        if len(matches) != 1:
            raise ValueError('exact labeled sentence offsets absent or duplicated')
        match = matches[0]
        if any(match[k] != unit[k] for k in ('text', 'display_text', 'argument_start', 'argument_end')):
            raise ValueError('sentence text/mapping differs from labeled unit')
        if 'sentence' not in match['granularities']:
            raise ValueError('labeled sentence matched nonsentence span')
        scores = dict(match['scores'])
        scores.update({'broadcast:' + name: value for name, value in score['response_scores'].items()})
        scores.update({'broadcast:published_fibs:' + arm: score['fibs'][arm]
                       for arm in ('no', 'yes') if arm in score.get('fibs', {})})
        scores.update({'broadcast:fibs_layer_' + layer: value
                       for layer, value in score.get('fibs', {}).get('layer_no_minus_yes_probability', {}).items()})
        if any(not np.isfinite(value) for value in scores.values()):
            raise ValueError('nonfinite probe score')
        meta = turn_meta[label['turn_id']]
        if score['response_body_sha256'] != meta['response_body_sha256']:
            raise ValueError('scored response differs from original labeled response')
        if meta['family'] != score['family'] or meta['arm'] != score['arm']:
            raise ValueError('family/arm differs from frozen source population')
        joined.append({'turn_id': label['turn_id'], 'unit_index': label['unit_index'],
                       **meta, 'primary_label': label['primary_label'],
                       'consensus_label': label['consensus_label'], 'scores': scores})
    return joined, missing


def metrics(y, scores, weights):
    keep = weights > 0
    y, scores, weights = y[keep], scores[keep], weights[keep]
    if len(np.unique(y)) < 2:
        return None
    return {'auroc': float(roc_auc_score(y, scores, sample_weight=weights)),
            'average_precision': float(average_precision_score(y, scores, sample_weight=weights)),
            'positive_prevalence': float(np.average(y, weights=weights))}


def estimate(rows, name, tier, bootstraps, seed=61):
    rows = [r for r in rows if r[tier] is not None and name in r['scores']]
    if not rows:
        return None
    families = sorted({r['family'] for r in rows})
    lookup = {family: i for i, family in enumerate(families)}
    fi = np.array([lookup[r['family']] for r in rows])
    counts = np.bincount(fi)
    weights = 1 / counts[fi]
    y = np.array([r[tier] for r in rows])
    scores = np.array([r['scores'][name] for r in rows])
    point = metrics(y, scores, weights)
    if point is None:
        return None
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstraps):
        mult = np.bincount(rng.integers(0, len(families), len(families)), minlength=len(families))
        result = metrics(y, scores, weights * mult[fi])
        if result:
            draws.append(result)
    ci = {key: np.quantile([d[key] for d in draws], [.025, .975]).tolist()
          for key in point} if draws else {}
    # Within-response discrimination excludes constant truth responses. It
    # exposes apparent localization from across-response/role differences.
    turns = defaultdict(list)
    for row in rows:
        turns[row['turn_id']].append(row)
    within = defaultdict(list)
    for items in turns.values():
        if len({r[tier] for r in items}) < 2:
            continue
        auc = roc_auc_score([r[tier] for r in items], [r['scores'][name] for r in items])
        within[items[0]['family']].append(float(auc))
    within_values = np.array([np.mean(v) for _, v in sorted(within.items())])
    within_ci = None
    if len(within_values) and bootstraps:
        within_draws = np.mean(within_values[rng.integers(len(within_values), size=(bootstraps, len(within_values)))], axis=1)
        within_ci = np.quantile(within_draws, [.025, .975]).tolist()
    return {'probe': name, 'tier': tier, 'units': len(rows), 'families': len(families),
            'turns': len(turns), 'contradicted': int(y.sum()), 'supported': int(len(y) - y.sum()),
            **point, 'ci95': ci, 'valid_bootstraps': len(draws),
            'mixed_truth_turns': sum(map(len, within.values())),
            'mixed_truth_families': len(within),
            'within_response_family_macro_auroc_ci95': within_ci,
            'within_response_family_macro_auroc': float(np.mean([np.mean(v) for v in within.values()])) if within else None,
            'localization': 'response-level broadcast' if name.startswith('broadcast:') or name == 'response_broadcast' else 'local span or control'}


def metadata(package_path, pairs_path):
    package = json.loads(package_path.read_text())
    pairs = json.loads(pairs_path.read_text())
    episodes = {}
    for pair in pairs:
        if pair['participant_model'] != 'qwen/qwen3.5-27b':
            continue
        for arm, key in [('A', 'no_feedback_episode_id'), ('B', 'feedback_episode_id')]:
            episodes[pair[key]] = {'family': pair['family'], 'arm': arm,
                                   'matched_question': pair['matched_question_id'],
                                   'exposure_stratum': pair['exposure_stratum']}
    result = {}
    audit_path = Path('runs/iteration6/raw_audit/turn_index.jsonl')
    audit_manifest = json.loads(audit_path.with_name('manifest.json').read_text())
    if digest(audit_path) != audit_manifest['outputs_sha256']['turn_index.jsonl']:
        raise ValueError('raw response audit changed')
    raw = {f"{r['episode_id']}:{r['turn_index']}": r for r in read_rows(audit_path)}
    for filename, expected in package['input_sha256'].items():
        path = Path(filename)
        if not path.name.startswith('episode_') or path.suffix != '.json':
            continue
        ep = json.loads(path.read_text())
        if ep['episode_id'] not in episodes:
            continue
        if digest(path) != expected:
            raise ValueError('frozen episode changed')
        for i, turn in enumerate(ep['turns']):
            raw_turn = raw[f"{ep['episode_id']}:{i}"]
            if raw_turn['request_sha256'] != turn['request_sha256']:
                raise ValueError('original response audit does not match episode')
            result[f"{ep['episode_id']}:{i}"] = {**episodes[ep['episode_id']],
                'role': 'correct_advocate' if turn['advocates_gold'] else 'wrong_advocate',
                'turn_index': i, 'response_body_sha256': raw_turn['response_body_sha256']}
    return result


def plots(results, out, coverage_note='', complete=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Fixed display set, not selected by observed performance.
    fixed = ['local3:local3', 'local3:random_direction', 'response_broadcast',
             'broadcast:adapted:dyl_no', 'broadcast:adapted:dyl_difference',
             'broadcast:apollo_data:response_mean', 'broadcast:published_fibs:no',
             'broadcast:published_fibs:yes', 'broadcast:published_fibs:no_minus_yes',
             'position', 'length']
    selected = [r for name in fixed for r in results
                if r['probe'] == name and r['group'] == 'all' and r['tier'] == 'primary_label']
    display_names = {
        'local3:local3': 'Local span probe (layer 35)',
        'local3:random_direction': 'Dev-selected best of 32 random directions (layer 35)',
        'response_broadcast': 'Adapted response mean',
        'broadcast:adapted:dyl_no': 'Adapted DYL: No',
        'broadcast:adapted:dyl_difference': 'Adapted DYL: No − Yes',
        'broadcast:apollo_data:response_mean': 'Apollo-trained response mean',
        'broadcast:published_fibs:no': 'FIBS: No follow-up',
        'broadcast:published_fibs:yes': 'FIBS: Yes follow-up',
        'broadcast:published_fibs:no_minus_yes': 'FIBS: No − Yes contrast',
        'position': 'Relative character position', 'length': 'Token-count control',
    }
    if not selected:
        return
    sample = selected[0]
    sample_note = (f"Eligible: {sample['units']} sentences in {sample['families']} families; "
                   f"within-argument analysis: {sample['mixed_truth_turns']} arguments in {sample['mixed_truth_families']} families")
    banner = '' if complete else 'INCOMPLETE DATA — pipeline check, not a result\n'
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, len(selected) * .46)), sharey=True)
    y = np.arange(len(selected))
    for ax, metric, title in zip(axes, ['auroc', 'within_response_family_macro_auroc'],
                                  ['Across sentences (family balanced)', 'Within arguments containing both label classes']):
        vals = [r[metric] if r[metric] is not None else np.nan for r in selected]
        ax.scatter(vals, y, s=40)
        if metric == 'auroc':
            for i, row in enumerate(selected):
                if 'auroc' in row['ci95']:
                    ax.plot(row['ci95']['auroc'], [i, i], linewidth=2, color='tab:blue')
        else:
            for i, row in enumerate(selected):
                if row['within_response_family_macro_auroc_ci95'] is not None:
                    ax.plot(row['within_response_family_macro_auroc_ci95'], [i, i], linewidth=2, color='tab:blue')
        ax.axvline(.5, linestyle='--', color='gray')
        ax.set_xlim(0, 1); ax.set_xlabel('AUROC (higher scores predict contradiction)')
        ax.set_title(title, fontsize=10); ax.grid(axis='x', alpha=.2)
    axes[0].set_yticks(y, [display_names[r['probe']] for r in selected])
    axes[0].invert_yaxis()
    fig.suptitle(banner + 'Qwen3.5-27B debate · automated primary factuality labels\nResponse-level scores are constant within each argument\n' + sample_note + '\n' + coverage_note, fontsize=9, color='black' if complete else '#9b1c1c')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'probe_factuality.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    main = {r['probe']: r for r in results if r['group'] == 'all' and r['tier'] == 'primary_label'}
    layers = [19, 27, 35, 43, 51, 58]
    for metric, marker, label in [('auroc', 'o', 'Across sentences'),
                                   ('within_response_family_macro_auroc', 's', 'Within mixed responses')]:
        values = [main.get('broadcast:fibs_layer_' + str(layer), {}).get(metric, np.nan) for layer in layers]
        axes[0].plot(layers, values, marker=marker, label=label)
    axes[0].set_title('Published FIBS: individual-layer DYL scores')
    for method in ['natural_bce', 'twins_bce', 'twins_pairwise', 'mixed_pairwise']:
        for strength, style in [('0.001', '-'), ('0.01', '--')]:
            ll = [27, 35, 43, 51]
            values = [main.get(f'local3_candidates:{method}:L{layer}:C{strength}', {}).get('within_response_family_macro_auroc', np.nan) for layer in ll]
            axes[1].plot(ll, values, linestyle=style, marker='.', label=f'{method}, C={strength}')
    axes[1].set_title('Frozen local heads: within mixed responses')
    for ax in axes:
        ax.axhline(.5, color='gray', linestyle=':'); ax.set_ylim(0, 1)
        ax.set_xlabel('Decoder layer'); ax.set_ylabel('AUROC')
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle(banner + 'Exploratory layer comparisons · automated primary labels · historically exposed families\n' + sample_note + '\n' + coverage_note, fontsize=9, color='black' if complete else '#9b1c1c')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'probe_layers.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--scores', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--bootstraps', type=int, default=300)
    p.add_argument('--package', type=Path, default=Path('runs/iteration5/judge_grid_v1/package.json'))
    p.add_argument('--pairs', type=Path, default=Path('runs/iteration5/judge_grid_v1/pairs.json'))
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('new evaluation output required')
    files = sorted(a.scores.glob('*.json'))
    labels = read_rows(a.labels)
    joined, missing = join_units(labels, [json.loads(f.read_text()) for f in files], metadata(a.package, a.pairs))
    names = sorted({name for row in joined for name in row['scores']})
    groups = {'all': joined}
    for key in ('arm', 'role', 'exposure_stratum'):
        for value in sorted({r[key] for r in joined}):
            groups[f'{key}:{value}'] = [r for r in joined if r[key] == value]
    results = []
    coverage = {group: {'units': len(rows), 'families': len({r['family'] for r in rows}),
                        'primary_eligible': sum(r['primary_label'] is not None for r in rows),
                        'consensus_eligible': sum(r['consensus_label'] is not None for r in rows)}
                for group, rows in groups.items()}
    for tier in ('primary_label', 'consensus_label'):
        for group, rows in groups.items():
            for name in names:
                result = estimate(rows, name, tier, a.bootstraps if group == 'all' else 0)
                if result:
                    results.append({'group': group, **result})
    a.out.mkdir(parents=True)
    (a.out / 'metrics.json').write_text(json.dumps(results, indent=2) + '\n')
    columns = ['group', 'tier', 'probe', 'units', 'families', 'turns', 'contradicted', 'supported',
               'auroc', 'auroc_lo', 'auroc_hi', 'average_precision', 'ap_lo', 'ap_hi',
               'positive_prevalence', 'mixed_truth_turns', 'mixed_truth_families',
               'within_response_family_macro_auroc', 'within_lo', 'within_hi', 'localization']
    with (a.out / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for result in results:
            row = {key: result.get(key) for key in columns}
            for key, prefix in [('auroc', 'auroc'), ('average_precision', 'ap')]:
                row[prefix + '_lo'], row[prefix + '_hi'] = result['ci95'].get(key, [None, None])
            row['within_lo'], row['within_hi'] = result['within_response_family_macro_auroc_ci95'] or [None, None]
            writer.writerow(row)
    (a.out / 'joined_units.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in joined))
    summary = {'schema': 'iteration6_probe_factuality_evaluation_v1',
               'labeled_population_units': len(labels), 'scored_units': len(joined),
               'units_missing_scores': missing, 'scored_turns': len(files),
               'complete_scores': missing == 0, 'primary_annotated_units': sum(r['primary_annotated'] for r in labels),
               'probes': names, 'metric_rows': len(results), 'bootstrap_replicates': a.bootstraps,
               'coverage_by_group': coverage,
               'weighting': 'Family-balanced pooled AUROC/AP: each source family has equal total unit weight within each eligible subset. This is not the mean of separate per-family AUROCs. Within-response AUROCs are averaged within source family, then across families.',
               'limits': ['Automated labels describe source-grounded factual contradiction, not intent.',
                          'Missing/invalid/unknown labels excluded; never treated as truthful.',
                          'Primary and dual-consensus results kept separate.',
                          'Consensus coverage mixes legacy convenience labels with the new frozen12-family audit; it is not representative of the full corpus. Independent-reader agreement uses only the frozen audit families.',
                          'All families previously observed; training exposure stratified, not clean held-out claims.',
                          'No probe signs selected from this evaluation; chance reversal is reported as observed.',
                          'FIBS No/Yes are the original published ensemble outputs on each forced follow-up; No-minus-Yes and individual-layer contrasts are derived views.',
                          'Multiple layers/heads are exploratory candidates, not independently confirmed winners.',
                          'Intervals resample source families conditional on retained labels; they do not capture annotation uncertainty or additional generation seeds.',
                          'Historical B context retains original feedback; replay cannot estimate changed live behavior.'],
               'inputs_sha256': {str(f): digest(f) for f in [a.labels, a.package, a.pairs, *files]},
               'script_sha256': digest(__file__)}
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    plots(results, a.out, f"Scored turns: {len(files)}/1992; primary annotated units: {summary['primary_annotated_units']}/{len(labels)}",
          complete=missing == 0 and all(r['primary_annotated'] for r in labels))
    print(json.dumps({k: v for k, v in summary.items() if k not in ('inputs_sha256', 'probes')}, indent=2))


if __name__ == '__main__':
    main()
