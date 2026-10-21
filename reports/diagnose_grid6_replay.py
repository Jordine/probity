"""Characterize all historical scalar differences after complete native replay."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from probity_benchmark.judge3_protocol import percentile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError('new output directory required')
    base = ROOT / 'runs/iteration6/gpu'
    plan_path = base / 'replay_v1/plan.json'
    plan = json.loads(plan_path.read_text())
    request_file = base / 'replay_v1' / plan['request_file']
    reference_path = ROOT / plan['reference_path']
    if sha(request_file) != plan['request_file_sha256'] or sha(reference_path) != plan['reference_sha256']:
        raise ValueError('frozen requests or percentile reference changed')
    reference = json.loads(reference_path.read_text())['references']['qwen/qwen3.5-27b']['scores']
    with gzip.open(request_file, 'rt') as handle:
        originals = [json.loads(line) for line in handle]
    files = sorted((base / 'results_v1/scalars').glob('*.json'))
    current = {json.loads(f.read_text())['job_id']: json.loads(f.read_text()) for f in files}
    if (len(files) != 1992 or len(originals) != 1992
            or set(current) != {r['request']['job_id'] for r in originals}):
        raise ValueError('complete exact 1992-job population required')
    rows = []
    for original in originals:
        request = original['request']
        result = current[request['job_id']]
        for key in ('source_request_sha256', 'response_body_sha256', 'argument_sha256', 'episode_id', 'turn_index'):
            if request[key] != result[key]:
                raise ValueError('scored context differs')
        old_raw, new_raw = original['expected_raw_score'], result['raw_score']
        old_percentile, new_percentile = percentile(old_raw, reference), percentile(new_raw, reference)
        if new_percentile != result['percentile']:
            raise ValueError('saved current percentile disagrees with frozen reference')
        rows.append({'job_id': request['job_id'], 'family': original['family'], 'arm': original['arm'],
                     'old_raw': old_raw, 'new_raw': new_raw, 'raw_difference': new_raw - old_raw,
                     'old_percentile': old_percentile, 'new_percentile': new_percentile,
                     'percentile_difference': round(new_percentile - old_percentile, 3)})
    counts = Counter(row['percentile_difference'] for row in rows)
    summary = {'complete': True, 'turns': len(rows),
               'raw_exact': sum(row['raw_difference'] == 0 for row in rows),
               'raw_within_1e_minus4': sum(abs(row['raw_difference']) <= 1e-4 for row in rows),
               'maximum_absolute_raw_difference': max(abs(row['raw_difference']) for row in rows),
               'display_percentile_changed': sum(row['percentile_difference'] != 0 for row in rows),
               'maximum_absolute_display_difference': max(abs(row['percentile_difference']) for row in rows),
               'display_difference_counts': dict(sorted(counts.items())),
               'limits': ['Historical feedback was preserved in every replay input.',
                          'Current shared-forward and original scalar implementations passed their separate 12-case same-run equivalence check.',
                          'This measures numerical reproducibility across runs; it does not identify a kernel or prove API/native equivalence.'],
               'inputs_sha256': {str(f): sha(f) for f in [plan_path, request_file, reference_path, *files]},
               'script_sha256': sha(Path(__file__))}
    args.out.mkdir(parents=True)
    with (args.out / 'turn_differences.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist([row['raw_difference'] for row in rows], bins=40, color='#3978a8')
    axes[0].set_xlabel('Current − historical raw probe score'); axes[0].set_ylabel('Public turns')
    axes[0].axvline(0, color='black', linewidth=.8)
    values = sorted(counts)
    axes[1].bar(range(len(values)), [counts[v] for v in values], color='#3978a8')
    axes[1].set_xticks(range(len(values)), [f'{v:+g}' for v in values], rotation=35)
    axes[1].set_xlabel('Change in displayed reference percentile'); axes[1].set_ylabel('Public turns (log scale)')
    axes[1].set_yscale('log')
    for i, v in enumerate(values):
        axes[1].text(i, counts[v] * 1.06, str(counts[v]), ha='center', fontsize=8)
    fig.suptitle(f"Historical replay numerical diagnostic · 1,992 full contexts\nDisplayed percentile changes in {summary['display_percentile_changed']} turns; original B feedback retained")
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(args.out / ('replay_differences.' + ext), dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items() if k != 'inputs_sha256'}, indent=2))


if __name__ == '__main__':
    main()
