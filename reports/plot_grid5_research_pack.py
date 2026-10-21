"""Offline, additive reader pack for completed grid5 aggregate summaries.

Never imports execution code or obtains outcomes. Original renderer stays frozen.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

CONTRASTS = {'B-A': ('B', 'A'), 'C-A': ('C', 'A'), 'D-B': ('D', 'B'), 'D-A': ('D', 'A')}
INTERACTION = '(D-B)-(C-A)'
LABELS = ['B − A: feedback trajectory comparison', 'C − A: display, no-feedback transcript',
          'D − B: display, feedback transcript', 'D − A: combined comparison',
          'Interaction: (D − B) − (C − A)']
CAPTION = ('Equal-weight source-family means; retained question/model/cell observations stay clustered within family.\n'
           'Unadjusted 95% percentile family-bootstrap intervals; exploratory, no multiplicity adjustment.\n'
           'Intervals describe family variation, not repeated-generation uncertainty. B−A/D−A also change trajectories;\n'
           'C−A/D−B compare score display with no display, without a matched neutral-display control.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name):
    require(type(value) in (int, float) and math.isfinite(value), 'nonfinite/non-numeric ' + name)
    return value


def integer(value, name):
    require(type(value) is int and value >= 0, 'invalid count ' + name)
    return value


def close(a, b, name):
    require(math.isclose(a, b, abs_tol=1e-10, rel_tol=1e-9), 'inconsistent ' + name)


def interval(row, limit):
    bounds = row['family_bootstrap_95_ci']
    require(isinstance(bounds, list) and len(bounds) == 2, 'interval requires two endpoints')
    lo, hi = [number(v, 'interval') for v in bounds]
    require(-limit <= lo <= hi <= limit, 'unordered/out-of-range interval')


def validate(data):
    """Validate published aggregate identities; cannot reconstruct CIs from aggregates."""
    try:
        require(data['schema'] == 'grid5_public_summary_v1', 'wrong summary schema')
        require((data['questions'], data['families']) == (166, 116), 'complete 166-question/116-family expansion required')
        require(data['matched_units_per_judge'] == 332, '332 matched question/model units required')
        require(data['bootstrap']['resamples'] == 10000, 'unexpected bootstrap resamples')
        require(data['bootstrap']['interval'] == 'unadjusted descriptive 95%', 'unexpected interval interpretation')
        require(data['accuracy'] and data['contrasts'], 'missing outcomes')
        accuracy, contrasts = {}, {}
        for rows, index, field in ((data['accuracy'], accuracy, 'cell'), (data['contrasts'], contrasts, 'contrast')):
            for row in rows:
                key = (row['judge'], row['cohort'], row['participant_model'])
                require(all(isinstance(v, str) and v for v in key), 'invalid group label')
                bucket = index.setdefault(key, {})
                require(row[field] not in bucket, 'duplicate aggregate row')
                bucket[row[field]] = row
                n, q, f = [integer(row[k], k) for k in ('episodes', 'questions', 'families')]
                require(0 < f <= q <= n, 'invalid group denominators')
                interval(row, 2 if row.get('contrast') == INTERACTION else 1)
        require(accuracy.keys() == contrasts.keys(), 'accuracy/contrast groups differ')
        for key, cells in accuracy.items():
            require(set(cells) == set('ABCD'), 'incomplete four-cell group')
            effects = contrasts[key]
            require(set(effects) == set(CONTRASTS) | {INTERACTION}, 'incomplete five-contrast group')
            counts = tuple(cells['A'][k] for k in ('episodes', 'questions', 'families'))
            for row in [*cells.values(), *effects.values()]:
                require(tuple(row[k] for k in ('episodes', 'questions', 'families')) == counts, 'unpaired group denominators')
            for row in cells.values():
                macro = number(row['family_macro_accuracy'], 'macro accuracy')
                micro = number(row['question_weighted_accuracy'], 'micro accuracy')
                require(0 <= macro <= 1 and 0 <= micro <= 1, 'accuracy outside [0,1]')
                require(row['family_bootstrap_95_ci'][0] >= 0, 'negative accuracy interval')
                close(number(row['accuracy'], 'accuracy'), macro, 'macro alias')
                correct = integer(row['correct'], 'correct')
                require(correct <= counts[0], 'correct exceeds units')
                close(micro, correct / counts[0], 'micro count ratio')
            for name, row in effects.items():
                macro = number(row['family_macro_difference'], 'macro difference')
                micro = number(row['question_weighted_difference'], 'micro difference')
                close(number(row['difference'], 'difference'), macro, 'difference alias')
                if name in CONTRASTS:
                    pos, neg = CONTRASTS[name]
                    close(macro, cells[pos]['family_macro_accuracy'] - cells[neg]['family_macro_accuracy'], name + ' macro')
                    close(micro, cells[pos]['question_weighted_accuracy'] - cells[neg]['question_weighted_accuracy'], name + ' micro')
                    fixes, harms, both, neither = [integer(row[k], k) for k in ('fixes', 'harms', 'both_correct', 'both_wrong')]
                    require(fixes + harms + both + neither == counts[0], 'paired counts do not sum to units')
                    require(fixes + both == cells[pos]['correct'] and harms + both == cells[neg]['correct'], 'paired counts disagree with cell counts')
                else:
                    for field in ('family_macro_difference', 'question_weighted_difference'):
                        close(row[field], effects['D-B'][field] - effects['C-A'][field], 'interaction ' + field)
            if key[1] in ('all', 'exposure:untouched'):
                q, f = (166, 116) if key[1] == 'all' else (35, 28)
                require(counts == (q * (2 if key[2] == 'all' else 1), q, f), 'incorrect full/untouched coverage')
        judges = sorted({key[0] for key in accuracy})
        require(len(judges) == 2, 'two complete judges required')
        for judge in judges:
            models = {key[2] for key in accuracy if key[:2] == (judge, 'all')} - {'all'}
            require(len(models) == 2, 'two participant models required')
            for cohort in {key[1] for key in accuracy if key[0] == judge}:
                require((judge, cohort, 'all') in accuracy, 'missing pooled group')
                pooled = accuracy[judge, cohort, 'all']
                present = [accuracy[judge, cohort, model] for model in models if (judge, cohort, model) in accuracy]
                require(present, 'missing model composition')
                for cell in 'ABCD':
                    require(sum(c[cell]['episodes'] for c in present) == pooled[cell]['episodes'], 'model units do not sum to pooled units')
                    require(sum(c[cell]['correct'] for c in present) == pooled[cell]['correct'], 'model correct counts do not sum')
            for cohort in ('all', 'exposure:untouched'):
                for model in {'all', *models}:
                    require((judge, cohort, model) in accuracy, 'missing full/untouched group')
        return accuracy, contrasts, judges
    except (KeyError, TypeError, ZeroDivisionError) as exc:
        raise ValueError('missing or malformed aggregate input: ' + str(exc)) from exc


def render(summary, out):
    data = json.loads(Path(summary).read_text())
    accuracy, contrasts, judges = validate(data)
    require(not Path(out).exists(), 'new output directory required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    out = Path(out)
    out.mkdir(parents=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    figures = []
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, judge in zip(axes, judges):
        cells = accuracy[judge, 'all', 'all']
        ax.set_xlim(-.5, 1.5); ax.set_ylim(1.5, -.5)
        ax.set_xticks([0, 1], ['Judge without scores', 'Judge with scores'])
        ax.set_yticks([0, 1], ['Debaters without\nfeedback', 'Debaters with\nfeedback'])
        ax.set_title(judge)
        for y, row in enumerate(('AC', 'BD')):
            for x, cell in enumerate(row):
                r = cells[cell]
                ax.text(x, y-.12, f"{cell}: {r['family_macro_accuracy']:.1%} family-macro", ha='center', fontsize=12)
                ax.text(x, y+.10, f"Micro: {r['question_weighted_accuracy']:.1%}\n{r['correct']}/{r['episodes']} question/model units", ha='center', fontsize=10)
    fig.suptitle('Judge accuracy: 166 questions / 116 source families\nDescriptive full corpus; includes 24 reused pilot questions', fontsize=15)
    fig.text(.03, .035, 'Families receive equal weight; micro percentages weight question/model units equally.\nA/C share a no-feedback transcript; B/D share a feedback transcript. Neither judge sees the story or private preparation.', fontsize=11)
    fig.subplots_adjust(left=.12, right=.98, top=.78, bottom=.20, wspace=.35)
    figures.append(('01_full_grid', fig))
    cohorts = ['all', 'exposure:untouched'] + sorted({key[1] for key in accuracy if key[1].startswith('secondary:')})
    for index, cohort in enumerate(cohorts, 2):
        fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=True)
        bounds = [v for judge in judges for r in contrasts[judge, cohort, 'all'].values() for v in [*r['family_bootstrap_95_ci'], r['family_macro_difference']]]
        low, high = min(0, min(bounds))*100, max(0, max(bounds))*100
        pad = max(3, (high-low)*.15)
        for ax, judge in zip(axes, judges):
            for y, name in enumerate([*CONTRASTS, INTERACTION]):
                r = contrasts[judge, cohort, 'all'][name]
                lo, hi = [v*100 for v in r['family_bootstrap_95_ci']]
                mean = r['family_macro_difference']*100
                # Draw interval endpoints directly: percentile CIs need not enclose the estimate.
                ax.hlines(y, lo, hi, color='#177E89', linewidth=2)
                ax.plot(mean, y, 'o', color='#177E89')
                ax.annotate(f'{mean:+.1f} [{lo:+.1f}, {hi:+.1f}]', (mean, y), xytext=(0, 12), textcoords='offset points', ha='center', fontsize=10)
            r = accuracy[judge, cohort, 'all']['A']
            ax.set_title(f"{judge}\n{r['families']} families / {r['questions']} questions / {r['episodes']} retained units")
            ax.set_yticks(range(5), LABELS); ax.set_ylim(4.6, -.6)
            ax.set_xlim(low-pad, high+pad); ax.axvline(0, color='gray', linewidth=1)
            ax.grid(axis='x', alpha=.2)
            ax.set_xlabel('Family-macro judge accuracy difference (percentage points)')
        title = {'all': 'Full corpus — descriptive paired comparisons', 'exposure:untouched': 'Untouched before expansion — 28 families / 35 questions',
                 'secondary:source_supported': 'Sensitivity: source-reviewer-supported questions',
                 'secondary:source_supported_and_eligible': 'Joint sensitivity: source support AND model-specific eligibility'}.get(cohort, cohort)
        fig.suptitle(title, fontsize=15)
        fig.text(.02, .02, CAPTION, fontsize=11)
        fig.tight_layout(rect=(0, .19, 1, .92), w_pad=3)
        figures.append((f'{index:02d}_{cohort.replace(":", "_")}', fig))
    with PdfPages(out / 'research_pack.pdf') as pdf:
        for name, fig in figures:
            pdf.savefig(fig)
            fig.savefig(out / (name + '.png'), dpi=150)
            plt.close(fig)
    with (out / 'cohort_composition.csv').open('w', newline='') as stream:
        fields = ['judge', 'cohort', 'participant_model', 'families', 'questions', 'episodes']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for key in sorted(accuracy):
            writer.writerow({k: accuracy[key]['A'][k] for k in fields})
    (out / 'READ_ME.txt').write_text(CAPTION + '\n\n' +
        'Composition is reported by judge, cohort and participant model in cohort_composition.csv.\n'
        'Pooled selected cohorts average retained units within family, then families equally; model balance can differ.\n'
        'Source support is a fallible reviewer diagnostic, not certified gold or local sentence truth.\n'
        'Joint eligibility additionally requires the initial knowledge check and public answer-field role retention\n'
        'in both trajectories; feedback inherits the initial knowledge observation. Selection is descriptive and\n'
        'post-generation, and is not an eligibility-only analysis. Cohorts overlap.\n'
        'Untouched means no recorded family use before this expansion, not an independent repeated-generation study.\n'
        'This pack does not contain attempt/missingness accounting: attach the audited recovery supplement before sharing.\n'
        'Intervals cannot be independently recomputed from aggregate summaries alone.\n')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())}
    manifest = {'schema': 'grid5_research_pack_v1', 'summary_sha256': hashlib.sha256(Path(summary).read_bytes()).hexdigest(),
                'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'files': hashes}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    render(args.summary, args.out)
