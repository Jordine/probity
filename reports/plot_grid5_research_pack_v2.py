"""Offline, additive reader pack for completed grid5 aggregate summaries.

Never imports execution code or obtains outcomes. Original renderer stays frozen.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path

CONTRASTS = {'B-A': ('B', 'A'), 'C-A': ('C', 'A'), 'D-B': ('D', 'B'), 'D-A': ('D', 'A')}
INTERACTION = '(D-B)-(C-A)'
LABELS = ['B − A\nFeedback-conditioned vs no-feedback trajectory',
          'C − A\nDisplay on same no-feedback transcript',
          'D − B\nDisplay on same feedback-conditioned transcript',
          'D − A\nCombined trajectory + display comparison',
          'Interaction\n(D − B) − (C − A)']
CELL_KEY = ('A: no feedback / no packet · B: feedback-conditioned / no packet · '
            'C: exact A transcript / packet · D: exact B transcript / packet.')
METHOD_KEY = ('Packet: frozen adapted response-mean logistic head, whole structured-response mean pooling; '
              'Qwen3.5 layer 43 / Qwen3.8 layer 35; prior-turn percentiles, speaker means, optional recommendation.')
CAPTION = (CELL_KEY + '\n' + METHOD_KEY + '\n'
           'Equal-weight source-family means. Intervals: unadjusted 95% percentile family bootstrap, 10,000 resamples.\n'
           'They describe family variation; one debate seed gives no repeated-generation uncertainty. '
           'B−A/D−A change trajectories; C−A/D−B hold public text fixed.')
JUDGE_NAMES = {'Qwen/Qwen3.5-4B': 'Native bf16 Qwen3.5-4B',
               'anthropic/claude-haiku-5.5': 'API Claude Haiku 5.5'}


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
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    for ax, judge in zip(axes, judges):
        cells = accuracy[judge, 'all', 'all']
        ax.set_xlim(-.5, 1.5); ax.set_ylim(1.5, -.5)
        ax.set_xticks([0, 1], ['Judge without scores', 'Judge with scores'])
        ax.set_yticks([0, 1], ['Debaters without\nfeedback', 'Debaters with\nfeedback'])
        ax.set_title(JUDGE_NAMES[judge], fontsize=14)
        for y, row in enumerate(('AC', 'BD')):
            for x, cell in enumerate(row):
                r = cells[cell]
                lo, hi = r['family_bootstrap_95_ci']
                ax.text(x, y-.15, f"{cell}: {r['family_macro_accuracy']:.1%}\n95% CI [{lo:.1%}, {hi:.1%}]",
                        ha='center', va='center', fontsize=12, fontweight='bold')
                ax.text(x, y+.17, f"Retained-unit weighted: {r['question_weighted_accuracy']:.1%}\n"
                                      f"{r['correct']}/{r['episodes']} question/model units",
                        ha='center', va='center', fontsize=10)
    fig.suptitle('Judge accuracy across the completed 2×2 grid\n166 questions · 116 source families · one trajectory seed', fontsize=16)
    fig.text(.025, .060, CELL_KEY, fontsize=10)
    fig.text(.025, .037, METHOD_KEY, fontsize=9.5)
    fig.text(.025, .014, 'Family-macro headline; retained-unit rate shown separately. Intervals are unadjusted 95% percentile family bootstrap, 10,000 resamples; no repeat-generation uncertainty.', fontsize=9.5)
    fig.subplots_adjust(left=.10, right=.98, top=.80, bottom=.18, wspace=.28)
    figures.append(('01_full_grid', fig))
    cohorts = ['all', 'exposure:untouched'] + sorted({key[1] for key in accuracy if key[1].startswith('secondary:')})
    for index, cohort in enumerate(cohorts, 2):
        fig, axes = plt.subplots(1, 2, figsize=(18, 10), sharex=True)
        bounds = [v for judge in judges for r in contrasts[judge, cohort, 'all'].values() for v in [*r['family_bootstrap_95_ci'], r['family_macro_difference']]]
        low, high = min(0, min(bounds))*100, max(0, max(bounds))*100
        pad = max(3, (high-low)*.15)
        for panel_index, (ax, judge) in enumerate(zip(axes, judges)):
            for y, name in enumerate([*CONTRASTS, INTERACTION]):
                r = contrasts[judge, cohort, 'all'][name]
                lo, hi = [v*100 for v in r['family_bootstrap_95_ci']]
                mean = r['family_macro_difference']*100
                # Draw interval endpoints directly: percentile CIs need not enclose the estimate.
                ax.hlines(y, lo, hi, color='#177E89', linewidth=2)
                ax.plot(mean, y, 'o', color='#177E89')
                ax.annotate(f'{mean:+.1f} [{lo:+.1f}, {hi:+.1f}]', (mean, y), xytext=(0, 12), textcoords='offset points', ha='center', fontsize=10)
            r = accuracy[judge, cohort, 'all']['A']
            ax.set_title(f"{JUDGE_NAMES[judge]}\n{r['families']} families / {r['questions']} questions / {r['episodes']} retained units")
            ax.set_yticks(range(5))
            ax.set_yticklabels(LABELS if panel_index == 0 else [], fontsize=10)
            ax.set_ylim(4.6, -.6)
            ax.set_xlim(low-pad, high+pad); ax.axvline(0, color='gray', linewidth=1)
            ax.grid(axis='x', alpha=.2)
            ax.set_xlabel('Family-macro judge accuracy difference (percentage points)')
        title = {'all': 'Full corpus — descriptive paired comparisons', 'exposure:untouched': 'Families untouched before expansion — descriptive sensitivity',
                 'secondary:source_supported': 'Sensitivity: source-reviewer-supported questions',
                 'secondary:source_supported_and_eligible': 'Joint sensitivity: source support AND model-specific eligibility'}.get(cohort, cohort)
        fig.suptitle(title, fontsize=15)
        cohort_note = {
            'secondary:source_supported':
                'Selection note: source support is a fallible reviewer diagnostic, not certified gold or local factual truth.',
            'secondary:source_supported_and_eligible':
                'Selection note: fallible source review plus post-generation, model-specific eligibility; 195 units / 98 questions because one question retains only one model.',
        }.get(cohort, '')
        footer = CAPTION + ('\n' + cohort_note if cohort_note else '')
        fig.text(.018, .014, footer, fontsize=9.2, linespacing=1.30)
        fig.subplots_adjust(left=.27, right=.98, top=.84, bottom=.24 if cohort_note else .22, wspace=.10)
        figures.append((f'{index:02d}_{cohort.replace(":", "_")}', fig))
    pdf_metadata = {
        'Title': 'Probity iteration 5 research figure pack v2',
        'Author': 'Probity benchmark',
        'CreationDate': datetime.datetime(2026, 10, 18, tzinfo=datetime.timezone.utc),
        'ModDate': datetime.datetime(2026, 10, 18, tzinfo=datetime.timezone.utc),
    }
    with PdfPages(out / 'research_pack.pdf', metadata=pdf_metadata) as pdf:
        for name, fig in figures:
            pdf.savefig(fig)
            fig.savefig(out / (name + '.png'), dpi=180, facecolor='white')
            plt.close(fig)
    with (out / 'cohort_composition.csv').open('w', newline='') as stream:
        fields = ['judge', 'cohort', 'participant_model', 'families', 'questions', 'episodes']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for key in sorted(accuracy):
            writer.writerow({k: accuracy[key]['A'][k] for k in fields})
    (out / 'READ_ME.txt').write_text(CAPTION + '\n\n' +
        'Completed corpus: 664 trajectories = 568 newly generated + 96 reused pilot; 332 matched question/model units per cell per judge; 1,328 decisions per judge.\n'
        'Composition is reported by judge, cohort and participant model in cohort_composition.csv.\n'
        'Pooled selected cohorts average retained units within family, then families equally; model balance can differ.\n'
        'The field named question_weighted_accuracy in the v1 source is a retained-unit-weighted rate; in the joint selected cohort one question retains only one model.\n'
        'Source support is a fallible reviewer diagnostic, not certified gold or local sentence truth.\n'
        'Joint eligibility additionally requires the initial knowledge check and public answer-field role retention\n'
        'in both trajectories; feedback inherits the initial knowledge observation. Selection is descriptive and\n'
        'post-generation, and is not an eligibility-only analysis. Cohorts overlap.\n'
        'Untouched means no recorded family use before this expansion, not an independent repeated-generation study.\n'
        'This pack does not contain attempt/missingness accounting: attach the audited recovery supplement before sharing.\n'
        'Intervals cannot be independently recomputed from aggregate summaries alone.\n')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir())}
    manifest = {'schema': 'grid5_research_pack_v2', 'summary_sha256': hashlib.sha256(Path(summary).read_bytes()).hexdigest(),
                'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'files': hashes}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    render(args.summary, args.out)
