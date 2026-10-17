"""Synthetic aggregate fixtures only; these are not experimental outcomes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from reports.plot_grid5_research_pack import validate, render, CONTRASTS, INTERACTION


def synthetic_fixture():
    data = {'schema': 'grid5_public_summary_v1', 'questions': 166, 'families': 116,
            'matched_units_per_judge': 332,
            'bootstrap': {'resamples': 10000, 'interval': 'unadjusted descriptive 95%'},
            'accuracy': [], 'contrasts': []}
    for judge in ('SYNTHETIC judge 1', 'SYNTHETIC judge 2'):
        for cohort, questions, families in (('all', 166, 116), ('exposure:untouched', 35, 28)):
            for model in ('all', 'SYNTHETIC model 1', 'SYNTHETIC model 2'):
                units = questions * (2 if model == 'all' else 1)
                base = dict(judge=judge, cohort=cohort, participant_model=model,
                            questions=questions, families=families, episodes=units)
                for cell in 'ABCD':
                    # Nonzero macro/micro discrepancy is valid for unequal family sizes.
                    correct = (10 if model != 'all' else 20) if cell in 'CD' else 0
                    macro = .2 if cell in 'CD' else 0
                    data['accuracy'].append(dict(base, cell=cell, correct=correct,
                        accuracy=macro, family_macro_accuracy=macro,
                        question_weighted_accuracy=correct/units,
                        family_bootstrap_95_ci=[max(0, macro-.1), macro+.1]))
                for name, (pos, neg) in CONTRASTS.items():
                    gain = pos in 'CD' and neg not in 'CD'
                    fixes = (10 if model != 'all' else 20) if gain else 0
                    data['contrasts'].append(dict(base, contrast=name,
                        difference=.2 if gain else 0, family_macro_difference=.2 if gain else 0,
                        question_weighted_difference=fixes/units,
                        family_bootstrap_95_ci=[.1, .3] if gain else [-.1, .1],
                        fixes=fixes, harms=0, both_correct=0, both_wrong=units-fixes))
                data['contrasts'].append(dict(base, contrast=INTERACTION, difference=0,
                    family_macro_difference=0, question_weighted_difference=0,
                    family_bootstrap_95_ci=[-.1, .1]))
    return data


class ResearchPackTests(unittest.TestCase):
    def test_valid_macro_micro_differ(self):
        accuracy, _, judges = validate(synthetic_fixture())
        self.assertEqual(len(judges), 2)
        row = accuracy[judges[0], 'all', 'all']['C']
        self.assertNotEqual(row['family_macro_accuracy'], row['question_weighted_accuracy'])

    def test_rejects_inconsistent_summaries(self):
        mutations = [
            lambda d: d['accuracy'].append(copy.deepcopy(d['accuracy'][0])),
            lambda d: d['accuracy'][0].update(family_macro_accuracy=float('nan')),
            lambda d: d['accuracy'][0].update(family_bootstrap_95_ci=[.5, .1]),
            lambda d: d['accuracy'][0].update(question_weighted_accuracy=.5),
            lambda d: d['contrasts'][0].update(family_macro_difference=.8, difference=.8),
            lambda d: d['contrasts'][4].update(family_macro_difference=.2, difference=.2),
            lambda d: d['contrasts'][1].update(fixes=0),
            lambda d: d['contrasts'].pop(),
            lambda d: d.update(questions=165),
            lambda d: d.pop('accuracy'),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                data = synthetic_fixture(); mutation(data)
                with self.assertRaises(ValueError):
                    validate(data)

    def test_missing_untouched_rejected(self):
        data = synthetic_fixture()
        for field in ('accuracy', 'contrasts'):
            data[field] = [r for r in data[field] if r['cohort'] != 'exposure:untouched']
        with self.assertRaises(ValueError):
            validate(data)

    def test_render_and_immutable_output(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_grid5_test_') as folder:
            root = Path(folder)
            summary = root / 'SYNTHETIC_ONLY.json'
            summary.write_text(json.dumps(synthetic_fixture()))
            out = root / 'SYNTHETIC_ONLY_render'
            manifest = render(summary, out)
            self.assertIn('research_pack.pdf', manifest['files'])
            self.assertEqual(len(list(out.glob('*.png'))), 3)
            self.assertIn('SYNTHETIC', (out / 'cohort_composition.csv').read_text())
            with self.assertRaises(ValueError):
                render(summary, out)

    def test_invalid_input_creates_no_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); summary = root / 'bad.json'; out = root / 'out'
            summary.write_text('{}')
            with self.assertRaises(ValueError):
                render(summary, out)
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
