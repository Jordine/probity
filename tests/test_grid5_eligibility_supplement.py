"""Synthetic offline selection/provenance tests; never experimental outcomes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from reports import grid5_eligibility_supplement as supplement


def fixture(full=False):
    question_count = 166 if full else 4
    diagnostics = {'schema': 'grid5_secondary_diagnostics_v1', 'outcome_blind_to_judges': True, 'rows': []}
    for q in range(question_count):
        diagnostics['rows'].append({'matched_question_id': f'q{q}',
            'source_status': 'contradicted' if q == 0 else 'supported',
            'eligibility_by_model': {'m1': True, 'm2': False if q == 0 else None if q == 1 else True}})
    rows = []
    for judge in ('judge1', 'judge2'):
        for q in range(question_count):
            for model in ('m1', 'm2'):
                rows.append(dict(judge=judge, matched_question_id=f'q{q}', participant_model=model,
                    family=f'f{q % 116}' if full else 'f1' if q < 3 else 'f2', no_feedback_episode_id=f'q{q}-{model}-no',
                    feedback_episode_id=f'q{q}-{model}-yes', no_feedback_knowledge_correct=True,
                    feedback_knowledge_correct=True, no_feedback_role_switches=[],
                    feedback_role_switches=[{'turn': 1}] if q == 0 and model == 'm2' else [],
                    secondary_diagnostics=copy.deepcopy(diagnostics['rows'][q]),
                    A=q < 3, B=True, C=True, D=True))
    analysis = dict(schema='grid5_corpus_evaluation_v1', questions=question_count, families=116 if full else 2, matched_units_per_judge=question_count * 2,
                    bootstrap={'resamples': 10000, 'seed': 2026101422, 'interval': 'unadjusted descriptive 95%'},
                    accuracy=[], contrasts=[])
    for judge in ('judge1', 'judge2'):
        all_rows = [r for r in rows if r['judge'] == judge]
        supported = [r for r in all_rows if r['secondary_diagnostics']['source_status'] == 'supported']
        for cohort, selected in [('all', all_rows), ('secondary:source_supported', supported),
                ('secondary:source_supported_and_eligible', [r for r in supported if supplement.eligible(r)])]:
            a, c = supplement.summarize(selected, judge, cohort)
            analysis['accuracy'].extend(a); analysis['contrasts'].extend(c)
    return analysis, rows, diagnostics


class EligibilitySupplementTests(unittest.TestCase):
    def test_selection_preserves_family_weighting_and_models(self):
        result = supplement.calculate(*fixture())
        pooled = next(r for r in result['accuracy'] if r['judge'] == 'judge1' and r['participant_model'] == 'all' and r['cell'] == 'A')
        self.assertEqual((pooled['episodes'], pooled['questions'], pooled['families']), (6, 4, 2))
        self.assertEqual(pooled['family_macro_accuracy'], .5)
        self.assertAlmostEqual(pooled['question_weighted_accuracy'], 4/6)
        models = {r['participant_model']: r for r in result['composition'] if r['judge'] == 'judge1'}
        self.assertEqual(models['m1']['retained_units'], 4)
        self.assertEqual(models['m2']['retained_units'], 2)
        self.assertEqual(models['m2']['unknown_eligibility_units'], 1)
        # Contradicted-source q0/m1 is retained, unlike the joint support filter.
        self.assertEqual(models['m1']['questions'], 4)
        contrasts = [r for r in result['contrasts'] if r['judge'] == 'judge1' and r['participant_model'] == 'all']
        self.assertEqual(len(contrasts), 5)
        delta = next(r for r in contrasts if r['contrast'] == 'C-A')
        self.assertEqual(delta['family_macro_difference'], .5)
        self.assertAlmostEqual(delta['question_weighted_difference'], 2/6)

    def test_missing_cell_duplicate_unit_and_coverage_fail(self):
        for mutate in (lambda rows: rows[0].pop('C'), lambda rows: rows.append(copy.deepcopy(rows[0])),
                       lambda rows: rows.pop(), lambda rows: rows[0].update(no_feedback_episode_id=rows[1]['no_feedback_episode_id'])):
            analysis, rows, diagnostics = fixture(); mutate(rows)
            with self.assertRaises(ValueError):
                supplement.calculate(analysis, rows, diagnostics)

    def test_diagnostic_provenance_and_actual_gate_fail(self):
        for field, value in (('secondary_diagnostics', {}), ('feedback_role_switches', [{'turn': 2}]),
                             ('no_feedback_knowledge_correct', False)):
            analysis, rows, diagnostics = fixture(); rows[0][field] = value
            with self.assertRaises(ValueError):
                supplement.calculate(analysis, rows, diagnostics)

    def test_disagreement_with_each_existing_cohort_fails(self):
        for cohort in supplement.REFERENCE_COHORTS:
            analysis, rows, diagnostics = fixture()
            row = next(r for r in analysis['contrasts'] if r['cohort'] == cohort)
            row['family_bootstrap_95_ci'][0] += .125
            with self.assertRaises(ValueError):
                supplement.calculate(analysis, rows, diagnostics)

    def test_empty_eligible_cohort_has_counts_not_estimates(self):
        analysis, rows, diagnostics = fixture()
        for entry in diagnostics['rows']:
            entry['eligibility_by_model'] = {'m1': None, 'm2': None}
        for row in rows:
            q = int(row['matched_question_id'][1:])
            row['secondary_diagnostics'] = copy.deepcopy(diagnostics['rows'][q])
        for field in ('accuracy', 'contrasts'):
            analysis[field] = [r for r in analysis[field] if r['cohort'] != 'secondary:source_supported_and_eligible']
        result = supplement.calculate(analysis, rows, diagnostics)
        self.assertEqual(result['accuracy'], [])
        self.assertEqual(result['contrasts'], [])
        self.assertTrue(all(r['retained_units'] == 0 for r in result['composition']))

    def test_completed_build_exports_only_aggregates_and_binds_marker(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_complete_eligibility_') as directory:
            root = Path(directory)
            analysis, rows, diagnostics = fixture(full=True)
            paths = {name: root / name for name in ('analysis', 'paired', 'diagnostics', 'completion')}
            paths['diagnostics'].write_text(json.dumps(diagnostics))
            analysis.update(package_sha256='synthetic-package', secondary_diagnostics={'sha256': supplement.digest(paths['diagnostics'])})
            paths['analysis'].write_text(json.dumps(analysis))
            paths['paired'].write_text(''.join(json.dumps(r)+'\n' for r in rows))
            completion = dict(analysis_sha256=supplement.digest(paths['analysis']),
                diagnostics_sha256=supplement.digest(paths['diagnostics']), package_sha256='synthetic-package',
                plot_manifest_sha256='synthetic-plot', public_summary_sha256='synthetic-summary',
                api_stage_receipts={'synthetic': True}, campaign_spending={'unsettled_entries': 0})
            paths['completion'].write_text(json.dumps(completion))
            hashes = {k: supplement.digest(p) for k, p in paths.items()}
            result = supplement.build(*paths.values(), hashes, root/'SYNTHETIC_ONLY_out')
            exported = (root/'SYNTHETIC_ONLY_out'/'eligibility_only.json').read_text()
            self.assertNotIn('matched_question_id', exported)
            self.assertNotIn('no_feedback_episode_id', exported)
            self.assertNotIn('secondary_diagnostics', exported)
            self.assertEqual(result['input_sha256'], hashes)
            completion['analysis_sha256'] = 'incorrect'
            paths['completion'].write_text(json.dumps(completion))
            hashes['completion'] = supplement.digest(paths['completion'])
            with self.assertRaises(ValueError):
                supplement.build(*paths.values(), hashes, root/'bad_out')
            self.assertFalse((root/'bad_out').exists())

    def test_build_requires_reviewed_hashes_and_full_completed_run(self):
        with tempfile.TemporaryDirectory(prefix='SYNTHETIC_eligibility_') as directory:
            root = Path(directory)
            analysis, rows, diagnostics = fixture()
            objects = {'analysis': analysis, 'diagnostics': diagnostics, 'completion': {}}
            paths = {}
            for name in ('analysis', 'paired', 'diagnostics', 'completion'):
                paths[name] = root / name
                paths[name].write_text(''.join(json.dumps(r)+'\n' for r in rows) if name == 'paired' else json.dumps(objects[name]))
            hashes = {k: supplement.digest(p) for k, p in paths.items()}
            for reviewed in ({}, hashes):
                with self.assertRaises(ValueError):
                    supplement.build(*paths.values(), reviewed, root/'out')
            self.assertFalse((root/'out').exists())


if __name__ == '__main__':
    unittest.main()
