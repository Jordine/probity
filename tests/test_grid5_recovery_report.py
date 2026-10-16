import unittest
from reports.build_grid5_recovery_report import aggregate, markdown, MODELS


class RecoveryReportTests(unittest.TestCase):
    def setUp(self):
        self.job = ('private-problem', MODELS[0], 'Red')
        self.expected = [self.job, ('missing-problem', MODELS[1], 'Blue')]
        self.ledger = {'cap_usd': 45, 'entries': {}}

    def receipt(self, valid=True, reused=False, basis='provider_reported', cost=.1):
        row = {'tag': dict(zip(('problem_id', 'model', 'speaker'), self.job)),
               'request_sha256': 'hash', 'attempt': 0, 'valid': valid,
               'http_status': 200 if valid else None, 'cost_basis': basis, 'cost_usd': cost}
        if reused:
            row['reused_from'] = '/private/origin'
            row['cost_usd'] = 0
        return row

    def batch(self, name, rows, events=(), guarded=False):
        return {'name': name, 'absolute_path': '/' + name, 'receipts': rows,
                'events': list(events), 'guarded': guarded}

    def test_duplicate_reuse_not_paid_attempt_or_extra_accepted(self):
        p, _ = aggregate(self.expected, [self.batch('v1', [self.receipt()]),
            self.batch('v2', [self.receipt(reused=True)])], self.ledger)
        self.assertEqual(p['totals']['accepted_unique_new_jobs'], 1)
        self.assertEqual(p['totals']['receipted_post_attempts'], 1)
        self.assertEqual(p['accounting']['preparation_provider_reported_charges_usd'], .1)
        self.assertEqual(p['totals']['missing_accepted_new_jobs'], 1)

    def test_valid_after_retry_preserves_unresolved_liability(self):
        self.ledger['entries']['unknown'] = {'kind': 'unknown_provider_liability',
            'status': 'settled', 'amount_usd': .6, 'preparation_stage': True,
            'actual_provider_charge_known': False}
        p, _ = aggregate(self.expected, [self.batch('v1', [self.receipt(False, basis='reservation_upper_bound', cost=.05)]),
            self.batch('v2', [self.receipt()])], self.ledger)
        self.assertEqual(p['totals']['accepted_after_retry_jobs'], 1)
        self.assertEqual(p['totals']['first_attempt_accepted_jobs'], 0)
        self.assertEqual(p['accounting']['unresolved_historical_liability_entries'], 1)
        self.assertAlmostEqual(p['accounting']['preparation_receipted_accounting_plus_topups_usd'], .75)

    def test_active_envelope_not_added_to_inner_liability_or_charges(self):
        self.ledger['entries']['api:/v4'] = {'kind': 'api', 'status': 'reserved', 'amount_usd': 4.18}
        event = {'state': 'post_intent', 'request_sha256': 'hash', 'attempt': 0,
                 'tag': self.receipt()['tag'], 'full_liability_usd': 2.2}
        p, _ = aggregate(self.expected, [self.batch('v4', [], [event], True)], self.ledger)
        self.assertEqual(p['totals']['pending_post_attempts'], 1)
        self.assertEqual(p['totals']['receipted_post_attempts'], 0)
        self.assertEqual(p['accounting']['campaign_committed_ledger_usd'], 4.18)
        p, _ = aggregate(self.expected, [self.batch('v4', [self.receipt()], [event], True)], self.ledger)
        self.assertEqual(p['totals']['pending_post_attempts'], 0)
        self.assertEqual(p['accounting']['campaign_committed_ledger_usd'], 4.18)

    def test_quarantine_charge_is_retained_not_accepted(self):
        row = self.receipt(False, cost=.128880375)
        row.update(http_status=200, usage={'completion_tokens': 82150})
        p, _ = aggregate(self.expected, [self.batch('v3', [row])], self.ledger)
        self.assertEqual(p['totals']['quarantined_overlimit_attempts'], 1)
        self.assertEqual(p['totals']['accepted_unique_new_jobs'], 0)
        self.assertEqual(p['accounting']['preparation_provider_reported_charges_usd'], .128880375)

    def test_prepost_guard_error_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'POST evidence'):
            aggregate(self.expected, [self.batch('v4', [self.receipt()], guarded=True)], self.ledger)

    def test_public_projection_excludes_private_identifiers(self):
        import json
        p, private = aggregate(self.expected, [self.batch('private-batch', [self.receipt()])], self.ledger)
        rendered = json.dumps(p)
        for secret in ('private-problem', 'missing-problem', 'private-batch', '/private/origin', 'hash'):
            self.assertNotIn(secret, rendered)
        self.assertIn('private-problem', json.dumps(private))

    def test_no_completion_without_verified_release_evidence(self):
        import json
        import tempfile
        from pathlib import Path
        from reports.build_grid5_recovery_report import build
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            coordinator = root / 'runs/iteration5/coordinator'
            coordinator.mkdir(parents=True)
            (coordinator / 'source_inventory_v2.json').write_text(json.dumps({
                'rows': [{'problem_id': str(i), 'prior24': False} for i in range(142)]}))
            (coordinator.parent / 'budget.json').write_text(json.dumps(self.ledger))
            public, _ = build(root)
            self.assertEqual(public['status'], 'in_progress_not_final')
            self.assertEqual(public['totals']['missing_accepted_new_jobs'], 568)
            (coordinator / 'final_analysis_complete_v1.json').write_text('{}')
            with self.assertRaises((FileNotFoundError, KeyError, ValueError)):
                build(root)

    def test_multiple_fresh_accepted_jobs_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'multiple fresh accepted'):
            aggregate(self.expected, [self.batch('v1', [self.receipt()]),
                      self.batch('v2', [self.receipt()])], self.ledger)

    def fixture(self, root, complete=False):
        import json
        from reports.build_grid5_recovery_report import digest
        def write(relative, obj):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(obj))
            return digest(path.read_bytes())
        run = 'runs/iteration5/'
        inv = write(run + 'coordinator/source_inventory_v2.json', {'rows': [
            {'problem_id': str(i), 'prior24': False} for i in range(142)]})
        ledger = {'cap_usd': 45, 'entries': {}}
        if complete:
            batch = root / run / 'preparations_v1'
            batch.mkdir()
            rows = []
            for i in range(142):
                for model in MODELS:
                    for speaker in ('Red', 'Blue'):
                        r = self.receipt(cost=0)
                        r['tag'] = {'problem_id': str(i), 'model': model, 'speaker': speaker}
                        r['request_sha256'] = str(len(rows))
                        rows.append(r)
            (batch / 'requests.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            ledger['entries']['api:' + str(batch)] = {'kind': 'api', 'status': 'settled', 'amount_usd': 0}
            paths = {'analysis_sha256': run + 'analysis_v1/analysis.json',
                     'diagnostics_sha256': run + 'coordinator/secondary_diagnostics_v1.json',
                     'public_summary_sha256': 'reports/data/iteration5/grid_summary_v1.json',
                     'plot_manifest_sha256': 'reports/figures/iteration5/grid_v1/manifest.json',
                     'package_sha256': run + 'judge_grid_v1/package.json'}
            marker = {k: write(p, {'inventory_sha256': inv} if k == 'package_sha256' else {}) for k, p in paths.items()}
            write(run + 'coordinator/final_analysis_complete_v1.json', marker)
            write(run + 'coordinator/rental_settlement_v1.json', {'instance_id': 1, 'verified_absent_at_unix': 1})
            write(run + 'coordinator/rental_v1.json', {'instance_id': 1})
        write(run + 'budget.json', ledger)
        return write, ledger

    def test_event_only_batch_is_pending(self):
        import tempfile
        from pathlib import Path
        from reports.build_grid5_recovery_report import build
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write, ledger = self.fixture(root)
            batch = root / 'runs/iteration5/preparations_v4'
            event = {'state': 'post_intent', 'request_sha256': 'first', 'attempt': 0,
                     'tag': {'problem_id': '0', 'model': MODELS[0], 'speaker': 'Red'}}
            write('runs/iteration5/preparations_v4/liability_events_v3.jsonl', event)
            ledger['entries']['api:' + str(batch)] = {'kind': 'api', 'status': 'reserved', 'amount_usd': 4.18}
            write('runs/iteration5/budget.json', ledger)
            p, _ = build(root)
            self.assertEqual(p['totals']['pending_post_attempts'], 1)
            self.assertEqual(p['totals']['jobs_without_recorded_post_attempt'], 567)

    def test_completion_gate_bindings_pending_active_and_rental(self):
        import tempfile
        from pathlib import Path
        from reports.build_grid5_recovery_report import build
        for change in ('none', 'hash', 'pending', 'active', 'rental'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                write, ledger = self.fixture(root, complete=True)
                if change == 'hash':
                    write('runs/iteration5/analysis_v1/analysis.json', {'changed': True})
                elif change == 'rental':
                    write('runs/iteration5/coordinator/rental_v1.json', {'instance_id': 2})
                elif change in ('active', 'pending'):
                    batch = root / 'runs/iteration5/preparations_v4'
                    ledger['entries']['api:' + str(batch)] = {'kind': 'api', 'status': 'reserved', 'amount_usd': 4.18}
                    write('runs/iteration5/budget.json', ledger)
                    if change == 'pending':
                        write('runs/iteration5/preparations_v4/liability_events_v3.jsonl', {
                            'state': 'post_intent', 'request_sha256': 'pending', 'attempt': 0,
                            'tag': {'problem_id': '0', 'model': MODELS[0], 'speaker': 'Red'}})
                if change == 'none':
                    self.assertEqual(build(root)[0]['status'], 'complete_with_verified_release_marker')
                else:
                    with self.assertRaises(ValueError):
                        build(root)

    def test_self_consistent_underfunded_liability_fails(self):
        import tempfile
        from pathlib import Path
        from reports.build_grid5_recovery_report import build, digest
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write, ledger = self.fixture(root)
            path = root / 'runs/iteration5/preparations_v1/requests.jsonl'
            row = self.receipt(False, basis='reservation_upper_bound', cost=.05)
            row['tag']['problem_id'] = '0'
            write('runs/iteration5/preparations_v1/requests.jsonl', row)
            proof = {'receipt_path': str(path), 'receipt_line': 1,
                     'receipt_line_sha256': digest(path.read_bytes()),
                     'request_sha256': 'hash', 'additional_liability_usd': .6}
            write('runs/iteration5/coordinator/preparation_liability_policy_v3.json', {'historical_unknowns': [proof]})
            ledger['entries']['api:' + str(path.parent)] = {'kind': 'api', 'status': 'settled', 'amount_usd': .05}
            entry = {**proof, 'kind': 'unknown_provider_liability', 'status': 'settled',
                     'actual_provider_charge_known': False, 'preparation_stage': True,
                     'amount_usd': .6, 'original_recorded_allowance_usd': .05, 'total_liability_bound_usd': .65}
            ledger['entries']['liability'] = entry
            write('runs/iteration5/budget.json', ledger)
            self.assertEqual(build(root)[0]['accounting']['preparation_unknown_liability_topups_usd'], .6)
            entry.update(amount_usd=.1, total_liability_bound_usd=.15)
            write('runs/iteration5/budget.json', ledger)
            with self.assertRaisesRegex(ValueError, 'funding bound mismatch'):
                build(root)

    def test_missing_liability_and_markdown_readability(self):
        import tempfile
        import json
        from pathlib import Path
        from reports.build_grid5_recovery_report import build
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write, ledger = self.fixture(root)
            batch = root / 'runs/iteration5/preparations_v1'
            row = self.receipt(False, basis='reservation_upper_bound', cost=.05)
            row['tag']['problem_id'] = '0'
            write('runs/iteration5/preparations_v1/requests.jsonl', row)
            ledger['entries']['api:' + str(batch)] = {'kind': 'api', 'status': 'settled', 'amount_usd': .05}
            write('runs/iteration5/budget.json', ledger)
            with self.assertRaisesRegex(ValueError, 'liability evidence'):
                build(root)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.fixture(root)
            text = markdown(build(root)[0])
            self.assertIn('Preparation: provider-reported charges', text)
            self.assertIn('not declared terminal failures', text)
            self.assertNotIn('campaign_committed_ledger_usd', text)

    def test_duplicate_fresh_attempt_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'duplicate fresh'):
            aggregate(self.expected, [self.batch('v1', [self.receipt(), self.receipt()])], self.ledger)


if __name__ == '__main__':
    unittest.main()
