"""Offline, additive preparation recovery/accounting snapshot; never makes API calls."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
from pathlib import Path

MODELS = ('qwen/qwen3.5-27b', 'qwen/qwen3.8-27b')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def key(tag):
    return tag['problem_id'], tag['model'], tag['speaker']


def aggregate(expected, batches, ledger):
    """Pure aggregation. Reuse rows never count as POSTs or new charges.

    A v3 post_intent records entry into the HTTP POST wrapper. Earlier receipt
    records are evidence of attempted POSTs, including unanswered transport calls.
    Neither establishes receipt by the provider. Pending POSTs remain separate.
    """
    expected = set(expected)
    attempts = defaultdict(list)
    pending = []
    reused = 0
    batch_details = []
    for batch in batches:
        posts = {(e['request_sha256'], e['attempt']): e for e in batch['events']
                 if e.get('state') == 'post_intent'}
        seen = set()
        fresh = []
        for line, row in enumerate(batch['receipts'], 1):
            job = key(row['tag'])
            if job not in expected:
                raise ValueError('receipt outside frozen new-question job inventory')
            if row.get('reused_from'):
                reused += 1
                continue
            identity = row['request_sha256'], row['attempt']
            if identity in seen:
                raise ValueError('duplicate fresh attempt identity within batch')
            seen.add(identity)
            # Guard failures before POST must not inflate actual attempt counts.
            if batch['guarded'] and identity not in posts:
                raise ValueError('fresh receipt without guarded POST evidence')
            item = {'batch': batch['name'], 'line': line, 'request_sha256': identity[0],
                    'attempt_in_batch': identity[1], 'valid': row.get('valid') is True,
                    'cost_usd': row['cost_usd'], 'cost_basis': row['cost_basis'],
                    'transport_unanswered': row.get('http_status') is None,
                    'quarantined_overlimit': (not row.get('valid') and
                        row.get('usage', {}).get('completion_tokens') == 82150)}
            attempts[job].append(item)
            fresh.append(item)
        for identity, event in posts.items():
            if identity not in seen:
                job = key(event['tag'])
                if job not in expected:
                    raise ValueError('POST outside new-question inventory')
                pending.append({'job': job, 'batch': batch['name'],
                                'request_sha256': identity[0], 'attempt_in_batch': identity[1]})
        batch_details.append({'batch': batch['name'], 'fresh_receipts': len(fresh),
                              'recorded_receipt_cost_usd': sum(x['cost_usd'] for x in fresh)})
    if any(sum(x['valid'] for x in history) > 1 for history in attempts.values()):
        raise ValueError('multiple fresh accepted responses for one canonical job')
    by_model = {}
    for model in MODELS:
        jobs = [j for j in expected if j[1] == model]
        histories = [attempts[j] for j in jobs]
        accepted = [h for h in histories if any(x['valid'] for x in h)]
        flat = [x for h in histories for x in h]
        by_model[model] = {
            'expected_new_jobs': len(jobs),
            'jobs_without_recorded_post_attempt': sum(not attempts[j] and not any(p['job'] == j for p in pending) for j in jobs),
            'unaccepted_jobs_with_recorded_attempt': sum(not any(x['valid'] for x in attempts[j]) and (bool(attempts[j]) or any(p['job'] == j for p in pending)) for j in jobs), 'accepted_unique_new_jobs': len(accepted),
            'missing_accepted_new_jobs': len(jobs) - len(accepted),
            'first_attempt_accepted_jobs': sum(h[0]['valid'] for h in accepted),
            'accepted_after_retry_jobs': sum(not h[0]['valid'] for h in accepted),
            'jobs_with_failed_first_attempt': sum(bool(h) and not h[0]['valid'] for h in histories),
            'jobs_with_multiple_receipted_attempts': sum(len(h) > 1 for h in histories),
            'receipted_post_attempts': len(flat),
            'failed_receipted_attempts': sum(not x['valid'] for x in flat),
            'unanswered_receipted_attempts': sum(x['transport_unanswered'] for x in flat),
            'quarantined_overlimit_attempts': sum(x['quarantined_overlimit'] for x in flat),
            'pending_post_attempts': sum(x['job'][1] == model for x in pending),
            'provider_reported_charges_usd': sum(x['cost_usd'] for x in flat if x['cost_basis'] == 'provider_reported'),
            'receipt_estimates_and_allowances_usd': sum(x['cost_usd'] for x in flat if x['cost_basis'] != 'provider_reported'),
        }
    entries = ledger['entries']
    active = sum(e['amount_usd'] for e in entries.values() if e['status'] != 'settled')
    settled = sum(e['amount_usd'] for e in entries.values() if e['status'] == 'settled')
    prep_keys = {f"api:{batch['absolute_path']}" for batch in batches}
    prep_entries = [e for k, e in entries.items() if k in prep_keys]
    unknown = [e for e in entries.values() if e.get('kind') == 'unknown_provider_liability'
               and e.get('preparation_stage') and e.get('actual_provider_charge_known') is not True]
    flat = [x for h in attempts.values() for x in h]
    known = sum(x['cost_usd'] for x in flat if x['cost_basis'] == 'provider_reported')
    estimated = sum(x['cost_usd'] for x in flat if x['cost_basis'] != 'provider_reported')
    allowances = sum(e['amount_usd'] for e in unknown)
    finance = {
        'preparation_provider_reported_charges_usd': known,
        'preparation_receipt_estimates_and_original_allowances_usd': estimated,
        'preparation_unknown_liability_topups_usd': allowances,
        'preparation_catalog_estimates_usd': sum(x['cost_usd'] for x in flat if x['cost_basis'] == 'catalog_estimate'),
        'preparation_original_unknown_allowances_usd': sum(x['cost_usd'] for x in flat if x['cost_basis'] == 'reservation_upper_bound'),
        'unresolved_historical_liability_entries': len(unknown),
        'preparation_receipted_accounting_plus_topups_usd': known + estimated + allowances,
        'preparation_settled_ledger_usd': sum(e['amount_usd'] for e in prep_entries if e['status'] == 'settled') + allowances,
        'preparation_active_outer_envelope_usd': sum(e['amount_usd'] for e in prep_entries if e['status'] != 'settled'),
        'campaign_settled_ledger_usd': settled,
        'campaign_active_outer_envelopes_usd': active,
        'campaign_committed_ledger_usd': settled + active,
        'campaign_ceiling_usd': ledger['cap_usd'],
        'prior_campaign_conservative_accounting_usd': ledger.get('prior_campaign_spend_usd'),
        'interpretation': 'Active outer envelopes already contain recorded active-batch charges and inner request guard liabilities. Do not add either to ledger committed totals. Settled ledger entries can include unknown-cost allowances; successful retries do not resolve earlier unanswered charges.'}
    return {'by_model': by_model, 'totals': {k: sum(v[k] for v in by_model.values()) for k in next(iter(by_model.values()))},
            'reused_receipt_copies_ignored': reused, 'accounting': finance}, {
                'job_attempts': [{'job': j, 'attempts': attempts[j]} for j in sorted(expected)],
                'pending_posts': pending, 'batches': batch_details,
                'unresolved_liability_entries': unknown}


def build(root):
    run = root / 'runs/iteration5'
    inputs = {}
    def read(path, lines=False):
        raw = path.read_bytes()
        inputs[str(path)] = {'sha256': digest(raw), 'bytes': len(raw)}
        # Fail closed on a partially appended JSON line; caller may rerun later.
        return [json.loads(x) for x in raw.splitlines() if x] if lines else json.loads(raw)
    inventory = read(run / 'coordinator/source_inventory_v2.json')
    expected = [(r['problem_id'], m, s) for r in inventory['rows'] if not r['prior24']
                for m in MODELS for s in ('Red', 'Blue')]
    if len(expected) != 568 or len(set(expected)) != 568:
        raise ValueError('expected exactly 568 new preparation jobs')
    ledger = read(run / 'budget.json')
    batches = []
    for directory in sorted(run.glob('preparations_v[0-9]*'), key=lambda p: int(p.name.split('_v')[-1])):
        receipt_path = directory / 'requests.jsonl'
        events_path = directory / 'liability_events_v3.jsonl'
        if not receipt_path.exists() and not events_path.exists():
            continue
        events = read(events_path, True) if events_path.exists() else []
        batches.append({'name': directory.name, 'absolute_path': str(directory.resolve()),
                        'receipts': read(receipt_path, True) if receipt_path.exists() else [], 'events': events,
                        'guarded': events_path.exists()})
    # Detect moving inputs. An in-progress report is a consistent byte snapshot,
    # not a claim that the live run has stopped advancing.
    public, private = aggregate(expected, batches, ledger)
    # The inner guard is a component of the active outer envelope, never additive.
    guard_details = []
    for batch in batches:
        path = Path(batch['absolute_path']) / 'liability_guard_status_v3.json'
        if path.exists():
            guard_details.append({'batch': batch['name'], 'status': read(path)})
    public['accounting']['inner_guard_snapshot_informational_only'] = [
        {k: item['status'][k] for k in ('raw_recorded_usd', 'persistent_unknown_extra_usd',
                                      'outstanding_liability_usd', 'batch_envelope_usd')
         if k in item['status']} for item in guard_details]
    private['guard_snapshots'] = guard_details
    for batch in private['batches']:
        path = str((run / batch['batch']).resolve())
        entry = ledger['entries'].get('api:' + path)
        if entry is None:
            raise ValueError('preparation batch missing ledger envelope')
        if entry['status'] == 'settled' and abs(entry['amount_usd'] - batch['recorded_receipt_cost_usd']) > 1e-8:
            raise ValueError('settled preparation ledger differs from receipt accounting')
    unresolved = {(str(Path(e['receipt_path']).resolve()), e['receipt_line']): e
                  for e in private['unresolved_liability_entries']}
    if len(unresolved) != len(private['unresolved_liability_entries']):
        raise ValueError('duplicate unresolved liability evidence')
    policy_path = run / 'coordinator/preparation_liability_policy_v3.json'
    policy = read(policy_path) if policy_path.exists() else {}
    historical = {(str(Path(e['receipt_path']).resolve()), e['receipt_line']): e
                  for e in policy.get('historical_unknowns', [])}
    active_required_extra = defaultdict(float)
    matched_liabilities = set()
    active_unsettled_unknowns = 0
    for batch in batches:
        path = Path(batch['absolute_path']) / 'requests.jsonl'
        raw_lines = path.read_bytes().splitlines() if path.exists() else []
        for line, row in enumerate(batch['receipts'], 1):
            if row.get('reused_from') or row.get('cost_basis') != 'reservation_upper_bound':
                continue
            identity = (str(path), line)
            entry = unresolved.get(identity)
            if identity in historical:
                proof = historical[identity]
                if proof['receipt_line_sha256'] != digest(raw_lines[line - 1]) or proof['request_sha256'] != row['request_sha256']:
                    raise ValueError('historical liability policy receipt mismatch')
                required_extra = proof['additional_liability_usd']
            elif 'request' in row:
                body = row['request']
                if digest(json.dumps(body, sort_keys=True).encode()) != row['request_sha256']:
                    raise ValueError('unknown receipt request identity mismatch')
                cap = body['provider']['max_price']
                context = {'qwen/qwen3.5-27b': 262144, 'qwen/qwen3.8-27b': 1000000}[row['model']]
                prompt_bytes = len(json.dumps([body['messages'], body['response_format']['json_schema']['schema']], ensure_ascii=False).encode()) + 2048
                bound = (Decimal(prompt_bytes) * Decimal(str(cap['prompt'])) + Decimal(context) * Decimal(str(cap['completion']))) / 1000000
                required_extra = float(max(Decimal(0), bound - Decimal(str(row['cost_usd']))).quantize(Decimal('0.000000001'), rounding=ROUND_CEILING))
            else:
                raise ValueError('unanswered receipt lacks liability evidence')
            if entry is None:
                outer = ledger['entries']['api:' + batch['absolute_path']]
                guard = next((x['status'] for x in guard_details if x['batch'] == batch['name']), {})
                if outer['status'] == 'settled' or guard.get('persistent_unknown_extra_usd', 0) <= 0:
                    raise ValueError('unanswered receipt lacks retained liability evidence')
                active_required_extra[batch['name']] += required_extra
                active_unsettled_unknowns += 1
                continue
            if abs(entry['amount_usd'] - required_extra) > 1e-8:
                raise ValueError('unknown liability funding bound mismatch')
            if (entry['request_sha256'] != row['request_sha256']
                    or entry['receipt_line_sha256'] != digest(raw_lines[line - 1])
                    or abs(entry['original_recorded_allowance_usd'] - row['cost_usd']) > 1e-8
                    or abs(entry['total_liability_bound_usd'] - row['cost_usd'] - entry['amount_usd']) > 1e-8):
                raise ValueError('unknown liability does not match original receipt')
            matched_liabilities.add(identity)
    for batch_name, required_extra in active_required_extra.items():
        guard = next(x['status'] for x in guard_details if x['batch'] == batch_name)
        if abs(guard['persistent_unknown_extra_usd'] - required_extra) > 1e-8:
            raise ValueError('active unknown liability funding bound mismatch')
    if matched_liabilities != set(unresolved):
        raise ValueError('extra or duplicate unresolved liability evidence')
    public['accounting']['active_unknown_receipts_awaiting_ledger_settlement'] = active_unsettled_unknowns
    completion = run / 'coordinator/final_analysis_complete_v1.json'
    complete = False
    if completion.exists():
        c = read(completion)
        bindings = {'analysis_sha256': run / 'analysis_v1/analysis.json',
                    'diagnostics_sha256': run / 'coordinator/secondary_diagnostics_v1.json',
                    'public_summary_sha256': root / 'reports/data/iteration5/grid_summary_v1.json',
                    'plot_manifest_sha256': root / 'reports/figures/iteration5/grid_v1/manifest.json',
                    'package_sha256': run / 'judge_grid_v1/package.json'}
        for field, path in bindings.items():
            read(path)
            if inputs[str(path)]['sha256'] != c[field]:
                raise ValueError('completion evidence hash mismatch')
        package = json.loads((run / 'judge_grid_v1/package.json').read_text())
        if package['inventory_sha256'] != inputs[str(run / 'coordinator/source_inventory_v2.json')]['sha256']:
            raise ValueError('completed package inventory binding mismatch')
        settlement = read(run / 'coordinator/rental_settlement_v1.json')
        rental = read(run / 'coordinator/rental_v1.json')
        complete = (public['totals']['missing_accepted_new_jobs'] == 0
                    and public['totals']['pending_post_attempts'] == 0
                    and public['accounting']['campaign_active_outer_envelopes_usd'] == 0
                    and bool(settlement.get('verified_absent_at_unix'))
                    and settlement['instance_id'] == rental['instance_id'])
        if not complete:
            raise ValueError('completion marker conflicts with preparation/accounting/rental evidence')
    for path, info in inputs.items():
        if digest(Path(path).read_bytes()) != info['sha256']:
            raise ValueError('input changed during snapshot; rerun into fresh output')
    public.update(schema='grid5_public_recovery_report_v1',
                  as_of_utc=datetime.now(timezone.utc).isoformat(),
                  status='complete_with_verified_release_marker' if complete else 'in_progress_not_final',
                  scope='568 NEW private preparation jobs; excludes the 96 old pilot preparation jobs. Reused receipt copies are not attempts, samples or new charges.',
                  attempt_definition='Receipted POST attempts plus separately reported pending POST-wrapper entries. Transport failure does not prove provider receipt. Accepted counts are unique canonical problem/model/speaker jobs.',
                  recovery_batches_observed=max(0, len(batches)-1))
    private.update(schema='grid5_private_recovery_audit_v1', public_summary=public, inputs=inputs,
                   builder_sha256=digest(Path(__file__).read_bytes()))
    return public, private


def markdown(report):
    t, f = report['totals'], report['accounting']
    status = 'Complete with verified release evidence' if report['status'].startswith('complete_') else 'In progress; not a final result'
    lines = ['# Preparation recovery and accounting', '',
             f"**{status}.** Snapshot: {report['as_of_utc']}.", '', report['scope'], '',
             '| Model | New jobs | Accepted | Awaiting accepted result | First attempt accepted | Accepted after retry |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for model, row in report['by_model'].items():
        fields = ['expected_new_jobs', 'accepted_unique_new_jobs', 'missing_accepted_new_jobs',
                  'first_attempt_accepted_jobs', 'accepted_after_retry_jobs']
        lines.append('| ' + model + ' | ' + ' | '.join(str(row[x]) for x in fields) + ' |')
    lines.extend(['', f"{t['jobs_with_failed_first_attempt']} jobs had a failed first receipted attempt; "
                  f"{t['accepted_after_retry_jobs']} subsequently obtained an accepted result. "
                  f"There are {t['failed_receipted_attempts']} failed receipted attempts in total, including "
                  f"{t['unanswered_receipted_attempts']} unanswered attempts and "
                  f"{t['quarantined_overlimit_attempts']} quarantined over-limit responses.", '',
                  f"{t['receipted_post_attempts']} POST attempts have receipts; "
                  f"{t['pending_post_attempts']} further POST-wrapper entries have no receipt yet. "
                  f"Of the {t['missing_accepted_new_jobs']} jobs without an accepted result, "
                  f"{t.get('jobs_without_recorded_post_attempt', 'unknown')} have no recorded POST attempt and "
                  f"{t.get('unaccepted_jobs_with_recorded_attempt', 'unknown')} have a recorded attempt. "
                  "These are snapshot coverage gaps, not declared terminal failures. Pending requests and not-yet-started jobs must not be counted as failed completed observations.", '',
                  report['attempt_definition'], '', '## Financial accounting', '',
                  'USD amounts below describe different scopes and overlap. Provider-reported charges are distinguished from estimates and retained unknown-charge allowances.', '',
                  '| Scope | Amount (USD) | Interpretation |', '| --- | ---: | --- |'])
    rows = [
        ('Preparation: provider-reported charges', 'preparation_provider_reported_charges_usd', 'All fresh receipted preparation attempts, including active batches and rejected responses.'),
        ('Preparation: catalog estimates', 'preparation_catalog_estimates_usd', 'Estimated charges where no provider cost was returned.'),
        ('Preparation: original unknown-call allowances', 'preparation_original_unknown_allowances_usd', 'Original receipt allowances for unanswered calls.'),
        ('Preparation: additional unknown-liability allowances', 'preparation_unknown_liability_topups_usd', 'Extra liability retained even after a successful retry.'),
        ('Preparation: settled ledger accounting', 'preparation_settled_ledger_usd', 'Settled batch accounting plus retained liability top-ups; includes estimates, not only measured charges.'),
        ('Preparation: active outer envelope', 'preparation_active_outer_envelope_usd', 'Reservation containing active-batch charges and inner guard liabilities.'),
        ('Campaign: settled ledger accounting', 'campaign_settled_ledger_usd', 'All campaign stages already settled, including preparation allowances.'),
        ('Campaign: active outer envelopes', 'campaign_active_outer_envelopes_usd', 'All active stage reservations, including the preparation envelope.'),
        ('Campaign: total committed', 'campaign_committed_ledger_usd', 'Campaign settled accounting + campaign active envelopes.'),
        ('Campaign: ceiling', 'campaign_ceiling_usd', 'Authorized campaign accounting limit.')]
    for label, field, meaning in rows:
        lines.append(f"| {label} | ${f[field]:.6f} | {meaning} |")
    lines.extend(['', '**Do not add these rows together.** Preparation is part of the campaign. '
                  'Active-batch recorded charges and per-request inner guards are already inside the active outer envelope. '
                  'The total committed is therefore campaign settled accounting plus campaign active envelopes, '
                  'not that total plus recorded active charges or guard values.', '',
                  f"{f['unresolved_historical_liability_entries']} unresolved historical liability entries remain. "
                  'A settled ledger status means the allowance has been recorded; it does not establish the provider’s actual charge. '
                  'Successful retries do not resolve earlier unanswered charges. Raw evidence and per-job identities remain private.', ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--out', type=Path, required=True, help='New local directory; does not publish')
    args = parser.parse_args()
    public, private = build(args.root.resolve())
    args.out.mkdir(parents=True, exist_ok=False)
    for name, value in [('private_audit.json', private), ('public_summary.json', public)]:
        (args.out / name).write_text(json.dumps(value, indent=2) + '\n')
    (args.out / 'public_summary.md').write_text(markdown(public))
    print(json.dumps({'status': public['status'], 'totals': public['totals']}))


if __name__ == '__main__':
    main()
