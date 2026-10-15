#!/usr/bin/env bash
set -euo pipefail
cd /root/projects/probity-benchmark
export PROBITY_PILOT_LEDGER=/root/projects/probity-benchmark/runs/pilot_budget.json
python3 - <<'CHECK'
import hashlib,json
from pathlib import Path
base=Path('runs/iteration3/final')
amendment=Path('runs/iteration3/environment/final_context_alias_amendment_v1.json')
for name,digest in json.loads(amendment.read_text())['input_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
context=json.loads((base/'context_alias_v1/context_receipt_audit.json').read_text())
assert context['status']=='complete_context_receipt_audit_with_six_derived_key_aliases'
assert context['complete_contexts']==96 and context['strict_schema_accepted']==90 and context['derived_alias_interpretations']==6
for name,digest in context['input_hashes'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
assert (base/'context_receipt_audit_v1.json').read_bytes()==(base/'context_alias_v1/context_receipt_audit.json').read_bytes()
public=json.loads((base/'complete_input_receipt_audit_v1.json').read_text())
assert public['episodes']==48 and public['turns']==288 and public['all_source_plan_history_and_output_bytes_match']
for name,digest in public['file_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
for name in ['prose_audit_v1','prose_audit_repair_v1']:
    assert not (base/name).exists(),'immutable new output directory required'
print('Complete public provenance and authorized context-alias audit verified. Original strict90/96 and both valid wrong answers remain preserved; no judge outcome inspected.')
CHECK
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'rc=$?; echo "$rc" > runs/iteration3/final/validity_audit_v1.exit' EXIT
python3 -m probity_benchmark.audit_advocacy \
  --dataset runs/iteration3/coordinator/holdout_families_source_v5.jsonl \
  --episodes runs/iteration3/final/debates_v1 \
  --out runs/iteration3/final/prose_audit_v1 \
  --budget 0.5
python3 -m probity_benchmark.repair_advocacy \
  --dataset runs/iteration3/coordinator/holdout_families_source_v5.jsonl \
  --episodes runs/iteration3/final/debates_v1 \
  --prior runs/iteration3/final/prose_audit_v1 \
  --out runs/iteration3/final/prose_audit_repair_v1 \
  --budget 0.3
python3 - <<'CHECK'
from collections import Counter
from datetime import datetime, timezone
import hashlib, json
from pathlib import Path
from probity_benchmark.api import assert_no_exposed_truncation
base = Path('runs/iteration3/final')
receipts = []
paths = []
cost = 0
for directory in ['prose_audit_v1', 'prose_audit_repair_v1']:
    path = base / directory
    paths += [path / 'requests.jsonl', path / 'summary.json', path / 'annotations.json']
    summary = json.loads((path / 'summary.json').read_text())
    assert not summary['errors']
    cost += summary.get('accounted_cost_upper_usd', summary.get('cost_upper_usd'))
    for line in (path / 'requests.jsonl').read_text().splitlines():
        row = json.loads(line)
        assert row['valid'] and row['finish_reason'] == 'stop'
        assert_no_exposed_truncation(json.loads(row['response_body']))
        receipts.append(row)
paths += [Path('probity_benchmark/audit_advocacy.py'), Path('probity_benchmark/repair_advocacy.py'), Path('probity_benchmark/judge3_context.py'), Path('probity_benchmark/api.py')]
report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'budget_workstream': 'judge_controls', 'workstream_ceiling_usd': 5, 'previous_judge_workstream_spend_usd': 0.2304779, 'separate_source_review_cost_usd': 0.0317168, 'this_audit_cost_usd': cost, 'finish_reasons': dict(Counter(r['finish_reason'] for r in receipts)), 'receipts': len(receipts), 'input_code_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, 'manual_review_pending': 'Read complete arguments for every repaired, disputed or unclear annotation; automatic role adherence is not factual truth.'}
out = base / 'validity_audit_cost_receipt_v1.json'
assert not out.exists()
out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
CHECK
