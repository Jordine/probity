#!/bin/bash
set -euo pipefail
cd /root/projects/probity-benchmark
if [ "$#" -ne 1 ]; then
  echo 'Usage: run_final_environment_v3.sh GLOBAL_FREEZE_JSON | --verify-only' >&2
  exit 2
fi
python3 - "$1" <<'CHECK'
import hashlib,json,sys
from pathlib import Path
p=Path('runs/iteration3/environment/final_environment_lock_v2.json')
lock=json.loads(p.read_text())
for name,expected in lock['code_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected, name
source=Path(lock['source']['dataset'])
assert hashlib.sha256(source.read_bytes()).hexdigest()==lock['source']['dataset_sha256']
for name,expected in lock['amendment_inputs_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
if sys.argv[1]!='--verify-only':
    assert Path(sys.argv[1])==Path('runs/iteration3/final/global_freeze_v2.json')
    global_lock=json.loads(Path(sys.argv[1]).read_text())
    assert global_lock['status']=='frozen_before_fresh_outcomes'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==global_lock['environment_lock_sha256']
    assert global_lock['source_dataset_sha256']==lock['source']['dataset_sha256']
    for directory in lock['outputs'].values():
        assert not Path(directory).exists(), 'immutable output directory already exists: '+directory
print('Environment protocol, current code and source hashes verified; no generation performed by this check.')
CHECK
if [ "$1" = '--verify-only' ]; then
  exit 0
fi
mkdir -p runs/iteration3/final
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'rc=$?; echo "$rc" > runs/iteration3/final/environment_v3.exit' EXIT
export PROBITY_PILOT_LEDGER=/root/projects/probity-benchmark/runs/pilot_budget.json
python3 -m probity_benchmark.preparation_recovery --dataset runs/iteration3/coordinator/holdout_families_source_v5.jsonl --out runs/iteration3/final/preparations_v3 --expected-split test --models qwen/qwen3.5-27b qwen/qwen3.8-27b --max-tokens 16384 --request-timeout-seconds 1800 --reuse-receipts runs/iteration3/final/preparations_v1/requests.jsonl runs/iteration3/final/preparations_v2/requests.jsonl --budget "${PROBITY_FINAL_PREPARATION_BUDGET:-3}" > runs/iteration3/final/preparations_v3.log 2>&1
python3 - <<'CHECK'
import json
from pathlib import Path
from probity_benchmark.api import assert_no_exposed_truncation
p=Path('runs/iteration3/final/preparations_v3');s=json.loads((p/'summary.json').read_text())
assert not s['errors'] and s['completed']==s['expected']==96
rows=list(map(json.loads,(p/'requests.jsonl').read_text().splitlines()))
assert rows and all(r['valid'] and r['finish_reason']=='stop' for r in rows)
for r in rows: assert_no_exposed_truncation(json.loads(r['response_body']))
expected=json.loads(Path('runs/iteration3/coordinator/preparation_recovery_offline_audit_v1.json').read_text())['reused']
key=lambda r:json.dumps(r['tag'],sort_keys=True)
actual={key(r):r for r in rows}
assert len(expected)==22
for e in expected:
    r=actual[key(e)]
    assert r['request_sha256']==e['request_sha256'] and r.get('reused_from')==e['reused_from']
    originals=[json.loads(line) for line in Path(e['reused_from']).read_text().splitlines()]
    original=next(v for v in originals if v['valid'] and key(v)==key(e) and v['request_sha256']==e['request_sha256'])
    assert r['request']==original['request'] and r['response_body']==original['response_body']
print('All96 private preparations completed;22 canonical original responses preserved; every accepted receipt passed full-generation gates.')
CHECK
python3 -m probity_benchmark.environment3_packets --dataset runs/iteration3/coordinator/holdout_families_source_v5.jsonl --out runs/iteration3/final/debates_v1 --expected-split test --unpaired --models qwen/qwen3.5-27b qwen/qwen3.8-27b --protocol explicit_reasoned_evidence_sequential --prepared-records runs/iteration3/final/preparations_v3/preparations.jsonl --rounds 3 --judge anthropic/claude-haiku-5.5 --budget "${PROBITY_FINAL_DEBATE_BUDGET:-3.5}" > runs/iteration3/final/debates_v1.log 2>&1
python3 - <<'CHECK'
import json
from pathlib import Path
from probity_benchmark.api import assert_no_exposed_truncation
p=Path('runs/iteration3/final/debates_v1');s=json.loads((p/'summary.json').read_text())
assert not s['errors'] and s['episodes']==s['expected_episodes']==48
assert s['turns']==s['answer_field_adherent']==288 and s['knowledge_correct']==48
assert s['all_receipts_valid'] and set(s['finish_reasons'])=={'stop'}
for r in map(json.loads,(p/'requests.jsonl').read_text().splitlines()): assert_no_exposed_truncation(json.loads(r['response_body']))
print('All48 debates and288 turns completed; full generation and source-only knowledge gates passed. No judge outcome aggregate inspected by this gate.')
CHECK
python3 -m probity_benchmark.environment3_context_check --dataset runs/iteration3/coordinator/holdout_families_source_v5.jsonl --episodes runs/iteration3/final/debates_v1 --out runs/iteration3/final/context_checks_v1 --budget "${PROBITY_FINAL_CONTEXT_BUDGET:-0.5}" > runs/iteration3/final/context_checks_v1.log 2>&1
python3 - <<'CHECK'
import json
from pathlib import Path
from probity_benchmark.api import assert_no_exposed_truncation
p=Path('runs/iteration3/final/context_checks_v1');s=json.loads((p/'summary.json').read_text())
assert not s['errors'] and s['completed']==s['expected']==96
for r in map(json.loads,(p/'requests.jsonl').read_text().splitlines()):
    assert r['valid'] and r['finish_reason']=='stop'
    assert_no_exposed_truncation(json.loads(r['response_body']))
print('All96 complete-context diagnostics returned. Prose/source validity review and global preregistered knowledge handling must precede final judge interpretation.')
CHECK
