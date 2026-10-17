#!/usr/bin/env bash
set -euo pipefail
cd /root/projects/probity-benchmark
if [ "$#" -gt 1 ] || { [ "$#" -eq 1 ] && [ "$1" != '--verify-only' ]; }; then
  echo 'Usage: run_final_environment_v4.sh [--verify-only]' >&2
  exit 2
fi
python3 - "${1:-run}" <<'CHECK'
import hashlib,json,sys
from pathlib import Path
amendment=Path('runs/iteration3/environment/final_environment_schema_retry_amendment_v1.json')
a=json.loads(amendment.read_text())
for name,expected in a['input_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
freeze=Path('runs/iteration3/final/global_freeze_v2.json')
assert hashlib.sha256(freeze.read_bytes()).hexdigest()=='35ce913f3837eb295d21eb118bfdafa7d9224f2c202aec7890741918e33a522c'
global_lock=json.loads(freeze.read_text())
p=Path('runs/iteration3/environment/final_environment_lock_v2.json')
assert hashlib.sha256(p.read_bytes()).hexdigest()==global_lock['environment_lock_sha256']
lock=json.loads(p.read_text())
for name,expected in {**lock['code_sha256'],**lock['amendment_inputs_sha256']}.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
assert hashlib.sha256(Path(lock['source']['dataset']).read_bytes()).hexdigest()==global_lock['source_dataset_sha256']
if sys.argv[1]!='--verify-only':
    assert Path('runs/iteration3/final/environment_v3.exit').read_text().strip()=='1', 'Wait for frozen launcher to stop at its archived raw-validity gate.'
    for directory in [lock['outputs']['debates'],lock['outputs']['context_checks']]:
        assert not Path(directory).exists(), 'immutable output directory already exists: '+directory
print('Original freeze, source, inference modules and versioned operational gate verified.')
CHECK
if [ "${1:-run}" = '--verify-only' ]; then
  exit 0
fi
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'rc=$?; echo "$rc" > runs/iteration3/final/environment_v4.exit' EXIT
export PROBITY_PILOT_LEDGER=/root/projects/probity-benchmark/runs/pilot_budget.json
python3 -m probity_benchmark.environment3_preparation_gate \
  --global-freeze runs/iteration3/final/global_freeze_v2.json \
  --dataset runs/iteration3/coordinator/holdout_families_source_v5.jsonl \
  --preparations runs/iteration3/final/preparations_v3 \
  --out runs/iteration3/final/preparation_completion_gate_v1.json
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
