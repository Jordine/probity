#!/bin/bash
set -euo pipefail
cd /root/projects/probity-benchmark
export PROBITY_PILOT_LEDGER=/root/projects/probity-benchmark/runs/pilot_budget.json
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'rc=$?; echo "$rc" > runs/iteration3/final/source_ambiguity_clean114_v1.exit' EXIT
python3 -m probity_benchmark.environment3_source_review --dataset runs/iteration3/environment/fresh_validity/clean114_source_v1.jsonl --out runs/iteration3/final/source_ambiguity_clean114_v1 --budget 0.05 > runs/iteration3/final/source_ambiguity_clean114_v1.log 2>&1
