#!/usr/bin/env bash
set -euo pipefail
cd /root/projects/probity-benchmark
export PYTHONPATH=/root/projects/probity-benchmark
export PROBITY_PILOT_LEDGER=/root/projects/probity-benchmark/runs/pilot_budget.json
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
trap 'code=$?; printf "%s\n" "$code" > runs/iteration3/final/local_clause_lossless_recovery_v1.exit' EXIT
python3 -u runs/iteration3/local_probes/clause_lossless_recovery_v1.py run --release "${1:?root release path required}" \
  > runs/iteration3/final/local_clause_lossless_recovery_v1.log 2>&1
python3 -u runs/iteration3/local_probes/finish_clause_lossless_recovery_v1.py audit \
  > runs/iteration3/final/local_clause_annotation_completion_audit_v2.log 2>&1
python3 -u runs/iteration3/local_probes/finish_clause_lossless_recovery_v1.py evaluate \
  > runs/iteration3/final/local_clause_recovered_evaluation_v1.log 2>&1
