#!/usr/bin/env bash
set -euo pipefail
cd /workspace/probity
export PYTHONPATH=/workspace/probity:/workspace/probity-fibs/src
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python - <<'PY'
import json
from pathlib import Path
freeze = json.loads(Path('runs/iteration3/final/global_freeze_v2.json').read_text())
if freeze['status'] != 'frozen_before_fresh_outcomes':
    raise ValueError('global freeze required')
summary = json.loads(Path('runs/iteration3/final/debates_v1/summary.json').read_text())
if summary['errors'] or summary['episodes'] != 48 or summary['expected_episodes'] != 48:
    raise ValueError('complete 48-episode corpus required')
PY
for model in qwen3.5 qwen3.8; do
  short=${model//./}
  python - "$short" <<'READY'
import json,sys,time
from pathlib import Path
p=Path('runs/iteration3/coordinator')
r=json.loads((p/'rental_resume.json').read_text())
while not (p/(sys.argv[1]+'_ready.json')).exists():
    if time.time()>r['destroy_at_unix']-1800:
        raise RuntimeError('Pinned model not ready before extraction deadline')
    time.sleep(10)
READY
  python -m probity_benchmark.prepare_debate_corpus \
    --episodes runs/iteration3/final/debates_v1 \
    --participant "qwen/${model}-27b" \
    --out "runs/iteration3/final/corpus_${short}.jsonl"
  bash runs/iteration3/coordinator/gpu_job.sh python -u -m probity_benchmark.extract_iteration \
    --corpus "runs/iteration3/final/corpus_${short}.jsonl" --model "$model" \
    --out "runs/iteration3/final/features_${short}" \
    > "runs/iteration3/final/extract_${short}.log" 2>&1
  python -m probity_benchmark.score_frozen_response \
    --features "runs/iteration3/final/features_${short}" \
    --heads "runs/iteration2/fitted_response_${short}/heads.pt" \
    --episodes runs/iteration3/final/debates_v1 \
    --out "runs/iteration3/final/fitted_response_${short}"
  python -m probity_benchmark.evaluate_response_probes \
    --features "runs/iteration3/final/features_${short}" \
    --apollo-heads "runs/iteration2/fitted_apollo_${short}/heads.pt" \
    --episodes runs/iteration3/final/debates_v1 \
    --out "runs/iteration3/final/response_probes_${short}"
done
python -m probity_benchmark.response3_bundle \
  --evaluation-root runs/iteration3/final --reference-root runs/iteration2 \
  --head-root runs/iteration2 --episodes runs/iteration3/final/debates_v1 \
  --out runs/iteration3/final/response_bundles
