#!/usr/bin/env bash
# Run only after root's explicit fresh-feature-ready notice, in remote tmux.
# Separate audit from replay/export so root can review new numerical diagnostics.
set -euo pipefail
cd /workspace/probity
export PYTHONPATH=/workspace/probity:/workspace/probity-fibs/src
export HF_HOME=/workspace/hf-cache
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

python - <<'PY'
import json
from pathlib import Path
ready = json.loads(Path('runs/iteration3/coordinator/fresh_features_ready_v1.json').read_text())
if ready.get('complete') is not True or ready.get('instance_id') != 56090219:
    raise ValueError('complete fresh features on the current task rental required')
PY

case "${1:-}" in
  audit)
    python -u runs/iteration3/local_probes/restore_reference_views_v1.py
    python -u runs/iteration3/local_probes/audit_fresh_local_features_v1.py
    ;;
  replay)
    python - <<'PY'
import json
from pathlib import Path
audit = json.loads(Path('runs/iteration3/final/local_feature_completion_audit_v1.json').read_text())
if audit['status'] != 'all_complete_sequence_invariants_passed':
    raise ValueError('complete feature audit required before reference replay')
PY
    python -u -m probity_benchmark.local3_bundle \
      --spec runs/iteration3/local_probes/final_handoff_v3/local3_bundle_spec.json \
      --references-only --out runs/iteration3/local_probes/restored_local3_reference_replay_v1
    python -u -m probity_benchmark.local3_bundle \
      --spec runs/iteration3/local_probes/final_handoff_v3/local2_bundle_spec.json \
      --references-only --detector-prefix local2_frozen \
      --out runs/iteration3/local_probes/restored_local2_reference_replay_v1
    python -u runs/iteration3/local_probes/verify_restored_references_v1.py
    ;;
  export)
    python - <<'PY'
import json
from pathlib import Path
proof = json.loads(Path('runs/iteration3/local_probes/restored_reference_identity_audit_v1.json').read_text())
if proof['status'] != 'exact_frozen_reference_replay_passed':
    raise ValueError('exact frozen reference replay required before export')
PY
    python -u runs/iteration3/local_probes/run_final_local_v3.py export
    python -u runs/iteration3/local_probes/audit_final_local_export_v1.py
    ;;
  retention-preview)
    python - <<'PY'
import json
from pathlib import Path
proof = json.loads(Path('runs/iteration3/final/local_score_export_audit_v1.json').read_text())
if proof['status'] != 'all_complete_score_text_and_display_invariants_passed':
    raise ValueError('complete audited frozen export required before retention preview')
PY
    for short in qwen35 qwen38; do
      python -u -m probity_benchmark.local3_retain_cache \
        --features "runs/iteration3/final/features_${short}" \
        --out "runs/iteration3/final/retention_preview_${short}_v1" \
        --policy final --episodes runs/iteration3/final/debates_v1 \
        --export-receipt runs/iteration3/final/local_export_receipt_v1.json \
        --preview-per-stratum 5
    done
    ;;
  *)
    printf '%s\n' 'Usage: bash run_fresh_local_finish_v1.sh audit|replay|export|retention-preview' >&2
    exit 2
    ;;
esac
