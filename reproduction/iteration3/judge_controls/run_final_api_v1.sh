#!/usr/bin/env bash
set -euo pipefail

# Run in tmux only after root releases this exact package hash and batch budget.
# No inference settings are supplied by the caller; they come from the frozen lock.
if [[ "$#" -ne 4 ]]; then
  echo 'usage: run_final_api_v1.sh haiku|opus REVIEWED_PACKAGE_SHA256 BUDGET_USD NEW_OUTPUT_DIRECTORY' >&2
  exit 2
fi
cd /root/projects/probity-benchmark
export PROBITY_PILOT_LEDGER=/root/projects/probity-benchmark/runs/pilot_budget.json
python3 - "$@" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

name, reviewed_hash, budget_text, output = sys.argv[1:]
if name not in {"haiku", "opus"}:
    raise ValueError("only frozen final Haiku and Opus runtimes are supported")
budget = float(budget_text)
ceiling = {"haiku": 2, "opus": 12}[name]
if not math.isfinite(budget) or not 0 < budget <= ceiling:
    raise ValueError("batch budget exceeds the planned workstream ceiling")
if Path(output).exists():
    raise ValueError("new immutable output directory required")
lock_path = Path("runs/iteration3/judge_controls/final_judge_code_lock_v2.json")
if hashlib.sha256(lock_path.read_bytes()).hexdigest() != "89ab74c57193b41b31c9d7f831ae52a904f4eef11b59948a0f5c9e5cf5c00c86":
    raise ValueError("frozen judge runtime lock changed")
lock = json.loads(lock_path.read_text())
for filename, expected in lock["files"].items():
    if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
        raise ValueError("frozen judge source changed: " + filename)
model = {"haiku": "anthropic/claude-haiku-5.5", "opus": "anthropic/claude-opus-5.5"}[name]
runtime = lock["api_runtimes"][model]
package = Path("runs/iteration3/final") / {"haiku": "judge_package_v1", "opus": "opus_package_v1"}[name]
if hashlib.sha256((package / "package.json").read_bytes()).hexdigest() != reviewed_hash:
    raise ValueError("reviewed release hash does not match the immutable package")
command = [
    sys.executable, "-m", "probity_benchmark.judge3_final", "run-api",
    "--package", str(package), "--frozen-package-sha256", reviewed_hash,
    "--judge", model, "--budget", budget_text,
    "--concurrency", str(runtime["concurrency"]),
    "--max-output-tokens", str(runtime["max_output_tokens"]),
    "--request-timeout-seconds", str(runtime["request_timeout_seconds"]),
    "--out", output,
]
if runtime["reasoning"].get("effort"):
    command += ["--reasoning-effort", runtime["reasoning"]["effort"]]
elif runtime["reasoning"] != {"enabled": False}:
    raise ValueError("unsupported frozen reasoning configuration")
print(json.dumps({"released_package_sha256": reviewed_hash, "command": command}, indent=2), flush=True)
subprocess.run(command, check=True)
PY
