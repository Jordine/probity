"""Destroy only the task-owned rental at its fixed deadline, from VPS tmux."""

import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("receipt", type=Path)
    args = ap.parse_args()
    while not args.receipt.exists():
        time.sleep(5)
    receipt = json.loads(args.receipt.read_text())
    instance_id = receipt["instance_id"]
    if receipt["label"] not in {
        "probity-qwen-pilot-20261014",
        "probity-iteration2-20261014",
        "probity-iteration3-20261014",
        "probity-iteration4-20261016",
    } or instance_id in {55765816, 55942119, 55949393}:
        raise RuntimeError("refusing a rental not owned by this pilot")
    print(json.dumps(receipt), flush=True)
    while time.time() < receipt["destroy_at_unix"]:
        time.sleep(max(0, min(30, receipt["destroy_at_unix"] - time.time())))
    for attempt in range(10):
        try:
            result = subprocess.run(
                ["vastai", "destroy", "instance", str(instance_id), "--yes", "--raw"],
                capture_output=True,
                text=True,
                timeout=45,
            )
            log = {
                "attempt": attempt,
                "time": time.time(),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired:
            log = {
                "attempt": attempt,
                "time": time.time(),
                "returncode": -1,
                "error": "destruction request timed out; retrying",
            }
        print(json.dumps(log), flush=True)
        with args.receipt.with_suffix(".cleanup.jsonl").open("a") as f:
            f.write(json.dumps(log) + "\n")
        if log["returncode"] == 0:
            return
        time.sleep(30)
    raise RuntimeError("rental destruction failed; manual cleanup required")


if __name__ == "__main__":
    main()
