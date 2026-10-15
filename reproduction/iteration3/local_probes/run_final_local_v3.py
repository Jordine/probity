"""Path-only adapter for the pre-debate source-footer correction.

All experimental modules, annotation requests, heads and references remain the
globally frozen versions. Only the manifest selection changes to handoff v3.
"""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys


BASE = Path("runs/iteration3/local_probes")
HANDOFF = BASE / "final_handoff_v3"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "verify",
            "annotate",
            "export",
            "evaluate",
            "secondary-plan",
            "secondary-annotate",
            "secondary-evaluate",
        ],
    )
    args = parser.parse_args()
    lock = json.loads((HANDOFF / "runner_lock.json").read_text())
    if digest(Path(__file__)) != lock["adapter_sha256"]:
        raise ValueError("path-only adapter differs from its versioned lock")
    if digest(HANDOFF / "manifest.json") != lock["handoff_sha256"]:
        raise ValueError("path-only handoff differs from its versioned lock")
    previous_lock = json.loads((BASE / "final_handoff_v2/runner_lock.json").read_text())
    for field in [
        "runner_sha256",
        "annotation_protocol_sha256",
        "secondary_protocol_sha256",
        "api_source_sha256",
    ]:
        if lock[field] != previous_lock[field]:
            raise ValueError("path-only amendment changed an experimental contract")
    old_freeze = json.loads(Path("runs/iteration3/final/global_freeze.json").read_text())
    for name, expected in old_freeze["source_hashes"].items():
        if Path(name).name.startswith("local3_") or Path(name).name == "api.py":
            if digest(Path(name)) != expected:
                raise ValueError(f"frozen local experiment source changed: {name}")
    for name in [
        "all_unit_score_spec.json",
        "local3_bundle_spec.json",
        "local2_bundle_spec.json",
    ]:
        if (HANDOFF / name).read_bytes() != (BASE / "final_handoff_v2" / name).read_bytes():
            raise ValueError("path-only amendment changed frozen model specifications")
    handoff = json.loads((HANDOFF / "manifest.json").read_text())
    freeze = json.loads(Path(handoff["global_freeze"]).read_text())
    if freeze["artifacts_sha256"].get(str(HANDOFF / "manifest.json")) != lock["handoff_sha256"]:
        raise ValueError("new global freeze must bind the exact versioned handoff")

    # Import before the secondary module, which binds HANDOFF at import time.
    runner = importlib.import_module("probity_benchmark.local3_final_runner")
    runner.HANDOFF = HANDOFF
    if args.stage == "verify":
        episodes, _ = runner.verify(handoff)
        print(json.dumps({"verified_episodes": len(episodes), "handoff": str(HANDOFF)}, indent=2))
        return
    if args.stage.startswith("secondary-"):
        secondary = importlib.import_module("probity_benchmark.local3_secondary_clauses")
        if secondary.HANDOFF != HANDOFF:
            raise ValueError("secondary stage did not inherit the versioned handoff")
        sys.argv = [str(Path(__file__)), args.stage.removeprefix("secondary-")]
        secondary.main()
    else:
        sys.argv = [str(Path(__file__)), args.stage]
        runner.main()


if __name__ == "__main__":
    main()
