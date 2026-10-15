"""Explicitly invoked final local-probe stages; no automatic paid work."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

BASE = Path("runs/iteration3/local_probes")
HANDOFF = BASE / "final_handoff_v2"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(handoff):
    freeze = Path(handoff["global_freeze"])
    if (
        not freeze.exists()
        or json.loads(freeze.read_text()).get("status") != handoff["required_status"]
    ):
        raise ValueError("root global freeze receipt is not released")
    if digest(handoff["dataset"]) != handoff["dataset_sha256"]:
        raise ValueError("final complete-source dataset changed")
    lock = json.loads((BASE / "local_artifact_lock_v1.json").read_text())
    for record in lock["artifacts"].values():
        if digest(record["path"]) != record["file_sha256"]:
            raise ValueError("frozen reference artifact changed")
    for key, name in [
        ("fresh_score_source_sha256", "local3_score_units.py"),
        ("fresh_evaluator_source_sha256", "local3_evaluate_units.py"),
        ("bundle_source_sha256", "local3_bundle.py"),
    ]:
        if digest(Path(__file__).with_name(name)) != lock[key]:
            raise ValueError("frozen scorer/evaluator/exporter source changed")
    annotation = json.loads(Path(handoff["annotation_protocol_lock"]).read_text())
    if (
        digest(Path(__file__).with_name("local3_annotate_units.py"))
        != annotation["script_sha256"]
    ):
        raise ValueError("frozen sentence annotation source changed")
    if digest(Path(__file__).with_name("api.py")) != annotation["api_source_sha256"]:
        raise ValueError("frozen annotation API runtime changed")
    for name, expected in handoff["spec_sha256"].items():
        if digest(HANDOFF / name) != expected:
            raise ValueError("final path/head specification changed")
    episode_dir = Path(handoff["episodes"])
    summary = json.loads((episode_dir / "summary.json").read_text())
    episodes = [json.loads(p.read_text()) for p in episode_dir.glob("episode_*.json")]
    if (
        summary["errors"]
        or summary["expected_episodes"] != handoff["expected_episodes"]
        or len(episodes) != handoff["expected_episodes"]
        or sum(len(e["turns"]) for e in episodes) != handoff["expected_turns"]
    ):
        raise ValueError(
            "all frozen final episodes/turns must complete before local stages"
        )
    return episodes, lock


def call(module, *args):
    subprocess.run([sys.executable, "-u", "-m", module, *map(str, args)], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["annotate", "export", "evaluate"])
    args = parser.parse_args()
    handoff = json.loads((HANDOFF / "manifest.json").read_text())
    episodes, lock = verify(handoff)
    if args.stage == "annotate":
        call(
            "probity_benchmark.local3_annotate_units",
            "--dataset",
            handoff["dataset"],
            "--episodes",
            handoff["episodes"],
            "--out",
            handoff["annotation_out"],
            "--global-freeze",
            handoff["global_freeze"],
            "--granularities",
            "sentence",
            "--budget",
            handoff["annotation_api_cap_usd"],
            "--concurrency",
            3,
        )
    elif args.stage == "export":
        call(
            "probity_benchmark.local3_score_units",
            "--spec",
            HANDOFF / "all_unit_score_spec.json",
            "--out",
            handoff["score_out"],
        )
        for prefix in ["local3", "local2"]:
            call(
                "probity_benchmark.local3_bundle",
                "--spec",
                HANDOFF / f"{prefix}_bundle_spec.json",
                "--episodes",
                handoff["episodes"],
                "--out",
                handoff[f"{prefix}_bundle_out"],
                "--detector-prefix",
                "local3" if prefix == "local3" else "local2_frozen",
            )
            for granularity in ["sentence", "clause"]:
                bundle = json.loads(
                    (
                        Path(handoff[f"{prefix}_bundle_out"]) / f"{granularity}.json"
                    ).read_text()
                )
                reference = json.loads(
                    Path(lock["artifacts"][bundle["detector"]]["path"]).read_text()
                )
                for field in ["head_sha256", "reference_sha256", "references"]:
                    if bundle[field] != reference[field]:
                        raise ValueError(
                            "final display changed frozen head/reference arrays"
                        )
                if len(bundle["rows"]) != handoff["expected_turns"]:
                    raise ValueError("final local display does not cover every turn")
        receipt = {
            "status": "all_fixed_units_exported",
            "episodes": len(episodes),
            "turns": handoff["expected_turns"],
            "scores_sha256": digest(Path(handoff["score_out"]) / "scores.jsonl"),
            "bundles": {
                f"{p}_{g}": digest(Path(handoff[f"{p}_bundle_out"]) / f"{g}.json")
                for p in ["local3", "local2"]
                for g in ["sentence", "clause"]
            },
        }
        output = Path("runs/iteration3/final/local_export_receipt_v1.json")
        if output.exists():
            raise ValueError("immutable export receipt required")
        output.write_text(json.dumps(receipt, indent=2) + "\n")
        print(json.dumps(receipt, indent=2), flush=True)
    else:
        call(
            "probity_benchmark.local3_evaluate_units",
            "--scores",
            handoff["score_out"],
            "--annotations",
            handoff["annotation_out"],
            "--family-map",
            "runs/iteration3/coordinator/source_family_map.json",
            "--out",
            handoff["evaluation_out"],
        )


if __name__ == "__main__":
    main()
