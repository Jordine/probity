"""Join unchanged controls with the source-only parent-114 followup, preserving provenance."""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from probity_benchmark.judge3_final import audit_completion
from probity_benchmark.judge_assistance import paired_stats


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "runs/iteration3"


def read_rows(path):
    raw = path.read_bytes()
    assert len(raw) == path.stat().st_size
    return [json.loads(line) for line in raw.splitlines()]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row):
    return row["parent_source_line"], row["kind"], row["swapped"]


def audit_run(directory, jobs_path, inputs):
    jobs = {r["job_id"]: r for r in read_rows(jobs_path)}
    evaluations_path = directory / "evaluations.jsonl"
    observations = read_rows(evaluations_path)
    inputs.add(evaluations_path)
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text())
    inputs.add(summary_path)
    assert summary.get("completed") == len(observations)
    assert not summary.get("errors")
    native = not (directory / "requests.jsonl").exists()
    runtime_path = directory / ("manifest.json" if native else "api_manifest.json")
    runtime = json.loads(runtime_path.read_text())
    inputs.add(runtime_path)
    if native:
        assert runtime["jobs_sha256"] == file_hash(jobs_path)
        assert runtime["contract"]["answer_only"] is True
        expected_models = {runtime["contract"]["model"]}
    else:
        receipts_path = directory / "requests.jsonl"
        inputs.add(receipts_path)
        receipts = {}
        for receipt in read_rows(receipts_path):
            if receipt.get("valid"):
                identity = receipt["request_sha256"], receipt["tag"]["job_id"]
                assert identity not in receipts
                receipts[identity] = receipt
        expected_models = set(runtime["models"])
    by_model = defaultdict(dict)
    for row in observations:
        model = runtime["contract"]["model"] if native else row["model"]
        job = jobs[row["job_id"]]
        assert row["valid"] and row["finish_reason"] == "stop"
        assert row["correct"] == (row["parsed"]["answer"] == job["gold"])
        assert row["parsed"]["answer"] in {"A", "B"}
        metadata = {k: v for k, v in job.items() if k != "messages"}
        assert all(row[k] == value for k, value in metadata.items())
        if native:
            assert row["completion_tokens"] < runtime["contract"]["max_new_tokens"]
            assert not any(row.get(flag) for flag in ("forced", "forced_final", "truncated"))
        else:
            receipt = receipts[(row["request_sha256"], row["job_id"])]
            audit_completion(receipt)
            assert receipt["request"]["messages"] == job["messages"]
            assert receipt["parsed"] == row["parsed"]
            assert receipt["model"] == model and receipt["tag"] == metadata
            assert hashlib.sha256(json.dumps(receipt["request"], sort_keys=True).encode()).hexdigest() == receipt["request_sha256"]
        assert key(row) not in by_model[model]
        by_model[model][key(row)] = (row, job, str(directory.relative_to(ROOT)))
    assert set(by_model) == expected_models
    for model, records in by_model.items():
        assert len(records) == len(jobs)
        assert {value[0]["job_id"] for value in records.values()} == set(jobs)
    return by_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-native", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("new immutable report required")
    old_jobs = BASE / "coordinator/final_control_jobs_v3/jobs.jsonl"
    new_jobs = BASE / "coordinator/final_control_jobs_v4/jobs.jsonl"
    follow_jobs = BASE / "judge_controls/source114_clean_controls_v1/followup_package/jobs.jsonl"
    inputs = {old_jobs, new_jobs, follow_jobs, BASE / "final/global_freeze_v2.json", BASE / "coordinator/source_metadata_cleanup_v1.json"}
    original, followup = {}, {}
    for directory in ("controls_weak_api_v1", "controls_opus_api_v2"):
        original.update(audit_run(BASE / "final" / directory, old_jobs, inputs))
    for directory in ("controls_clean114_weak_v1", "controls_clean114_opus_v1"):
        followup.update(audit_run(BASE / "final" / directory, follow_jobs, inputs))
    if args.include_native:
        original.update(audit_run(BASE / "final/controls_native_v1", old_jobs, inputs))
        followup.update(audit_run(BASE / "final/controls_clean114_native_v1", follow_jobs, inputs))
    assert set(original) == set(followup)
    expected_jobs = {key(j): j for j in read_rows(new_jobs)}
    assert len(expected_jobs) == 96
    normalized, cells = [], []
    for model in sorted(original):
        combined = dict(original[model])
        assert set(followup[model]) == {(114, "full_story", False), (114, "full_story", True)}
        combined.update(followup[model])
        assert set(combined) == set(expected_jobs)
        model_rows = []
        for identity, (row, source_job, run) in sorted(combined.items()):
            expected = expected_jobs[identity]
            assert source_job["messages"] == expected["messages"]
            assert all(source_job[k] == v for k, v in expected.items() if k not in ("job_id", "messages"))
            normalized_row = {
                "model": model, "kind": expected["kind"], "group": expected["group"],
                "parent_source_line": expected["parent_source_line"], "swapped": expected["swapped"],
                "short_options": expected["short_options"], "gold": expected["gold"],
                "answer": row["parsed"]["answer"], "correct": row["correct"],
                "original_run": run, "original_job_id": row["job_id"],
                "source_v5_reference_job_id": expected["job_id"],
                "original_request_sha256": row["request_sha256"],
                "complete_messages_equal_to_source_v5": True,
            }
            model_rows.append(normalized_row)
            normalized.append(normalized_row)
        for kind in ("no_story", "full_story"):
            for subgroup in ("all", "short", "descriptive"):
                chosen = [r for r in model_rows if r["kind"] == kind and (subgroup == "all" or r["short_options"] == (subgroup == "short"))]
                stats = paired_stats([{"group": r["group"], "difference": int(r["correct"])} for r in chosen])
                cells.append({
                    "model": model, "kind": kind, "subgroup": subgroup,
                    "accuracy": stats["mean_difference"], "correct": sum(r["correct"] for r in chosen),
                    "episodes": len(chosen), "parent_groups": stats["parent_groups"],
                    "parent_bootstrap_95_ci": stats["parent_bootstrap_95_ci"],
                })
    report = {
        "schema": "probity_corrected_source_control_results_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete_models": sorted(original),
        "native_clean_followup_included": args.include_native,
        "dataset": "runs/iteration3/coordinator/holdout_families_source_v5.jsonl",
        "dataset_sha256": file_hash(BASE / "coordinator/holdout_families_source_v5.jsonl"),
        "complete_input_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(inputs)},
        "source_selection": "For every model and family, use the original complete source-control observation unless it is parent 114 full-story; use the root-released cleaned-source followup for both orders of that case. All 24 families retained. No response selected based on correctness.",
        "receipt_policy": "Original request bodies, tags, job IDs and responses remain immutable. The explicit analysis mapping to the v5 reference is not a rewritten receipt. Every complete message list is verified equal to its v5 reference.",
        "cells": cells,
        "observations": normalized,
        "limits": [
            "A source-only bug correction occurred after original control outcomes were observed; chronology is explicit in global_freeze_v2.",
            "Parent 114 followup uses later requests; model IDs/settings are unchanged, but catalog price filters, provider routing and stochastic variation may differ. Its old/new answer difference alone is not a causal estimate.",
            "Both answer orders share one source family. Existing 10,000-draw family bootstrap and seed are retained.",
            "These are source-reading controls, not debate-judge performance or probe-assistance uplift.",
            "Perfect finite-sample accuracy does not establish perfect population accuracy.",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "models": report["complete_models"], "observations": len(normalized), "full_story_cells": [c for c in cells if c["kind"] == "full_story" and c["subgroup"] == "all"]}, indent=2))


if __name__ == "__main__":
    main()
