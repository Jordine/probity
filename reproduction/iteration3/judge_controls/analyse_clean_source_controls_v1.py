"""Source-footer audit of original controls; no calls and no outcome selection."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from probity_benchmark.judge3_final import audit_completion
from probity_benchmark.judge_assistance import paired_stats


ROOT = Path(__file__).resolve().parents[3]
FINAL = ROOT / "runs/iteration3/final"
JOBS = ROOT / "runs/iteration3/coordinator/final_control_jobs_v3/jobs.jsonl"
OUT = Path(__file__).with_name("source_control_clean23_audit_v1.json")


def rows(path):
    raw = path.read_bytes()
    assert len(raw) == path.stat().st_size
    parsed = [json.loads(line) for line in raw.splitlines()]
    assert len(parsed) == len(raw.splitlines())
    return parsed


def main():
    jobs = {j["job_id"]: j for j in rows(JOBS)}
    assert len(jobs) == 96
    inputs = [JOBS]
    model_rows = defaultdict(list)
    audits = []
    for dirname in ("controls_weak_api_v1", "controls_opus_api_v2", "controls_native_v1"):
        directory = FINAL / dirname
        evaluations = rows(directory / "evaluations.jsonl")
        inputs.append(directory / "evaluations.jsonl")
        native = dirname == "controls_native_v1"
        if native:
            runtime_path = directory / "manifest.json"
            runtime = json.loads(runtime_path.read_text())
            assert runtime["jobs_sha256"] == hashlib.sha256(JOBS.read_bytes()).hexdigest()
            assert runtime["contract"]["answer_only"] is True
        else:
            runtime_path = directory / "api_manifest.json"
            runtime = json.loads(runtime_path.read_text())
            receipt_path = directory / "requests.jsonl"
            inputs.append(receipt_path)
            valid = {}
            for receipt in rows(receipt_path):
                if receipt.get("valid"):
                    key = (receipt["request_sha256"], receipt["tag"]["job_id"])
                    assert key not in valid
                    valid[key] = receipt
        inputs.append(runtime_path)
        for row in evaluations:
            job = jobs[row["job_id"]]
            model = runtime["contract"]["model"] if native else row["model"]
            assert row["valid"] and row["finish_reason"] == "stop"
            for key, value in job.items():
                if key != "messages":
                    assert row[key] == value
            assert row["correct"] == (row["parsed"]["answer"] == job["gold"])
            if native:
                assert row["completion_tokens"] < runtime["contract"]["max_new_tokens"]
                assert not any(row.get(k) for k in ("forced", "forced_final", "truncated"))
            else:
                receipt = valid[(row["request_sha256"], row["job_id"])]
                audit_completion(receipt)
                assert receipt["request"]["messages"] == job["messages"]
                assert receipt["parsed"] == row["parsed"]
                assert receipt["model"] == model
                assert receipt["tag"] == {k: v for k, v in job.items() if k != "messages"}
                assert hashlib.sha256(json.dumps(receipt["request"], sort_keys=True).encode()).hexdigest() == receipt["request_sha256"]
            model_rows[model].append(row)
        audits.append({"directory": str(directory.relative_to(ROOT)), "evaluations": len(evaluations), "finish_reasons": dict(Counter(r["finish_reason"] for r in evaluations))})
    assert len(model_rows) == 5
    cells = []
    excluded = []
    for model, observations in sorted(model_rows.items()):
        assert len(observations) == 96 and {r["job_id"] for r in observations} == set(jobs)
        chosen = [r for r in observations if r["kind"] == "full_story" and r["parent_source_line"] != 114]
        assert len(chosen) == 46 and len({r["group"] for r in chosen}) == 23
        for subgroup in ("all", "short", "descriptive"):
            subset = [r for r in chosen if subgroup == "all" or r["short_options"] == (subgroup == "short")]
            stats = paired_stats([{"group": r["group"], "difference": int(r["correct"])} for r in subset])
            cells.append({
                "model": model, "kind": "full_story", "subgroup": subgroup,
                "presentations": len(subset), "families": stats["parent_groups"],
                "correct": sum(r["correct"] for r in subset),
                "accuracy": stats["mean_difference"],
                "family_bootstrap_95_ci": stats["parent_bootstrap_95_ci"],
            })
        excluded.append({
            "model": model,
            "parent_source_line": 114,
            "complete_original_full_story_rows": [r for r in observations if r["kind"] == "full_story" and r["parent_source_line"] == 114],
        })
    for filename in ("source_metadata_cleanup_v1.json", "final_source_lock_v3.json"):
        inputs.append(ROOT / "runs/iteration3/coordinator" / filename)
    report = {
        "schema": "judge3_source_control_clean23_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "Include full-story evaluations for all original final families except parent 114, whose supplied source contained an appended literal question/options/correct-answer metadata footer. The exclusion is a post hoc source-integrity diagnostic authorized by root, not an outcome-based change to the final 24-family debate sample.",
        "original_evaluations_preserved": True,
        "no_story_controls": "All original no-story prompts are unaffected; keep their original 24-family results.",
        "clean_parent114_followup": "Pending root release and generation; do not combine unrun followup outcomes with this 23-family report.",
        "input_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "complete_receipt_audits": audits,
        "cells": cells,
        "excluded_original_observations": excluded,
        "inference": "Existing paired_stats procedure: 10,000 family bootstrap draws, seed 2026101422; both option orders remain together. These controls estimate source-reading accuracy, not probe assistance uplift. A perfect finite sample gives a degenerate bootstrap interval, not perfect population accuracy.",
    }
    if OUT.exists():
        raise ValueError("immutable output already exists")
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"path": str(OUT), "cells": cells, "audits": audits}, indent=2))


if __name__ == "__main__":
    main()
