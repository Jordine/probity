"""Post hoc source-control order diagnostic; never changes the frozen task."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
FINAL = ROOT / "runs/iteration3/final"
JOBS = ROOT / "runs/iteration3/coordinator/final_control_jobs_v3/jobs.jsonl"
OUT = Path(__file__).with_name("source_control_order_audit_v1.json")


def read_rows(path):
    raw = path.read_bytes()
    if len(raw) != path.stat().st_size:
        raise ValueError("incomplete file read")
    lines = raw.splitlines()
    rows = [json.loads(line) for line in lines]
    if len(rows) != len(lines):
        raise ValueError("row count changed")
    return rows


def main():
    jobs = {job["job_id"]: job for job in read_rows(JOBS)}
    if len(jobs) != 96:
        raise ValueError("expected all 96 frozen source-control jobs")
    by_model = defaultdict(list)
    inputs = [JOBS]
    for dirname in ("controls_weak_api_v1", "controls_opus_api_v2"):
        path = FINAL / dirname / "evaluations.jsonl"
        inputs.append(path)
        for row in read_rows(path):
            by_model[row["model"]].append(row)
    path = FINAL / "controls_native_v1/evaluations.jsonl"
    inputs.append(path)
    by_model["Qwen/Qwen3.5-4B"] = read_rows(path)
    model_set = {
        "Qwen/Qwen3.5-4B", "anthropic/claude-haiku-5.5",
        "qwen/qwen3.5-27b", "qwen/qwen3.8-27b", "anthropic/claude-opus-5.5",
    }
    if set(by_model) != model_set:
        raise ValueError("model coverage differs from completed controls")
    cells, pair_records = [], []
    for model, rows in sorted(by_model.items()):
        if len(rows) != len(jobs) or {r["job_id"] for r in rows} != set(jobs):
            raise ValueError("incomplete or duplicate control coverage")
        pairs = defaultdict(dict)
        for row in rows:
            job = jobs[row["job_id"]]
            answer = row["parsed"]["answer"]
            if not row["valid"] or row["finish_reason"] != "stop":
                raise ValueError("incomplete completion in source controls")
            if answer not in ("A", "B") or row["correct"] != (answer == job["gold"]):
                raise ValueError("source-control correctness mismatch")
            for field in ("kind", "group", "parent_source_line", "swapped", "gold", "short_options"):
                if row[field] != job[field]:
                    raise ValueError("source-control metadata differs from frozen job")
            key = (job["kind"], job["group"])
            if job["swapped"] in pairs[key]:
                raise ValueError("duplicate answer order")
            pairs[key][job["swapped"]] = row
        records = []
        for (kind, family), order in sorted(pairs.items()):
            if set(order) != {False, True}:
                raise ValueError("both answer orders required")
            original, swapped = order[False], order[True]
            left, right = original["parsed"]["answer"], swapped["parsed"]["answer"]
            same_content = left != right
            correct_count = int(original["correct"]) + int(swapped["correct"])
            if same_content != (correct_count in (0, 2)):
                raise ValueError("content/order mapping is inconsistent")
            record = {
                "model": model, "kind": kind, "family": family,
                "parent_source_line": original["parent_source_line"],
                "short_options": original["short_options"],
                "original_answer": left, "swapped_answer": right,
                "same_option_content": same_content,
                "correct_in_both_orders": correct_count == 2,
                "wrong_in_both_orders": correct_count == 0,
                "correct_in_one_order": correct_count == 1,
                "original_correct": original["correct"],
                "swapped_correct": swapped["correct"],
            }
            records.append(record)
            pair_records.append(record)
        for kind in ("no_story", "full_story"):
            for subgroup in ("all", "short", "descriptive"):
                chosen = [
                    r for r in records if r["kind"] == kind and
                    (subgroup == "all" or r["short_options"] == (subgroup == "short"))
                ]
                correct_both = sum(r["correct_in_both_orders"] for r in chosen)
                wrong_both = sum(r["wrong_in_both_orders"] for r in chosen)
                one_order = sum(r["correct_in_one_order"] for r in chosen)
                cells.append({
                    "model": model, "kind": kind, "subgroup": subgroup,
                    "independent_families": len(chosen),
                    "presentations": 2 * len(chosen),
                    "correct_in_both_orders": correct_both,
                    "wrong_in_both_orders": wrong_both,
                    "correct_in_one_order": one_order,
                    "same_option_content_families": correct_both + wrong_both,
                    "same_option_content_rate": (correct_both + wrong_both) / len(chosen),
                    "original_order_correct": sum(r["original_correct"] for r in chosen),
                    "swapped_order_correct": sum(r["swapped_correct"] for r in chosen),
                    "displayed_answer_counts": dict(Counter(
                        answer for r in chosen for answer in
                        (r["original_answer"], r["swapped_answer"])
                    )),
                })
    report = {
        "schema": "judge3_source_control_order_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_status": "Post hoc descriptive diagnostic of already completed frozen source controls; no new calls and no change to final design.",
        "complete_input_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        "finish_reasons": {model: dict(Counter(r["finish_reason"] for r in rows)) for model, rows in by_model.items()},
        "cells": cells,
        "families": pair_records,
        "interpretation": [
            "Both orders are the same source family and are not independent examples.",
            "Same option content means the displayed answer letter flips when the options flip.",
            "Order-sensitive answers conflate option-position sensitivity and stochastic/provider variation; this is not an identical-prompt noise experiment.",
            "Correct in both orders is evidence of order-stable item-level success, not proof of contamination, source memory or calibrated certainty.",
            "Final judge packages independently retain identical-message neutral repeats for descriptive answer repeatability.",
        ],
    }
    if OUT.exists():
        raise ValueError("immutable output already exists")
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"path": str(OUT), "cells": cells}, indent=2))


if __name__ == "__main__":
    main()
