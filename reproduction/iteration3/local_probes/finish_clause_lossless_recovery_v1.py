"""Audit complete descriptive recovery before unchanged frozen evaluation."""

import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import sys

LOCAL = Path("runs/iteration3/local_probes")
spec = importlib.util.spec_from_file_location(
    "clause_recovery", LOCAL / "clause_lossless_recovery_v1.py"
)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def audit():
    r = recovery
    if r.AUDIT.exists():
        raise ValueError("immutable complete recovery audit required")
    plan = r.verify_plan()
    runtime = json.loads((r.NEW / "api_manifest.json").read_text())
    if any(
        runtime[field] != value
        for field, value in {
            "budget": plan["new_cap_usd"],
            "max_tokens": 32768,
            "request_timeout_seconds": 1200,
            "reasoning_by_model": r.secondary.ANNOTATION_REASONING,
            "structured_output": True,
        }.items()
    ):
        raise ValueError("approved recovery runtime changed")
    operational_path = r.NEW / "lossless_recovery_release_receipt.json"
    operational = json.loads(operational_path.read_text())
    release_path = Path(operational["root_release_path"])
    release = json.loads(release_path.read_text())
    if (
        operational["root_release_sha256"] != r.digest(release_path)
        or operational["plan_sha256"] != r.digest(r.PLAN)
        or release["status"] != "released"
        or release["descriptive_only"] is not True
    ):
        raise ValueError("matching descriptive root release required")
    summary = json.loads((r.NEW / "summary.json").read_text())
    if summary["errors"] or any(
        summary[key] != value
        for key, value in {
            "expected_rows": 48,
            "completed_rows": 48,
            "expected_units": 771,
            "completed_units": 771,
            "corpus_sha256": r.digest(r.NEW / "corpus.jsonl"),
        }.items()
    ):
        raise ValueError("complete fixed recovered corpus required")
    originals = {
        r.key(record["model"], record["tag"]): record
        for record in (
            json.loads(line)
            for line in (r.OLD / "requests.jsonl").read_text().splitlines()
        )
        if record["valid"]
    }
    canonical = json.loads(r.CANONICAL.read_text())
    records = [
        json.loads(line) for line in (r.NEW / "requests.jsonl").read_text().splitlines()
    ]
    successful, paid, reused, canonical_count = {}, [], 0, 0
    for record in records:
        identity = r.key(record["model"], record["tag"])
        if (
            identity not in plan["expected_jobs"]
            or record["request"] != plan["expected_jobs"][identity]["request"]
            or record["request_sha256"]
            != r.api.sha256(json.dumps(record["request"], sort_keys=True))
        ):
            raise ValueError("recovered request differs from complete frozen body")
        if identity in originals:
            original = originals[identity]
            if (
                record.get("reused_from") != str(r.OLD / "requests.jsonl")
                or record["cost_usd"] != 0
                or record["original_cost_usd"] != original["cost_usd"]
            ):
                raise ValueError("successful original request was not exactly reused")
            for field in ["request", "response_body", "usage", "parsed"]:
                if record[field] != original[field]:
                    raise ValueError("original valid response changed")
            reused += 1
        elif identity == plan["canonical_reuse_job"]:
            original = canonical["original_receipt"]
            if (
                record.get("canonical_source_sha256") != r.digest(r.CANONICAL)
                or record["response_body"] != original["response_body"]
                or record["parsed"] != canonical["parsed_losslessly"]
                or record["cost_usd"] != 0
                or record["original_cost_usd"] != original["cost_usd"]
                or record.get("valid_under_original_reason_limit") is not False
            ):
                raise ValueError("canonical original content or accounting changed")
            canonical_count += 1
        elif identity == plan["missing_paid_job"]:
            if (
                "reused_from" in record
                or record["attempt"] > 1
                or record["cost_usd"] > plan["full_per_call_reservation_usd"]
            ):
                raise ValueError("missing Opus call exceeds approved recovery scope")
            paid.append(record)
        else:
            raise ValueError("unapproved recovery record")
        if record["valid"]:
            if identity in successful:
                raise ValueError("duplicate successful job")
            parsed, _ = r.validate_payload(
                record, identity == plan["canonical_reuse_job"]
            )
            if parsed != record["parsed"]:
                raise ValueError("parsed complete response changed")
            successful[identity] = record
        elif identity != plan["missing_paid_job"]:
            raise ValueError(
                "only the originally missing job may have new failed receipts"
            )
    if (
        set(successful) != set(plan["expected_jobs"])
        or reused != 48
        or canonical_count != 1
        or not 1 <= len(paid) <= 2
    ):
        raise ValueError("complete exact reuse/canonical/new-call coverage required")
    rows, sources, _, _ = r.fixed_jobs()
    annotated_list = [
        json.loads(line) for line in (r.NEW / "corpus.jsonl").read_text().splitlines()
    ]
    annotated = {row["id"]: row for row in annotated_list}
    if len(annotated_list) != 48 or set(annotated) != {row["id"] for row in rows}:
        raise ValueError("recovered response coverage changed")
    for row in rows:
        actual = annotated[row["id"]]
        if any(
            actual[name] != value for name, value in row.items() if name != "spans"
        ) or len(actual["annotations"]) != len(row["units"]):
            raise ValueError("complete original source/context/units changed")
    count = 0
    for problem, index, items in r.chunks(rows, 48):
        tag = {
            "stage": "blind_secondary_fixed_opening_clauses",
            "source_problem": problem,
            "chunk": index,
        }
        pair = [successful[r.key(model, tag)] for model in r.MODELS]
        lookups = [
            {check["index"]: check for check in receipt["parsed"]["checks"]}
            for receipt in pair
        ]
        paragraphs = sources[problem].split("\n\n")
        for item_index, (row, unit_index, unit) in enumerate(items):
            checks = [
                {
                    **lookup[item_index],
                    "evidence_paragraphs": [
                        paragraphs[i] for i in lookup[item_index]["story_paragraph_ids"]
                    ],
                }
                for lookup in lookups
            ]
            label, reasons = r.reconcile_unit(unit, checks)
            expected = {
                "row_id": row["id"],
                "unit_index": unit_index,
                "unit": unit,
                "label": label,
                "rejection_reasons": reasons,
                "checks": checks,
                "request_sha256": [receipt["request_sha256"] for receipt in pair],
            }
            if annotated[row["id"]]["annotations"][unit_index] != expected:
                raise ValueError(
                    "frozen reconciliation, full reason or source evidence changed"
                )
            count += 1
    if count != 771:
        raise ValueError("incomplete exact-unit audit")
    new_cost = sum(record["cost_usd"] for record in records)
    ledger = json.loads(Path("runs/pilot_budget.json").read_text())
    ledger_key = "api:" + str(r.NEW.resolve())
    entry = ledger["entries"][ledger_key]
    if (
        entry["status"] != "settled"
        or not math.isclose(entry["amount_usd"], new_cost, abs_tol=1e-9)
        or not math.isclose(summary["cost_upper_usd"], new_cost, abs_tol=1e-9)
        or new_cost > plan["new_cap_usd"]
    ):
        raise ValueError("recovery cost or shared-ledger settlement mismatch")
    paths = [Path(__file__), r.PLAN, r.CANONICAL, operational_path, release_path]
    paths += [
        r.NEW / name
        for name in [
            "api_manifest.json",
            "requests.jsonl",
            "manifest.json",
            "summary.json",
            "corpus.jsonl",
        ]
    ]
    result = {
        "schema": "local3_descriptive_clause_recovery_completion_audit_v1",
        "status": "all_complete_lossless_reuse_unit_evidence_cost_invariants_passed",
        "descriptive_only": True,
        "verified_rows": 48,
        "verified_units": 771,
        "successful_jobs": len(successful),
        "original_valid_jobs_reused": reused,
        "canonical_earliest_qwen_jobs": canonical_count,
        "new_paid_requests_including_retries": len(paid),
        "new_paid_finish_reasons": dict(
            Counter(record.get("finish_reason") for record in paid)
        ),
        "new_paid_cost_usd": new_cost,
        "original_stage_cost_usd": plan["original_stage_cost_usd"],
        "combined_stage_cost_usd": plan["original_stage_cost_usd"] + new_cost,
        "ledger_entry": entry,
        "ledger_key": ledger_key,
        "artifacts_sha256": {str(path): r.digest(path) for path in paths},
        "scope": "Full fixed secondary corpus only. Original failed batch remains immutable. Exactly one historical Qwen explanatory reason retains all976 characters under a descriptive-only local schema amendment; labels, evidence and copied text are unchanged. No truncated or shortened content. Only the genuinely missing Opus job was paid again. Original primary sentence evaluation unchanged.",
    }
    r.AUDIT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def evaluate():
    r = recovery
    proof = json.loads(r.AUDIT.read_text())
    if (
        proof["status"]
        != "all_complete_lossless_reuse_unit_evidence_cost_invariants_passed"
        or proof["descriptive_only"] is not True
    ):
        raise ValueError("complete descriptive recovery audit required")
    for path, expected in proof["artifacts_sha256"].items():
        if r.digest(path) != expected:
            raise ValueError("audited recovered input changed")
    r.secondary.ANNOTATIONS = r.NEW
    r.secondary.evaluate()
    descriptor_path = LOCAL / "summarize_truth_by_advocacy_v2.py"
    spec = importlib.util.spec_from_file_location(
        "clause_descriptive_companion", descriptor_path
    )
    descriptor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(descriptor)
    original = descriptor.STAGES["clause"]
    descriptor.STAGES["clause"] = (original[0], r.NEW, *original[2:])
    old_argv = sys.argv
    try:
        sys.argv = [str(descriptor_path), "--stage", "clause"]
        descriptor.main()
    finally:
        sys.argv = old_argv
    paths = [
        r.FINAL / "local_opening_clause_evaluation_v1/results.json",
        r.FINAL / "local_opening_clause_evaluation_v1/predictions_with_labels.jsonl",
        r.FINAL / "local_truth_by_advocacy_clause_v2.json",
        r.AUDIT,
        Path(__file__),
        descriptor_path,
    ]
    receipt = {
        "schema": "local3_descriptive_clause_recovery_evaluation_scope_v1",
        "descriptive_only": True,
        "frozen_evaluator_unchanged": True,
        "annotation_directory_override": str(r.NEW),
        "primary_sentence_results_unchanged": True,
        "artifacts_sha256": {str(path): r.digest(path) for path in paths},
    }
    (r.FINAL / "local_clause_recovered_evaluation_scope_v1.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["audit", "evaluate"])
    args = parser.parse_args()
    if args.stage == "audit":
        audit()
    else:
        evaluate()


if __name__ == "__main__":
    main()
