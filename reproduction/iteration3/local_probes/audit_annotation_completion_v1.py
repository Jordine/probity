"""Read-only exact-payload, unit, receipt and cost reconciliation at completion."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from probity_benchmark.api import assert_no_exposed_truncation
from probity_benchmark.local3_annotate_units import chunks, load_inputs, reconcile_unit
from probity_benchmark.local3_secondary_clauses import opening_ids

BASE = Path("runs/iteration3")
FINAL = BASE / "final"
LOCAL = BASE / "local_probes"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["sentence", "clause"])
    args = parser.parse_args()
    amendment_path = LOCAL / "annotation_allowance_amendment_v2.json"
    amendment = json.loads(amendment_path.read_text())
    spec = amendment["stages"][args.stage]
    directory = Path(spec["out"])
    out = FINAL / f"local_{args.stage}_annotation_completion_audit_v1.json"
    if out.exists():
        raise ValueError("immutable annotation completion audit required")
    recount_path = Path(amendment["payload_recount"])
    recount = json.loads(recount_path.read_text())
    workload = recount["stages"][args.stage]
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text())
    if (
        summary["errors"]
        or summary["expected_rows"] != workload["rows"]
        or summary["completed_rows"] != workload["rows"]
        or summary["expected_units"] != workload["units"]
        or summary["completed_units"] != workload["units"]
        or summary["corpus_sha256"] != digest(directory / "corpus.jsonl")
    ):
        raise ValueError("complete fixed annotation stage required before audit")
    for name, expected in amendment["input_sha256"].items():
        if digest(name) != expected:
            raise ValueError(f"reviewed annotation input changed: {name}")
    api_path = directory / "api_manifest.json"
    api = json.loads(api_path.read_text())
    if any(
        api[key] != value
        for key, value in {
            "max_tokens": 32768,
            "request_timeout_seconds": 1200,
            "budget": spec["budget_usd"],
            "reasoning_by_model": amendment["reasoning_by_model"],
        }.items()
    ):
        raise ValueError("actual annotation runtime differs from reviewed amendment")
    operational_path = directory / "operational_amendment_receipt.json"
    operational = json.loads(operational_path.read_text())
    if (
        operational["amendment_sha256"] != digest(amendment_path)
        or operational["adapter_sha256"] != amendment["adapter_sha256"]
        or operational["payload_recount_sha256"] != digest(recount_path)
        or operational["root_release_sha256"]
        != digest(operational["root_release_path"])
        or operational["stage"] != args.stage
    ):
        raise ValueError("actual operational provenance changed")
    release = json.loads(Path(operational["root_release_path"]).read_text())
    if args.stage not in release["approved_stages"] or release["status"] != "released":
        raise ValueError("annotation stage lacks matching release")
    for name, expected in release["validity_receipts_sha256"].items():
        if digest(name) != expected:
            raise ValueError("released validity receipt changed")
    requests_path = directory / "requests.jsonl"
    receipts = [json.loads(line) for line in requests_path.read_text().splitlines()]
    jobs = {(job["source_problem"], job["chunk"]): job for job in workload["jobs"]}
    expected_keys = {
        (model, problem, index)
        for model in amendment["models"]
        for problem, index in jobs
    }
    successful = {}
    for receipt in receipts:
        body = receipt["request"]
        model, tag = receipt["model"], receipt["tag"]
        key = model, tag["source_problem"], tag["chunk"]
        if key not in expected_keys or tag["stage"] != spec["request_stage"]:
            raise ValueError("unexpected annotation request identity")
        schema = body["response_format"]["json_schema"]["schema"]
        payload_hash = text_digest(
            json.dumps([body["messages"], schema], ensure_ascii=False)
        )
        if (
            payload_hash != jobs[key[1:]]["payload_sha256"]
            or receipt["request_sha256"]
            != text_digest(json.dumps(body, sort_keys=True))
            or body["max_tokens"] != 32768
            or body["reasoning"] != amendment["reasoning_by_model"][model]
            or body["model"] != model
        ):
            raise ValueError("complete request payload/runtime changed")
        if receipt["valid"]:
            if key in successful:
                raise ValueError("duplicate successful annotation job")
            if (
                receipt["finish_reason"] != "stop"
                or receipt["usage"]["completion_tokens"] >= 32768
                or receipt["http_status"] != 200
            ):
                raise ValueError("incomplete successful generation")
            assert_no_exposed_truncation(json.loads(receipt["response_body"]))
            successful[key] = receipt
    if set(successful) != expected_keys:
        raise ValueError("successful receipts do not cover the complete fixed jobs")
    rows, sources = load_inputs(
        Path(amendment["dataset"]), [Path(amendment["episodes"])], [args.stage]
    )
    if args.stage == "clause":
        episodes = [
            json.loads(path.read_text())
            for path in sorted(Path(amendment["episodes"]).glob("episode_*.json"))
        ]
        selected = opening_ids(episodes)
        rows = [row for row in rows if row["id"] in selected]
    originals = {row["id"]: row for row in rows}
    annotated_rows = [
        json.loads(line)
        for line in (directory / "corpus.jsonl").read_text().splitlines()
    ]
    annotated = {row["id"]: row for row in annotated_rows}
    if (
        len(annotated_rows) != workload["rows"]
        or len(annotated) != workload["rows"]
        or set(annotated) != set(originals)
    ):
        raise ValueError("annotation corpus does not cover exactly the original rows")
    for identity, row in annotated.items():
        if any(
            row[key] != value
            for key, value in originals[identity].items()
            if key != "spans"
        ) or len(row["annotations"]) != len(row["units"]):
            raise ValueError(
                "complete source/argument/context or unit coverage changed"
            )
    verified_units = 0
    for problem, chunk, items in chunks(rows, 48):
        pair = [successful[model, problem, chunk] for model in amendment["models"]]
        paragraphs = sources[problem].split("\n\n")
        lookups = [
            {check["index"]: check for check in receipt["parsed"]["checks"]}
            for receipt in pair
        ]
        for index, (row, unit_index, unit) in enumerate(items):
            record = annotated[row["id"]]["annotations"][unit_index]
            checks = [
                dict(
                    lookup[index],
                    evidence_paragraphs=[
                        paragraphs[j] for j in lookup[index]["story_paragraph_ids"]
                    ],
                )
                for lookup in lookups
            ]
            label, reasons = reconcile_unit(unit, checks)
            if record != {
                "row_id": row["id"],
                "unit_index": unit_index,
                "unit": unit,
                "label": label,
                "rejection_reasons": reasons,
                "checks": checks,
                "request_sha256": [receipt["request_sha256"] for receipt in pair],
            }:
                raise ValueError(
                    "complete exact-text check/evidence or frozen acceptance changed"
                )
            verified_units += 1
    if verified_units != workload["units"]:
        raise ValueError("unit verification is incomplete")
    total_cost = sum(receipt["cost_usd"] for receipt in receipts)
    if (
        not math.isclose(total_cost, summary["cost_upper_usd"], abs_tol=1e-8)
        or total_cost > spec["budget_usd"]
    ):
        raise ValueError("annotation cost summary or stage cap mismatch")
    ledger_path = Path(amendment["ledger"])
    ledger = json.loads(ledger_path.read_text())
    ledger_key = "api:" + str(directory.resolve())
    entry = ledger["entries"][ledger_key]
    if entry["status"] != "settled" or not math.isclose(
        entry["amount_usd"], total_cost, abs_tol=1e-8
    ):
        raise ValueError("annotation shared-ledger entry has not settled exactly")
    statistics = {}
    for model in amendment["models"]:
        records = [record for record in receipts if record["model"] == model]
        valid = [record for record in records if record["valid"]]
        statistics[model] = {
            "requests": len(records),
            "successful_jobs": len(valid),
            "cost_usd": sum(record["cost_usd"] for record in records),
            "finish_reasons": dict(
                Counter(record.get("finish_reason") for record in records)
            ),
            "max_duration_seconds": max(
                record["duration_seconds"] for record in records
            ),
            "completion_tokens_quantiles": dict(
                zip(
                    ["min", "p50", "p95", "max"],
                    np.quantile(
                        [record["usage"]["completion_tokens"] for record in valid],
                        [0, 0.5, 0.95, 1],
                    ).tolist(),
                )
            ),
            "max_prompt_tokens": max(
                record["usage"]["prompt_tokens"] for record in valid
            ),
        }
    paths = [
        Path(__file__),
        amendment_path,
        recount_path,
        summary_path,
        operational_path,
        api_path,
        requests_path,
        directory / "corpus.jsonl",
        directory / "manifest.json",
        Path(operational["root_release_path"]),
    ]
    result = {
        "schema": "local3_annotation_completion_audit_v1",
        "status": "complete_exact_payload_unit_evidence_and_cost_invariants_passed",
        "stage": args.stage,
        "verified_rows": len(rows),
        "verified_units": verified_units,
        "expected_successful_jobs": len(expected_keys),
        "requests_including_retries": len(receipts),
        "failed_receipts_retained": [
            {
                "model": record["model"],
                "tag": record["tag"],
                "request_sha256": record["request_sha256"],
                "error": record.get("error"),
                "cost_usd": record["cost_usd"],
            }
            for record in receipts
            if not record["valid"]
        ],
        "cost_usd": total_cost,
        "runtime": {"max_tokens": 32768, "transport_timeout_seconds": 1200},
        "models": statistics,
        "ledger_key": ledger_key,
        "ledger_entry": entry,
        "artifacts_sha256": {str(path): digest(path) for path in paths},
        "scope": "Complete immutable stage only. Every original source/context/unit and copied exact-text check is preserved, all accepted labels follow the frozen two-annotator reconciliation, all original receipt evidence is retained, and all API costs are settled in the shared ledger. No fitting, exclusions or relabeling.",
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
