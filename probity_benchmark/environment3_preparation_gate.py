"""Audit accepted private plans and every retained schema-retry attempt.

The frozen API permits two same-request schema retries. Failed attempts stay in
the immutable receipt archive; only a fully validated terminal result satisfies
the 96-job preparation gate. No valid answer is regenerated or selected here.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import jsonschema

from .api import assert_no_exposed_truncation, strict_json
from .environment3_prepared import preparation_messages, preparation_schema
from .preflight import load_problems, sha256


def job_key(row):
    return json.dumps(row["tag"], sort_keys=True)


def validate_attempt(row, schema):
    request = row["request"]
    if sha256(json.dumps(request, sort_keys=True)) != row["request_sha256"]:
        raise ValueError("request hash/body mismatch")
    if row["model"] != request["model"] or row.get("http_status") != 200:
        raise ValueError("model mismatch or unsuccessful HTTP response")
    if (
        type(row.get("valid")) is not bool
        or type(row.get("attempt")) is not int
        or row["attempt"] < 0
        or not isinstance(row.get("cost_usd"), (int, float))
        or not math.isfinite(row["cost_usd"])
        or row["cost_usd"] < 0
    ):
        raise ValueError("invalid attempt or charge accounting")
    payload = strict_json(row["response_body"])
    assert_no_exposed_truncation(payload)
    if len(payload["choices"]) != 1:
        raise ValueError("exactly one complete response required")
    choice = payload["choices"][0]
    usage = payload["usage"]
    if (
        row.get("finish_reason") != "stop"
        or choice.get("finish_reason") != "stop"
        or usage != row["usage"]
        or not all(
            type(usage.get(k)) is int and usage[k] >= 0
            for k in ("prompt_tokens", "completion_tokens")
        )
        or usage["completion_tokens"] >= request["max_tokens"]
    ):
        raise ValueError("incomplete generation or invalid token accounting")
    failure_type = None
    parsed = None
    try:
        parsed = strict_json(choice["message"]["content"])
        jsonschema.validate(parsed, schema)
    except (json.JSONDecodeError, jsonschema.ValidationError) as exc:
        failure_type = type(exc).__name__
    if row["valid"]:
        if failure_type or parsed != row["parsed"]:
            raise ValueError("accepted receipt fails the complete schema")
    elif (
        failure_type is None
        or row.get("error_type") != failure_type
        or failure_type not in ("JSONDecodeError", "ValidationError")
    ):
        raise ValueError("failed attempt is not an allowed schema retry")


def audit_receipts(rows, expected_tags, schema, schema_retries=2):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        validate_attempt(row, schema)
        groups[job_key(row)].append((index, row))
    expected = {json.dumps(tag, sort_keys=True) for tag in expected_tags}
    if set(groups) != expected:
        raise ValueError("missing or extra canonical preparation job")
    accepted = []
    failed = []
    for key in sorted(groups):
        attempts = groups[key]
        successes = [(i, r) for i, r in attempts if r["valid"]]
        if len(successes) != 1:
            raise ValueError("each job requires exactly one accepted result")
        accepted_index, result = successes[0]
        history = [(i, r) for i, r in attempts if not r["valid"]]
        retry_number = result.get("attempt", 0)
        if type(retry_number) is not int or not 0 <= retry_number <= schema_retries:
            raise ValueError("schema-retry count exceeds frozen policy")
        if result.get("reused_from") and retry_number and not history:
            prior = Path(result["reused_from"])
            history = [
                (-1, r)
                for r in map(json.loads, prior.read_text().splitlines())
                if job_key(r) == key
                and r["request_sha256"] == result["request_sha256"]
                and not r["valid"]
                and r.get("attempt", 0) < retry_number
            ]
            for _, row in history:
                validate_attempt(row, schema)
        if [r["attempt"] for _, r in history] != list(range(retry_number)):
            raise ValueError("missing, duplicate or unordered schema-retry history")
        for index, row in history:
            if (
                row["request"] != result["request"]
                or row["request_sha256"] != result["request_sha256"]
                or row["tag"] != result["tag"]
                or index >= accepted_index
            ):
                raise ValueError("schema retry changed request/tag or result ordering")
            failed.append(
                {
                    "tag": row["tag"],
                    "request_sha256": row["request_sha256"],
                    "attempt": row.get("attempt", 0),
                    "error_type": row["error_type"],
                    "finish_reason": row["finish_reason"],
                    "cost_usd": row["cost_usd"],
                    "receipt_index": index,
                    "external_history": index == -1,
                    "accepted_attempt": retry_number,
                }
            )
        accepted.append(result)
    return accepted, failed


def run(args):
    if args.out.exists():
        raise ValueError("immutable new audit file required")
    frozen = json.loads(args.global_freeze.read_text())
    if frozen["source_dataset_sha256"] != sha256(args.dataset.read_text()):
        raise ValueError("source differs from amended global freeze")
    problems, source_audit = load_problems(args.dataset)
    source_rows = list(map(json.loads, args.dataset.read_text().splitlines()))
    if source_audit["errors"] or len(problems) != 24:
        raise ValueError("complete 24-family source required")
    summary = json.loads((args.preparations / "summary.json").read_text())
    if summary["errors"] or summary["completed"] != summary["expected"]:
        raise ValueError("preparation summary is incomplete")
    if summary["completed"] != 96 or summary["expected"] != 96:
        raise ValueError("96 accepted private plans required")
    rows = list(
        map(json.loads, (args.preparations / "requests.jsonl").read_text().splitlines())
    )
    expected = []
    inputs = {}
    for problem, source in zip(problems, source_rows):
        for model in frozen["participants"]:
            for speaker in ("Red", "Blue"):
                tag = dict(
                    problem_id=problem.problem_id,
                    model=model,
                    speaker=speaker,
                    stage="reasoned_preparation",
                )
                expected.append(tag)
                inputs[json.dumps(tag, sort_keys=True)] = (
                    problem,
                    bool(source["schedule_parent_index"] % 2),
                )
    accepted, failed = audit_receipts(rows, expected, preparation_schema())
    if not math.isclose(
        sum(r["cost_usd"] for r in rows), summary["cost_usd"], abs_tol=1e-12
    ):
        raise ValueError("raw attempt charges do not equal settled preparation cost")
    recovery_audit = Path(
        "runs/iteration3/coordinator/preparation_recovery_offline_audit_v1.json"
    )
    canonical = json.loads(recovery_audit.read_text())["reused"]
    by_key = {job_key(r): r for r in accepted}
    if len(canonical) != 22 or sum(bool(r.get("reused_from")) for r in accepted) != 22:
        raise ValueError("canonical 22 original responses were not preserved")
    for expected_reuse in canonical:
        receipt = by_key[job_key(expected_reuse)]
        if (
            receipt["request_sha256"] != expected_reuse["request_sha256"]
            or receipt.get("reused_from") != expected_reuse["reused_from"]
            or receipt["cost_usd"] != 0
        ):
            raise ValueError("original canonical request or zero-rebilling changed")
        originals = map(
            json.loads, Path(expected_reuse["reused_from"]).read_text().splitlines()
        )
        original = next(
            r
            for r in originals
            if r["valid"]
            and job_key(r) == job_key(expected_reuse)
            and r["request_sha256"] == expected_reuse["request_sha256"]
        )
        if any(
            receipt[key] != original[key]
            for key in (
                "request",
                "response_body",
                "tag",
                "model",
                "usage",
                "parsed",
                "attempt",
            )
        ):
            raise ValueError("canonical original request/response bytes changed")
    plan_rows = list(
        map(
            json.loads,
            (args.preparations / "preparations.jsonl").read_text().splitlines(),
        )
    )
    plans = {(r["problem_id"], r["model"], r["speaker"]): r for r in plan_rows}
    if len(plans) != len(plan_rows) or len(plans) != 96:
        raise ValueError("duplicate or missing accepted private plan")
    for receipt in accepted:
        tag = receipt["tag"]
        problem, swapped = inputs[job_key(receipt)]
        assigned = "A" if tag["speaker"] == "Red" else "B"
        request = receipt["request"]
        plan = plans[(tag["problem_id"], tag["model"], tag["speaker"])]
        expected_request = {
            "model": tag["model"],
            "messages": preparation_messages(
                problem, swapped, tag["speaker"], assigned, explicit_truth=True
            ),
            "max_tokens": 16384,
            "reasoning": {"enabled": True},
            "usage": {"include": True},
            "provider": {
                "require_parameters": True,
                "max_price": request["provider"]["max_price"],
            },
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "diagnostic",
                    "strict": True,
                    "schema": preparation_schema(),
                },
            },
        }
        if (
            request != expected_request
            or plan["request_sha256"] != receipt["request_sha256"]
            or plan["parsed"] != receipt["parsed"]
            or plan["source_story_sha256"] != sha256(problem.story)
            or plan["swapped"] != swapped
            or plan["assigned_answer"] != assigned
            or plan["finish_reason"] != "stop"
            or plan["max_tokens"] != 16384
            or plan["reasoning"] != {"enabled": True}
            or plan["completion_tokens"] != receipt["usage"]["completion_tokens"]
        ):
            raise ValueError("accepted plan differs from frozen inference contract")
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "all_96_canonical_preparation_jobs_complete",
        "accepted_jobs": len(accepted),
        "raw_receipts": len(rows),
        "canonical_original_reuses": len(canonical),
        "retained_schema_failures": failed,
        "all_attempts_accounted_cost_usd": summary["cost_usd"],
        "receipt_charges_sum_usd": sum(r["cost_usd"] for r in rows),
        "policy": "Frozen two-schema-retry policy; exact request and full tag retained. One accepted result per canonical job. No failed attempt discarded or valid answer regenerated by this gate.",
        "accepted_receipts": [
            {"tag": r["tag"], "request_sha256": r["request_sha256"]} for r in accepted
        ],
        "input_sha256": {
            str(path): sha256(path.read_text())
            for path in (
                args.global_freeze,
                args.dataset,
                args.preparations / "summary.json",
                args.preparations / "requests.jsonl",
                args.preparations / "preparations.jsonl",
                Path(__file__),
                recovery_audit,
            )
        },
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("accepted_receipts", "input_sha256")
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    for name in ("global-freeze", "dataset", "preparations", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
