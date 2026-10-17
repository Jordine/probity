"""Recount complete final Haiku prompts without issuing completion calls."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import urllib.request

import tiktoken

from probity_benchmark.api import object_schema
from probity_benchmark.controls3 import ANSWER
from probity_benchmark.judge3_final import audit_completion, check_package


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--frozen-package-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("new immutable cost report required")
    package, jobs_path = check_package(args.package, args.frozen_package_sha256, True)
    lock_path = Path("runs/iteration3/judge_controls/final_judge_code_lock_v2.json")
    assert (
        file_hash(lock_path)
        == "89ab74c57193b41b31c9d7f831ae52a904f4eef11b59948a0f5c9e5cf5c00c86"
    )
    lock = json.loads(lock_path.read_text())
    scope = lock["packages"]["full"]
    if package["jobs"] != scope["jobs_expected"] or {
        s["detector"] for s in package["signals"]
    } != set(scope["detectors"]):
        raise ValueError("expected the frozen full final Haiku package")
    jobs = [json.loads(line) for line in jobs_path.read_text().splitlines()]
    assert len(jobs) == 1032
    calibration = Path(
        "runs/iteration3/judge_controls/haiku_reasoned_candidates_v1/requests.jsonl"
    )
    encoding = tiktoken.get_encoding("cl100k_base")
    receipts = [json.loads(line) for line in calibration.read_text().splitlines()]
    assert len(receipts) == 64
    model_id = "anthropic/claude-haiku-5.5"
    runtime = lock["api_runtimes"][model_id]
    offsets, ratios, completions = [], [], []
    for receipt in receipts:
        audit_completion(receipt)
        assert receipt["model"] == model_id
        assert receipt["request"]["reasoning"] == runtime["reasoning"]
        assert receipt["request"]["max_tokens"] == runtime["max_output_tokens"]
        proxy = len(
            encoding.encode(
                json.dumps(receipt["request"]["messages"], ensure_ascii=False)
            )
        )
        observed = receipt["usage"]["prompt_tokens"]
        offsets.append(observed - proxy)
        ratios.append(observed / proxy)
        completions.append(receipt["usage"]["completion_tokens"])
    offset, ratio = max(offsets), max(ratios)
    expected_completion = max(completions)
    catalog = json.load(
        urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=30)
    )
    model = next(m for m in catalog["data"] if m["id"] == model_id)
    prices = {
        key: float(model["pricing"].get(key, 0))
        for key in ("prompt", "completion", "request")
    }
    assert all(math.isfinite(value) and value >= 0 for value in prices.values())
    context = model["top_provider"].get("context_length") or model["context_length"]
    output_allowance = runtime["max_output_tokens"]
    rows, conditions = [], defaultdict(list)
    schema = object_schema({"answer": ANSWER})
    for job in jobs:
        serialized = json.dumps(job["messages"], ensure_ascii=False)
        proxy = len(encoding.encode(serialized))
        estimated_tokens = math.ceil(max(proxy + offset, proxy * ratio))
        complete_bound = (
            len(json.dumps([job["messages"], schema], ensure_ascii=False).encode())
            + 2048
        )
        assert complete_bound + output_allowance <= context
        expected_cost = (
            estimated_tokens * prices["prompt"]
            + expected_completion * prices["completion"]
            + prices["request"]
        )
        reserved_upper = (
            complete_bound * prices["prompt"]
            + output_allowance * prices["completion"]
            + prices["request"]
        )
        row = {
            "job_id": job["job_id"],
            "kind": job["kind"],
            "complete_message_json_characters": len(serialized),
            "complete_message_json_utf8_bytes": len(serialized.encode()),
            "cl100k_proxy_tokens": proxy,
            "estimated_prompt_tokens": estimated_tokens,
            "api_input_byte_upper_bound": complete_bound,
            "estimated_cost_usd": expected_cost,
            "api_per_call_reserved_upper_usd": reserved_upper,
        }
        rows.append(row)
        conditions[job["kind"]].append(row)
    estimated_total = sum(row["estimated_cost_usd"] for row in rows)
    four_call_reserve = sum(
        sorted((r["api_per_call_reserved_upper_usd"] for r in rows), reverse=True)[
            : runtime["concurrency"]
        ]
    )
    report = {
        "schema": "judge3_final_haiku_cost_recount_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_sha256": args.frozen_package_sha256,
        "jobs_sha256": file_hash(jobs_path),
        "jobs": len(rows),
        "calibration_sha256": file_hash(calibration),
        "calibration_calls": len(receipts),
        "runtime_lock_sha256": file_hash(lock_path),
        "token_estimator": "Complete final message JSON counted with cl100k_base, using the greater-of-additive-offset/multiplicative-ratio envelope from all 64 old development Haiku calls. A cost forecast, not an exact Claude tokenizer or billing guarantee.",
        "observed_offsets": offsets,
        "observed_ratios": ratios,
        "observed_completion_tokens": completions,
        "offset_used": offset,
        "ratio_used": ratio,
        "live_model_catalog": model,
        "prices_per_token": prices,
        "runtime": runtime,
        "expected_completion_tokens_assumed": expected_completion,
        "estimated_total_prompt_tokens": sum(
            r["estimated_prompt_tokens"] for r in rows
        ),
        "estimated_total_usd": estimated_total,
        "estimated_total_plus25percent_usd": 1.25 * estimated_total,
        "maximum_per_call_reservation_usd": max(
            r["api_per_call_reserved_upper_usd"] for r in rows
        ),
        "largest_four_call_reservation_usd": four_call_reserve,
        "estimated_total_plus_four_largest_call_reservations_usd": estimated_total
        + four_call_reserve,
        "extra_cost_per100_mean_completion_tokens_usd": 100
        * len(rows)
        * prices["completion"],
        "by_condition": {
            kind: {
                "jobs": len(values),
                "estimated_prompt_tokens": sum(
                    v["estimated_prompt_tokens"] for v in values
                ),
                "estimated_cost_usd": sum(v["estimated_cost_usd"] for v in values),
            }
            for kind, values in sorted(conditions.items())
        },
        "complete_job_recount": rows,
        "limits": [
            "No completion calls made; root still must release the package and budget.",
            "The shared ledger and conservative per-call reservations enforce the actual approved spending cap.",
            "Output allowance remains 8192. A budget limit never justifies scoring a truncated generation.",
            "Provider routing, hidden overhead, or longer outputs may make actual cost differ from this forecast.",
        ],
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key
                not in (
                    "complete_job_recount",
                    "live_model_catalog",
                    "observed_offsets",
                    "observed_ratios",
                    "observed_completion_tokens",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
