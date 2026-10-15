"""Recount complete frozen final prompts; forecast cost without completion calls."""

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
    if package["jobs"] != 384 or {s["detector"] for s in package["signals"]} != {"adapted_primary", "local3_sentence"}:
        raise ValueError("expected the frozen reduced final Opus package")
    jobs = [json.loads(line) for line in jobs_path.read_text().splitlines()]
    assert len(jobs) == 384
    calibration = Path("runs/iteration3/judge_controls/opus_runtime_smoke_v1/requests.jsonl")
    prior_projection = Path("runs/iteration3/judge_controls/opus_runtime_cost_projection_v2.json")
    old = json.loads(prior_projection.read_text())
    encoding = tiktoken.get_encoding("cl100k_base")
    receipts = [json.loads(line) for line in calibration.read_text().splitlines()]
    assert len(receipts) == 8
    offsets, ratios = [], []
    for receipt in receipts:
        audit_completion(receipt)
        assert receipt["model"] == "anthropic/claude-opus-5.5"
        proxy = len(encoding.encode(json.dumps(receipt["request"]["messages"], ensure_ascii=False)))
        observed = receipt["usage"]["prompt_tokens"]
        offsets.append(observed - proxy)
        ratios.append(observed / proxy)
    assert offsets == old["observed_offsets"] and ratios == old["observed_ratios"]
    offset, ratio = max(offsets), max(ratios)
    catalog = json.load(urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=30))
    model = next(m for m in catalog["data"] if m["id"] == "anthropic/claude-opus-5.5")
    prices = {key: float(model["pricing"].get(key, 0)) for key in ("prompt", "completion", "request")}
    assert all(math.isfinite(value) and value >= 0 for value in prices.values())
    context = model["top_provider"].get("context_length") or model["context_length"]
    output_allowance = 32768
    expected_completion = 11
    rows, conditions = [], defaultdict(list)
    schema = object_schema({"answer": ANSWER})
    for job in jobs:
        serialized = json.dumps(job["messages"], ensure_ascii=False)
        proxy = len(encoding.encode(serialized))
        estimated_tokens = math.ceil(max(proxy + offset, proxy * ratio))
        complete_bound = len(json.dumps([job["messages"], schema], ensure_ascii=False).encode()) + 2048
        assert complete_bound + output_allowance <= context
        expected_cost = estimated_tokens * prices["prompt"] + expected_completion * prices["completion"] + prices["request"]
        reserved_upper = complete_bound * prices["prompt"] + output_allowance * prices["completion"] + prices["request"]
        row = {
            "job_id": job["job_id"], "kind": job["kind"],
            "complete_message_json_characters": len(serialized),
            "complete_message_json_utf8_bytes": len(serialized.encode()),
            "cl100k_proxy_tokens": proxy, "estimated_prompt_tokens": estimated_tokens,
            "api_input_byte_upper_bound": complete_bound,
            "estimated_cost_usd": expected_cost,
            "api_per_call_reserved_upper_usd": reserved_upper,
        }
        rows.append(row)
        conditions[job["kind"]].append(row)
    estimated_total = sum(row["estimated_cost_usd"] for row in rows)
    report = {
        "schema": "judge3_final_opus_cost_recount_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_sha256": args.frozen_package_sha256,
        "jobs_sha256": file_hash(jobs_path), "jobs": len(rows),
        "calibration_sha256": file_hash(calibration),
        "previous_cost_projection_sha256": file_hash(prior_projection),
        "token_estimator": "Complete final message JSON counted with cl100k_base, using the unchanged greater-of-additive-offset/multiplicative-ratio envelope from all eight old development Opus calls. This is a cost forecast, not an exact Claude tokenizer or billing guarantee.",
        "offset_used": offset, "ratio_used": ratio,
        "live_model_catalog": model,
        "prices_per_token": prices,
        "runtime": {"reasoning_effort": "low", "max_output_tokens": output_allowance, "request_timeout_seconds": 900, "concurrency": 1},
        "expected_completion_tokens_assumed": expected_completion,
        "estimated_total_prompt_tokens": sum(r["estimated_prompt_tokens"] for r in rows),
        "estimated_total_usd": estimated_total,
        "estimated_total_plus25percent_usd": 1.25 * estimated_total,
        "maximum_per_call_reservation_usd": max(r["api_per_call_reserved_upper_usd"] for r in rows),
        "estimated_total_plus_one_maximum_call_reservation_usd": estimated_total + max(r["api_per_call_reserved_upper_usd"] for r in rows),
        "extra_cost_per100_mean_completion_tokens_usd": 100 * len(rows) * prices["completion"],
        "by_condition": {kind: {"jobs": len(values), "estimated_prompt_tokens": sum(v["estimated_prompt_tokens"] for v in values), "estimated_cost_usd": sum(v["estimated_cost_usd"] for v in values)} for kind, values in sorted(conditions.items())},
        "complete_job_recount": rows,
        "limits": ["No completion calls made; root still must release the package and budget.", "The shared ledger and conservative per-call reservations enforce the actual approved spending cap.", "The output allowance remains 32768; budget limits never justify scoring a truncated generation.", "Provider routing, hidden overhead, or longer reasoning may make actual cost differ from this forecast."],
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in ("complete_job_recount", "live_model_catalog")}, indent=2))


if __name__ == "__main__":
    main()
