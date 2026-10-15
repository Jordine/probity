"""Read-only generation provenance for frozen judge analyses.

Native exact-prompt cache hits are logical jobs, not additional generations.
This module leaves inference, original receipts and frozen statistics unchanged.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .judge3_final import (
    audit_completion,
    check_package,
    file_hash,
    prompt_repeatability,
    read_rows,
)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def answer(row):
    parsed = row.get("parsed", {}).get("answer", row.get("answer"))
    if parsed not in ("A", "B") or row.get("answer", parsed) != parsed:
        raise ValueError("evaluation answer differs from parsed answer")
    return parsed


def validate_rows(jobs, rows):
    indexed = {row["job_id"]: row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != set(jobs):
        raise ValueError("complete unique job coverage required")
    for job_id, row in indexed.items():
        value = answer(row)
        if (
            not row.get("valid")
            or row.get("finish_reason") != "stop"
            or row["gold"] != jobs[job_id]["gold"]
            or row["correct"] != (value == row["gold"])
        ):
            raise ValueError("incomplete or inconsistent evaluation")
    return indexed


def read_api_receipts(path, jobs, rows, judge, inputs):
    """Resolve recovery chains to original complete receipt identities."""
    cache = {}

    def load(source):
        source = Path(source).resolve()
        if source not in cache:
            inputs[str(source)] = file_hash(source)
            cache[source] = read_rows(source)
        return source, cache[source]

    def key(record):
        return record["request_sha256"], json.dumps(record["tag"], sort_keys=True)

    def find(source, identity):
        source, records = load(source)
        matches = [
            (line, record)
            for line, record in enumerate(records, start=1)
            if record.get("valid") and key(record) == identity
        ]
        if len(matches) != 1:
            raise ValueError("one original valid receipt per request and job required")
        return source, *matches[0]

    def origin(source, line, record, seen):
        marker = str(source), line
        if marker in seen:
            raise ValueError("cyclic receipt recovery provenance")
        audit_completion(record)
        if not record.get("reused_from"):
            return source, line, record
        earlier_source, earlier_line, earlier = find(record["reused_from"], key(record))
        for field in ("request", "response_body", "parsed", "model", "usage"):
            if earlier[field] != record[field]:
                raise ValueError("recovered receipt changed original generation")
        return origin(earlier_source, earlier_line, earlier, seen | {marker})

    _, records = load(path)
    by_job, origins = {}, {}
    for row in rows:
        job_id = row["job_id"]
        matches = [
            (line, receipt)
            for line, receipt in enumerate(records, start=1)
            if receipt.get("valid")
            and receipt["request_sha256"] == row["request_sha256"]
            and receipt["tag"].get("job") == job_id
        ]
        if len(matches) != 1:
            raise ValueError("one valid API receipt per logical job required")
        line, receipt = matches[0]
        audit_completion(receipt)
        # Frozen API hashes use ASCII-escaped canonical JSON.
        request_hash = hashlib.sha256(
            json.dumps(receipt["request"], sort_keys=True).encode()
        ).hexdigest()
        if (
            request_hash != receipt["request_sha256"]
            or receipt["request"]["messages"] != jobs[job_id]["messages"]
            or receipt["model"] != judge
            or receipt["parsed"]["answer"] != answer(row)
            or receipt["tag"]["kind"] != jobs[job_id]["kind"]
        ):
            raise ValueError("API generation receipt differs from frozen job")
        original_path, original_line, original = origin(
            Path(path).resolve(), line, receipt, set()
        )
        payload = json.loads(original["response_body"])
        # Distinct requests with identical bodies remain distinct observations.
        # A repeated provider response ID denotes the same recorded response.
        response_id = payload.get("id")
        identity = (
            "api_response:" + digest([judge, response_id])
            if response_id
            else "api_receipt:"
            + digest([str(original_path), inputs[str(original_path)], original_line])
        )
        by_job[job_id] = receipt
        origins[job_id] = {
            "generation_id": identity,
            "fresh_in_current_run": not bool(receipt.get("reused_from")),
            "origin_resolved": True,
            "original_receipt_path": str(original_path),
            "original_receipt_line": original_line,
            "provider_response_id_as_reported": response_id,
        }
    return by_job, origins


def generation_repeatability(jobs, rows, manifest, receipts=None, origins=None):
    """Count distinct generation observations separately from logical jobs."""
    validate_rows(jobs, rows)
    native = receipts is None
    kind = "native" if native else "api"
    normalized = [{**row, "answer": answer(row)} for row in rows]
    logical = prompt_repeatability(jobs, normalized, receipts)
    groups = defaultdict(list)
    fresh_by_native_key = Counter()
    native_values = {}
    if native:
        if not manifest["contract"]["answer_only"]:
            raise ValueError("native frozen answer-only runtime required")
        for row in rows:
            if type(row.get("reused")) is not bool:
                raise ValueError("native receipt must explicitly identify cache reuse")
            if (
                not 0
                < row["completion_tokens"]
                < manifest["contract"]["max_new_tokens"]
            ):
                raise ValueError("incomplete native generation")
            key = row["request_sha256"]
            value = (digest(jobs[row["job_id"]]["messages"]), answer(row), row["raw"])
            if key in native_values and native_values[key] != value:
                raise ValueError("native cached receipt changed prompt or generation")
            native_values[key] = value
            if not row["reused"]:
                fresh_by_native_key[key] += 1
        if any(count > 1 for count in fresh_by_native_key.values()):
            raise ValueError(
                "frozen native runner cannot generate the same cache key twice"
            )
        origins = {
            row["job_id"]: {
                "generation_id": "native_cache_key:" + row["request_sha256"],
                "fresh_in_current_run": not row["reused"],
                "origin_resolved": bool(fresh_by_native_key[row["request_sha256"]]),
            }
            for row in rows
        }
    elif set(origins or {}) != {row["job_id"] for row in rows}:
        raise ValueError("resolved API origin required for every evaluation")
    generation_messages = {}
    for row in rows:
        job = jobs[row["job_id"]]
        generation_id = origins[row["job_id"]]["generation_id"]
        message_hash = digest(job["messages"])
        if (
            generation_id in generation_messages
            and generation_messages[generation_id] != message_hash
        ):
            raise ValueError(
                "one recorded generation cannot belong to different prompts"
            )
        generation_messages[generation_id] = message_hash
        groups[message_hash].append(
            {
                "job_id": row["job_id"],
                "kind": job["kind"],
                "answer": answer(row),
                "request_sha256": row["request_sha256"],
                **origins[row["job_id"]],
            }
        )
    repeated, generation_matches, generation_pairs = [], 0, 0
    all_ids = set()
    for message_hash, observations in sorted(groups.items()):
        distinct = {}
        for observation in observations:
            generation_id = observation["generation_id"]
            if (
                generation_id in distinct
                and distinct[generation_id]["answer"] != observation["answer"]
            ):
                raise ValueError(
                    "one recorded generation cannot have different answers"
                )
            distinct[generation_id] = observation
            all_ids.add(generation_id)
        if len(observations) < 2:
            continue
        # Native cache-only receipts with an unavailable prior origin are not
        # promoted to new independently observed generations.
        measured = [item for item in distinct.values() if item["origin_resolved"]]
        counts = Counter(item["answer"] for item in measured)
        pairs = len(measured) * (len(measured) - 1) // 2
        matches = sum(n * (n - 1) // 2 for n in counts.values())
        generation_matches += matches
        generation_pairs += pairs
        repeated.append(
            {
                "messages_sha256": message_hash,
                "logical_jobs": len(observations),
                "fresh_generations_in_run": sum(
                    item["fresh_in_current_run"] for item in observations
                ),
                "reused_logical_jobs": sum(
                    not item["fresh_in_current_run"] for item in observations
                ),
                "generation_receipts_represented": len(distinct),
                "resolved_generation_observations": len(measured),
                "generation_answer_counts": dict(counts),
                "generation_answer_disagreement": len(counts) > 1 if pairs else None,
                "generation_answer_pairs": pairs,
                "generation_matching_answer_pairs": matches,
                "generation_pairwise_agreement": matches / pairs if pairs else None,
                "jobs": observations,
            }
        )
    if generation_pairs:
        status = "descriptive_distinct_recorded_generations"
    elif native and repeated:
        status = "cached_repeats_only_sampling_repeatability_unmeasured"
    else:
        status = "no_repeated_generations_unmeasured"
    summary = {
        "runtime_kind": kind,
        "measurement_status": status,
        "logical_jobs_total": len(rows),
        "fresh_generations_in_run": sum(
            item["fresh_in_current_run"] for item in origins.values()
        ),
        "reused_logical_jobs": sum(
            not item["fresh_in_current_run"] for item in origins.values()
        ),
        "generation_receipts_represented": len(all_ids),
        "unresolved_native_cached_origins": len(
            {
                item["generation_id"]
                for item in origins.values()
                if not item["origin_resolved"]
            }
        ),
        "generation_repeated_prompt_groups": sum(
            g["generation_answer_pairs"] > 0 for g in repeated
        ),
        "generation_groups_with_answer_disagreement": sum(
            g["generation_answer_disagreement"] is True for g in repeated
        ),
        "generation_matching_answer_pairs": generation_matches,
        "generation_all_answer_pairs": generation_pairs,
        "generation_pairwise_agreement": (
            generation_matches / generation_pairs if generation_pairs else None
        ),
    }
    return {
        "schema": "judge3_generation_repeatability_v1",
        "summary": summary,
        "legacy_logical_summary": {
            key: value for key, value in logical.items() if key != "groups"
        },
        "groups": repeated,
        "limits": [
            "Logical agreement includes native cached copies and must not be described as measured sampling repeatability.",
            "No repeated distinct recorded generations means unmeasured repeatability, not zero sampling noise.",
            "Fresh generations in run counts recorded non-reused native generations or API responses; recovered API origins are retained as existing observations, not new calls.",
            "API request equality does not establish equal provider routing or independent hardware or sampling; provider response caching beyond the returned metadata is unknown.",
            "Native greedy execution is not a stochastic sampling experiment. Caching prevents observing repeat variation under that runtime.",
            "These naturally repeated messages do not add independent source families. No extra model calls or bootstrap calculations were performed.",
        ],
    }


def prepare_native_view(args):
    package, jobs_path = check_package(args.package, args.frozen_package_sha256)
    if args.out.exists():
        raise ValueError("new immutable native analysis view required")
    manifest_path = args.results / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest["jobs_sha256"] != package["jobs_sha256"]
        or not manifest["contract"]["answer_only"]
    ):
        raise ValueError("native results differ from frozen jobs/runtime")
    jobs = {job["job_id"]: job for job in read_rows(jobs_path)}
    rows = read_rows(args.results / "evaluations.jsonl")
    validate_rows(jobs, rows)
    if any("answer" in row for row in rows):
        raise ValueError(
            "original native rows unexpectedly already contain top-level answers"
        )
    derived = [{**row, "answer": row["parsed"]["answer"]} for row in rows]
    assert all(
        {k: v for k, v in new.items() if k != "answer"} == old
        for old, new in zip(rows, derived)
    )
    args.out.mkdir(parents=True)
    (args.out / "manifest.json").write_bytes(manifest_path.read_bytes())
    (args.out / "evaluations.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in derived)
    )
    provenance = {
        "schema": "judge3_native_analysis_view_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_sha256": args.frozen_package_sha256,
        "original_results": str(args.results.resolve()),
        "original_evaluations_sha256": file_hash(args.results / "evaluations.jsonl"),
        "original_manifest_sha256": file_hash(manifest_path),
        "view_evaluations_sha256": file_hash(args.out / "evaluations.jsonl"),
        "view_manifest_sha256": file_hash(args.out / "manifest.json"),
        "rows": len(rows),
        "only_change": "Each row adds top-level answer exactly equal to its existing parsed.answer. Every original field including reused and request_sha256 is unchanged; manifest bytes are copied unchanged.",
        "source_sha256": file_hash(__file__),
    }
    (args.out / "native_analysis_view_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(json.dumps(provenance, indent=2))


def audit(args):
    package, jobs_path = check_package(args.package, args.frozen_package_sha256)
    if args.out.exists():
        raise ValueError("new immutable generation audit required")
    manifest_path = args.results / "manifest.json"
    rows_path = args.results / "evaluations.jsonl"
    manifest = json.loads(manifest_path.read_text())
    if manifest["jobs_sha256"] != package["jobs_sha256"]:
        raise ValueError("results used different frozen jobs")
    jobs = {job["job_id"]: job for job in read_rows(jobs_path)}
    rows = read_rows(rows_path)
    inputs = {
        str(path.resolve()): file_hash(path)
        for path in [
            args.package / "package.json",
            jobs_path,
            manifest_path,
            rows_path,
            args.analysis / "prompt_repeatability.json",
        ]
    }
    receipts = origins = None
    judge = manifest.get("judge_model") or manifest["contract"]["model"]
    if (args.results / "requests.jsonl").exists():
        receipts, origins = read_api_receipts(
            args.results / "requests.jsonl", jobs, rows, judge, inputs
        )
    report = generation_repeatability(jobs, rows, manifest, receipts, origins)
    frozen_repeat = json.loads(
        (args.analysis / "prompt_repeatability.json").read_text()
    )
    if report["legacy_logical_summary"] != {
        key: value for key, value in frozen_repeat.items() if key != "groups"
    }:
        raise ValueError("generation audit differs from frozen logical-prompt report")
    report.update(
        judge=judge,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        package_sha256=args.frozen_package_sha256,
        input_sha256=inputs,
        source_sha256=file_hash(__file__),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "groups"}, indent=2
        )
    )


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["prepare-native-view", "audit"]:
        part = sub.add_parser(name)
        part.add_argument("--package", type=Path, required=True)
        part.add_argument("--frozen-package-sha256", required=True)
        part.add_argument("--results", type=Path, required=True)
        part.add_argument("--out", type=Path, required=True)
        if name == "audit":
            part.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare-native-view":
        prepare_native_view(args)
    else:
        audit(args)


if __name__ == "__main__":
    main()
