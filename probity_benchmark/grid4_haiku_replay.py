"""Fresh identical-request A/C replay: a post-result routing/noise sensitivity.

This diagnostic never replaces the primary cached A/C results. Freeze all 96
original A/C jobs before calling, retain every new provider response, and keep
the already observed B/D results fixed. No cache reuse or automatic retries.
"""

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from .api import ANSWER, BudgetedChat, object_schema
from .grid4_evaluate import (
    api_request_body, audit_results, check_package, grid_statistics, keyed,
    prior_jobs_hash, read_rows,
)
from .grid4_generate import atomic_json, file_hash, run_lock
from .judge3_final import audit_completion
from .judge_assistance import paired_stats
from .preflight import sha256


MODEL = "anthropic/claude-haiku-5.5"
CAP = .10
RUNTIME = {"reasoning": {"enabled": False}, "max_tokens": 8192,
           "request_timeout_seconds": 300.}
SCHEMA = "grid4_haiku_ac_replay_v1"
SOURCE_FILES = ["probity_benchmark/grid4_haiku_replay.py", "probity_benchmark/api.py",
                "probity_benchmark/grid4_evaluate.py", "probity_benchmark/grid4_generate.py",
                "probity_benchmark/judge3_final.py", "probity_benchmark/judge_assistance.py",
                "probity_benchmark/budget.py", "probity_benchmark/preflight.py"]
ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write_rows(path, values):
    path = Path(path)
    require(not path.exists(), "immutable output already exists")
    with path.open("x") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def request_hash(body):
    return sha256(json.dumps(body, sort_keys=True))


def validate_selection(jobs, pairs):
    require(len(jobs) == 96 and len(keyed(jobs)) == 96 and len(pairs) == 48,
            "all 96 original A/C jobs and all 48 paired trajectories are required")
    expected = {p["jobs"][cell] for p in pairs for cell in ("A", "C")}
    require(set(keyed(jobs)) == expected and len(expected) == 96, "A/C selection differs from frozen pairs")
    require(len({p["family"] for p in pairs}) == 24, "24 original story families required")
    require(Counter(j["kind"] for j in jobs) == {"none": 48, "adapted_primary:real": 48},
            "complete original A/C cells required")
    require(Counter(p["participant_model"] for p in pairs) == {
        "qwen/qwen3.5-27b": 24, "qwen/qwen3.8-27b": 24}, "both original participant models required")
    require(len({(p["family"], p["participant_model"]) for p in pairs}) == 48,
            "one trajectory per original story family and participant is required")


def bind_original_requests(jobs, prior_outcomes, receipts):
    by_receipt = {}
    for receipt in receipts:
        if receipt.get("valid"):
            key = (receipt["request_sha256"], receipt["tag"]["job"])
            require(key not in by_receipt, "ambiguous original valid receipt")
            by_receipt[key] = receipt
    frozen = []
    for job in jobs:
        old = prior_outcomes[job["job_id"]]
        receipt = by_receipt[(old["request_sha256"], old.get("receipt_job_id", job["job_id"]))]
        body = receipt["request"]
        require(body["messages"] == job["messages"] and request_hash(body) == old["request_sha256"],
                "original request does not match the frozen original job")
        require(body["model"] == MODEL and body["reasoning"] == RUNTIME["reasoning"] and
                body["max_tokens"] == 8192 and "temperature" not in body,
                "original request differs from declared frozen diagnostic contract")
        require(body["response_format"] == {"type": "json_schema", "json_schema": {
            "name": "diagnostic", "strict": True, "schema": object_schema({"answer": ANSWER})}},
            "original structured-answer schema changed")
        audit_completion(receipt)
        frozen.append({"job_id": job["job_id"], "request_sha256": old["request_sha256"],
                       "request": body, "original_outcome": old, "original_receipt": receipt})
    return frozen


def validate_client_requests(api, jobs, frozen):
    require(api.max_tokens == 8192 and api.reasoning == {"enabled": False} and
            api.request_timeout_seconds == 300 and api.schema_retries == 0 and
            not api.reusable and not api.unavailable, "replay must use fresh fixed-runtime calls without retries/cache")
    by_id = keyed(frozen)
    for job in jobs:
        body = api_request_body(api, MODEL, job["messages"])
        require(body == by_id[job["job_id"]]["request"] and
                request_hash(body) == by_id[job["job_id"]]["request_sha256"],
                "current request builder differs from frozen original; no calls authorized")


def prepare(args):
    require(not args.out.exists(), "new diagnostic plan directory required")
    package = check_package(args.package, args.frozen_package_sha256)
    jobs = read_rows(args.package / "prior_jobs.jsonl")
    pairs = read(args.package / "pairs.json")
    validate_selection(jobs, pairs)
    old, old_audit = audit_results(args.prior_results, keyed(jobs), prior_jobs_hash(package, args.prior_results), True)
    _, bd_audit = audit_results(args.feedback_results, keyed(read_rows(args.package / "jobs.jsonl")), package["jobs_sha256"])
    require(old_audit["runtime_identity"] == bd_audit["runtime_identity"], "original and fixed B/D runtime contracts differ")
    require(old_audit["runtime_identity"]["model"] == MODEL and
            old_audit["runtime_identity"]["runtime"] == RUNTIME, "prior Haiku runtime differs")
    frozen = bind_original_requests(jobs, old, read_rows(args.prior_results / "requests.jsonl"))
    inputs = [args.package / name for name in ("package.json", "prior_jobs.jsonl", "jobs.jsonl", "pairs.json")]
    inputs += [directory / name for directory in (args.prior_results, args.feedback_results)
               for name in ("manifest.json", "api_manifest.json", "requests.jsonl", "evaluations.jsonl")]
    args.out.mkdir(parents=True)
    # Preserve original job bytes, not a reformatted selection of favorable rows.
    (args.out / "jobs.jsonl").write_bytes((args.package / "prior_jobs.jsonl").read_bytes())
    write_rows(args.out / "frozen_requests.jsonl", frozen)
    atomic_json(args.out / "pairs.json", pairs, immutable=True)
    plan = {"schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection": "All 96 original A/C jobs, fixed before any new diagnostic response; no outcome-based selection.",
            "reason": "Post-result routing/noise sensitivity: cached Haiku A/C reported mixed providers while fresh B/D reported Azure, with missing system fingerprints.",
            "scope": "Secondary diagnostic only; never overwrite or replace primary cached A/C or original grid analysis.",
            "jobs_expected": 96, "families": 24, "model": MODEL, "runtime": RUNTIME,
            "local_cap_usd": CAP, "schema_retries": 0, "receipt_cache_reuse": False,
            "unique_request_hashes": len({r["request_sha256"] for r in frozen}),
            "jobs_sha256": file_hash(args.out / "jobs.jsonl"),
            "frozen_requests_sha256": file_hash(args.out / "frozen_requests.jsonl"),
            "pairs_sha256": file_hash(args.out / "pairs.json"),
            "package": str(args.package.resolve()), "package_sha256": args.frozen_package_sha256,
            "prior_results": str(args.prior_results.resolve()), "feedback_results": str(args.feedback_results.resolve()),
            "input_sha256": {str(path.resolve()): file_hash(path) for path in inputs},
            "source_sha256": {name: file_hash(ROOT / name) for name in SOURCE_FILES},
            "original_runtime_audit": old_audit, "fixed_BD_runtime_audit": bd_audit,
            "fresh_directory": "fresh", "analysis_directory": "analysis",
            "limitations": ["The request omits temperature exactly as before; observed drift is not isolated sampling noise.",
                            "Provider routing is observed, not randomized or forced. Matching answers or provider names does not establish identical backend weights/hardware.",
                            "Missing fingerprints limit model/backend identity claims. Same-family intervals are descriptive post-result sensitivities."]}
    atomic_json(args.out / "plan.json", plan, immutable=True)
    return {"prepared": True, "jobs": 96, "families": 24, "local_cap_usd": CAP,
            "plan_sha256": file_hash(args.out / "plan.json"), "output": str(args.out)}


def validate_plan(out):
    plan = read(out / "plan.json")
    require(plan["schema"] == SCHEMA and plan["local_cap_usd"] == CAP and plan["runtime"] == RUNTIME and
            plan["schema_retries"] == 0 and plan["receipt_cache_reuse"] is False, "diagnostic contract changed")
    for name, expected in plan["input_sha256"].items():
        require(file_hash(name) == expected, "frozen diagnostic input changed: " + name)
    for name, expected in plan["source_sha256"].items():
        require(file_hash(ROOT / name) == expected, "diagnostic source changed after pre-call freeze: " + name)
    for name, key in (("jobs.jsonl", "jobs_sha256"), ("frozen_requests.jsonl", "frozen_requests_sha256"), ("pairs.json", "pairs_sha256")):
        require(file_hash(out / name) == plan[key], "frozen diagnostic plan artifact changed")
    jobs, frozen, pairs = read_rows(out / "jobs.jsonl"), read_rows(out / "frozen_requests.jsonl"), read(out / "pairs.json")
    validate_selection(jobs, pairs)
    require(len(frozen) == 96 and set(keyed(frozen)) == set(keyed(jobs)), "frozen original request coverage changed")
    return plan, jobs, frozen, pairs


async def run(args):
    plan, jobs, frozen, _ = validate_plan(args.out)
    require(os.environ.get("PROBITY_PILOT_LEDGER"), "shared global ledger is mandatory")
    require(not (args.out / "started.json").exists() and not (args.out / "fresh").exists(),
            "diagnostic has started; never automatically repeat calls")
    results = []
    originals = keyed(frozen)
    async with BudgetedChat(args.out / "fresh", [MODEL], CAP, max_tokens=8192,
                            schema_retries=0, concurrency=args.concurrency,
                            reasoning={"enabled": False}, request_timeout_seconds=300,
                            reuse_receipts=()) as api:
        validate_client_requests(api, jobs, frozen)  # Validate every body before dispatching any.
        atomic_json(args.out / "started.json", {"plan_sha256": file_hash(args.out / "plan.json"),
            "global_budget_key": api.budget_key, "reserved_usd": CAP,
            "at_utc": datetime.now(timezone.utc).isoformat()}, immutable=True)
        atomic_json(args.out / "fresh/manifest.json", {
            "schema": SCHEMA, "plan_sha256": file_hash(args.out / "plan.json"),
            "jobs_sha256": plan["jobs_sha256"], "judge_model": MODEL, "jobs_expected": 96,
            "runtime": RUNTIME, "source_sha256": file_hash(__file__),
            "cache_reuse": False, "automatic_retries": 0}, immutable=True)

        async def one(job):
            original = originals[job["job_id"]]
            atomic_json(args.out / "call_intents" / (job["job_id"] + ".json"), {
                "job_id": job["job_id"], "request_sha256": original["request_sha256"],
                "plan_sha256": file_hash(args.out / "plan.json"), "global_budget_key": api.budget_key}, immutable=True)
            receipt = await api.complete(MODEL, job["messages"], object_schema({"answer": ANSWER}),
                                         {"job": job["job_id"], "kind": job["kind"]})
            require(receipt["request"] == original["request"] and
                    receipt["request_sha256"] == original["request_sha256"] and
                    not receipt.get("reused_from") and receipt["attempt"] == 0,
                    "fresh receipt identity or single-attempt contract failed")
            completion = audit_completion(receipt)
            answer = receipt["parsed"]["answer"]
            row = {**{k: v for k, v in job.items() if k != "messages"},
                   "judge_model": MODEL, "answer": answer, "correct": answer == job["gold"],
                   "request_sha256": receipt["request_sha256"], "receipt_job_id": job["job_id"],
                   "receipt_reused": False, "exact_request_alias": False,
                   "valid": True, "finish_reason": "stop", "completion_audit": completion}
            results.append(row)
            with (args.out / "fresh/evaluations.jsonl").open("a") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

        outcomes = await asyncio.gather(*(one(job) for job in jobs), return_exceptions=True)
        summary = {"jobs_expected": 96, "complete": len(results) == 96,
                   "completed": len(results), "fresh_receipts": len(api.records),
                   "accounted_cost_usd": api.spent + api.reserved,
                   "errors": [str(value) for value in outcomes if isinstance(value, BaseException)]}
        atomic_json(args.out / "fresh/summary.json", summary, immutable=True)
    require(summary["complete"] and not summary["errors"] and summary["fresh_receipts"] == 96,
            "replay incomplete; all received outcomes/charges retained, no automatic retry")
    return summary


def route(receipt):
    payload = json.loads(receipt["response_body"])
    return {"provider": str(payload.get("provider")), "reported_model": str(payload.get("model")),
            "fingerprint": payload.get("system_fingerprint")}


def validate_fresh_receipts(jobs, frozen, receipts):
    originals = keyed(frozen)
    require(len(receipts) == len(jobs) and all(r.get("valid") is True and
            not r.get("reused_from") and r["attempt"] == 0 for r in receipts),
            "one fresh complete single-attempt receipt per frozen job required")
    by_job = {r["tag"]["job"]: r for r in receipts}
    require(len(by_job) == len(receipts) and set(by_job) == set(keyed(jobs)),
            "fresh receipt coverage differs from all original jobs")
    old_ids = {json.loads(r["original_receipt"]["response_body"]).get("id") for r in frozen}
    old_ids.discard(None)
    new_ids = []
    for job_id, receipt in by_job.items():
        require(receipt["request"] == originals[job_id]["request"] and
                receipt["request_sha256"] == originals[job_id]["request_sha256"],
                "fresh diagnostic request differs from original")
        response_id = json.loads(receipt["response_body"]).get("id")
        if response_id:
            require(response_id not in old_ids and response_id not in new_ids,
                    "new provider response ID repeats old or other diagnostic receipt")
            new_ids.append(response_id)
    return by_job, {"new_provider_response_ids_observed": len(new_ids),
                    "new_ids_distinct_from_originals_and_each_other": True,
                    "missing_provider_response_ids": len(receipts) - len(new_ids)}


def analyse(args):
    plan, jobs, frozen, pairs = validate_plan(args.out)
    destination = args.out / "analysis"
    require(not destination.exists(), "new immutable diagnostic analysis required")
    fresh, fresh_audit = audit_results(args.out / "fresh", keyed(jobs), plan["jobs_sha256"])
    require(fresh_audit["runtime_identity"] == plan["original_runtime_audit"]["runtime_identity"],
            "fresh and old declared request/runtime identities differ")
    receipts = read_rows(args.out / "fresh/requests.jsonl")
    by_job, generation_id_evidence = validate_fresh_receipts(jobs, frozen, receipts)
    require(len({r["request_sha256"] for r in receipts}) == plan["unique_request_hashes"],
            "diagnostic unique request coverage changed")
    old = {r["job_id"]: r["original_outcome"] for r in frozen}
    originals = keyed(frozen)
    comparisons = []
    for job in jobs:
        job_id = job["job_id"]
        original, new = originals[job_id], by_job[job_id]
        require(new["request"] == original["request"] and new["request_sha256"] == original["request_sha256"],
                "fresh diagnostic request differs from original")
        old_answer, answer = original["original_receipt"]["parsed"]["answer"], new["parsed"]["answer"]
        comparisons.append({"job_id": job_id, "family": job["group"], "participant_model": job["participant_model"],
                            "cell": "A" if job["kind"] == "none" else "C", "gold": job["gold"],
                            "old_answer": old_answer, "fresh_answer": answer,
                            "old_correct": old_answer == job["gold"], "fresh_correct": answer == job["gold"],
                            "answer_changed": old_answer != answer, "request_sha256": new["request_sha256"],
                            "old_route": route(original["original_receipt"]), "fresh_route": route(new)})
    package = read(Path(plan["package"]) / "package.json")
    bd, bd_audit = audit_results(Path(plan["feedback_results"]),
        keyed(read_rows(Path(plan["package"]) / "jobs.jsonl")), package["jobs_sha256"])
    require(bd_audit == plan["fixed_BD_runtime_audit"], "B/D results changed after diagnostic freeze")
    original_grid = grid_statistics(pairs, {**old, **bd}, MODEL)
    sensitivity_grid = grid_statistics(pairs, {**fresh, **bd}, MODEL)
    aggregates = []
    for participant in ["all", *sorted({j["participant_model"] for j in jobs})]:
        for cell in ("all", "A", "C"):
            selected = [r for r in comparisons if (participant == "all" or r["participant_model"] == participant)
                        and (cell == "all" or r["cell"] == cell)]
            stats = paired_stats([{"group": r["family"], "difference": int(r["fresh_correct"]) - int(r["old_correct"])}
                                  for r in selected])
            aggregates.append({"participant_model": participant, "cell": cell, "judgments": len(selected),
                "families": stats["parent_groups"], "old_correct": sum(r["old_correct"] for r in selected),
                "fresh_correct": sum(r["fresh_correct"] for r in selected),
                "old_accuracy": sum(r["old_correct"] for r in selected) / len(selected),
                "fresh_accuracy": sum(r["fresh_correct"] for r in selected) / len(selected),
                "answer_disagreements": sum(r["answer_changed"] for r in selected),
                "fixes": stats["improved"], "harms": stats["harmed"],
                "accuracy_difference": stats["mean_difference"], "family_bootstrap_95_ci": stats["parent_bootstrap_95_ci"],
                "old_provider_counts": dict(Counter(r["old_route"]["provider"] for r in selected)),
                "fresh_provider_counts": dict(Counter(r["fresh_route"]["provider"] for r in selected)),
                "provider_transition_counts": [{"old": a, "fresh": b, "judgments": n}
                    for (a, b), n in sorted(Counter((r["old_route"]["provider"], r["fresh_route"]["provider"]) for r in selected).items())],
                "old_missing_fingerprints": sum(not r["old_route"]["fingerprint"] for r in selected),
                "fresh_missing_fingerprints": sum(not r["fresh_route"]["fingerprint"] for r in selected)})
    # Only aggregate fields: safe to publish without prompts, IDs, hashes or paths.
    public = {"schema": "grid4_haiku_ac_replay_public_summary_v1", "model": MODEL,
              "post_result_sensitivity": True, "primary_grid_unchanged": True,
              "selection": plan["selection"], "matching_original_request_hashes": 96,
              "fresh_receipts": 96, "families": 24, "no_cache_reuse": True,
              "generation_id_evidence": generation_id_evidence,
              "local_cap_usd": CAP, "accounted_cost_usd": read(args.out / "fresh/summary.json")["accounted_cost_usd"],
              "original_vs_fresh_AC": aggregates,
              "cached_original_grid": {"accuracy": original_grid[0], "contrasts": original_grid[1]},
              "fresh_AC_with_fixed_BD_sensitivity": {"accuracy": sensitivity_grid[0], "contrasts": sensitivity_grid[1]},
              "limits": plan["limitations"]}
    destination.mkdir()
    atomic_json(destination / "public_summary.json", public, immutable=True)
    atomic_json(destination / "audit.json", {"schema": SCHEMA, "complete": True,
        "plan_sha256": file_hash(args.out / "plan.json"), "fresh_runtime_audit": fresh_audit,
        "fixed_BD_runtime_audit": bd_audit, "matching_original_request_hashes": 96}, immutable=True)
    write_rows(destination / "comparisons.jsonl", comparisons)
    write_rows(destination / "sensitivity_paired_episodes.jsonl", sensitivity_grid[2])
    return {"complete": True, "matching_original_request_hashes": 96,
            "answer_disagreements": sum(r["answer_changed"] for r in comparisons),
            "public_summary": str(destination / "public_summary.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "analyse"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--package", type=Path, default=Path("runs/iteration4/judge_package_v1"))
    parser.add_argument("--frozen-package-sha256")
    parser.add_argument("--prior-results", type=Path, default=Path("runs/iteration3/final/haiku_judge_v1"))
    parser.add_argument("--feedback-results", type=Path, default=Path("runs/iteration4/haiku_judge_v1"))
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    require(args.concurrency in (1, 2, 3, 4), "bounded concurrency 1 through 4 required")
    if args.command == "prepare":
        require(bool(args.frozen_package_sha256), "explicit frozen package hash required")
        result = prepare(args)
    else:
        with run_lock(args.out):
            result = asyncio.run(run(args)) if args.command == "run" else analyse(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
