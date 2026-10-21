"""Frozen full-source/no-source controls on both answer orders per family."""

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path

from .api import ANSWER, BudgetedChat, object_schema
from .build_native_judge_jobs import OUTPUT
from .judge_assistance import paired_stats
from .judge3_final import audit_completion
from .preflight import load_problems, sha256


def build(args):
    if args.out.exists():
        raise ValueError("new immutable output required")
    problems, audit = load_problems(args.dataset)
    source = list(map(json.loads, args.dataset.read_text().splitlines()))
    lock = json.loads(args.source_lock.read_text())
    if audit["errors"] or sha256(args.dataset.read_text()) != lock["sha256"]:
        raise ValueError("dataset differs from source freeze")
    if len({r["family_id"] for r in source}) != len(source):
        raise ValueError("one question per independent family required")
    jobs = []
    for problem, row in zip(problems, source):
        for swapped in (False, True):
            for with_story in (False, True):
                messages, gold = problem.view(with_story, swapped, structured=True)
                messages[0]["content"] = (
                    "Answer the reading-comprehension question. If no story is supplied, make your best choice from the question and options alone."
                    + OUTPUT
                )
                jobs.append(
                    {
                        "job_id": sha256(
                            problem.problem_id + str(swapped) + str(with_story)
                        ),
                        "kind": "full_story" if with_story else "no_story",
                        "group": row["family_id"],
                        "parent_problem_id": row["parent_problem_id"],
                        "parent_source_line": row["parent_source_line"],
                        "split": "test",
                        "swapped": swapped,
                        "gold": gold,
                        "messages": messages,
                        "short_options": max(
                            map(lambda x: len(x.split()), problem.options)
                        )
                        <= 6,
                    }
                )
    args.out.mkdir(parents=True)
    path = args.out / "jobs.jsonl"
    path.write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in jobs))
    manifest = {
        "jobs": len(jobs),
        "families": len(source),
        "jobs_sha256": sha256(path.read_text()),
        "source_lock_sha256": sha256(args.source_lock.read_text()),
        "dataset_sha256": lock["sha256"],
        "answer_orders": 2,
        "constructed_chance": False,
        "subgroup_rule": "Both options at most six whitespace-separated words; fixed before outcomes.",
        "runner_sha256": sha256(Path(__file__).read_text()),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


async def run(args):
    freeze = json.loads(args.global_freeze.read_text())
    if freeze.get("status") != "frozen_before_fresh_outcomes":
        raise ValueError("global pre-outcome freeze required")
    jobs_path = args.package / "jobs.jsonl"
    manifest = json.loads((args.package / "manifest.json").read_text())
    if manifest["jobs_sha256"] != sha256(jobs_path.read_text()):
        raise ValueError("control job package changed")
    if (
        freeze["control_jobs_sha256"] != manifest["jobs_sha256"]
        or freeze["source_dataset_sha256"] != manifest["dataset_sha256"]
    ):
        raise ValueError("control package differs from global freeze")
    jobs = list(map(json.loads, jobs_path.read_text().splitlines()))
    async with BudgetedChat(
        args.out,
        args.models,
        args.budget,
        schema_retries=2,
        reuse_receipts=args.reuse_receipts,
        max_tokens=args.max_output_tokens,
        reasoning=(
            {"enabled": False}
            if args.reasoning_effort == "disabled"
            else {"enabled": True, "effort": args.reasoning_effort}
        ),
        request_timeout_seconds=args.request_timeout_seconds,
        concurrency=args.concurrency,
    ) as api:
        results = []

        async def one(model, job):
            metadata = {k: v for k, v in job.items() if k != "messages"}
            receipt = await api.complete(
                model, job["messages"], object_schema({"answer": ANSWER}), metadata
            )
            audit_completion(receipt)
            row = {
                **metadata,
                "model": model,
                "parsed": receipt["parsed"],
                "valid": receipt["valid"],
                "finish_reason": receipt["finish_reason"],
                "correct": receipt["parsed"]["answer"] == job["gold"],
                "request_sha256": receipt["request_sha256"],
            }
            results.append(row)
            with (args.out / "evaluations.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")

        outcomes = await asyncio.gather(
            *(one(m, j) for m in args.models for j in jobs), return_exceptions=True
        )
        errors = [str(o) for o in outcomes if isinstance(o, BaseException)]
        summary = {
            "finish_reasons": dict(
                Counter(r.get("finish_reason", "error") for r in api.records)
            ),
            "expected": len(jobs) * len(args.models),
            "completed": len(results),
            "errors": errors,
            "accounted_cost_upper_usd": api.spent,
            "package_sha256": sha256((args.package / "manifest.json").read_text()),
            "freeze_sha256": sha256(args.global_freeze.read_text()),
        }
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        if errors or summary["expected"] != summary["completed"]:
            raise RuntimeError("incomplete controls; no accuracy aggregate")


def analyse(args):
    job_text = (args.package / "jobs.jsonl").read_text()
    jobs = list(map(json.loads, job_text.splitlines()))
    rows = list(map(json.loads, args.evaluations.read_text().splitlines()))
    if not rows:
        raise ValueError("empty control results")
    by_id = {j["job_id"]: j for j in jobs}
    if len(by_id) != len(jobs):
        raise ValueError("duplicate control job")
    if all("model" in r for r in rows):
        receipts = {
            (r["request_sha256"], r["tag"]["job_id"]): r
            for r in map(
                json.loads,
                args.evaluations.with_name("requests.jsonl").read_text().splitlines(),
            )
            if r.get("valid")
        }
        for row in rows:
            receipt = receipts[(row["request_sha256"], row["job_id"])]
            audit_completion(receipt)
            if (
                receipt["request"]["messages"] != by_id[row["job_id"]]["messages"]
                or receipt["model"] != row["model"]
                or receipt["parsed"] != row["parsed"]
                or receipt["finish_reason"] != "stop"
            ):
                raise ValueError("control outcome differs from complete API receipt")
        runtime = json.loads(
            args.evaluations.with_name("api_manifest.json").read_text()
        )
        if {r["model"] for r in rows} != set(runtime["models"]):
            raise ValueError("missing or unexpected control model")
        models = sorted(runtime["models"])
        native_model = None
    else:
        if any("model" in r for r in rows):
            raise ValueError("mixed API and native control results")
        runtime = json.loads(args.evaluations.with_name("manifest.json").read_text())
        if (
            runtime["jobs_sha256"] != sha256(job_text)
            or not runtime["contract"]["answer_only"]
        ):
            raise ValueError("native controls ran a different package or answer format")
        native_model = runtime["contract"]["model"]
        models = [native_model]
    cells = []
    for model in models:
        selected = [r for r in rows if r.get("model", native_model) == model]
        if len(selected) != len(jobs) or {r["job_id"] for r in selected} != set(by_id):
            raise ValueError("exact control coverage required")
        for r in selected:
            j = by_id[r["job_id"]]
            if not r["valid"] or r["finish_reason"] != "stop" or r["gold"] != j["gold"]:
                raise ValueError("failed control receipt")
            if any(r[k] != j[k] for k in ("kind", "group", "swapped", "short_options")):
                raise ValueError("control result metadata differs from job")
            if r["correct"] != (r["parsed"]["answer"] == j["gold"]):
                raise ValueError("incorrect control answer mapping")
            if native_model is not None and (
                r["completion_tokens"] >= runtime["contract"]["max_new_tokens"]
                or any(r.get(k) for k in ("forced", "truncated", "forced_final"))
            ):
                raise ValueError("incomplete native control generation")
        for kind in ("no_story", "full_story"):
            for subgroup in ("all", "short", "descriptive"):
                cell = [
                    r
                    for r in selected
                    if r["kind"] == kind
                    and (
                        subgroup == "all"
                        or by_id[r["job_id"]]["short_options"] == (subgroup == "short")
                    )
                ]
                if not cell:
                    continue
                stats = paired_stats(
                    [
                        {"group": r["group"], "difference": int(r["correct"])}
                        for r in cell
                    ]
                )
                cells.append(
                    {
                        "model": model,
                        "kind": kind,
                        "subgroup": subgroup,
                        "accuracy": stats.pop("mean_difference"),
                        **stats,
                    }
                )
    result = {
        "finish_reasons": dict(Counter(r["finish_reason"] for r in rows)),
        "cells": cells,
        "runtime": runtime,
        "limit": "Both answer orders share a family. Family-bootstrap intervals can degenerate in all-correct samples; no claim of perfect population accuracy. No-story semantics are not neutralized by order swapping.",
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "runtime"}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="action", required=True)
    b = subs.add_parser("build")
    b.add_argument("--dataset", type=Path, required=True)
    b.add_argument("--source-lock", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    r = subs.add_parser("run-api")
    r.add_argument("--package", type=Path, required=True)
    r.add_argument("--global-freeze", type=Path, required=True)
    r.add_argument("--models", nargs="+", required=True)
    r.add_argument("--budget", type=float, required=True)
    r.add_argument(
        "--reasoning-effort", choices=["disabled", "low"], default="disabled"
    )
    r.add_argument("--max-output-tokens", type=int, default=8192)
    r.add_argument("--request-timeout-seconds", type=float, default=300)
    r.add_argument("--concurrency", type=int, default=4)
    r.add_argument("--reuse-receipts", type=Path, nargs="*", default=[])
    r.add_argument("--out", type=Path, required=True)
    a = subs.add_parser("analyse")
    a.add_argument("--package", type=Path, required=True)
    a.add_argument("--evaluations", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "run-api":
        asyncio.run(run(args))
    else:
        {"build": build, "analyse": analyse}[args.action](args)


if __name__ == "__main__":
    main()
