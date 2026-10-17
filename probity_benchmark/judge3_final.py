"""Package frozen judge jobs and run/audit one judge without changing displays.

Preparation is local and free. Root releases paid execution by supplying the
reviewed package hash; the same jobs.jsonl is used by the native judge runner.
"""

import argparse
import asyncio
from collections import Counter, defaultdict
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

from .api import ANSWER, BudgetedChat, object_schema
from . import judge3_protocol as protocol


def read_rows(path):
    raw = Path(path).read_bytes()
    if len(raw) != Path(path).stat().st_size:
        raise ValueError("incomplete input read")
    return [json.loads(line) for line in raw.decode().splitlines() if line.strip()]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prompt_repeatability(jobs, rows, receipts_by_job=None):
    groups = defaultdict(list)
    for row in rows:
        job = jobs[row["job_id"]]
        key = hashlib.sha256(
            json.dumps(job["messages"], sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        observation = {
            "job_id": row["job_id"],
            "kind": job["kind"],
            "answer": row["answer"],
        }
        if receipts_by_job is not None:
            receipt = receipts_by_job[row["job_id"]]
            payload = json.loads(receipt["response_body"])
            observation.update(
                request_sha256=receipt["request_sha256"],
                provider_as_reported=payload.get("provider"),
                model_as_reported=payload.get("model"),
                system_fingerprint_as_reported=payload.get("system_fingerprint"),
            )
        groups[key].append(observation)
    repeated = []
    matching_pairs = total_pairs = 0
    for key, group in sorted(groups.items()):
        if len(group) < 2:
            continue
        counts = Counter(r["answer"] for r in group)
        pairs = len(group) * (len(group) - 1) // 2
        matches = sum(n * (n - 1) // 2 for n in counts.values())
        total_pairs += pairs
        matching_pairs += matches
        repeated.append(
            {
                "messages_sha256": key,
                "jobs": group,
                "answer_counts": dict(counts),
                "answer_disagreement": len(counts) > 1,
                "complete_request_hashes": sorted(
                    {r["request_sha256"] for r in group if r.get("request_sha256")}
                ),
            }
        )
    return {
        "repeated_prompt_groups": len(repeated),
        "logical_jobs_in_repeated_groups": sum(len(g["jobs"]) for g in repeated),
        "groups_with_answer_disagreement": sum(
            g["answer_disagreement"] for g in repeated
        ),
        "matching_answer_pairs": matching_pairs,
        "all_answer_pairs": total_pairs,
        "pairwise_agreement": matching_pairs / total_pairs if total_pairs else None,
        "groups": repeated,
        "scope": "Descriptive agreement across byte-identical message lists within one audited judge runtime. Complete request hashes and available provider metadata are reported separately; equal requests do not establish equal provider routing or hardware. Missing provider metadata is unknown. These naturally repeated neutral presentations are not extra independent families or a separate repeated-trial experiment.",
    }


def audit_completion(receipt):
    """Reject exposed truncation even when the normalized finish reason is stop."""
    if not receipt.get("valid") or receipt.get("finish_reason") != "stop":
        raise ValueError("judge receipt is incomplete")
    payload = json.loads(receipt["response_body"])
    flags = []

    def visit(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                here = path + [key]
                if key in ("forced", "truncated", "forced_final") and item:
                    flags.append({"path": here, "value": item})
                if key in (
                    "native_finish_reason",
                    "stop_reason",
                    "finish_reason",
                ) and item in (
                    "length",
                    "max_tokens",
                    "max_output_tokens",
                    "truncated",
                ):
                    flags.append({"path": here, "value": item})
                visit(item, here)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, path + [index])

    visit(payload, [])
    if flags:
        raise ValueError(
            "provider exposes truncated judge output: " + json.dumps(flags)
        )
    usage = receipt["usage"]
    cap = receipt["request"]["max_tokens"]
    reasoning_tokens = usage.get("completion_tokens_details", {}).get(
        "reasoning_tokens"
    )
    if usage["completion_tokens"] >= cap or (reasoning_tokens or 0) >= cap:
        raise ValueError("judge completion reached its allowance")
    return {
        "native_finish_reason": payload["choices"][0].get("native_finish_reason"),
        "completion_tokens": usage["completion_tokens"],
        "reasoning_tokens": reasoning_tokens,
        "max_tokens": cap,
        "exposed_truncation_flags": flags,
    }


def selected_episodes(sources, split):
    episodes = {}
    for source in sources:
        for path in sorted(Path(source["episodes"]).glob("episode_*.json")):
            episode = json.loads(path.read_text())
            if episode["split"] != split:
                continue
            if (
                source.get("parent_source_lines")
                and episode["parent_source_line"] not in source["parent_source_lines"]
            ):
                continue
            if episode["episode_id"] in episodes:
                raise ValueError("duplicate selected episode")
            episodes[episode["episode_id"]] = episode
    return episodes


def check_families(episodes, index, expected_episodes, expected_families, models):
    if len(episodes) != expected_episodes:
        raise ValueError("selected episode count differs from frozen design")
    if {e["model"] for e in episodes.values()} != set(models):
        raise ValueError("participant models differ from frozen design")
    family_parents = defaultdict(set)
    counts = defaultdict(Counter)
    mapping = {}
    for eid, episode in episodes.items():
        parent = episode["parent_problem_id"]
        family = index["by_parent"][parent]
        family_parents[family].add(parent)
        counts[family][episode["model"]] += 1
        mapping[eid] = family
    if len(family_parents) != expected_families:
        raise ValueError("source family count differs from frozen design")
    if any(len(parents) != 1 for parents in family_parents.values()):
        raise ValueError("primary design requires one question per source family")
    if len({tuple(sorted(c.items())) for c in counts.values()}) != 1:
        raise ValueError("participant coverage differs across source families")
    return mapping


def check_signal(spec, bundle, episodes, index, episode_families):
    for field in ["head_sha256", "reference_sha256"]:
        if bundle[field] != spec[field]:
            raise ValueError("score bundle differs from frozen " + field)
    if spec.get("reference_artifact"):
        artifact_path = Path(spec["reference_artifact"])
        if file_hash(artifact_path) != spec["reference_artifact_sha256"]:
            raise ValueError("frozen reference artifact changed")
        artifact = json.loads(artifact_path.read_text())
        for field in [
            "detector",
            "target",
            "granularity",
            "orientation",
            "head_sha256",
            "reference_sha256",
            "references",
        ]:
            if artifact[field] != bundle[field]:
                raise ValueError("score bundle differs from frozen reference " + field)
    models = set(spec["participant_models"])
    if not models or not models <= {e["model"] for e in episodes.values()}:
        raise ValueError("invalid declared detector model coverage")
    _, covered = protocol.validate_bundle(bundle, episodes)
    expected = {eid for eid, e in episodes.items() if e["model"] in models}
    if covered != expected:
        raise ValueError("detector does not cover every episode in its declared models")
    if set(bundle["references"]) != models:
        raise ValueError("detector references differ from declared model coverage")
    final_families = set(episode_families.values())
    reference_families = set()
    for model in models:
        for parent in bundle["references"][model]["parents"]:
            reference_families.add(index["by_parent"][parent])
    if final_families & reference_families:
        raise ValueError("source family leaked into detector normalization")
    return {
        "detector": bundle["detector"],
        "participant_models": sorted(models),
        "episodes": len(covered),
        "turns": len(bundle["rows"]),
        "units": sum(len(r["scores"]) for r in bundle["rows"]),
        "granularity": bundle["granularity"],
        "target": bundle["target"],
        "head_sha256": bundle["head_sha256"],
        "reference_sha256": bundle["reference_sha256"],
        "reference_artifact_sha256": spec.get("reference_artifact_sha256"),
        "reference_families": sorted(reference_families),
        "reference_population": bundle.get("reference_population"),
        "limitation": bundle.get("limitation"),
    }


def prepare(args):
    if args.out.exists():
        raise ValueError("immutable new package directory required")
    design = json.loads(args.design.read_text())
    if design["schema"] != "judge3_final_design_v1" or design["split"] != "test":
        raise ValueError("explicit final test design required")
    locked = json.loads(Path(design["protocol_lock"]).read_text())
    for path, expected in locked["files"].items():
        if file_hash(path) != expected:
            raise ValueError("frozen judge protocol changed")
    sources = json.loads(Path(design["sources"]).read_text())
    episodes = selected_episodes(sources, design["split"])
    index = json.loads(Path(design["family_index"]).read_text())
    families = check_families(
        episodes,
        index,
        design["expected_episodes"],
        design["expected_families"],
        design["participant_models"],
    )
    signals = []
    for spec in design["score_bundles"]:
        bundle = json.loads(Path(spec["path"]).read_text())
        signals.append(check_signal(spec, bundle, episodes, index, families))
    if not signals:
        raise ValueError("final assistance package requires frozen signals")
    inputs = {
        str(args.design.resolve()): file_hash(args.design),
        **{
            str(Path(design[key]).resolve()): file_hash(design[key])
            for key in ["protocol_lock", "sources", "family_index"]
        },
    }
    for spec in design["score_bundles"]:
        inputs[str(Path(spec["path"]).resolve())] = file_hash(spec["path"])
        if spec.get("reference_artifact"):
            reference = spec["reference_artifact"]
            inputs[str(Path(reference).resolve())] = file_hash(reference)
    for path in design["validity_artifacts"]:
        inputs[str(Path(path).resolve())] = file_hash(path)
    if not design["validity_artifacts"]:
        raise ValueError("source, knowledge and prose validity evidence required")
    protocol.build(
        SimpleNamespace(
            sources=Path(design["sources"]),
            splits=[design["split"]],
            out=args.out / "compiled",
            score_bundles=[Path(s["path"]) for s in design["score_bundles"]],
            antioracle=False,
        )
    )
    jobs_path = args.out / "compiled" / "jobs.jsonl"
    jobs = read_rows(jobs_path)
    expected_jobs = 2 * len(episodes) + 3 * sum(s["episodes"] for s in signals)
    if len(jobs) != expected_jobs:
        raise ValueError("compiled condition coverage differs from frozen design")
    manifest = {
        "schema": "judge3_final_package_v1",
        "jobs_path": "compiled/jobs.jsonl",
        "jobs_sha256": file_hash(jobs_path),
        "compiled_manifest_sha256": file_hash(args.out / "compiled/manifest.json"),
        "episodes": len(episodes),
        "families": len(set(families.values())),
        "episode_families": families,
        "jobs": len(jobs),
        "conditions": dict(Counter(j["kind"] for j in jobs)),
        "signals": signals,
        "input_hashes": inputs,
        "runtime_source_hashes": {
            str(Path(path).resolve()): file_hash(path)
            for path in [
                __file__,
                "probity_benchmark/api.py",
                "probity_benchmark/judge3_protocol.py",
                "probity_benchmark/run_local_judge.py",
            ]
        },
        "validity_note": "Evidence artifacts are bound by hash; their semantic review remains explicit in root's source/protocol lock. This wrapper does not infer source truth from answer-field adherence.",
        "primary": "Real minus matched neutral display, paired within episode and resampled by source family. All primary questions are in different families.",
        "limits": [
            "Judge API and native runtimes differ even with identical message text.",
            "A perfect sample has a degenerate bootstrap interval, not population certainty.",
            "Real minus shuffled can reflect damage from shuffled advice; show shuffled minus neutral and fixes/harms.",
            "No final outcome-dependent exclusions or detector selection.",
        ],
    }
    path = args.out / "package.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "package": str(path),
                "package_sha256": file_hash(path),
                "jobs": len(jobs),
                "episodes": len(episodes),
                "families": manifest["families"],
                "conditions": manifest["conditions"],
                "execution": "Prepared only; no API or native outcome calls made.",
            },
            indent=2,
        )
    )


def check_package(directory, expected_hash, check_runtime=False):
    path = directory / "package.json"
    if file_hash(path) != expected_hash:
        raise ValueError("package hash differs from the reviewed release")
    manifest = json.loads(path.read_text())
    jobs_path = directory / manifest["jobs_path"]
    if file_hash(jobs_path) != manifest["jobs_sha256"]:
        raise ValueError("frozen judge jobs changed")
    if (
        file_hash(directory / "compiled/manifest.json")
        != manifest["compiled_manifest_sha256"]
    ):
        raise ValueError("compiled shuffle/coverage manifest changed")
    if check_runtime:
        for filename, expected in manifest["runtime_source_hashes"].items():
            if file_hash(filename) != expected:
                raise ValueError("runner source changed after package freeze")
    return manifest, jobs_path


async def run_api(args):
    manifest, jobs_path = check_package(args.package, args.frozen_package_sha256, True)
    if not os.environ.get("PROBITY_PILOT_LEDGER"):
        raise ValueError("shared pilot budget ledger is mandatory")
    jobs = read_rows(jobs_path)
    results = []
    reasoning = (
        {"effort": args.reasoning_effort}
        if args.reasoning_effort
        else {"enabled": False}
    )
    async with BudgetedChat(
        args.out,
        [args.judge],
        args.budget,
        schema_retries=2,
        concurrency=args.concurrency,
        reuse_receipts=args.reuse_receipts,
        max_tokens=args.max_output_tokens,
        reasoning=reasoning,
        request_timeout_seconds=args.request_timeout_seconds,
    ) as api:
        if api.models[args.judge].get("reasoning", {}).get(
            "mandatory"
        ) and reasoning == {"enabled": False}:
            raise ValueError("judge requires an explicit supported reasoning effort")
        (args.out / "manifest.json").write_text(
            json.dumps(
                {
                    "package_sha256": args.frozen_package_sha256,
                    "jobs_sha256": manifest["jobs_sha256"],
                    "judge_model": args.judge,
                    "jobs_expected": len(jobs),
                    "repeats": 1,
                    "source_sha256": file_hash(__file__),
                    "runtime": {
                        "reasoning": reasoning,
                        "max_tokens": args.max_output_tokens,
                        "request_timeout_seconds": args.request_timeout_seconds,
                    },
                },
                indent=2,
            )
            + "\n"
        )

        async def one(job):
            receipt = await api.complete(
                args.judge,
                job["messages"],
                object_schema({"answer": ANSWER}),
                {"job": job["job_id"], "kind": job["kind"]},
            )
            try:
                completion = audit_completion(receipt)
            except ValueError:
                api.stopped = True
                raise
            answer = receipt["parsed"]["answer"]
            row = {
                **{k: v for k, v in job.items() if k != "messages"},
                "judge_model": args.judge,
                "answer": answer,
                "correct": answer == job["gold"],
                "request_sha256": receipt["request_sha256"],
                "valid": receipt["valid"],
                "finish_reason": receipt["finish_reason"],
                "completion_audit": completion,
            }
            results.append(row)
            with (args.out / "evaluations.jsonl").open("a") as file:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

        outcomes = await asyncio.gather(*(one(j) for j in jobs), return_exceptions=True)
        summary = {
            "finish_reasons": dict(
                Counter(r.get("finish_reason") for r in api.records)
            ),
            "expected": len(jobs),
            "completed": len(results),
            "errors": [str(o) for o in outcomes if isinstance(o, BaseException)],
            "cost_upper_usd": api.spent,
        }
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        if summary["errors"] or len(results) != len(jobs):
            raise RuntimeError("incomplete final run; retained receipts, no aggregate")
    analyse(
        SimpleNamespace(
            package=args.package,
            frozen_package_sha256=args.frozen_package_sha256,
            results=args.out,
            out=args.out / "analysis",
        )
    )


def analyse(args):
    package, jobs_path = check_package(args.package, args.frozen_package_sha256)
    if args.out.exists():
        raise ValueError("immutable new analysis directory required")
    run_manifest = json.loads((args.results / "manifest.json").read_text())
    if run_manifest["jobs_sha256"] != package["jobs_sha256"]:
        raise ValueError("judge run used a different jobs file")
    jobs = {j["job_id"]: j for j in read_rows(jobs_path)}
    rows = read_rows(args.results / "evaluations.jsonl")
    receipts_by_job = None
    if (args.results / "requests.jsonl").exists():
        records = read_rows(args.results / "requests.jsonl")
        valid_receipts = {}
        for receipt in records:
            if receipt.get("valid"):
                key = (receipt["request_sha256"], receipt["tag"]["job"])
                if key in valid_receipts:
                    raise ValueError("multiple valid receipts for one judge job")
                valid_receipts[key] = receipt
        receipts_by_job = {}
        for row in rows:
            receipt = valid_receipts[(row["request_sha256"], row["job_id"])]
            audit_completion(receipt)
            job = jobs[row["job_id"]]
            if (
                receipt["request"]["messages"] != job["messages"]
                or hashlib.sha256(
                    json.dumps(receipt["request"], sort_keys=True).encode()
                ).hexdigest()
                != receipt["request_sha256"]
                or receipt["finish_reason"] != "stop"
                or receipt["parsed"]["answer"] != row["answer"]
                or receipt["model"] != run_manifest["judge_model"]
                or receipt["tag"]["kind"] != job["kind"]
            ):
                raise ValueError("API receipt differs from the frozen evaluation")
            runtime = run_manifest.get("runtime")
            if runtime and any(
                receipt["request"][key] != runtime[key]
                for key in ["reasoning", "max_tokens"]
            ):
                raise ValueError("API receipt differs from declared judge runtime")
            receipts_by_job[row["job_id"]] = receipt
        judge = run_manifest["judge_model"]
    else:
        records = rows
        if not run_manifest["contract"]["answer_only"]:
            raise ValueError("native judge must use the frozen answer-only format")
        judge = run_manifest["contract"]["model"]
    args.out.mkdir(parents=True)
    with contextlib.redirect_stdout(io.StringIO()):
        protocol.analyse(
            SimpleNamespace(
                jobs=jobs_path,
                evaluations=args.results / "evaluations.jsonl",
                out=args.out / "analysis.json",
            )
        )
    report = json.loads((args.out / "analysis.json").read_text())
    repeatability = prompt_repeatability(jobs, rows, receipts_by_job)
    (args.out / "prompt_repeatability.json").write_text(
        json.dumps(repeatability, indent=2) + "\n"
    )
    audit = {
        "finish_reasons": dict(Counter(r.get("finish_reason") for r in records)),
        "judge": judge,
        "expected": len(jobs),
        "completed": len(rows),
        "families": package["families"],
        "package_sha256": args.frozen_package_sha256,
        "evaluations_sha256": file_hash(args.results / "evaluations.jsonl"),
        "run_manifest_sha256": file_hash(args.results / "manifest.json"),
        "prompt_repeatability": {
            k: v for k, v in repeatability.items() if k != "groups"
        },
        "claim_limits": package["limits"],
        "primary_comparisons": [
            row
            for row in report["comparisons"]
            if row["condition"].endswith(":real")
            and row["reference"] == row["condition"].removesuffix(":real") + ":neutral"
        ],
    }
    (args.out / "receipt_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--design", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run-api")
    run.add_argument("--judge", default="anthropic/claude-haiku-5.5")
    run.add_argument("--budget", type=float, required=True)
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--reuse-receipts", type=Path, nargs="*", default=[])
    run.add_argument(
        "--reasoning-effort", choices=["low", "medium", "high", "xhigh", "max"]
    )
    run.add_argument("--max-output-tokens", type=int, default=8192)
    run.add_argument("--request-timeout-seconds", type=float, default=300)
    analysis = sub.add_parser("analyse")
    analysis.add_argument("--results", type=Path, required=True)
    for command in [run, analysis]:
        command.add_argument("--package", type=Path, required=True)
        command.add_argument("--frozen-package-sha256", required=True)
        command.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run-api":
        asyncio.run(run_api(args))
    else:
        analyse(args)


if __name__ == "__main__":
    main()
