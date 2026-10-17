"""Paired source-family comparisons of frozen judge assistance effects."""

import argparse
from collections import defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
from types import SimpleNamespace

from .judge3_final import analyse, check_package, file_hash, read_rows
from .judge_assistance import paired_stats


def shared_jobs(first_package, first_jobs, second_package, second_jobs):
    """Permit a frozen condition subset, never changed inputs or episode sets."""
    if first_package["episode_families"] != second_package["episode_families"]:
        raise ValueError("crossjudge packages use different episode-family sets")
    first = {j["job_id"]: j for j in first_jobs}
    second = {j["job_id"]: j for j in second_jobs}
    common = sorted(set(first) & set(second))
    if not common:
        raise ValueError("crossjudge packages have no shared jobs")
    for key in common:
        if first[key] != second[key]:
            raise ValueError("shared judge job changed messages or score provenance")
    jobs = [first[key] for key in common]
    baseline_episodes = {j["episode_id"] for j in jobs if j["kind"] == "none"}
    oracle_episodes = {j["episode_id"] for j in jobs if j["kind"] == "oracle"}
    expected = set(first_package["episode_families"])
    if baseline_episodes != expected or oracle_episodes != expected:
        raise ValueError(
            "every shared episode needs identical baseline and oracle jobs"
        )
    signals = {j["detector"] for j in jobs if j.get("detector")}
    first_signals = {s["detector"]: s for s in first_package["signals"]}
    second_signals = {s["detector"]: s for s in second_package["signals"]}
    if any(first_signals[name] != second_signals[name] for name in signals):
        raise ValueError("shared detector provenance differs across packages")
    for job in jobs:
        if job.get("condition") == "real" and job["baseline_job_id"] not in common:
            raise ValueError("shared real advice is missing its matched neutral")
    return jobs


def interaction_rows(jobs, first, second, families):
    """Second judge's real-minus-neutral uplift minus the first judge's."""
    grouped = defaultdict(list)
    for job in jobs:
        if job.get("condition") != "real":
            continue
        if job.get("baseline_condition") != job["detector"] + ":neutral":
            raise ValueError("interaction requires the matched neutral reference")
        real, neutral = job["job_id"], job["baseline_job_id"]
        first_effect = int(first[real]["correct"]) - int(first[neutral]["correct"])
        second_effect = int(second[real]["correct"]) - int(second[neutral]["correct"])
        for participant in [job["participant_model"], "all"]:
            key = (
                job["environment_protocol"],
                job["split"],
                participant,
                job["detector"],
            )
            grouped[key].append(
                {
                    "group": families[job["episode_id"]],
                    "difference": second_effect - first_effect,
                }
            )
    return grouped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, nargs="+", required=True)
    parser.add_argument("--frozen-package-sha256", nargs="+", required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("immutable new comparison directory required")
    if len(args.results) < 2:
        raise ValueError("at least two separately audited judges required")
    if len(args.package) not in (1, len(args.results)) or len(
        args.frozen_package_sha256
    ) != len(args.package):
        raise ValueError("provide one shared package or one package/hash per result")
    packages = (
        args.package * len(args.results) if len(args.package) == 1 else args.package
    )
    hashes = (
        args.frozen_package_sha256 * len(args.results)
        if len(args.frozen_package_sha256) == 1
        else args.frozen_package_sha256
    )
    judges = []
    args.out.mkdir(parents=True)
    for index, directory in enumerate(args.results):
        package, jobs_path = check_package(packages[index], hashes[index])
        jobs = read_rows(jobs_path)
        destination = args.out / ("judge_" + str(index))
        analyse(
            SimpleNamespace(
                package=packages[index],
                frozen_package_sha256=hashes[index],
                results=directory,
                out=destination,
            )
        )
        audit = json.loads((destination / "receipt_audit.json").read_text())
        name = audit["judge"]
        if name in {j["name"] for j in judges}:
            raise ValueError("judge identities must differ; repeats are not new judges")
        judges.append(
            {
                "name": name,
                "directory": str(directory),
                "evaluation_sha256": file_hash(directory / "evaluations.jsonl"),
                "package_sha256": hashes[index],
                "package_directory": str(packages[index]),
                "package": package,
                "jobs": jobs,
                "rows": {
                    r["job_id"]: r for r in read_rows(directory / "evaluations.jsonl")
                },
                "finish_reasons": audit["finish_reasons"],
            }
        )
    interactions, accuracies, pairing_audits = [], [], []
    for first, second in combinations(judges, 2):
        names = {"judge_first": first["name"], "judge_second": second["name"]}
        jobs = shared_jobs(
            first["package"], first["jobs"], second["package"], second["jobs"]
        )
        families = first["package"]["episode_families"]
        conditions = {j["kind"] for j in jobs}
        pairing_audits.append(
            {
                **names,
                "jobs": len(jobs),
                "episodes": len(families),
                "conditions": sorted(conditions),
                "first_only_conditions": sorted(
                    {j["kind"] for j in first["jobs"]} - conditions
                ),
                "second_only_conditions": sorted(
                    {j["kind"] for j in second["jobs"]} - conditions
                ),
                "shared_jobs_sha256": hashlib.sha256(
                    json.dumps(jobs, sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest(),
            }
        )
        grouped = interaction_rows(jobs, first["rows"], second["rows"], families)
        for key, rows in sorted(grouped.items()):
            stats = paired_stats(rows)
            stats["mean_uplift_difference_second_minus_first"] = stats.pop(
                "mean_difference"
            )
            stats["positive_interaction_episodes"] = stats.pop("improved")
            stats["negative_interaction_episodes"] = stats.pop("harmed")
            interactions.append(
                {
                    **names,
                    "protocol": key[0],
                    "split": key[1],
                    "participant": key[2],
                    "detector": key[3],
                    **stats,
                }
            )
        grouped = defaultdict(list)
        for job in jobs:
            for participant in [job["participant_model"], "all"]:
                key = (
                    job["environment_protocol"],
                    job["split"],
                    participant,
                    job["kind"],
                )
                grouped[key].append(
                    {
                        "group": families[job["episode_id"]],
                        "difference": int(second["rows"][job["job_id"]]["correct"])
                        - int(first["rows"][job["job_id"]]["correct"]),
                    }
                )
        for key, rows in sorted(grouped.items()):
            stats = paired_stats(rows)
            stats["mean_accuracy_difference_second_minus_first"] = stats.pop(
                "mean_difference"
            )
            stats["second_only_correct"] = stats.pop("improved")
            stats["first_only_correct"] = stats.pop("harmed")
            accuracies.append(
                {
                    **names,
                    "protocol": key[0],
                    "split": key[1],
                    "participant": key[2],
                    "condition": key[3],
                    **stats,
                }
            )
    report = {
        "finish_reasons_by_judge": {j["name"]: j["finish_reasons"] for j in judges},
        "package_sha256_by_judge": {j["name"]: j["package_sha256"] for j in judges},
        "judges_in_directional_order": [j["name"] for j in judges],
        "input_results": [
            {k: v for k, v in j.items() if k not in ("rows", "jobs", "package")}
            for j in judges
        ],
        "shared_condition_audits": pairing_audits,
        "independent_unit": "source family; all episode/model/world observations resampled together",
        "estimand": "(second real - second matched neutral) - (first real - first matched neutral)",
        "interpretation": "When the second judge is designated stronger before outcomes, negative values mean smaller observed uplift. Also inspect both judges' raw and neutral accuracy, oracle headroom, and individual fixes/harms. This is a run-specific paired comparison, not a general claim about model size or capability.",
        "uplift_interactions": interactions,
        "paired_accuracy_differences": accuracies,
        "limits": sorted(
            {limit for judge in judges for limit in judge["package"]["limits"]}
            | {
                "These intervals are unadjusted across frozen detector/judge contrasts; no significance-based detector selection.",
                "Reduced judge packages only support crossjudge claims for their exact shared conditions, selected before fresh outcomes.",
            }
        ),
    }
    (args.out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "finish_reasons_by_judge": report["finish_reasons_by_judge"],
                "judge_pairs": len(list(combinations(judges, 2))),
                "uplift_contrasts": len(interactions),
                "accuracy_contrasts": len(accuracies),
                "complete_report": str(args.out / "comparison.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
