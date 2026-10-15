"""Export complete audited final judge analyses without rerunning inference or statistics."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from .judge3_final import check_package


def file_hash(path):
    raw = Path(path).read_bytes()
    if len(raw) != Path(path).stat().st_size:
        raise ValueError("incomplete artifact read")
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    raw = Path(path).read_bytes()
    if len(raw) != Path(path).stat().st_size:
        raise ValueError("incomplete JSON read")
    return json.loads(raw)


def condition_parts(value):
    return value.rsplit(":", 1) if ":" in value else ("", value)


def project_analysis(report, audit, repeatability, generation=None):
    """Copy reported estimates and denominators; calculate no new uncertainty."""
    judge = audit["judge"]
    if (
        report["expected"] != report["completed"]
        or audit["expected"] != audit["completed"]
    ):
        raise ValueError("complete audited judge results required")
    if report["completed"] != audit["completed"]:
        raise ValueError("analysis and receipt audit coverage differ")
    if report["finish_reasons"] != {"stop": report["completed"]}:
        raise ValueError("all evaluated generations must complete normally")
    primary = [
        r
        for r in report["comparisons"]
        if r["condition"].endswith(":real")
        and r["reference"] == r["condition"].removesuffix(":real") + ":neutral"
    ]
    if primary != audit["primary_comparisons"]:
        raise ValueError("paired effects differ from the receipt audit")
    if audit["prompt_repeatability"] != {
        k: v for k, v in repeatability.items() if k != "groups"
    }:
        raise ValueError("repeatability differs from the receipt audit")
    generation_groups = {}
    if generation is not None:
        if generation["legacy_logical_summary"] != audit["prompt_repeatability"]:
            raise ValueError("generation audit differs from frozen logical agreement")
        generation_groups = {g["messages_sha256"]: g for g in generation["groups"]}
        if set(generation_groups) != {
            g["messages_sha256"] for g in repeatability["groups"]
        }:
            raise ValueError("generation audit has different repeated message groups")
    accuracy_rows, cells = [], {}
    for row in report["cells"]:
        identity = tuple(
            row[k] for k in ("protocol", "split", "participant", "condition")
        )
        if identity in cells:
            raise ValueError("duplicate accuracy cell")
        cells[identity] = row
        detector, condition = condition_parts(row["condition"])
        accuracy_rows.append(
            {
                "judge": judge,
                "protocol": row["protocol"],
                "split": row["split"],
                "participant": row["participant"],
                "detector": detector,
                "condition": condition,
                "condition_id": row["condition"],
                "n_presentations": row["episodes"],
                "n_families": row["parent_groups"],
                "n_correct": row["correct"],
                "accuracy": row["accuracy"],
                "family_bootstrap_ci_low": row["parent_bootstrap_95_ci"][0],
                "family_bootstrap_ci_high": row["parent_bootstrap_95_ci"][1],
            }
        )
    contrast_rows, primary_rows = [], []
    for row in report["comparisons"]:
        detector, condition = condition_parts(row["condition"])
        common = tuple(row[k] for k in ("protocol", "split", "participant"))
        numerator = cells[(*common, row["condition"])]
        reference = cells[(*common, row["reference"])]
        contrast = {
            "judge": judge,
            "protocol": row["protocol"],
            "split": row["split"],
            "participant": row["participant"],
            "detector": detector,
            "condition": condition,
            "condition_id": row["condition"],
            "reference_condition_id": row["reference"],
            "n_presentations": row["episodes"],
            "n_families": row["parent_groups"],
            "mean_difference": row["mean_difference"],
            "family_bootstrap_ci_low": row["parent_bootstrap_95_ci"][0],
            "family_bootstrap_ci_high": row["parent_bootstrap_95_ci"][1],
            "fixes": row["improved"],
            "harms": row["harmed"],
        }
        contrast_rows.append(contrast)
        if row in primary:
            if (numerator["episodes"], numerator["parent_groups"]) != (
                row["episodes"],
                row["parent_groups"],
            ) or (reference["episodes"], reference["parent_groups"]) != (
                row["episodes"],
                row["parent_groups"],
            ):
                raise ValueError("matched neutral denominator differs")
            if not math.isclose(
                numerator["accuracy"] - reference["accuracy"],
                row["mean_difference"],
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "paired effect does not match the balanced accuracy cells"
                )
            primary_rows.append(
                {
                    **contrast,
                    "real_accuracy": numerator["accuracy"],
                    "neutral_accuracy": reference["accuracy"],
                    "prespecified_primary_signal": detector
                    in {"adapted_primary", "local3_sentence"},
                }
            )
    groups = []
    for group in repeatability["groups"]:
        answers = Counter(job["answer"] for job in group["jobs"])
        if (
            dict(answers) != group["answer_counts"]
            or (len(answers) > 1) != group["answer_disagreement"]
        ):
            raise ValueError("repeatability group is inconsistent")
        api_metadata = any("request_sha256" in job for job in group["jobs"])
        reported = lambda field: json.dumps(
            sorted(
                {job.get(field) for job in group["jobs"]},
                key=lambda value: (value is not None, str(value)),
            ),
            ensure_ascii=False,
        )
        groups.append(
            {
                "judge": judge,
                "messages_sha256": group["messages_sha256"],
                "n_logical_jobs": len(group["jobs"]),
                "distinct_answers": len(answers),
                "logical_answer_counts_json": json.dumps(
                    group["answer_counts"], sort_keys=True
                ),
                "logical_answer_disagreement": group["answer_disagreement"],
                "condition_ids_json": json.dumps(
                    sorted({job["kind"] for job in group["jobs"]})
                ),
                "api_request_metadata_present": api_metadata,
                "complete_request_hash_count": len(group["complete_request_hashes"]),
                "complete_request_hashes_json": json.dumps(
                    group["complete_request_hashes"]
                ),
                "providers_as_reported_json": (
                    reported("provider_as_reported") if api_metadata else ""
                ),
                "models_as_reported_json": (
                    reported("model_as_reported") if api_metadata else ""
                ),
                "fingerprints_as_reported_json": (
                    reported("system_fingerprint_as_reported") if api_metadata else ""
                ),
                **{
                    key: generation_groups.get(group["messages_sha256"], {}).get(key)
                    for key in (
                        "fresh_generations_in_run",
                        "reused_logical_jobs",
                        "generation_receipts_represented",
                        "resolved_generation_observations",
                        "generation_answer_disagreement",
                        "generation_answer_pairs",
                        "generation_matching_answer_pairs",
                        "generation_pairwise_agreement",
                    )
                },
            }
        )
    if len(groups) != repeatability["repeated_prompt_groups"]:
        raise ValueError("repeatability group count differs")
    generation_summary = (
        generation["summary"]
        if generation is not None
        else {
            "measurement_status": "generation_provenance_not_supplied_unmeasured",
            "runtime_kind": "unknown",
            "logical_jobs_total": report["completed"],
            **{
                key: None
                for key in (
                    "fresh_generations_in_run",
                    "reused_logical_jobs",
                    "generation_receipts_represented",
                    "unresolved_native_cached_origins",
                    "generation_repeated_prompt_groups",
                    "generation_groups_with_answer_disagreement",
                    "generation_matching_answer_pairs",
                    "generation_all_answer_pairs",
                    "generation_pairwise_agreement",
                )
            },
        }
    )
    noise = {
        "judge": judge,
        **generation_summary,
        "logical_repeated_prompt_groups": repeatability["repeated_prompt_groups"],
        "logical_jobs_in_repeated_groups": repeatability[
            "logical_jobs_in_repeated_groups"
        ],
        "logical_groups_with_answer_disagreement": repeatability[
            "groups_with_answer_disagreement"
        ],
        "logical_matching_answer_pairs": repeatability["matching_answer_pairs"],
        "logical_all_answer_pairs": repeatability["all_answer_pairs"],
        "logical_pairwise_agreement": repeatability["pairwise_agreement"],
        "independent_evaluation_families": audit["families"],
    }
    return {
        "accuracies": accuracy_rows,
        "all_paired_contrasts": contrast_rows,
        "real_minus_neutral": primary_rows,
        "exact_prompt_summary": [noise],
        "exact_prompt_groups": groups,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("new immutable export directory required")
    spec = read_json(args.spec)
    if spec.get("schema") != "judge3_table_export_spec_v1" or not spec.get("runs"):
        raise ValueError("explicit audited-run export specification required")
    inputs = {str(args.spec.resolve()): file_hash(args.spec)}
    tables, judges = {}, set()
    for entry in spec["runs"]:
        package, _ = check_package(Path(entry["package"]), entry["package_sha256"])
        directory, results = Path(entry["analysis"]), Path(entry["results"])
        report_path = directory / "analysis.json"
        audit_path = directory / "receipt_audit.json"
        repeat_path = directory / "prompt_repeatability.json"
        report, audit, repeatability = map(
            read_json, (report_path, audit_path, repeat_path)
        )
        generation_path = Path(entry["generation_audit"])
        generation = read_json(generation_path)
        if (
            generation["judge"] != audit["judge"]
            or generation["package_sha256"] != entry["package_sha256"]
        ):
            raise ValueError("generation audit differs from the released judge package")
        for filename, expected in generation["input_sha256"].items():
            if file_hash(filename) != expected:
                raise ValueError("generation audit input changed: " + filename)
        original = Path(entry.get("original_results", entry["results"]))
        for path in (
            original / "evaluations.jsonl",
            original / "manifest.json",
            repeat_path,
        ):
            if generation["input_sha256"].get(str(path.resolve())) != file_hash(path):
                raise ValueError(
                    "generation audit did not bind the original run and analysis"
                )
        if original.resolve() != results.resolve():
            provenance_path = results / "native_analysis_view_provenance.json"
            provenance = read_json(provenance_path)
            if (
                provenance["original_results"] != str(original.resolve())
                or provenance["original_evaluations_sha256"]
                != file_hash(original / "evaluations.jsonl")
                or provenance["view_evaluations_sha256"]
                != file_hash(results / "evaluations.jsonl")
                or provenance["original_manifest_sha256"]
                != file_hash(original / "manifest.json")
                or provenance["view_manifest_sha256"]
                != file_hash(results / "manifest.json")
                or provenance["package_sha256"] != entry["package_sha256"]
            ):
                raise ValueError("native analysis view provenance changed")
            inputs[str(provenance_path.resolve())] = file_hash(provenance_path)
        inputs.update(generation["input_sha256"])
        inputs[str(generation_path.resolve())] = file_hash(generation_path)
        if (
            audit["package_sha256"] != entry["package_sha256"]
            or audit["completed"] != package["jobs"]
            or audit["families"] != package["families"]
        ):
            raise ValueError("analysis does not match the released package")
        if (
            file_hash(results / "evaluations.jsonl") != audit["evaluations_sha256"]
            or file_hash(results / "manifest.json") != audit["run_manifest_sha256"]
        ):
            raise ValueError("audited run artifacts changed")
        if audit["judge"] in judges:
            raise ValueError("one audited final run per judge required")
        judges.add(audit["judge"])
        for path in (
            report_path,
            audit_path,
            repeat_path,
            results / "evaluations.jsonl",
            results / "manifest.json",
            Path(entry["package"]) / "package.json",
        ):
            inputs[str(path.resolve())] = file_hash(path)
        for name, rows in project_analysis(
            report, audit, repeatability, generation
        ).items():
            tables.setdefault(name, []).extend(rows)
    args.out.mkdir(parents=True)
    outputs = {}
    for name, rows in tables.items():
        path = args.out / (name + ".csv")
        if rows:
            columns = list(rows[0])
        else:
            # Empty repeat groups remain explicitly unmeasured in the summary.
            columns = [
                "judge",
                "messages_sha256",
                "n_logical_jobs",
                "logical_answer_disagreement",
            ]
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        with path.open(newline="") as stream:
            if len(list(csv.DictReader(stream))) != len(rows):
                raise ValueError("CSV export lost rows")
        outputs[name] = {
            "path": str(path),
            "rows": len(rows),
            "sha256": file_hash(path),
        }
    manifest = {
        "schema": "judge3_audited_table_export_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "judges": sorted(judges),
        "input_sha256": inputs,
        "projector_source_sha256": file_hash(__file__),
        "outputs": outputs,
        "inference_or_bootstrap_rerun": False,
        "interpretation": [
            "Estimates, family bootstrap intervals, fixes, harms and denominators are copied from the complete audited analyses; no outcome-based selection.",
            "Participant all pools only the detector's declared model coverage. Published FIBS has Qwen3.5 coverage only.",
            "The matched-neutral table includes every frozen detector; the prespecified-primary flag identifies adapted_primary and local3_sentence without selecting on outcomes.",
            "All paired contrasts retain their own matched denominators; do not subtract an unmatched pooled raw baseline cell for a restricted detector.",
            "Exact-message repeats are descriptive observations within a judge run, not extra independent families. Zero repeated groups means unmeasured repeatability, not zero noise.",
            "Logical agreement includes cached native jobs. Generation-aware agreement counts only distinct resolved generation receipts; cache-only native repeats leave sampling repeatability unmeasured.",
            "Original native evaluations and manifest remain immutable. If a native analysis view is used, its sole added field is answer equal to the existing parsed.answer, with original and derived hashes recorded.",
            "Complete-request hashes and provider/model/fingerprint metadata remain distinct from message equality. Missing metadata is unknown.",
        ],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {"out": str(args.out), "judges": sorted(judges), "tables": outputs},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
