"""Assemble factual sentence labels without mixing single-reader and consensus tiers."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


PRIMARY = "qwen/qwen3.8-27b"
SECONDARY = "anthropic/claude-opus-5.5"


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def eligible_label(check, text):
    if check["exact_text"] != text:
        raise ValueError("annotation text differs from exact unit")
    if check["verdict"] not in ("supported", "contradicted", "unknown"):
        raise ValueError("unknown annotation verdict")
    if (check["verdict"] == "unknown" or check["claim_kind"] != "checkable_fact"
            or not check["local_label_valid"] or not check["story_paragraph_ids"]):
        return None
    return int(check["verdict"] == "contradicted")


def resolve(unit, checks):
    labels = {model: eligible_label(record["check"], unit["display_text"])
              for model, record in checks.items()}
    primary = labels.get(PRIMARY)
    dual = PRIMARY in checks and SECONDARY in checks
    consensus = primary if dual and primary is not None and primary == labels[SECONDARY] else None
    if not dual:
        status = "not_dual_annotated"
    elif primary is not None and primary == labels[SECONDARY]:
        status = "eligible_agreement"
    elif checks[PRIMARY]["check"]["verdict"] != checks[SECONDARY]["check"]["verdict"]:
        status = "verdict_disagreement"
    else:
        status = "unknown_or_invalid_scope"
    return {"primary_annotated": PRIMARY in checks, "primary_label": primary,
            "secondary_annotated": SECONDARY in checks,
            "consensus_label": consensus, "consensus_status": status,
            "checks_by_model": checks}


def assemble(units, metadata, reused, annotations, legacy_models):
    if legacy_models != [PRIMARY, SECONDARY]:
        raise ValueError("legacy check order differs from verified model contract")
    indexed = {(row["id"], i): unit for row in units for i, unit in enumerate(row["units"])}
    if len(indexed) != sum(len(row["units"]) for row in units):
        raise ValueError("duplicate turn/unit identity")
    checks = {key: {} for key in indexed}

    def add(key, model, check, request_sha256, provenance):
        if key not in indexed or model not in (PRIMARY, SECONDARY):
            raise ValueError("annotation outside declared population/models")
        eligible_label(check, indexed[key]["display_text"])
        record = {"check": check, "request_sha256": request_sha256, "provenance": provenance}
        previous = checks[key].get(model)
        if previous is not None and previous != record:
            raise ValueError("conflicting duplicate annotation; adjudicate explicitly")
        checks[key][model] = record

    for row in reused:
        if len(row["annotations"]) != len(row["units"]):
            raise ValueError("incomplete legacy turn")
        seen = set()
        for annotation in row["annotations"]:
            i = annotation["unit_index"]
            key = (row["id"], i)
            if i in seen or annotation["unit"] != indexed[key]:
                raise ValueError("legacy unit mismatch or duplicate")
            seen.add(i)
            if len(annotation["checks"]) != 2 or len(annotation["request_sha256"]) != 2:
                raise ValueError("legacy dual-reader provenance missing")
            for model, check, request in zip(legacy_models, annotation["checks"], annotation["request_sha256"]):
                add(key, model, check, request, "verified_iteration3_reuse")

    jobs = {job["job_id"]: job for job in metadata}
    if len(jobs) != len(metadata):
        raise ValueError("duplicate annotation metadata job")
    for annotation in annotations:
        job = jobs[annotation["job_id"]]
        mapping = {item["index"]: item for item in job["units"]}
        responses = annotation["parsed"]["checks"]
        if len(responses) != len(mapping) or {r["index"] for r in responses} != set(mapping):
            raise ValueError("incomplete or repeated annotation check indices")
        for check in responses:
            item = mapping[check["index"]]
            key = (item["turn_id"], item["unit_index"])
            if any(indexed[key][field] != item[field] for field in ("start", "end")):
                raise ValueError("compiled annotation offsets changed")
            provenance = annotation.get("provenance", "iteration6_annotation")
            waiver = annotation.get("unit_waivers", {}).get(str(check["index"]))
            if waiver is not None:
                if not isinstance(provenance, dict) or waiver["turn_id"] != key[0] or waiver["unit_index"] != key[1]:
                    raise ValueError("waiver unit/provenance mismatch")
                provenance = {**provenance, "administrative_waiver": waiver}
            add(key, annotation["model"], check, annotation["request_sha256"], provenance)

    output = []
    for row in units:
        for i, unit in enumerate(row["units"]):
            output.append({"turn_id": row["id"], "unit_index": i,
                           "source_problem": row["source_problem"],
                           "source_story_sha256": row["source_story_sha256"],
                           "generation_request_sha256": row["request_sha256"],
                           "unit": unit, **resolve(unit, checks[row["id"], i])})
    summary = {
        "schema": "iteration6_factual_label_tiers_v1", "units": len(output),
        "turns": len(units), "primary_model": PRIMARY, "secondary_model": SECONDARY,
        "primary_annotated_units": sum(r["primary_annotated"] for r in output),
        "secondary_annotated_units": sum(r["secondary_annotated"] for r in output),
        "primary_label_counts": dict(Counter(str(r["primary_label"]) for r in output)),
        "consensus_label_counts": dict(Counter(str(r["consensus_label"]) for r in output)),
        "consensus_status_counts": dict(Counter(r["consensus_status"] for r in output)),
        "primary_complete": all(r["primary_annotated"] for r in output),
        "administrative_waived_checks": sum(isinstance(c["provenance"], dict) and "administrative_waiver" in c["provenance"]
                                           for r in output for c in r["checks_by_model"].values()),
        "label_limit": "Model factuality labels, not human truth or intent; primary and consensus must be reported separately. Missing and invalid units are not truthful negatives. No interaction labels.",
    }
    return output, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planning", type=Path, default=Path("runs/iteration6/planning"))
    parser.add_argument("--legacy-manifest", type=Path,
                        default=Path("runs/iteration3/final/local_sentence_annotations_v1/manifest.json"))
    parser.add_argument("--annotations", type=Path, nargs="*", default=[])
    parser.add_argument("--metadata-overlay", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    files = {name: args.planning / f"annotation_{name}.jsonl"
             for name in ("units_private", "metadata_private", "reused_verified")}
    plan_path = args.planning / "annotation_plan.json"
    plan = json.loads(plan_path.read_text())
    for path in files.values():
        if hashlib.sha256(path.read_bytes()).hexdigest() != plan["files_sha256"][path.name]:
            raise ValueError("compiled annotation input hash mismatch")
    metadata = {row["job_id"]: row for row in read_rows(files["metadata_private"])}
    for path in args.metadata_overlay:
        for row in read_rows(path):
            previous = metadata.get(row["job_id"])
            if previous is not None and previous != row:
                raise ValueError("conflicting metadata overlay")
            metadata[row["job_id"]] = row
    output, summary = assemble(
        read_rows(files["units_private"]), list(metadata.values()),
        read_rows(files["reused_verified"]),
        [row for path in args.annotations for row in read_rows(path)],
        json.loads(args.legacy_manifest.read_text())["models"],
    )
    if args.out.exists():
        raise ValueError("new assembly output directory required")
    args.out.mkdir(parents=True)
    (args.out / "units.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in output))
    summary["input_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [*files.values(), plan_path, args.legacy_manifest, *args.annotations, *args.metadata_overlay]}
    summary["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "input_sha256"}, indent=2))


if __name__ == "__main__":
    main()
