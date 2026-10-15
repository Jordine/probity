"""Read-only score/text/display audit after the complete frozen feature export.

No truth annotations, inference, fitting, or score-based case selection are used.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from probity_benchmark.judge3_protocol import validate_bundle
from probity_benchmark.local3_annotate_units import load_inputs

BASE = Path("runs/iteration3")
HANDOFF = BASE / "local_probes/final_handoff_v3"
OUT = BASE / "final/local_score_export_audit_v1.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def validate_score_row(actual, original, published_available):
    metadata = {
        key: original[key]
        for key in ["id", "group", "participant_model", "source_episode", "turn_index"]
    }
    from probity_benchmark.evaluate_debate_transfer import argument_mapping

    argument = argument_mapping(original["response"])[0]
    metadata.update(
        response_sha256=text_digest(original["response"]),
        argument_sha256=text_digest(argument),
    )
    if set(actual) != set(metadata) | {"units"} or any(
        actual[key] != value for key, value in metadata.items()
    ):
        raise ValueError("all-unit score row identity/text changed")
    if len(actual["units"]) != len(original["units"]):
        raise ValueError("all-unit score coverage is incomplete")
    score_keys = {
        "local3",
        "random_direction",
        "frozen_local2",
        "position",
        "negative_position",
        "length",
        "negative_length",
        "fixed_word_tfidf",
        "response_broadcast",
    }
    if published_available:
        score_keys.add("published_dyl_broadcast")
    for predicted, unit in zip(actual["units"], original["units"]):
        if {key: value for key, value in predicted.items() if key != "scores"} != unit:
            raise ValueError("fixed complete unit text/coordinates changed")
        scores = predicted["scores"]
        if set(scores) != score_keys or any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in scores.values()
        ):
            raise ValueError("all-unit score controls missing or nonfinite")
        if (
            scores["position"] != unit["start"] / len(original["response"])
            or scores["negative_position"] != -scores["position"]
            or scores["length"] <= 0
            or scores["negative_length"] != -scores["length"]
        ):
            raise ValueError("frozen position/length control identity changed")


def main():
    if OUT.exists():
        raise ValueError("immutable new export audit required")
    handoff = json.loads((HANDOFF / "manifest.json").read_text())
    freeze_path = Path(handoff["global_freeze"])
    freeze = json.loads(freeze_path.read_text())
    for name, expected in freeze["source_hashes"].items():
        if digest(name) != expected:
            raise ValueError("frozen source changed before score audit")
    import probity_benchmark.local3_final_runner as runner

    runner.HANDOFF = HANDOFF
    episode_rows, lock = runner.verify(handoff)
    episodes = {episode["episode_id"]: episode for episode in episode_rows}
    source_rows, _ = load_inputs(
        Path(handoff["dataset"]), [Path(handoff["episodes"])], ["sentence", "clause"]
    )
    original = {row["id"]: row for row in source_rows}
    feature_audit_path = BASE / "final/local_feature_completion_audit_v1.json"
    feature_audit = json.loads(feature_audit_path.read_text())
    if (
        feature_audit["status"] != "all_complete_sequence_invariants_passed"
        or feature_audit["global_freeze_sha256"] != digest(freeze_path)
        or feature_audit["dataset_sha256"] != digest(handoff["dataset"])
    ):
        raise ValueError("required complete feature audit is missing or changed")
    audited_ids = {
        row["id"] for model in feature_audit["models"] for row in model["rows"]
    }
    if len(original) != 288 or audited_ids != set(original):
        raise ValueError("feature audit does not cover the exact complete corpus")
    score_dir = Path(handoff["score_out"])
    manifest = json.loads((score_dir / "manifest.json").read_text())
    specs = json.loads((HANDOFF / "all_unit_score_spec.json").read_text())["models"]
    if (
        manifest["spec"] != specs
        or manifest["script_sha256"]
        != freeze["source_hashes"]["probity_benchmark/local3_score_units.py"]
    ):
        raise ValueError("scorer source/specification changed")
    text_control_path = BASE / "local_probes/text_control_provenance_v1.json"
    text_control = {
        row["participant"]: row["sha256"]
        for row in json.loads(text_control_path.read_text())["artifacts"]
    }
    availability = {}
    for spec in specs:
        model = spec["participant"]
        metadata = manifest["models"][model]
        for name in ["head", "old_head", "response_head", "text_control"]:
            expected = (
                text_control[model]
                if name == "text_control"
                else freeze["artifacts_sha256"][spec[name]]
            )
            if (
                metadata["artifact_sha256"][name] != expected
                or digest(spec[name]) != expected
            ):
                raise ValueError("score artifact differs from frozen provenance")
        feature_manifests = metadata["feature_manifests"]
        if len(feature_manifests) != 1:
            raise ValueError("one complete feature manifest per model required")
        feature_manifest_path = Path(spec["evaluation_features"][0]) / "manifest.json"
        proof = next(
            row for row in feature_audit["models"] if row["participant"] == model
        )
        if (
            digest(feature_manifest_path) != proof["manifest_sha256"]
            or json.loads(feature_manifest_path.read_text()) != feature_manifests[0]
        ):
            raise ValueError("score features differ from completed-feature audit")
        availability[model] = feature_manifests[0]["published_probe_available"]
        for detector, ref_key in [
            ("local3", "local_reference_bundle"),
            ("frozen_local2", "old_reference_bundle"),
        ]:
            path = Path(spec[ref_key])
            reference = json.loads(path.read_text())
            expected = float(
                np.quantile(
                    reference["references"][model]["scores"], 0.9, method="linear"
                )
            )
            if metadata["thresholds"][detector] != expected or metadata[
                "threshold_reference_artifacts"
            ][detector] != digest(path):
                raise ValueError("frozen score threshold changed")
    rows = [
        json.loads(line)
        for line in (score_dir / "scores.jsonl").read_text().splitlines()
    ]
    predicted = {row["id"]: row for row in rows}
    if len(rows) != 288 or len(predicted) != 288 or set(predicted) != set(original):
        raise ValueError("all-unit scores do not cover all 288 unique turns")
    for identity, row in predicted.items():
        validate_score_row(
            row, original[identity], availability[row["participant_model"]]
        )
    summary = json.loads((score_dir / "summary.json").read_text())
    if summary != {
        "rows": 288,
        "episodes": 48,
        "units": sum(len(row["units"]) for row in rows),
        "scores_sha256": digest(score_dir / "scores.jsonl"),
    }:
        raise ValueError("all-unit score summary differs from complete rows")
    export_path = BASE / "final/local_export_receipt_v1.json"
    export = json.loads(export_path.read_text())
    if (
        export["status"] != "all_fixed_units_exported"
        or export["episodes"] != 48
        or export["turns"] != 288
        or export["scores_sha256"] != digest(score_dir / "scores.jsonl")
    ):
        raise ValueError("frozen export receipt mismatch")
    bundle_proofs = []
    for prefix, score_name in [("local3", "local3"), ("local2", "frozen_local2")]:
        for granularity in ["sentence", "clause"]:
            path = Path(handoff[f"{prefix}_bundle_out"]) / f"{granularity}.json"
            bundle = json.loads(path.read_text())
            reference = json.loads(
                Path(lock["artifacts"][bundle["detector"]]["path"]).read_text()
            )
            if any(
                bundle[field] != reference[field]
                for field in ["head_sha256", "reference_sha256", "references"]
            ):
                raise ValueError("display changed a frozen head/reference array")
            keyed, covered = validate_bundle(bundle, episodes)
            if len(keyed) != 288 or covered != set(episodes):
                raise ValueError("display does not cover all complete episodes/turns")
            if export["bundles"][f"{prefix}_{granularity}"] != digest(path):
                raise ValueError("display bundle differs from export receipt")
            for row in rows:
                display = keyed[row["source_episode"], row["turn_index"]]
                units = [
                    unit
                    for unit in row["units"]
                    if granularity in unit["granularities"]
                ]
                if display["spans"] != [
                    [u["argument_start"], u["argument_end"]] for u in units
                ]:
                    raise ValueError("score/display coordinate disagreement")
                if display["scores"] != [u["scores"][score_name] for u in units]:
                    raise ValueError(
                        "score/display numerical mismatch; stop before any renormalization"
                    )
            bundle_proofs.append(
                {
                    "path": str(path),
                    "sha256": digest(path),
                    "turns": len(keyed),
                    "episodes": len(covered),
                    "units": sum(len(row["scores"]) for row in bundle["rows"]),
                    "reference_sha256": bundle["reference_sha256"],
                }
            )
    result = {
        "schema": "local3_score_export_audit_v1",
        "status": "all_complete_score_text_and_display_invariants_passed",
        "audit_source_sha256": digest(__file__),
        "global_freeze_sha256": digest(freeze_path),
        "dataset_sha256": digest(handoff["dataset"]),
        "feature_audit_sha256": digest(feature_audit_path),
        "text_control_provenance_sha256": digest(text_control_path),
        "export_receipt_sha256": digest(export_path),
        "score_manifest_sha256": digest(score_dir / "manifest.json"),
        "score_summary": summary,
        "bundles": bundle_proofs,
        "truth_annotations_loaded": False,
        "model_inference_performed": False,
        "scope": "All complete score-row identities, raw/display text, sentence/clause coordinates, finite controls, fixed development thresholds and four bundle/reference identities agree exactly. No truth-label join or empirical detector performance is involved.",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
