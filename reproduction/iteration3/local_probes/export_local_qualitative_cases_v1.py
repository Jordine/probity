"""Deterministic complete-context review candidates after frozen evaluation.

Select at most two false positives and two false negatives per participant.
Selection never changes labels, the evaluation population, or any metric.
"""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from probity_benchmark.local3_annotate_units import load_inputs, reconcile_unit

BASE = Path("runs/iteration3")
FINAL = BASE / "final"
ANNOTATIONS = FINAL / "local_sentence_annotations_v1"
EVALUATION = FINAL / "local_sentence_evaluation_v1"
SCORES = FINAL / "local_all_unit_scores_v1"
OUT = FINAL / "local_qualitative_cases_v1.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(row):
    return row["id"], row["argument_start"], row["argument_end"]


def choose_cases(predictions, thresholds):
    """Use frozen threshold semantics and stable family/id/offset ties."""
    groups = []
    for model in sorted(thresholds):
        threshold = thresholds[model]
        if not math.isfinite(threshold):
            raise ValueError("nonfinite threshold")
        population = [row for row in predictions if row["model"] == model]
        if not population:
            raise ValueError("participant missing from the complete population")
        for kind, label, sign in [("false_positive", 0, -1), ("false_negative", 1, 1)]:
            candidates = []
            for row in population:
                score = row["scores"]["local3"]
                if not math.isfinite(score):
                    raise ValueError("nonfinite local score")
                flagged = score >= threshold
                if row["y"] == label and flagged == (kind == "false_positive"):
                    if row["rejection_reasons"]:
                        raise ValueError("accepted unit has rejection reasons")
                    candidates.append(row)
            candidates.sort(
                key=lambda row: (
                    sign * row["scores"]["local3"],
                    row["family"],
                    row["id"],
                    row["argument_start"],
                    row["argument_end"],
                )
            )
            selected = []
            for row in candidates:
                if len(selected) == 2:
                    break
                selected.append(row)
            groups.append(
                {
                    "participant": model,
                    "kind": kind,
                    "threshold": threshold,
                    "eligible_units": len(candidates),
                    "selected": selected,
                }
            )
    return groups


def main():
    if OUT.exists():
        raise ValueError("immutable qualitative casebook required")
    summary_path = ANNOTATIONS / "summary.json"
    summary = json.loads(summary_path.read_text())
    if (
        summary["errors"]
        or summary["expected_rows"] != 288
        or summary["completed_rows"] != 288
        or summary["expected_units"] != 1977
        or summary["completed_units"] != 1977
        or summary["corpus_sha256"] != digest(ANNOTATIONS / "corpus.jsonl")
    ):
        raise ValueError("complete fixed sentence annotations required")
    export_path = FINAL / "local_score_export_audit_v1.json"
    export = json.loads(export_path.read_text())
    if (
        export["status"] != "all_complete_score_text_and_display_invariants_passed"
        or export["score_summary"]["scores_sha256"] != digest(SCORES / "scores.jsonl")
        or export["score_manifest_sha256"] != digest(SCORES / "manifest.json")
    ):
        raise ValueError("complete audited frozen scores required")
    results_path = EVALUATION / "results.json"
    results = json.loads(results_path.read_text())
    manifest = json.loads((SCORES / "manifest.json").read_text())
    if (
        results["annotation_summary"] != summary
        or results["score_manifest"] != manifest
    ):
        raise ValueError("evaluation inputs differ from completed artifacts")
    family_path = BASE / "coordinator/source_family_map.json"
    families = json.loads(family_path.read_text())["by_parent"]
    if results["family_map_sha256"] != digest(family_path):
        raise ValueError("evaluation source families changed")
    dataset = BASE / "coordinator/holdout_families_source_v5.jsonl"
    original_rows, sources = load_inputs(dataset, [FINAL / "debates_v1"], ["sentence"])
    originals = {row["id"]: row for row in original_rows}
    annotation_rows = [
        json.loads(line)
        for line in (ANNOTATIONS / "corpus.jsonl").read_text().splitlines()
    ]
    annotations = {row["id"]: row for row in annotation_rows}
    if len(annotation_rows) != 288 or set(annotations) != set(originals):
        raise ValueError("annotation rows differ from the fixed complete corpus")
    annotated_units = {}
    for row in annotation_rows:
        original = originals[row["id"]]
        if any(row[key] != value for key, value in original.items() if key != "spans"):
            raise ValueError("annotated response or complete context changed")
        if len(row["annotations"]) != len(row["units"]):
            raise ValueError("incomplete response annotations")
        for index, annotated in enumerate(row["annotations"]):
            unit = annotated["unit"]
            if annotated["unit_index"] != index or unit != row["units"][index]:
                raise ValueError("annotation coordinates or exact text changed")
            label, reasons = reconcile_unit(unit, annotated["checks"])
            if label != annotated["label"] or reasons != annotated["rejection_reasons"]:
                raise ValueError("frozen acceptance rule was changed")
            key = row["id"], unit["argument_start"], unit["argument_end"]
            if key in annotated_units:
                raise ValueError("duplicate unit identity")
            annotated_units[key] = annotated
    prediction_path = EVALUATION / "predictions_with_labels.jsonl"
    predictions = [
        json.loads(line) for line in prediction_path.read_text().splitlines()
    ]
    keyed = {identity(row): row for row in predictions}
    if (
        len(predictions) != 1977
        or len(keyed) != 1977
        or set(keyed) != set(annotated_units)
    ):
        raise ValueError("complete evaluated fixed sentence coverage required")
    score_rows = {
        row["id"]: row
        for row in map(json.loads, (SCORES / "scores.jsonl").read_text().splitlines())
    }
    for row in predictions:
        original = originals[row["id"]]
        annotation = annotated_units[identity(row)]
        scored = next(
            unit
            for unit in score_rows[row["id"]]["units"]
            if (unit["argument_start"], unit["argument_end"]) == identity(row)[1:]
        )
        expected = {
            "id": original["id"],
            "source_parent_id": original["group"],
            "family": families[original["group"]],
            "model": original["participant_model"],
            "granularity": "sentence",
            "text": annotation["unit"]["display_text"],
            "argument_start": annotation["unit"]["argument_start"],
            "argument_end": annotation["unit"]["argument_end"],
            "y": annotation["label"],
            "rejection_reasons": annotation["rejection_reasons"],
            "scores": scored["scores"],
        }
        if row != expected:
            raise ValueError("evaluated unit differs from frozen scores or annotation")
    thresholds = {
        model: values["thresholds"]["local3"]
        for model, values in manifest["models"].items()
    }
    groups = choose_cases(predictions, thresholds)
    used_rows, used_sources = {}, {}
    for group in groups:
        cases = []
        for prediction in group.pop("selected"):
            row = annotations[prediction["id"]]
            annotation = annotated_units[identity(prediction)]
            used_rows[row["id"]] = {
                "annotation_row": row,
                "complete_argument": json.loads(row["response"])["argument"],
                "source_episode_path": str(
                    FINAL / "debates_v1" / f"episode_{row['source_episode']}.json"
                ),
            }
            used_sources[row["source_problem"]] = sources[row["source_problem"]]
            cases.append(
                {
                    "prediction": prediction,
                    "annotation": annotation,
                    "source_problem": row["source_problem"],
                    "source_episode": row["source_episode"],
                    "turn_index": row["turn_index"],
                    "source_request_sha256": row["source_request_sha256"],
                }
            )
        group["cases"] = cases
    paths = [
        Path(__file__),
        dataset,
        family_path,
        summary_path,
        ANNOTATIONS / "corpus.jsonl",
        ANNOTATIONS / "manifest.json",
        ANNOTATIONS / "requests.jsonl",
        results_path,
        prediction_path,
        SCORES / "scores.jsonl",
        SCORES / "manifest.json",
        export_path,
        FINAL / "local_feature_completion_audit_v1.json",
        FINAL / "global_freeze_v2.json",
    ]
    artifact = {
        "schema": "local3_deterministic_qualitative_cases_v1",
        "scope": "Post-evaluation diagnostic review only. No exclusions, relabeling, refitting, or change to frozen metrics. Silver agreement is not human truth or intent.",
        "selection": "Per participant, up to two highest local3-scoring accepted supported sentences at or above the frozen threshold, and up to two lowest local3-scoring accepted contradicted sentences below it. Ties use family, response id, argument start, argument end. Empty groups stay empty; no forced diversity.",
        "complete_population": {
            "responses": len(annotations),
            "sentence_units": len(predictions),
            "labels": dict(Counter(str(row["y"]) for row in predictions)),
        },
        "artifacts_sha256": {str(path): digest(path) for path in paths},
        "groups": groups,
        "complete_response_contexts": used_rows,
        "complete_source_stories": used_sources,
        "annotator_order": json.loads((ANNOTATIONS / "manifest.json").read_text())[
            "models"
        ],
    }
    OUT.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "artifact": str(OUT),
                "sha256": digest(OUT),
                "groups": [
                    {key: value for key, value in group.items() if key != "cases"}
                    | {"selected_units": len(group["cases"])}
                    for group in groups
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
