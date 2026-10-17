"""Descriptive local-truth coverage by advocate role, after complete annotation.

This reads no probe scores and changes no labels. It is separate from the frozen
primary local-ranking and flagging estimands.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

BASE = Path("runs/iteration3/final")
STAGES = {
    "sentence": (
        "primary_sentence_all_turns",
        BASE / "local_sentence_annotations_v1",
        288,
        1977,
    ),
    "clause": (
        "secondary_clause_wrong_openings",
        BASE / "local_opening_clause_annotations_v1",
        48,
        771,
    ),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_complete(path, expected_rows, expected_units=None):
    summary = json.loads((path / "summary.json").read_text())
    if (
        summary["errors"]
        or summary["expected_rows"] != expected_rows
        or summary["completed_rows"] != expected_rows
        or summary["expected_units"] != summary["completed_units"]
        or (expected_units is not None and summary["expected_units"] != expected_units)
    ):
        raise ValueError("the requested complete fixed annotation set is required")
    rows = [
        json.loads(line) for line in (path / "corpus.jsonl").read_text().splitlines()
    ]
    if len(rows) != expected_rows or len({row["id"] for row in rows}) != expected_rows:
        raise ValueError("annotation rows do not match complete fixed coverage")
    if sum(len(row["annotations"]) for row in rows) != summary["expected_units"]:
        raise ValueError("annotation unit count differs from completeness receipt")
    if digest(path / "corpus.jsonl") != summary["corpus_sha256"]:
        raise ValueError("annotation corpus changed after completion")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=["sentence", "clause", "both"], required=True
    )
    args = parser.parse_args()
    requested = list(STAGES) if args.stage == "both" else [args.stage]
    output = BASE / f"local_truth_by_advocacy_{args.stage}_v2.json"
    if output.exists():
        raise ValueError("immutable descriptive analysis required")
    family_path = Path("runs/iteration3/coordinator/source_family_map.json")
    families = json.loads(family_path.read_text())["by_parent"]
    episode_path = BASE / "debates_v1"
    episode_summary = json.loads((episode_path / "summary.json").read_text())
    episodes = {
        episode["episode_id"]: episode
        for episode in (
            json.loads(path.read_text())
            for path in sorted(episode_path.glob("episode_*.json"))
        )
    }
    if episode_summary["errors"] or len(episodes) != 48:
        raise ValueError("complete source episodes required")
    annotation_dirs = {STAGES[name][0]: STAGES[name][1:] for name in requested}
    complete = {
        stage: read_complete(path, expected_rows, expected_units)
        for stage, (path, expected_rows, expected_units) in annotation_dirs.items()
    }
    result = {
        "schema": "local3_descriptive_truth_by_advocacy_v2",
        "requested_stages": requested,
        "unrequested_stages": [name for name in STAGES if name not in requested],
        "script_sha256": digest(Path(__file__)),
        "scope": "Companion description of the complete fixed annotation population. No probe scores read, no label overrides, no case selection, and no change to frozen primary local AUROC/flagging metrics. Assigned-answer correctness is a role, not sustained prose advocacy, a local factual label, or intent.",
        "labels": "Supported and contradicted mean independent Qwen/Opus agreement with valid local assertion scope and source evidence. Unknown includes ambiguous, mixed, interpretive, disputed and invalid-scope units. These are silver labels, not intent or human truth labels.",
        "selection": {
            "primary_sentence_all_turns": "Every sentence unit in all 288 turns, both advocate roles.",
            "secondary_clause_wrong_openings": "Every clause unit in all 48 predetermined first wrong-advocate openings; there is no corresponding correct-advocate clause sample.",
        },
        "artifacts_sha256": {
            str(family_path): digest(family_path),
            str(episode_path / "manifest.json"): digest(episode_path / "manifest.json"),
            **{
                str(path / "corpus.jsonl"): digest(path / "corpus.jsonl")
                for path, _, _ in annotation_dirs.values()
            },
        },
        "cells": [],
    }
    for stage, rows in complete.items():
        grouped = defaultdict(list)
        for row in rows:
            episode = episodes[row["source_episode"]]
            turn = episode["turns"][row["turn_index"]]
            if (
                row["participant_model"] != episode["model"]
                or row["group"] != episode["parent_problem_id"]
            ):
                raise ValueError("annotation/episode identity mismatch")
            if json.loads(row["response"])["argument"] != turn["argument"]:
                raise ValueError("complete annotated argument changed")
            role = (
                "correct_answer_advocate"
                if turn["assigned_answer"] == episode["gold"]
                else "wrong_answer_advocate"
            )
            if stage.startswith("secondary"):
                wrong_turns = [
                    i
                    for i, item in enumerate(episode["turns"])
                    if item["assigned_answer"] != episode["gold"]
                ]
                if (
                    role != "wrong_answer_advocate"
                    or row["turn_index"] != wrong_turns[0]
                ):
                    raise ValueError("secondary row is outside fixed opening selection")
            for annotation in row["annotations"]:
                if annotation["label"] not in (None, 0, 1):
                    raise ValueError("unexpected silver label")
                grouped[row["participant_model"], role].append(
                    {
                        "row_id": row["id"],
                        "family": families[row["group"]],
                        "label": annotation["label"],
                        "rejection_reasons": annotation["rejection_reasons"],
                    }
                )
        for (model, role), units in sorted(grouped.items()):
            by_family = defaultdict(list)
            by_response = defaultdict(list)
            for unit in units:
                by_family[unit["family"]].append(unit)
                by_response[unit["row_id"]].append(unit["label"])
            counts = Counter(unit["label"] for unit in units)
            known = counts[0] + counts[1]
            family_rows = []
            for family, members in sorted(by_family.items()):
                labels = Counter(unit["label"] for unit in members)
                denominator = labels[0] + labels[1]
                family_rows.append(
                    {
                        "family": family,
                        "all_units": len(members),
                        "responses": len({unit["row_id"] for unit in members}),
                        "supported": labels[0],
                        "contradicted": labels[1],
                        "unknown": labels[None],
                        "contradicted_fraction_among_accepted": (
                            labels[1] / denominator if denominator else None
                        ),
                    }
                )
            eligible_rates = [
                record["contradicted_fraction_among_accepted"]
                for record in family_rows
                if record["contradicted_fraction_among_accepted"] is not None
            ]
            result["cells"].append(
                {
                    "stage": stage,
                    "participant": model,
                    "advocate_role": role,
                    "all_units": len(units),
                    "responses": len(by_response),
                    "families": len(by_family),
                    "supported": counts[0],
                    "contradicted": counts[1],
                    "unknown": counts[None],
                    "accepted_fraction": known / len(units),
                    "contradicted_fraction_among_accepted_pooled": (
                        counts[1] / known if known else None
                    ),
                    "contradicted_fraction_of_all_fixed_units": counts[1] / len(units),
                    "family_macro_contradicted_fraction_among_accepted": (
                        sum(eligible_rates) / len(eligible_rates)
                        if eligible_rates
                        else None
                    ),
                    "families_with_accepted_labels": len(eligible_rates),
                    "families_with_contradicted_units": sum(
                        record["contradicted"] > 0 for record in family_rows
                    ),
                    "responses_with_contradicted_units": sum(
                        1 in labels for labels in by_response.values()
                    ),
                    "responses_with_both_supported_and_contradicted": sum(
                        0 in labels and 1 in labels for labels in by_response.values()
                    ),
                    "unknown_reason_counts_overlapping": dict(
                        Counter(
                            reason
                            for unit in units
                            for reason in unit["rejection_reasons"]
                        )
                    ),
                    "per_family": family_rows,
                }
            )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "artifact": str(output),
                "cells": [
                    {key: value for key, value in cell.items() if key != "per_family"}
                    for cell in result["cells"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
