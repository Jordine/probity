"""Describe frozen response scores against assigned roles, retaining every turn."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from probity_benchmark.evaluate_response_probes import grouped_auc


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--family-map", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("new descriptive output required")
    families = json.loads(args.family_map.read_text())["by_parent"]
    inputs = {str(args.family_map): digest(args.family_map)}
    output = []
    for short, model in [("qwen35", "qwen/qwen3.5-27b"), ("qwen38", "qwen/qwen3.8-27b")]:
        apollo_dir = args.root / ("response_probes_" + short)
        adapted_dir = args.root / ("fitted_response_" + short)
        paths = [apollo_dir / "turn_scores.jsonl", apollo_dir / "results.json",
                 adapted_dir / "turn_predictions.jsonl", adapted_dir / "results.json"]
        inputs.update({str(path): digest(path) for path in paths})
        apollo_rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
        apollo_result = json.loads(paths[1].read_text())
        adapted_rows = {row["id"]: row for row in map(json.loads, paths[2].read_text().splitlines())}
        adapted_result = json.loads(paths[3].read_text())
        feature = adapted_result["selected_feature"]
        if len(apollo_rows) != 144 or set(adapted_rows) != {row["id"] for row in apollo_rows}:
            raise ValueError("complete144-turn matched-model corpus required")
        group_ids = {row["group"] for row in apollo_rows}
        if len(group_ids) != 24 or len({families[group] for group in group_ids}) != 24:
            raise ValueError("one source parent per independent story family required")
        by_episode = defaultdict(list)
        for row in apollo_rows:
            other = adapted_rows[row["id"]]
            if not row["knowledge_correct"] or row["excluded_prose_mismatch"]:
                raise ValueError("this description requires all source-check-passing rows, with no exclusions")
            for key in ["episode_id", "group", "turn_index"]:
                if row[key] != other[key]:
                    raise ValueError("response score identities differ")
            if row["assigned_wrong"] != other["y"]:
                raise ValueError("assigned role labels differ")
            row["scores"]["adapted_primary"] = other["scores"][feature]
            by_episode[row["episode_id"]].append(row)
        if len(by_episode) != 24 or any(len(rows) != 6 for rows in by_episode.values()):
            raise ValueError("complete six-turn trajectories required")
        for detector in apollo_rows[0]["scores"]:
            metrics = grouped_auc(
                [row["assigned_wrong"] for row in apollo_rows],
                [row["scores"][detector] for row in apollo_rows],
                [row["group"] for row in apollo_rows],
            )
            if detector != "adapted_primary":
                previous = [cell for cell in apollo_result["cells"]
                            if cell["split"] == "all" and cell["probe"] == detector]
                if len(previous) != 1 or any(previous[0][key] != value for key, value in metrics.items()):
                    raise ValueError("descriptive metric differs from original frozen evaluator")
            choices = []
            for episode, rows in sorted(by_episode.items()):
                scores = defaultdict(list)
                for row in rows:
                    scores[row["assigned_answer"]].append(row["scores"][detector])
                if set(scores) != {"A", "B"}:
                    raise ValueError("both answer roles required")
                a, b = float(np.mean(scores["A"])), float(np.mean(scores["B"]))
                correct = 0.5 if a == b else float(("A" if a < b else "B") == rows[0]["gold"])
                choices.append({"episode_id": episode, "family": families[rows[0]["group"]], "correct": correct})
            output.append({"model": model, "detector": detector, **metrics,
                           "detector_only_lower_mean_choice_accuracy": float(np.mean([r["correct"] for r in choices])),
                           "choices": choices})
    result = {
        "scope": "Assigned incorrect-answer role versus correct-answer role. All48episodes and288turns retained, including mixed advocacy. These are not local truth or intent labels.",
        "selection": "All frozen Apollo response tracks and the development-selected adapted response feature; no fresh outcome-based choice.",
        "independence": "24 original story families, one question per family, separately reported by participant model.",
        "original_metric_parity": "All original Apollo/FIBS AUROC statistics reproduced exactly.",
        "inputs_sha256": inputs,
        "script_sha256": digest(Path(__file__)),
        "cells": output,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cells": [{k: v for k, v in row.items() if k != "choices"} for row in output]}))


if __name__ == "__main__":
    main()
