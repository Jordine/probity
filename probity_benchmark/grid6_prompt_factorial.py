"""Complete paired 2x2 prompt contrasts for behavior and both final judges."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from probity_benchmark import grid6_live as live

BEHAVIOR = ROOT / "reports/data/iteration6/prompt_behavior_v1/trajectories.csv"
JUDGMENTS = ROOT / "reports/data/iteration6/live_judges_v1/judgments.csv"
DESTINATION = ROOT / "reports/data/iteration6/prompt_factorial_v3"


def read_csv(path):
    return list(csv.DictReader(path.open()))


def summarize(values, rng):
    values = np.asarray(values, dtype=float)
    bootstrap = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
    result = {
        "families": len(values),
        "estimate": float(values.mean()),
        "lo95": float(np.percentile(bootstrap, 2.5)),
        "hi95": float(np.percentile(bootstrap, 97.5)),
    }
    if set(values).issubset({-1.0, 0.0, 1.0}):
        result.update(fixes=int((values > 0).sum()), harms=int((values < 0).sum()), ties=int((values == 0).sum()))
    return result


def main():
    if DESTINATION.exists():
        raise ValueError("new versioned destination required")
    behavior = read_csv(BEHAVIOR)
    judgments = [row for row in read_csv(JUDGMENTS) if row["group"].startswith("prompt:")]
    judge_lookup = {(row["family"], row["group"].split(":", 1)[1], row["policy"]): row for row in judgments}
    rows = []
    for row in behavior:
        key = (row["family"], row["level"], row["feedback"])
        judge = judge_lookup[key]
        merged = dict(row)
        merged["haiku_correct"] = float(judge["haiku_correct"])
        merged["native_correct"] = float(judge["native_correct"])
        rows.append(merged)
    families = sorted({row["family"] for row in rows})
    lookup = {(row["family"], row["level"], row["feedback"]): row for row in rows}
    if len(families) != 8 or len(rows) != 32 or any((f, level, policy) not in lookup for f in families for level in ("selective", "explicit") for policy in ("none", "bundle")):
        raise ValueError("incomplete paired 2x2")
    metrics = (
        "haiku_correct",
        "native_correct",
        "words",
        "exact_self_repeat",
        "assigned_answer_deviations",
        "authentic_assignment_mean_percentile",
        "incorrect_assignment_mean_percentile",
    )
    rng = np.random.default_rng(8266)
    contrasts = []
    for metric in metrics:
        value = lambda family, level, policy: float(lookup[family, level, policy][metric])
        definitions = [
            ("feedback_bundle_minus_none_at_selective", [value(f, "selective", "bundle") - value(f, "selective", "none") for f in families]),
            ("feedback_bundle_minus_none_at_explicit", [value(f, "explicit", "bundle") - value(f, "explicit", "none") for f in families]),
            ("instruction_explicit_minus_selective_at_none", [value(f, "explicit", "none") - value(f, "selective", "none") for f in families]),
            ("instruction_explicit_minus_selective_at_bundle", [value(f, "explicit", "bundle") - value(f, "selective", "bundle") for f in families]),
            (
                "instruction_by_feedback_interaction",
                [
                    (value(f, "explicit", "bundle") - value(f, "explicit", "none"))
                    - (value(f, "selective", "bundle") - value(f, "selective", "none"))
                    for f in families
                ],
            ),
        ]
        for contrast, values in definitions:
            contrasts.append({"metric": metric, "contrast": contrast, **summarize(values, rng)})
    DESTINATION.mkdir(parents=True)
    with (DESTINATION / "contrasts.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in contrasts for key in row}))
        writer.writeheader()
        writer.writerows(contrasts)
    summary = {
        "schema": "grid6_prompt_factorial_v2",
        "families": 8,
        "cells": 4,
        "contrasts": contrasts,
        "interpretation": (
            "Complete paired 2x2 descriptive contrasts. Correct-role instructions are honest in every cell; "
            "selective/explicit changes the wrong-role instruction. Intervals are empirical percentile ranges "
            "from 10,000 source-family bootstrap draws (seed 8266). With n=8, boundary intervals and zero "
            "endpoints do not establish certainty. One trajectory per cell; no generation-seed variance."
        ),
        "provenance": {
            "behavior_sha256": live.file_hash(BEHAVIOR),
            "judgments_sha256": live.file_hash(JUDGMENTS),
            "script_sha256": live.file_hash(__file__),
        },
    }
    live.durable(DESTINATION / "summary.json", summary)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = {
        "feedback_bundle_minus_none_at_selective": "Feedback effect\nat selective",
        "feedback_bundle_minus_none_at_explicit": "Feedback effect\nat explicit",
        "instruction_explicit_minus_selective_at_none": "Instruction effect\nwithout feedback",
        "instruction_explicit_minus_selective_at_bundle": "Instruction effect\nwith feedback",
        "instruction_by_feedback_interaction": "Interaction",
    }
    order = tuple(names)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, metric, title, color in (
        (axes[0], "haiku_correct", "Haiku 5.5", "#ae5b1a"),
        (axes[1], "native_correct", "Native Qwen3.5-4B", "#216d91"),
    ):
        selected = {row["contrast"]: row for row in contrasts if row["metric"] == metric}
        for index, contrast in enumerate(order):
            row = selected[contrast]
            estimate = 100 * row["estimate"]
            axis.errorbar(estimate, index, xerr=[[estimate - 100 * row["lo95"]], [100 * row["hi95"] - estimate]], fmt="o", capsize=4, color=color)
            axis.text(estimate, index - 0.16, f"{row['fixes']} fix / {row['harms']} harm", ha="center", fontsize=8)
        axis.set_yticks(range(len(order)), [names[item] for item in order])
        axis.invert_yaxis()
        axis.axvline(0, color="gray", lw=0.8)
        axis.set_xlabel("Accuracy difference (percentage points)")
        axis.set_title(title)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Complete prompt 2×2 contrasts · 8 paired families\nempirical 95% family-bootstrap ranges; fix/harm counts shown")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(DESTINATION / ("prompt_factorial_judges." + extension), dpi=180, bbox_inches="tight")
    print(json.dumps({"families": len(families), "contrasts": len(contrasts), "destination": str(DESTINATION)}, indent=2))


if __name__ == "__main__":
    main()
