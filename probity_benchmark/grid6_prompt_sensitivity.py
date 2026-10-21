"""Whole-family sensitivity for the one confirmed incomplete prompt argument."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from probity_benchmark import grid6_live as live

FAMILY = "6ac15f0338e70a96ba9068660aec78e1b76caed14ed1d83a27349c0d8ca9a91d"
METRICS = (
    "haiku_correct",
    "native_correct",
    "words",
    "exact_self_repeat",
    "assigned_answer_deviations",
    "authentic_assignment_mean_percentile",
    "incorrect_assignment_mean_percentile",
)


def interval(values, rng):
    values = np.asarray(values, dtype=float)
    boots = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
    return {
        "families": len(values),
        "estimate": float(values.mean()),
        "lower95": float(np.percentile(boots, 2.5)),
        "upper95": float(np.percentile(boots, 97.5)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    source = ROOT / "reports/data/iteration6/prompt_behavior_v1/trajectories.csv"
    screen_path = ROOT / "reports/data/iteration6/prompt_prose_audit_v1/screen.json"
    rows = list(csv.DictReader(source.open()))
    judgments_path = ROOT / "reports/data/iteration6/live_judges_v1/judgments.csv"
    judgments = [row for row in csv.DictReader(judgments_path.open()) if row["group"].startswith("prompt:")]
    judge_lookup = {(row["family"], row["group"].split(":", 1)[1], row["policy"]): row for row in judgments}
    for row in rows:
        judge = judge_lookup[row["family"], row["level"], row["feedback"]]
        row["haiku_correct"] = judge["haiku_correct"]
        row["native_correct"] = judge["native_correct"]
    screen = live.read(screen_path)
    cases = screen["cases"]
    if len(cases) != 1 or cases[0]["family"] != FAMILY:
        raise ValueError("prompt prose screen changed; manual review required")
    if not cases[0]["argument"].rstrip().endswith("The phrase"):
        raise ValueError("candidate text changed; manual review required")
    review = {
        "schema": "prompt_prose_review_v1",
        "screen_sha256": live.file_hash(screen_path),
        "confirmed_incomplete_arguments": 1,
        "excluded_families": [FAMILY],
        "manual_basis": (
            "The exact raw argument ends with the unfinished clause 'The phrase'. "
            "Its raw provider content is identical to the parsed argument, the provider "
            "reported finish_reason=stop, and no parser-side clipping occurred."
        ),
        "policy": (
            "The primary eight-family analysis is unchanged. The sensitivity removes this "
            "family from all four cells, rather than removing or regenerating one response."
        ),
    }
    retained = sorted({row["family"] for row in rows} - {FAMILY})
    lookup = {(row["family"], row["level"], row["feedback"]): row for row in rows}
    if len(rows) != 32 or len(retained) != 7:
        raise ValueError("expected complete 8-family prompt grid")
    effects = []
    rng = np.random.default_rng(8167)
    for metric in METRICS:
        for level in ("selective", "explicit"):
            values = [
                float(lookup[f, level, "bundle"][metric])
                - float(lookup[f, level, "none"][metric])
                for f in retained
            ]
            effects.append(
                {
                    "metric": metric,
                    "level": level,
                    "contrast": "feedback_on_minus_off",
                    **interval(values, rng),
                }
            )
    args.destination.mkdir(parents=True, exist_ok=False)
    live.durable(args.destination / "review.json", review)
    with (args.destination / "contrasts.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(effects[0]))
        writer.writeheader()
        writer.writerows(effects)
    summary = {
        "schema": "prompt_complete_prose_sensitivity_v1",
        "primary_families": 8,
        "retained_families": 7,
        "retained_debates": 28,
        "excluded_families": [FAMILY],
        "contrasts": effects,
        "interpretation": (
            "Post-hoc descriptive sensitivity. It is not a replacement for the primary "
            "eight-family estimates. Intervals use 10,000 paired family-bootstrap draws "
            "with seed 8167 and do not include generation-seed uncertainty."
        ),
        "provenance": {
            "trajectories_sha256": live.file_hash(source),
            "judgments_sha256": live.file_hash(judgments_path),
            "screen_sha256": live.file_hash(screen_path),
            "script_sha256": live.file_hash(__file__),
        },
    }
    live.durable(args.destination / "summary.json", summary)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = live.read(ROOT / "reports/data/iteration6/prompt_behavior_v1/summary.json")
    primary_judges = live.read(ROOT / "reports/data/iteration6/live_judges_v1/summary.json")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.ravel()
    specs = (
        ("haiku_correct", "Haiku accuracy difference (pp)", 100),
        ("native_correct", "Native 4B accuracy difference (pp)", 100),
        ("words", "Mean words difference", 1),
        ("incorrect_assignment_mean_percentile", "Incorrect-role score difference", 1),
    )
    for axis, (metric, label, scale) in zip(axes, specs):
        for index, level in enumerate(("selective", "explicit")):
            if metric in ("haiku_correct", "native_correct"):
                judge_name = "haiku" if metric == "haiku_correct" else "native"
                p = next(x for x in primary_judges["policy_effects"] if x["group"] == "prompt:" + level and x["judge"] == judge_name)
                p = {"estimate": p["estimate"], "lower95": p["lo95"], "upper95": p["hi95"]}
            else:
                p = next(x for x in primary["contrasts"] if x["metric"] == metric and x["level"] == level)
            s = next(x for x in effects if x["metric"] == metric and x["level"] == level)
            for item, offset, color, name in ((p, -0.12, "#9aa1a6", "All 8"), (s, 0.12, "#1f6f8b", "Complete 7")):
                estimate = item["estimate"] * scale
                axis.errorbar(
                    estimate,
                    index + offset,
                    xerr=[[estimate - item["lower95"] * scale], [item["upper95"] * scale - estimate]],
                    fmt="o",
                    capsize=3,
                    color=color,
                    label=name if index == 0 else None,
                )
        axis.set_yticks((0, 1), ("selective", "explicit"))
        axis.axvline(0, color="gray", lw=0.8)
        axis.set_xlabel(label)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend()
    fig.suptitle("Prompt screen sensitivity: feedback on minus off\npaired family bootstrap; primary n=8, complete-prose n=7")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(args.destination / ("prompt_sensitivity." + extension), dpi=180, bbox_inches="tight")
    print(json.dumps({"retained_families": 7, "destination": str(args.destination)}, indent=2))


if __name__ == "__main__":
    main()
