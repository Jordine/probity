"""Plot every frozen local-score track without selecting tracks by results."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRACKS = [
    ("local3", "New local probe"),
    ("frozen_local2", "Previous local probe"),
    ("fixed_word_tfidf", "Fixed text classifier"),
    ("random_direction", "Development-selected random direction"),
    ("length", "Unit length"),
    ("negative_length", "Negative unit length"),
    ("position", "Unit position"),
    ("negative_position", "Negative unit position"),
    ("response_broadcast", "Response score copied to each unit"),
    ("published_dyl_broadcast", "FIBS DYL score copied to each unit"),
]
MODELS = [("qwen/qwen3.5-27b", "Qwen3.5-27B"), ("qwen/qwen3.8-27b", "Qwen3.8-27B")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--granularity", choices=["sentence", "clause"], default="sentence")
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    raw = args.results.read_bytes()
    result = json.loads(raw)
    annotation = result["annotation_summary"]
    if annotation["errors"] or annotation["expected_units"] != annotation["completed_units"]:
        raise ValueError("Complete independent annotations required")
    if annotation["expected_rows"] != annotation["completed_rows"]:
        raise ValueError("Complete response coverage required")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    plotted = []
    coverage_lines = []
    known_tracks = {name for name, _ in TRACKS}
    for ax, (model, label) in zip(axes, MODELS):
        cell = result["cells"][model][args.granularity]
        if set(cell["metrics"]) - known_tracks:
            raise ValueError("Unrecognized track would be omitted")
        cov = cell["coverage"]
        coverage_lines.append(
            f"{label}: {cov['accepted_units']}/{cov['all_units']} units with accepted truth labels; "
            f"{cov['all_responses']} complete responses; {cov['all_families']} source families."
        )
        primary = cell["metrics"].get("local3", {})
        ax.set_title(
            f"{label}\n{primary.get('mixed_responses', 0)} mixed responses in "
            f"{primary.get('mixed_parent_groups', 0)} eligible families",
            fontsize=13,
            pad=12,
        )
        for index, (track, _) in enumerate(TRACKS):
            metric = cell["metrics"].get(track, {})
            stat = metric.get("within_response_auroc")
            record = {"model": model, "track": track, "metric": stat}
            plotted.append(record)
            if stat is None:
                ax.text(0.02, index, "Unavailable" if not metric else "No mixed-label responses", va="center", color="#737373", fontsize=9)
                continue
            value = stat["parent_macro_mean"]
            low, high = stat["parent_bootstrap_95_ci"]
            if not all(math.isfinite(x) and 0 <= x <= 1 for x in [value, low, high]):
                raise ValueError("Invalid AUROC or interval")
            color = "#1766a4" if track == "local3" else "#2b8b79" if track == "frozen_local2" else "#777777"
            ax.hlines(index, low, high, color=color, linewidth=2)
            ax.plot(value, index, "o", color=color, markersize=7 if track == "local3" else 5)
        ax.axvline(0.5, color="#bbbbbb", linestyle="--", linewidth=1, zorder=0)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(len(TRACKS) - 0.5, -0.5)
        ax.set_yticks(range(len(TRACKS)), [label for _, label in TRACKS])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xlabel("Within-response AUROC, mean across source families")
        ax.grid(axis="x", color="#eeeeee", linewidth=0.6)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    title = f"Fresh local probe evaluation — {args.granularity} units"
    if args.fixture:
        title = "SYNTHETIC LAYOUT FIXTURE — NOT RESULTS"
        fig.text(0.6, 0.5, "FIXTURE", fontsize=70, alpha=0.08, rotation=25, ha="center")
    fig.suptitle(title, fontsize=16, y=0.98)
    footer = "\n".join(coverage_lines + [
        "Intervals resample source families (95%). Only responses containing both accepted supported and contradicted units contribute to local ranking.",
        "Unknown/mixed units stay in coverage counts. Labels are independent model judgments, not human ground truth. All frozen controls are shown.",
    ])
    fig.text(0.02, 0.02, footer, fontsize=9, va="bottom", linespacing=1.5)
    fig.subplots_adjust(left=0.29, right=0.98, top=0.84, bottom=0.22, wspace=0.12)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in [".png", ".svg"]:
        path = args.out_prefix.with_suffix(suffix)
        fig.savefig(path, dpi=180, facecolor="white")
        outputs.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    plt.close(fig)
    manifest = {
        "fixture": args.fixture,
        "input": str(args.results),
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "granularity": args.granularity,
        "selection": "All frozen score tracks in a fixed order; no outcome-based sorting or filtering.",
        "plotted_statistics": plotted,
        "coverage": {m: result["cells"][m][args.granularity]["coverage"] for m, _ in MODELS},
        "outputs": outputs,
    }
    args.out_prefix.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"outputs": outputs, "fixture": args.fixture}))


if __name__ == "__main__":
    main()
