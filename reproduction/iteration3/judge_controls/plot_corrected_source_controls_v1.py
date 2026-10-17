"""Project all audited source-control cells into a table and fixed-order figure."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


MODELS = [
    "Qwen/Qwen3.5-4B",
    "anthropic/claude-haiku-5.5",
    "qwen/qwen3.5-27b",
    "qwen/qwen3.8-27b",
    "anthropic/claude-opus-5.5",
]
LABELS = ["Qwen3.5 4B", "Haiku5.5", "Qwen3.5 27B", "Qwen3.8 27B", "Opus5.5 (low)"]
SUBGROUPS = [
    ("all", "All families", 24),
    ("short", "Short options", 13),
    ("descriptive", "Descriptive options", 11),
]


def sha256(path):
    data = path.read_bytes()
    assert len(data) == path.stat().st_size
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_bytes())
    assert report["schema"] == "probity_corrected_source_control_results_v1"
    assert report["native_clean_followup_included"] is True
    assert set(report["complete_models"]) == set(MODELS)
    assert len(report["observations"]) == 480
    cells = {
        (r["model"], r["kind"], r["subgroup"]): r
        for r in report["cells"]
    }
    assert len(cells) == len(report["cells"]) == 30
    ordered = []
    for subgroup, _, families in SUBGROUPS:
        for model in MODELS:
            for kind in ("no_story", "full_story"):
                row = cells[model, kind, subgroup]
                assert row["parent_groups"] == families
                assert row["episodes"] == 2 * families
                assert row["accuracy"] == row["correct"] / row["episodes"]
                low, high = row["parent_bootstrap_95_ci"]
                assert 0 <= low <= row["accuracy"] <= high <= 1
                ordered.append({
                    "model": model,
                    "kind": kind,
                    "subgroup": subgroup,
                    "correct": row["correct"],
                    "presentations": row["episodes"],
                    "source_families": row["parent_groups"],
                    "accuracy": row["accuracy"],
                    "family_bootstrap_ci_low": low,
                    "family_bootstrap_ci_high": high,
                })
    assert not args.out.exists(), "new immutable output directory required"
    args.out.mkdir(parents=True)
    table_path = args.out / "source_control_cells.csv"
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ordered[0]))
        writer.writeheader()
        writer.writerows(ordered)

    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 5.3), sharey=True)
    for ax, (subgroup, title, families) in zip(axes, SUBGROUPS):
        for offset, kind, color, label in [
            (-0.15, "no_story", "#c8641d", "No story"),
            (0.15, "full_story", "#136f9c", "Full story"),
        ]:
            selected = [cells[model, kind, subgroup] for model in MODELS]
            errors = [
                [row["accuracy"] - row["parent_bootstrap_95_ci"][0] for row in selected],
                [row["parent_bootstrap_95_ci"][1] - row["accuracy"] for row in selected],
            ]
            ax.errorbar(
                [row["accuracy"] for row in selected],
                [i + offset for i in range(len(MODELS))],
                xerr=errors, fmt="o", markersize=5.5, color=color,
                capsize=3, linewidth=1.4, label=label,
            )
        ax.set_title(f"{title}\n{families} families / {2 * families} presentations", fontsize=11)
        ax.set_xlim(0, 1.04)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.axvline(0.5, color="#888888", linestyle=":", linewidth=1)
        ax.grid(axis="x", alpha=0.15)
        ax.set_xlabel("Answer accuracy")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(len(MODELS)), LABELS)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.56, 0.90), ncol=2, frameon=False)
    fig.suptitle("Source-reading controls after source metadata cleanup", fontsize=14, y=0.985)
    fig.text(0.56, 0.925, "All 24 held-out source families retained; both answer orders", ha="center", fontsize=10)
    fig.text(0.5, 0.091, "Points: observed accuracy. Bars: 95% bootstrap intervals over source families (10,000 draws).", ha="center", fontsize=8.5)
    fig.text(0.5, 0.061, "Only source 114's full-story controls were rerun after removing an appended answer-key footer; original records remain intact.", ha="center", fontsize=8.5)
    fig.text(0.5, 0.031, "Source controls do not measure debate judging or probe uplift. Perfect sample accuracy does not imply population certainty.", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.12, 1, 0.845))
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(args.out / f"source_controls.{suffix}", dpi=200)
    plt.close(fig)
    outputs = [table_path] + [args.out / f"source_controls.{s}" for s in ("png", "svg", "pdf")]
    manifest = {
        "schema": "probity_corrected_source_control_display_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "report": str(args.report), "report_sha256": sha256(args.report),
        "renderer": str(Path(__file__).resolve()), "renderer_sha256": sha256(Path(__file__)),
        "dataset": report["dataset"], "dataset_sha256": report["dataset_sha256"],
        "rows_in": len(report["cells"]), "rows_out": len(ordered),
        "model_order": MODELS, "subgroup_order": [s[0] for s in SUBGROUPS],
        "projection": "All 30 audited aggregate cells copied in fixed order, without recalculating statistics or selecting outcomes.",
        "outputs_sha256": {str(p): sha256(p) for p in outputs},
        "limits": report["limits"],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
