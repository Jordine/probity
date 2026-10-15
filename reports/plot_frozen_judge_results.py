"""Plot all prespecified judge assistance contrasts from audited CSV exports."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

PARTICIPANTS = ["qwen/qwen3.5-27b", "qwen/qwen3.8-27b"]
DETECTORS = [
    ("adapted_primary", "Adapted response head"),
    ("apollo_response_mean", "Apollo response mean"),
    ("apollo_dyl_difference", "Apollo DYL difference"),
    ("published_fibs_dyl", "Published FIBS DYL"),
    ("local3_sentence", "New local sentence"),
    ("local3_clause", "New local clause"),
    ("local2_frozen_sentence", "Previous local sentence"),
]
COLORS = ["#176B9A", "#C76C21", "#6F4C9B"]
MARKERS = ["o", "s", "D"]


def read_csv(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty audited table: {path}")
    return rows


def unique_row(rows, **fields):
    matched = [row for row in rows if all(row.get(k) == v for k, v in fields.items())]
    if len(matched) != 1:
        raise ValueError(f"expected one row for {fields}, found {len(matched)}")
    return matched[0]


def interval(ax, value, low, high, y, color, marker):
    if low > high:
        raise ValueError("reversed confidence interval")
    ax.hlines(y, low, high, color=color, linewidth=1.5, alpha=0.8)
    ax.plot(value, y, marker=marker, color=color, markersize=6, linestyle="none")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table",
        nargs=2,
        action="append",
        metavar=("JUDGE", "DIRECTORY"),
        required=True,
    )
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument(
        "--fixture", action="store_true", help="Watermark non-result layout data."
    )
    args = parser.parse_args()
    if len(args.table) != 3 or len({name for name, _ in args.table}) != 3:
        raise ValueError(
            "provide the three distinct prespecified judges in increasing-strength order"
        )

    inputs, tables = {}, []
    for name, directory in args.table:
        directory = Path(directory)
        paths = {
            key: directory / (key + ".csv")
            for key in ("accuracies", "real_minus_neutral")
        }
        for path in paths.values():
            inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        tables.append((name, {key: read_csv(path) for key, path in paths.items()}))

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 10), gridspec_kw={"height_ratios": [2.1, 1]}
    )
    limits = [0.0]
    family_counts, plotted = set(), []
    for pi, participant in enumerate(PARTICIPANTS):
        ax = axes[0, pi]
        for ji, (name, table) in enumerate(tables):
            for di, (detector, _) in enumerate(DETECTORS):
                available = not (
                    participant.endswith("3.8-27b") and detector == "published_fibs_dyl"
                )
                available &= ji < 2 or detector in {
                    "adapted_primary",
                    "local3_sentence",
                }
                if not available:
                    continue
                row = unique_row(
                    table["real_minus_neutral"],
                    participant=participant,
                    detector=detector,
                )
                value, low, high = [
                    100 * float(row[k])
                    for k in (
                        "mean_difference",
                        "family_bootstrap_ci_low",
                        "family_bootstrap_ci_high",
                    )
                ]
                interval(
                    ax, value, low, high, di + (ji - 1) * 0.20, COLORS[ji], MARKERS[ji]
                )
                limits.extend([value, low, high])
                family_counts.add(int(row["n_families"]))
                plotted.append(
                    {
                        "judge": name,
                        "participant": participant,
                        "detector": detector,
                        "mean_pp": value,
                        "ci_low_pp": low,
                        "ci_high_pp": high,
                        "families": int(row["n_families"]),
                        "presentations": int(row["n_presentations"]),
                    }
                )
        ax.axvline(0, color="#555555", linewidth=1, linestyle="--", zorder=0)
        ax.set_yticks(range(len(DETECTORS)), [label for _, label in DETECTORS])
        ax.set_ylim(len(DETECTORS) - 0.5, -0.5)
        ax.set_title(
            "Qwen " + ("3.5" if pi == 0 else "3.8") + " 27B debaters",
            fontsize=13,
            pad=14,
        )
        ax.set_xlabel("Real − matched-neutral accuracy\n(percentage points)")
        ax.grid(axis="x", alpha=0.16)

        lower = axes[1, pi]
        for ji, (name, table) in enumerate(tables):
            for offset, condition, marker in [
                (-0.10, "none", MARKERS[ji]),
                (0.10, "oracle", "^"),
            ]:
                row = unique_row(
                    table["accuracies"], participant=participant, condition_id=condition
                )
                value, low, high = [
                    100 * float(row[k])
                    for k in (
                        "accuracy",
                        "family_bootstrap_ci_low",
                        "family_bootstrap_ci_high",
                    )
                ]
                interval(lower, value, low, high, ji + offset, COLORS[ji], marker)
                family_counts.add(int(row["n_families"]))
        lower.set_yticks(range(len(tables)), [name for name, _ in tables])
        lower.set_ylim(len(tables) - 0.5, -0.5)
        lower.set_xlim(-2, 102)
        lower.xaxis.set_major_formatter(PercentFormatter(100))
        lower.set_xlabel("Judge accuracy")
        lower.set_title("Raw debate baseline and labelled oracle", fontsize=11, pad=10)
        lower.grid(axis="x", alpha=0.16)

    margin = max(3, (max(limits) - min(limits)) * 0.08)
    for ax in axes[0]:
        ax.set_xlim(max(-102, min(limits) - margin), min(102, max(limits) + margin))
    legend = [
        Line2D(
            [0], [0], color=COLORS[i], marker=MARKERS[i], linestyle="none", label=name
        )
        for i, (name, _) in enumerate(tables)
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.55, 0.965),
        frameon=False,
    )
    fig.suptitle(
        "Probity: frozen probe assistance comparison", fontsize=17, x=0.55, y=0.995
    )
    footnote = (
        "Intervals: unadjusted 95% source-family bootstrap; "
        + ", ".join(map(str, sorted(family_counts)))
        + " families per participant cell.\n"
        "All frozen detector contrasts are shown. Opus evaluates only the two prespecified primary signals; FIBS is Qwen3.5 only.\n"
        "Lower panels: judge-specific marker = raw baseline; triangle = labelled oracle. Perfect sample scores do not imply perfect population accuracy."
    )
    fig.text(0.03, 0.018, footnote, fontsize=9, color="#444444", va="bottom")
    if args.fixture:
        fig.text(
            0.5,
            0.5,
            "SYNTHETIC LAYOUT CHECK — NOT RESULTS",
            color="#A51D29",
            alpha=0.18,
            fontsize=28,
            ha="center",
            rotation=22,
        )
    fig.subplots_adjust(
        left=0.18, right=0.98, top=0.89, bottom=0.15, hspace=0.42, wspace=0.62
    )
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(
            args.out_prefix.with_suffix(suffix),
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)
    manifest = {
        "schema": "frozen_judge_figure_v1",
        "synthetic_fixture": args.fixture,
        "inputs_sha256": inputs,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cells": plotted,
    }
    args.out_prefix.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "figure_prefix": str(args.out_prefix),
                "effect_cells": len(plotted),
                "synthetic_fixture": args.fixture,
            }
        )
    )


if __name__ == "__main__":
    main()
