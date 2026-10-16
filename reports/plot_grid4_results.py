"""Plot an audited four-cell analysis without fitting or selecting new results."""

import argparse
import csv
import json
from pathlib import Path
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from reports.plot_analysis_summary import JUDGES, digest, unique_index, clean_axis


MODELS = [("all", "Both Qwen 27B participants"),
          ("qwen/qwen3.5-27b", "Qwen 3.5 27B debates"),
          ("qwen/qwen3.8-27b", "Qwen 3.8 27B debates")]
CONTRASTS = [
    ("B-A", "B − A · Debaters receive feedback"),
    ("C-A", "C − A · Judge receives scores"),
    ("D-B", "D − B · Judge scores after feedback"),
    ("D-A", "D − A · Both receive scores"),
    ("(D-B)-(C-A)", "Interaction · (D − B) − (C − A)"),
]


def validate(report):
    accuracy = unique_index(report["accuracy"], ["judge", "participant_model", "cell"])
    effects = unique_index(report["contrasts"], ["judge", "participant_model", "contrast"])
    judges = [j for j in JUDGES if any(r["judge"] == j[0] for r in report["accuracy"])]
    if not judges or {r["judge"] for r in report["accuracy"]} != {j[0] for j in judges}:
        raise ValueError("Unexpected or empty judge coverage")
    for judge, _, _ in judges:
        for model, _ in MODELS:
            n = 48 if model == "all" else 24
            cells = {}
            for cell in "ABCD":
                row = accuracy[judge, model, cell]
                if row["episodes"] != n or row["families"] != 24:
                    raise ValueError("Incomplete matched story coverage")
                if not np.isclose(row["accuracy"], row["correct"] / n):
                    raise ValueError("Accuracy/count disagreement")
                cells[cell] = row["accuracy"]
            expected = {"B-A": cells["B"] - cells["A"],
                        "C-A": cells["C"] - cells["A"],
                        "D-B": cells["D"] - cells["B"],
                        "D-A": cells["D"] - cells["A"]}
            expected["(D-B)-(C-A)"] = expected["D-B"] - expected["C-A"]
            for name, value in expected.items():
                row = effects[judge, model, name]
                if row["episodes"] != n or row["families"] != 24 or not np.isclose(row["difference"], value):
                    raise ValueError("Paired effect disagrees with cell accuracies")
                if name != "(D-B)-(C-A)" and not np.isclose((row["fixes"] - row["harms"]) / n, value):
                    raise ValueError("Corrections/harms disagree with paired effect")
    return accuracy, effects, judges


def frame(fig, heading, subtitle, footer):
    fig.suptitle(heading, x=0.045, y=0.97, ha="left", fontsize=19, weight="bold")
    fig.text(0.045, 0.90, subtitle, ha="left", va="top", fontsize=11, color="#475360")
    fig.text(0.045, 0.025, footer, ha="left", va="bottom", fontsize=9.5,
             color="#475360", linespacing=1.4)


def grid_figure(accuracy, judges, model, label):
    fig, axes = plt.subplots(1, len(judges), figsize=(12.8, 6.4), squeeze=False)
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.21, top=0.76, wspace=0.29)
    for ax, (judge, name, _) in zip(axes[0], judges):
        cells = [["A", "C"], ["B", "D"]]
        values = np.array([[accuracy[judge, model, c]["accuracy"] for c in row] for row in cells])
        ax.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        ax.set_title(name, fontsize=13, pad=15, weight="bold")
        ax.set_xticks([0, 1], ["No scores", "Gets scores"], fontsize=10)
        ax.set_xlabel("Judge", fontsize=11, labelpad=10)
        ax.set_yticks([0, 1], ["No feedback", "Gets feedback"], fontsize=10)
        if ax is axes[0][0]:
            ax.set_ylabel("Debaters", fontsize=11, labelpad=10)
        else:
            ax.set_yticklabels([])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.axhline(0.5, color="white", lw=2)
        ax.axvline(0.5, color="white", lw=2)
        for y in range(2):
            for x in range(2):
                row = accuracy[judge, model, cells[y][x]]
                color = "white" if values[y, x] > 0.55 else "#17324d"
                ax.text(x, y - 0.18, cells[y][x], ha="center", va="center", color=color,
                        fontsize=13, weight="bold")
                ax.text(x, y + 0.02, f"{row['accuracy']:.1%}", ha="center", va="center",
                        color=color, fontsize=20, weight="bold")
                ax.text(x, y + 0.24, f"{row['correct']}/{row['episodes']}", ha="center",
                        va="center", color=color, fontsize=10)
    frame(fig, "The completed 2×2 grid", label + " · 24 matched original story families",
          "A/C share the original transcript; B/D share a new feedback transcript. Counts are correct judge answers.\n"
          "Frozen adapted response probe: a wrong-answer advocacy signal, not a local-lie probability. Cell color spans 0–100% accuracy.")
    return fig


def effect_figure(effects, judges, model, label):
    fig, ax = plt.subplots(figsize=(12.8, 7.5))
    fig.subplots_adjust(left=0.34, right=0.94, bottom=0.22, top=0.77)
    clean_axis(ax, zero=0)
    all_limits = []
    for j, (judge, name, color) in enumerate(judges):
        for i, (contrast, _) in enumerate(CONTRASTS):
            row = effects[judge, model, contrast]
            lo, hi = np.array(row["family_bootstrap_95_ci"]) * 100
            value = row["difference"] * 100
            if not np.isfinite([lo, hi, value]).all() or lo > hi:
                raise ValueError("Invalid interval")
            y = i + (j - (len(judges) - 1) / 2) * 0.22
            ax.plot([lo, hi], [y, y], color=color, lw=2)
            ax.scatter([value], [y], c=color, s=35, label=name if i == 0 else None, zorder=3)
            ax.annotate(f"{value:+.1f}", (hi, y), xytext=(5, 0), textcoords="offset points",
                        va="center", color=color, fontsize=9)
            all_limits.extend([lo, hi])
    ax.set_xlim(min(-10, min(all_limits) - 6), max(10, max(all_limits) + 12))
    ax.set_ylim(4.6, -0.6)
    ax.set_yticks(range(len(CONTRASTS)), [c[1] for c in CONTRASTS], fontsize=10.5)
    ax.set_xlabel("Difference in judge accuracy (percentage points)", fontsize=11)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.01, 1.04), ncol=len(judges), frameon=False)
    frame(fig, "Paired effects across the four cells", label + " · point estimates and family bootstrap 95% intervals",
          "The same 24 families are resampled jointly across cells and participant models. Intervals are unadjusted pilot intervals.\n"
          "Raw no-score A/B versus real-score C/D; these contrasts include display effects. Reused A/C and fresh B/D may differ in API routing.")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("New output directory required")
    report = json.loads(args.analysis.read_text())
    accuracy, effects, judges = validate(report)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "probity-grid4-v1",
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    args.out.mkdir(parents=True)
    summary = {key: report[key] for key in ["schema", "episodes_per_judge", "families", "detector",
               "baseline", "accuracy", "contrasts", "independent_unit", "limits"]}
    (args.out / "grid_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for key in ["accuracy", "contrasts"]:
        rows = summary[key]
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (args.out / (key + ".csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            writer.writerows(rows)
    metadata = {"CreationDate": None, "ModDate": None, "Creator": "Probity audited aggregate plotting"}
    with PdfPages(args.out / "probity_2x2_plots.pdf", metadata=metadata) as pdf:
        for index, (model, label) in enumerate(MODELS):
            for kind, function in [("grid", grid_figure), ("effects", effect_figure)]:
                figure = function(accuracy if kind == "grid" else effects, judges, model, label)
                stem = f"{index + 1:02d}_{kind}_{model.split('/')[-1]}"
                figure.savefig(args.out / (stem + ".png"), dpi=170)
                figure.savefig(args.out / (stem + ".svg"), metadata={"Date": None})
                figure.savefig(args.out / (stem + ".pdf"), metadata=metadata)
                pdf.savefig(figure)
                plt.close(figure)
    manifest = {"analysis_sha256": digest(args.analysis), "renderer_sha256": digest(__file__),
                "accuracy_cells": len(accuracy), "contrasts": len(effects),
                "outputs": {p.name: digest(p) for p in sorted(args.out.iterdir())}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(args.out / "probity_2x2_plots.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(args.out.iterdir()):
            if path.suffix != ".zip":
                archive.write(path, path.name)
    print(json.dumps({"accuracy_cells": len(accuracy), "contrasts": len(effects), "pdf_pages": 6}))


if __name__ == "__main__":
    main()
