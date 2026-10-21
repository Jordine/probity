"""Render aggregate-only family-macro corpus-grid figures, without private text."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np


LABELS = {"all": "All source families", "exposure:previously_used": "Previously used families",
          "exposure:untouched": "Untouched before expansion", "exposure:training": "Detector training",
          "exposure:development_reference": "Development / reference", "exposure:other_tuning": "Other prior tuning",
          "exposure:previously_evaluated": "Earlier evaluation families",
          "secondary:source_supported": "Reviewer-supported sources",
          "secondary:source_supported_and_eligible": "Reviewer-supported + eligible"}
CONTRASTS = ("B-A", "C-A", "D-B", "D-A", "(D-B)-(C-A)")
CONTRAST_LABELS = ("B − A: debater feedback", "C − A: judge scores, original text",
                   "D − B: judge scores, feedback text", "D − A: combined comparison", "Interaction: (D − B) − (C − A)")
CELL_LAYOUT = (("A", "C"), ("B", "D"))
COLORS = ("#177E89", "#8A4F9D", "#CC6633", "#376596")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def judge_label(value):
    return "Native Qwen3.5-4B" if "4b" in value.lower() else "Haiku5.5" if "haiku" in value.lower() else value


def participant_label(value):
    return "Qwen3.5-27B" if "3.5" in value else "Qwen3.8-27B" if "3.8" in value else value


def rows_for(rows, judge, cohort="all", model="all"):
    return [r for r in rows if r["judge"] == judge and r["cohort"] == cohort and r["participant_model"] == model]


def footnote(fig, extra=""):
    text = "Equal-weight source-family means; all questions and both participant models stay paired within family."
    if extra:
        text += "\n" + extra
    fig.text(.03, .015, text, fontsize=8, color="#444444", va="bottom")


def grid_figure(data, judges):
    fig, axes = plt.subplots(1, len(judges), figsize=(6.4 * len(judges), 5.4), squeeze=False)
    for column, judge in enumerate(judges):
        ax = axes[0, column]
        rows = {r["cell"]: r for r in rows_for(data["accuracy"], judge)}
        values = np.array([[rows[c]["family_macro_accuracy"] * 100 for c in line] for line in CELL_LAYOUT])
        ax.imshow(values, vmin=0, vmax=100, cmap="Blues")
        for y, line in enumerate(CELL_LAYOUT):
            for x, cell in enumerate(line):
                row = rows[cell]
                color = "white" if values[y, x] > 65 else "#102733"
                ax.text(x, y - .13, f"{cell}: {100 * row['family_macro_accuracy']:.1f}%",
                        ha="center", va="center", fontsize=14, color=color)
                ax.text(x, y + .03, f"{row['correct']}/{row['episodes']} correct",
                        ha="center", va="center", fontsize=10, color=color)
                ax.text(x, y + .17, f"Question mean: {100 * row['question_weighted_accuracy']:.1f}%",
                        ha="center", va="center", fontsize=10, color=color)
        ax.set_xticks([0, 1], ["Judge without scores", "Judge with scores"])
        ax.set_yticks([0, 1], ["Debaters without\nfeedback", "Debaters with\nfeedback"])
        ax.tick_params(length=0)
        ax.set_title(judge_label(judge) + " — family-macro accuracy", pad=12, fontsize=13)
    fig.suptitle(f"Expanded matched grid: {data['questions']} questions / {data['families']} source families", fontsize=15)
    fig.subplots_adjust(left=.12, right=.98, bottom=.15, top=.86, wspace=.38)
    footnote(fig, "A/C share a transcript; B/D share a transcript. Counts are descriptive question/model presentations.")
    return fig


def effects_panel(ax, rows, color, title):
    by_contrast = {r["contrast"]: r for r in rows}
    selected = [by_contrast[c] for c in CONTRASTS if c in by_contrast]
    if not selected:
        ax.text(.5, .5, "No complete cohort", transform=ax.transAxes, ha="center")
        return
    y = np.arange(len(selected))[::-1]
    mean = np.array([r["family_macro_difference"] for r in selected]) * 100
    bounds = np.array([r["family_bootstrap_95_ci"] for r in selected]) * 100
    ax.errorbar(mean, y, xerr=np.maximum(0, np.vstack((mean - bounds[:, 0], bounds[:, 1] - mean))),
                fmt="o", color=color, capsize=4, linewidth=1.5)
    ax.axvline(0, color="#888888", linewidth=.8)
    ax.set_yticks(y, [CONTRAST_LABELS[CONTRASTS.index(r["contrast"])] for r in selected])
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Paired family-macro difference (percentage points)")
    ax.grid(axis="x", alpha=.2)
    for point, level, bound in zip(mean, y, bounds):
        ax.annotate(f"{point:+.1f} [{bound[0]:+.1f}, {bound[1]:+.1f}]", (point, level),
                    xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8)
    ax.margins(x=.30, y=.18)


def effects_figure(data, judges, by_model=False):
    models = sorted({r["participant_model"] for r in data["accuracy"] if r["participant_model"] != "all"}) if by_model else ["all"]
    fig, axes = plt.subplots(len(models), len(judges), figsize=(8 * len(judges), 4.6 * len(models) + .6), squeeze=False)
    for i, model in enumerate(models):
        for j, judge in enumerate(judges):
            title = judge_label(judge) + (" / " + participant_label(model) if model != "all" else "")
            effects_panel(axes[i, j], rows_for(data["contrasts"], judge, model=model), COLORS[j], title)
    fig.suptitle("Paired grid contrasts" + (" by participant model" if by_model else " — full corpus"), fontsize=15)
    fig.tight_layout(rect=(.01, .10 if not by_model else .07, .99, .95), w_pad=3)
    footnote(fig, "Unadjusted 95% family bootstrap intervals. Feedback comparisons retain generation and intervention-bundle limitations.")
    return fig


def exposure_grid(data, judges):
    cohorts = [c for c in ("all", "exposure:previously_used", "exposure:untouched")
               if any(r["cohort"] == c for r in data["accuracy"])]
    fig, axes = plt.subplots(len(judges), len(cohorts), figsize=(5 * len(cohorts), 3.4 * len(judges) + .8), squeeze=False)
    for i, judge in enumerate(judges):
        for j, cohort in enumerate(cohorts):
            ax = axes[i, j]
            rows = {r["cell"]: r for r in rows_for(data["accuracy"], judge, cohort)}
            points = [rows[c]["family_macro_accuracy"] * 100 for c in "ABCD"]
            lower = [rows[c]["family_bootstrap_95_ci"][0] * 100 for c in "ABCD"]
            upper = [rows[c]["family_bootstrap_95_ci"][1] * 100 for c in "ABCD"]
            ax.errorbar(range(4), points, yerr=np.maximum(0, [np.array(points) - lower, np.array(upper) - points]),
                        color=COLORS[i], fmt="o-", capsize=4)
            for x, y in enumerate(points):
                ax.annotate(f"{y:.1f}", (x, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
            ax.set_xticks(range(4), list("ABCD"))
            ax.set_ylim(0, 105); ax.set_xlim(-.3, 3.3); ax.grid(axis="y", alpha=.2)
            ax.set_title(f"{LABELS[cohort]}\n{rows['A']['families']} families / {rows['A']['questions']} questions", fontsize=11)
            ax.set_ylabel(judge_label(judge) + "\nFamily-macro accuracy (%)")
    fig.suptitle("Prior exposure and untouched-family generalization", fontsize=15)
    fig.tight_layout(rect=(.01, .11, .99, .94), w_pad=2)
    footnote(fig, "Untouched means no use before this expansion; the full corpus includes training, development and earlier evaluation families.")
    return fig


def strata_effects(data, judges):
    cohorts = [c for c in LABELS if c not in ("all", "exposure:previously_used")
               and any(r["cohort"] == c for r in data["contrasts"])]
    fig, axes = plt.subplots(1, len(judges), figsize=(8 * len(judges), .65 * len(cohorts) + 2.6), squeeze=False)
    for j, judge in enumerate(judges):
        ax = axes[0, j]
        labels = []
        for y, cohort in enumerate(cohorts[::-1]):
            rows = {r["contrast"]: r for r in rows_for(data["contrasts"], judge, cohort)}
            labels.append(f"{LABELS[cohort]} ({rows['D-A']['families']} families)")
            for contrast, offset, color in (("B-A", -.12, COLORS[0]), ("D-A", .12, COLORS[1])):
                row = rows[contrast]
                mean = row["family_macro_difference"] * 100
                lo, hi = np.array(row["family_bootstrap_95_ci"]) * 100
                ax.errorbar(mean, y + offset, xerr=[[max(0, mean - lo)], [max(0, hi - mean)]],
                            fmt="o", color=color, capsize=3, label=contrast if y == 0 else None)
        ax.set_yticks(range(len(labels)), labels)
        ax.axvline(0, color="#888888", linewidth=.8); ax.grid(axis="x", alpha=.2)
        ax.set_xlabel("Paired family-macro difference (percentage points)")
        ax.set_title(judge_label(judge)); ax.legend(loc="best", frameon=False)
        ax.margins(x=.15, y=.12)
    fig.suptitle("Feedback comparisons by exposure and secondary diagnostic cohorts", fontsize=14)
    fig.tight_layout(rect=(.01, .13, .99, .93), w_pad=3)
    footnote(fig, "Reviewer-supported is a source diagnostic, not gold certification. Secondary diagnostic cohorts may overlap exposure strata.")
    return fig


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def render(summary, out):
    if out.exists():
        raise ValueError("new immutable plot directory required")
    data = json.loads(summary.read_text())
    if data.get("schema") != "grid5_public_summary_v1":
        raise ValueError("aggregate-only expansion summary required")
    for row in data["accuracy"]:
        if row["accuracy"] != row["family_macro_accuracy"]:
            raise ValueError("plot input confuses question-weighted and family-macro accuracy")
    judges = sorted({r["judge"] for r in data["accuracy"]}, key=lambda x: ("4b" not in x.lower(), x))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": "probity-grid5-v1",
                         "axes.spines.top": False, "axes.spines.right": False})
    out.mkdir(parents=True)
    figures = [("01_full_grid", grid_figure(data, judges)),
               ("02_paired_effects", effects_figure(data, judges)),
               ("03_exposure_grid", exposure_grid(data, judges)),
               ("04_participant_effects", effects_figure(data, judges, True)),
               ("05_strata_effects", strata_effects(data, judges))]
    with PdfPages(out / "probity_expanded_grid.pdf", metadata={"CreationDate": None, "ModDate": None}) as pdf:
        for name, fig in figures:
            pdf.savefig(fig)
            for suffix in ("png", "svg", "pdf"):
                metadata = {"Date": None} if suffix == "svg" else {"CreationDate": None, "ModDate": None} if suffix == "pdf" else {}
                fig.savefig(out / (name + "." + suffix), dpi=160, metadata=metadata)
            plt.close(fig)
    write_csv(out / "accuracy.csv", data["accuracy"])
    write_csv(out / "contrasts.csv", data["contrasts"])
    files = sorted(out.iterdir())
    with zipfile.ZipFile(out / "probity_expanded_grid_plots.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            item = zipfile.ZipInfo(path.name, date_time=(1980, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(item, path.read_bytes())
    manifest = {"schema": "grid5_figure_manifest_v1", "summary_sha256": digest(summary),
                "renderer_sha256": digest(__file__), "pdf_pages": len(figures),
                "primary_estimator": "equal-weight original source-family mean",
                "files": {p.name: digest(p) for p in sorted(out.iterdir())}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"figures": len(figures), "files": len(manifest["files"]), "out": str(out)}))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.summary, args.out)


if __name__ == "__main__":
    main()
