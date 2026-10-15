"""Render the completed Probity pilot from audited public aggregates only.

No inference, relabelling, new bootstrap, or result-dependent track selection.
Produces standalone figures plus a PDF and ZIP containing the full plot set.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


JUDGES = [
    ("Qwen/Qwen3.5-4B", "Qwen 3.5 4B", "#166b91"),
    ("anthropic/claude-haiku-5.5", "Haiku 5.5", "#b56b24"),
    ("anthropic/claude-opus-5.5", "Opus 5.5", "#7853a3"),
]
MODELS = [
    ("qwen/qwen3.5-27b", "Qwen 3.5 27B", "#166b91"),
    ("qwen/qwen3.8-27b", "Qwen 3.8 27B", "#b56b24"),
]
DETECTORS = [
    ("adapted_primary", "Adapted response probe"),
    ("apollo_response_mean", "Apollo response mean"),
    ("apollo_dyl_difference", "Apollo DYL difference"),
    ("published_fibs_dyl", "Published FIBS DYL*"),
    ("local3_sentence", "New local sentence"),
    ("local3_clause", "New local clause"),
    ("local2_frozen_sentence", "Previous local sentence"),
]
BLUE, ORANGE, GREY = "#166b91", "#bd593b", "#8b949e"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def unique_index(rows, fields):
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in result:
            raise ValueError("Duplicate aggregate cell: " + str(key))
        result[key] = row
    return result


def clean_axis(ax, zero=None):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#e8edf1", linewidth=0.8)
    ax.set_axisbelow(True)
    if zero is not None:
        ax.axvline(zero, color="#87919b", linestyle="--", linewidth=1.1)


def title(fig, heading, subtitle, footer):
    fig.suptitle(heading, x=0.035, y=0.97, ha="left", fontsize=19, weight="bold")
    fig.text(0.035, 0.905, subtitle, fontsize=11.5, color="#475360", va="top")
    fig.text(0.035, 0.025, footer, fontsize=10, color="#475360", va="bottom", linespacing=1.4)


def interval(ax, value, lo, hi, y, color, marker="o", size=7):
    if not np.isfinite([value, lo, hi]).all() or not lo - 1e-10 <= value <= hi + 1e-10:
        raise ValueError("Invalid plotted interval")
    ax.hlines(y, lo, hi, color=color, linewidth=2.2)
    ax.plot(value, y, marker=marker, color=color, markersize=size, zorder=3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("reports/data/iteration3"))
    parser.add_argument("--existing-figures", type=Path, default=Path("reports/figures/iteration3"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Use a fresh output directory to preserve earlier plots")
    args.out.mkdir(parents=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11.5,
        "axes.labelsize": 11.5, "axes.titlesize": 13,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    inputs = [args.data / name for name in [
        "judges/real_minus_neutral.csv", "judges/accuracies.csv",
        "judges/exact_prompt_summary.csv", "response_role_summary_v1.json",
        "local_truth_by_advocacy_sentence_v2.json",
    ]]
    effects = [r for r in read_csv(inputs[0]) if r["participant"] == "all"]
    effect = unique_index(effects, ["judge", "detector"])
    accuracy_rows = [r for r in read_csv(inputs[1]) if r["participant"] == "all"]
    accuracy = unique_index(accuracy_rows, ["judge", "condition_id"])
    noise = unique_index(read_csv(inputs[2]), ["judge"])
    roles = json.loads(inputs[3].read_text())["cells"]
    role = unique_index(roles, ["model", "detector"])
    truth = json.loads(inputs[4].read_text())["cells"]
    if len(effects) != 16 or len(roles) != 9 or len(truth) != 4:
        raise ValueError("Unexpected frozen aggregate coverage")
    for row in effects:
        n = int(row["n_presentations"])
        if int(row["n_families"]) != 24 or n != (24 if row["detector"] == "published_fibs_dyl" else 48):
            raise ValueError("Unexpected independence unit or presentation count")
        if not np.isclose(float(row["mean_difference"]), (int(row["fixes"]) - int(row["harms"])) / n):
            raise ValueError("Effect disagrees with paired correction counts")
        if not np.isclose(float(row["mean_difference"]), float(row["real_accuracy"]) - float(row["neutral_accuracy"])):
            raise ValueError("Effect disagrees with real and neutral accuracy")
    outputs, figures = [], []

    def save(fig, stem, caption, cells):
        files = []
        for suffix in ("png", "svg", "pdf"):
            path = args.out / f"{stem}.{suffix}"
            fig.savefig(path, dpi=190)
            files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest(path)})
        figures.append({"stem": stem, "caption": caption, "plotted_cells": cells, "files": files})
        outputs.extend(args.out / x["name"] for x in files)
        plt.close(fig)

    # The two prespecified primary signals, pooled across both debater models.
    fig, ax = plt.subplots(figsize=(11, 6.4))
    primary = [("adapted_primary", "Adapted response probe", BLUE, "o"),
               ("local3_sentence", "New local sentence probe", ORANGE, "s")]
    cells = []
    for j, (judge, label, _) in enumerate(JUDGES):
        for k, (detector, _, color, marker) in enumerate(primary):
            row = effect[judge, detector]; cells.append(row)
            y = j + (-0.16 if k == 0 else 0.16)
            value, lo, hi = [100 * float(row[x]) for x in ["mean_difference", "family_bootstrap_ci_low", "family_bootstrap_ci_high"]]
            interval(ax, value, lo, hi, y, color, marker)
            ax.text(36, y, f"{value:+.1f} pp", va="center", fontsize=11, color=color)
    ax.set(yticks=range(3), yticklabels=[x[1] for x in JUDGES], ylim=(2.6, -0.6), xlim=(-35, 47))
    ax.set_xticks([-30, -20, -10, 0, 10, 20, 30])
    ax.set_xlabel("Change in judge answer accuracy vs matched neutral display (percentage points)")
    clean_axis(ax, 0)
    ax.legend(handles=[Line2D([], [], color=c, marker=m, label=l, linewidth=2) for _, l, c, m in primary],
              loc="lower left", bbox_to_anchor=(0, 1.04), ncol=2, frameon=False)
    title(fig, "Probe assistance depends on the judge", "Prespecified primary comparisons · both Qwen 27B debater models pooled",
          "48 debate presentations from 24 original story families per point. Bars: unadjusted 95% family bootstrap intervals.\nPositive values improve judge accuracy; negative values reduce it. These are paired effects, not raw baseline accuracy.")
    fig.subplots_adjust(left=0.19, right=0.97, top=0.74, bottom=0.23)
    save(fig, "01_primary_uplift", "The adapted response signal helps the small judge; the new local sentence display harms it.", cells)

    # Baseline and oracle headroom, with exact answer counts.
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 6.4), sharey=True)
    conditions = [("none", "No signal", GREY), ("adapted_primary:neutral", "Response neutral", "#6b9cba"),
                  ("adapted_primary:real", "Response probe", BLUE), ("local3_sentence:neutral", "Sentence neutral", "#d9a08a"),
                  ("local3_sentence:real", "Sentence probe", ORANGE), ("oracle", "Labelled oracle", "#917921")]
    cells = []
    for ax, (judge, label, _) in zip(axes, JUDGES):
        for y, (cid, _, color) in enumerate(conditions):
            row = accuracy[judge, cid]; cells.append(row)
            n, correct = int(row["n_presentations"]), int(row["n_correct"])
            assert n == 48 and np.isclose(correct / n, float(row["accuracy"]))
            value, lo, hi = [100 * float(row[x]) for x in ["accuracy", "family_bootstrap_ci_low", "family_bootstrap_ci_high"]]
            interval(ax, value, lo, hi, y, color)
            ax.text(106, y, f"{correct}/{n}", va="center", fontsize=10.5, color=color)
        ax.set(title=label, xlim=(0, 127), ylim=(5.6, -0.6), yticks=range(6), yticklabels=[x[1] for x in conditions])
        ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("Judge answer accuracy")
        clean_axis(ax, 50)
    title(fig, "Stronger judges start closer to the ceiling", "Raw baseline, each probe's neutral control, assisted accuracy, and oracle",
          "48 presentations / 24 story families per cell. Bars: unadjusted 95% family bootstrap intervals.\nOracle is explicitly labelled. A perfect sample score does not establish perfect population accuracy.")
    fig.subplots_adjust(left=0.16, right=0.98, top=0.79, bottom=0.23, wspace=0.14)
    save(fig, "02_judge_accuracy", "The high baseline measures judge answers: 37/48, 43/48 and 45/48 without probe assistance.", cells)

    # Every measured detector, in the same fixed order for all judges.
    fig, axes = plt.subplots(1, 3, figsize=(14, 7), sharey=True)
    for ax, (judge, label, color) in zip(axes, JUDGES):
        for y, (detector, _) in enumerate(DETECTORS):
            row = effect.get((judge, detector))
            if row is None:
                ax.text(-29, y, "Not evaluated", va="center", color=GREY, fontsize=10)
                continue
            value, lo, hi = [100 * float(row[x]) for x in ["mean_difference", "family_bootstrap_ci_low", "family_bootstrap_ci_high"]]
            interval(ax, value, lo, hi, y, color)
            ax.text(34, y, f"{value:+.1f}", va="center", fontsize=10, color=color)
        ax.set(title=label, xlim=(-35, 47), ylim=(6.6, -0.6), yticks=range(7), yticklabels=[x[1] for x in DETECTORS])
        ax.set_xticks([-30, -15, 0, 15, 30]); ax.set_xlabel("Real − neutral (pp)")
        clean_axis(ax, 0)
    title(fig, "All evaluated detector effects", "Paired changes in judge accuracy · pooled across debater models where available",
          "Bars: unadjusted 95% source-family bootstrap intervals; 24 independent families. All 16 pooled effects are shown.\n*FIBS: Qwen 3.5 debaters only, 24 presentations. Other cells: 48. Opus evaluated only the two prespecified primary signals.")
    fig.subplots_adjust(left=0.21, right=0.98, top=0.79, bottom=0.21, wspace=0.11)
    save(fig, "03_all_detector_uplift", "Every measured detector is retained, including unsuccessful comparisons and unavailable Opus cells.", effects)

    fig, axes = plt.subplots(1, 3, figsize=(14, 7), sharey=True)
    for ax, (judge, label, _) in zip(axes, JUDGES):
        for y, (detector, _) in enumerate(DETECTORS):
            row = effect.get((judge, detector))
            if row is None:
                ax.text(-8, y, "Not evaluated", va="center", color=GREY, fontsize=10); continue
            fixes, harms = int(row["fixes"]), int(row["harms"])
            ax.barh(y, fixes, height=0.48, color=BLUE)
            ax.barh(y, -harms, height=0.48, color=ORANGE)
            ax.text(-harms - 0.35, y, str(harms), ha="right", va="center", fontsize=10, color=ORANGE)
            ax.text(fixes + 0.35, y, str(fixes), ha="left", va="center", fontsize=10, color=BLUE)
        ax.set(title=label, xlim=(-11, 11), ylim=(6.6, -0.6), yticks=range(7), yticklabels=[x[1] for x in DETECTORS])
        ax.set_xticks([-10, -5, 0, 5, 10], ["10", "5", "0", "5", "10"])
        ax.set_xlabel("Introduced errors ←     → Fixed errors")
        clean_axis(ax, 0)
    fig.legend(handles=[Patch(color=ORANGE, label="Neutral correct → real wrong"), Patch(color=BLUE, label="Neutral wrong → real correct")],
               loc="upper center", bbox_to_anchor=(0.61, 0.86), ncol=2, frameon=False, fontsize=10.5)
    title(fig, "Which answers changed when the probe was shown?", "Corrections and new errors relative to the matched neutral display",
          "Counts are paired descriptive outcomes; unchanged answers are omitted. Net uplift = (fixes − harms) / presentations.\n48 presentations per cell; *FIBS uses 24 Qwen 3.5 presentations. Original story families, not presentations, are independent.")
    fig.subplots_adjust(left=0.21, right=0.98, top=0.74, bottom=0.21, wspace=0.11)
    save(fig, "04_fixes_and_harms", "The small judge gains eight fixes with the response probe and incurs nine new errors with the sentence probe.", effects)

    tracks = [("adapted_primary", "Adapted response probe"), ("apollo_response_mean", "Apollo response mean"),
              ("apollo_dyl_difference", "Apollo DYL difference"), ("apollo_dyl_no", "Apollo DYL No"),
              ("published_fibs_dyl", "Published FIBS DYL")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), sharey=True)
    compact_roles = []
    for ax, (model, label, color) in zip(axes, MODELS):
        for y, (detector, _) in enumerate(tracks):
            row = role.get((model, detector))
            if row is None:
                ax.text(0.08, y, "Unavailable", va="center", color=GREY); continue
            compact_roles.append({k: row[k] for k in ["model", "detector", "rows", "positives", "parent_groups", "auroc", "parent_bootstrap_95_ci"]})
            interval(ax, row["auroc"], *row["parent_bootstrap_95_ci"], y, color)
            ax.text(1.04, y, f"{row['auroc']:.3f}", va="center", fontsize=10.5, color=color)
        ax.set(title=label, xlim=(0, 1.2), ylim=(4.6, -0.6), yticks=range(5), yticklabels=[x[1] for x in tracks])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xlabel("Assigned-role AUROC")
        clean_axis(ax, 0.5)
    title(fig, "Whole-response scores distinguish assigned roles", "Incorrect-answer assignment versus correct-answer assignment",
          "144 complete turns per debater model, 72 in each role, across 24 original story families. Bars: 95% family bootstrap intervals.\nThese labels describe assigned roles. They do not establish which individual sentences are false or whether the model intended to deceive.")
    fig.subplots_adjust(left=0.25, right=0.97, top=0.78, bottom=0.23, wspace=0.13)
    save(fig, "05_response_role_detection", "High role AUROC is a separate measurement from sentence truth and from judge uplift.", compact_roles)

    fig, ax = plt.subplots(figsize=(12, 6.6))
    labels = []
    compact_truth = []
    for y, row in enumerate(truth):
        total = row["all_units"]
        assert sum(row[k] for k in ["supported", "contradicted", "unknown"]) == total
        model_label = next(x[1] for x in MODELS if x[0] == row["participant"])
        role_label = "correct answer assigned" if row["advocate_role"] == "correct_answer_advocate" else "wrong answer assigned"
        labels.append(model_label + "\n" + role_label)
        left = 0
        for key, color in [("supported", BLUE), ("contradicted", ORANGE), ("unknown", "#d9dfe4")]:
            width = row[key] / total * 100
            ax.barh(y, width, left=left, height=0.55, color=color, edgecolor="white", linewidth=1)
            if width >= 8:
                ax.text(left + width / 2, y, f"{row[key]}\n({width:.1f}%)", va="center", ha="center", color="white" if key != "unknown" else "#34414d", fontsize=10.5)
            elif row[key]:
                ax.annotate(str(row[key]), (left + width / 2, y - 0.25), xytext=(left + width / 2, y - 0.43), ha="center", fontsize=10, color=color)
            left += width
        ax.text(102, y, f"n={total}", va="center", fontsize=10.5)
        compact_truth.append({k: row[k] for k in ["participant", "advocate_role", "all_units", "responses", "families", "supported", "contradicted", "unknown"]})
    assert sum(x["all_units"] for x in truth) == 1977
    ax.set(xlim=(0, 117), ylim=(3.7, -0.7), yticks=range(4), yticklabels=labels, xlabel="Share of all fixed sentence units")
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    clean_axis(ax)
    ax.legend(handles=[Patch(color=c, label=l) for l, c in [("Supported", BLUE), ("Contradicted", ORANGE), ("Unknown / unaccepted", "#d9dfe4")]],
              loc="lower left", bbox_to_anchor=(0, 1.04), ncol=3, frameon=False)
    title(fig, "Assigned role and local truth are different labels", "All 1,977 fixed sentence units, including unknowns and annotator disagreement",
          "Descriptive unit counts, not independent observations. Each row spans 72 responses from 24 original story families.\nAccepted labels require independent model judgments and evidence; they are not human ground truth. Unknown units are not counted as truthful.")
    fig.subplots_adjust(left=0.25, right=0.97, top=0.72, bottom=0.23)
    save(fig, "06_sentence_truth_coverage", "Wrong-assigned responses still contain supported claims; correct-assigned responses can contain contradicted claims.", compact_truth)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.5), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    for y, (judge, label, _) in enumerate(JUDGES):
        row = noise[judge,]
        fresh, reused = int(row["fresh_generations_in_run"]), int(row["reused_logical_jobs"])
        assert fresh + reused == int(row["logical_jobs_total"])
        ax.barh(y, fresh, color=BLUE, height=0.48)
        ax.barh(y, reused, left=fresh, color="#d9dfe4", height=0.48)
        ax.text(fresh / 2, y, str(fresh), ha="center", va="center", color="white")
        if reused: ax.text(fresh + reused / 2, y, str(reused), ha="center", va="center", color="#34414d")
    ax.set(title="How logical judgments were obtained", yticks=range(3), yticklabels=[x[1] for x in JUDGES], ylim=(2.6, -0.6), xlim=(0, 1100), xlabel="Number of judgments")
    clean_axis(ax)
    ax.legend(handles=[Patch(color=BLUE, label="New generation"), Patch(color="#d9dfe4", label="Cached reuse")], loc="lower left", bbox_to_anchor=(0, 1.07), frameon=False, ncol=2, fontsize=10)
    row = noise["anthropic/claude-haiku-5.5",]
    pairs, same = int(row["generation_all_answer_pairs"]), int(row["generation_matching_answer_pairs"])
    groups, differing_groups = int(row["generation_repeated_prompt_groups"]), int(row["generation_groups_with_answer_disagreement"])
    ax = axes[1]
    values = [("Answer pairs", pairs - same, pairs), ("Prompt groups", differing_groups, groups)]
    for y, (label, different, total) in enumerate(values):
        percent = 100 * different / total
        ax.barh(y, percent, color=ORANGE, height=0.38)
        ax.text(percent + 0.15, y, f"{different}/{total} ({percent:.2f}%)", va="center", fontsize=11)
    ax.set(title="Haiku identical-request disagreements", yticks=range(2), yticklabels=[v[0] for v in values], ylim=(1.7, -0.7), xlim=(0, 5.1), xlabel="Disagreement rate (%) · axis shown from 0 to 5")
    ax.set_xticks([0, 1, 2, 3, 4, 5]); clean_axis(ax)
    title(fig, "The available repeatability check is limited", "Cached agreement does not measure generation noise",
          "Native Qwen: cached repeats only; fresh-repeat variability unmeasured. Opus: no repeated identical prompts.\nHaiku calls span Anthropic/Azure routes without fingerprints. This measures routing/runtime variability, not isolated sampling noise.\nPairs within a prompt group overlap; the two rates are descriptive summaries, not independent trials or confidence intervals.")
    fig.subplots_adjust(left=0.14, right=0.98, top=0.71, bottom=0.26, wspace=0.5)
    save(fig, "07_repeatability", "Only Haiku provides distinct repeated generations; three of 265 answer pairs differ, within two of 97 groups.", list(noise.values()))

    # Preserve the existing detailed plots, including every local control and
    # participant-specific judge effect. Convert their original SVGs to vector PDF.
    import cairosvg
    import fitz

    appendix = []
    for stem in ["local_sentence_results", "local_clause_results", "source_controls", "judge_results"]:
        source = args.existing_figures / f"{stem}.svg"
        path = args.out / f"appendix_{stem}.pdf"
        raw_pdf = cairosvg.svg2pdf(url=str(source))
        with fitz.open(stream=raw_pdf, filetype="pdf") as original:
            annotated = fitz.open()
            rect = original[0].rect
            page = annotated.new_page(width=rect.width, height=rect.height + 32)
            page.show_pdf_page(fitz.Rect(0, 32, rect.width, rect.height + 32), original, 0)
            scope = {
                "local_sentence_results": "Primary local evaluation: all 288 turns; accepted mixed-label responses determine within-response ranking.",
                "local_clause_results": "Secondary diagnostic only: 48 wrong-assigned openings. Distinct from the all-turn clause displays used by judges.",
                "source_controls": "Fresh corrected source controls: question-only versus full-story answers, with short-option subgroup reported separately.",
                "judge_results": "Detailed judge comparisons: every frozen participant-specific effect, plus raw no-signal baselines and labelled oracle.",
            }[stem]
            page.insert_text((16, 20), scope, fontsize=10, color=(0.25, 0.30, 0.35))
            annotated.save(path)
            annotated.close()
        outputs.append(path)
        appendix.append({"source": str(source), "source_sha256": digest(source), "pdf": path.name})
    combined = fitz.open()
    pdf_order = [args.out / f"{entry['stem']}.pdf" for entry in figures] + [args.out / entry["pdf"] for entry in appendix]
    for path in pdf_order:
        with fitz.open(path) as document:
            assert len(document) == 1
            combined.insert_pdf(document)
    combined_path = args.out / "probity_analysis_plots.pdf"
    combined.set_metadata({"title": "Probity analysis plots", "subject": "Completed Qwen pilot, frozen aggregates and scope notes"})
    combined.save(combined_path)
    combined.close()
    outputs.append(combined_path)
    manifest = {
        "schema": "probity_analysis_plot_collection_v1", "inference_calls": 0,
        "data_policy": "Existing public aggregates only; no new statistical analysis, omitted measured detector cells, or changed frozen metrics.",
        "script_sha256": digest(__file__), "inputs": [{"path": str(p), "sha256": digest(p)} for p in inputs],
        "checks": {"pooled_effect_cells": len(effects), "role_cells": len(roles), "truth_cells": len(truth), "primary_sentences": 1977,
                   "paired_effect_equals_fix_minus_harm": True, "paired_effect_equals_accuracy_difference": True,
                   "unknown_in_all_sentence_denominators": True, "unmeasured_noise_distinct_from_zero": True},
        "figures": figures, "appendix": appendix, "pdf_pages": len(pdf_order),
        "output_files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": digest(p)} for p in outputs],
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(args.out / "probity_analysis_plots.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in outputs + [manifest_path]:
            archive.write(path, path.name)
        for stem in ["local_sentence_results", "local_clause_results", "source_controls", "judge_results"]:
            for suffix in ["png", "svg"]:
                path = args.existing_figures / f"{stem}.{suffix}"
                archive.write(path, "appendix_" + path.name)
    print(json.dumps({"output_directory": str(args.out), "new_figures": len(figures), "pdf_pages": len(pdf_order),
                      "pdf_bytes": combined_path.stat().st_size, "zip_bytes": (args.out / 'probity_analysis_plots.zip').stat().st_size,
                      "checks": manifest["checks"]}, indent=2))


if __name__ == "__main__":
    main()
