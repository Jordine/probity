"""Plot the complete Haiku A/C replay as a separate post-result sensitivity."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reports.plot_analysis_summary import clean_axis, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--primary-pdf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("New output directory required")
    report = json.loads(args.summary.read_text())
    if not (report["post_result_sensitivity"] and report["primary_grid_unchanged"]
            and report["matching_original_request_hashes"] == 96
            and report["fresh_receipts"] == 96 and report["families"] == 24):
        raise ValueError("Incomplete identical-request sensitivity")
    rows = {r["cell"]: r for r in report["original_vs_fresh_AC"]
            if r["participant_model"] == "all"}
    for cell in ("A", "C"):
        row = rows[cell]
        if row["judgments"] != 48 or not np.isclose(
                row["fresh_correct"] - row["old_correct"], row["fixes"] - row["harms"]):
            raise ValueError("Replay counts disagree")
    if rows["all"]["judgments"] != 96 or rows["all"]["answer_disagreements"] != sum(
            rows[cell]["answer_disagreements"] for cell in ("A", "C")):
        raise ValueError("Incomplete disagreement counts")
    grids = [report["cached_original_grid"], report["fresh_AC_with_fixed_BD_sensitivity"]]
    effects = []
    for grid in grids:
        accuracy = {r["cell"]: r for r in grid["accuracy"] if r["participant_model"] == "all"}
        effect = next(r for r in grid["contrasts"] if r["participant_model"] == "all" and r["contrast"] == "D-A")
        if not np.isclose(effect["difference"], (accuracy["D"]["correct"] - accuracy["A"]["correct"]) / 48):
            raise ValueError("D-A disagrees with accuracy counts")
        effects.append(effect)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.hashsalt": "probity-grid4-replay-v1"})
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.8, 6.4))
    fig.subplots_adjust(left=.09, right=.95, top=.72, bottom=.25, wspace=.58)
    old, fresh = "#176e94", "#b96e25"
    clean_axis(left)
    for i, cell in enumerate(("A", "C")):
        row = rows[cell]
        values = [100 * row["old_correct"] / 48, 100 * row["fresh_correct"] / 48]
        left.plot(values, [i, i], color="#8b939c", lw=2)
        for value, color, count, offset, marker in zip(values, [old, fresh],
                [row["old_correct"], row["fresh_correct"]], [-18, 12], ["o", "D"]):
            left.scatter([value], [i], color=color, marker=marker, s=55, zorder=3)
            left.annotate(f"{count}/48", (value, i), xytext=(0, offset),
                          textcoords="offset points", ha="center", color=color, fontsize=10)
    left.set(xlim=(80, 100), ylim=(1.7, -.6), xlabel="Judge accuracy (%)",
             yticks=[0, 1], yticklabels=["A · no scores", "C · scores"])
    left.set_title("Repeat the exact original A/C requests", fontsize=12, pad=38, weight="bold")
    left.scatter([], [], color=old, label="Original", marker="o")
    left.scatter([], [], color=fresh, label="Fresh", marker="D")
    left.legend(loc="lower left", bbox_to_anchor=(-.02, 1.01), ncol=2, frameon=False)
    clean_axis(right, zero=0)
    for i, (effect, color) in enumerate(zip(effects, [old, fresh])):
        lo, hi = np.array(effect["family_bootstrap_95_ci"]) * 100
        value = effect["difference"] * 100
        right.plot([lo, hi], [i, i], color=color, lw=2)
        right.scatter([value], [i], color=color, s=55, zorder=3)
        right.annotate(f"{value:+.2f}", (hi, i), xytext=(6, 0),
                       textcoords="offset points", va="center", color=color, fontsize=10)
    right.set(xlim=(-1, 21), ylim=(1.7, -.6), xlabel="D − A (percentage points)",
              yticks=[0, 1], yticklabels=["Cached A/C\n(primary)", "Fresh A/C\n(sensitivity)"])
    right.set_title("Hold the new B/D results fixed", fontsize=12, pad=38, weight="bold")
    fig.suptitle("Haiku replay: how much does the reused baseline move?",
                 x=.045, y=.97, ha="left", fontsize=18, weight="bold")
    fig.text(.045, .89, "Post-result diagnostic · all 96 original A/C requests · the same 24 story families",
             fontsize=11, color="#475360")
    changes = rows["all"]["answer_disagreements"]
    fixes, harms = rows["all"]["fixes"], rows["all"]["harms"]
    fig.text(.045, .045, f"{changes}/96 answers changed: {fixes} corrections, {harms} new errors. All requests match; no receipt reuse. The primary grid is unchanged.\n"
             "Original routing: Anthropic 56 / Azure 40; replay and B/D: Azure. No backend fingerprints were returned.\n"
             "95% intervals resample the 24 families. Provider, time and sampling variation are not separately identified.",
             fontsize=9.5, color="#475360", linespacing=1.4)
    args.out.mkdir(parents=True)
    metadata = {"CreationDate": None, "ModDate": None, "Creator": "Probity audited aggregate plotting"}
    for suffix in ("png", "svg", "pdf"):
        kwargs = {"dpi": 170} if suffix == "png" else {"metadata": {"Date": None} if suffix == "svg" else metadata}
        fig.savefig(args.out / ("haiku_ac_replay." + suffix), **kwargs)
    plt.close(fig)
    import fitz
    with fitz.open(args.primary_pdf) as combined, fitz.open(args.out / "haiku_ac_replay.pdf") as extra:
        if len(combined) != 6 or len(extra) != 1:
            raise ValueError("Expected six primary pages and one supplementary page")
        combined.insert_pdf(extra)
        combined.set_metadata({"title": "Probity feedback grid and Haiku replay", "creator": "Probity"})
        combined.save(args.out / "probity_grid_and_replay.pdf", no_new_id=True)
    manifest = {"summary_sha256": digest(args.summary), "renderer_sha256": digest(__file__),
                "primary_pdf_sha256": digest(args.primary_pdf),
                "outputs": {p.name: digest(p) for p in sorted(args.out.iterdir())}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"primary_pages": 6, "supplementary_pages": 1, "answer_changes": changes}))


if __name__ == "__main__":
    main()
