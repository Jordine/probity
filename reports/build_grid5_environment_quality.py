"""Audit exact public-turn repetition in the completed iteration-5 grid.

This is an additive, text-only diagnostic. It does not modify canonical episodes,
judge inputs, outcomes, probe scores, or the frozen analysis population.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ARMS = ("no_feedback", "feedback")
ARM_LABELS = {"no_feedback": "No feedback (A/C)", "feedback": "Live feedback (B/D)"}
PROVENANCE_DIRS = {
    ("new_iteration5", "no_feedback"): Path("runs/iteration5/debates_v4"),
    ("new_iteration5", "feedback"): Path("runs/iteration5/feedback_v1"),
    ("reused_iteration3_and_4", "no_feedback"): Path("runs/iteration3/final/debates_v1"),
    ("reused_iteration3_and_4", "feedback"): Path("runs/iteration4/feedback_v1"),
}
BLANK_LINE = re.compile(r"\n\s*\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def episode_path(root: Path, provenance: str, arm: str, episode_id: str) -> Path:
    try:
        directory = root / PROVENANCE_DIRS[(provenance, arm)]
    except KeyError as exc:
        raise ValueError(f"unknown provenance/arm: {provenance}/{arm}") from exc
    path = directory / f"episode_{episode_id}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def inspect_episode(pair: dict, arm: str, path: Path) -> dict:
    episode = read_json(path)
    expected_id = pair["no_feedback_episode_id" if arm == "no_feedback" else "feedback_episode_id"]
    if episode.get("episode_id") != expected_id:
        raise ValueError(f"episode identity mismatch: {path}")
    if episode.get("model") != pair["participant_model"]:
        raise ValueError(f"participant identity mismatch: {path}")
    if episode.get("parent_problem_id") != pair["parent_problem_id"]:
        raise ValueError(f"question identity mismatch: {path}")
    turns = episode.get("turns")
    if not episode.get("complete") or not isinstance(turns, list) or len(turns) != 6:
        raise ValueError(f"incomplete six-turn episode: {path}")

    by_speaker = defaultdict(list)
    blank_line_turns = 0
    for turn in turns:
        argument = turn.get("argument")
        speaker = turn.get("speaker")
        if not isinstance(argument, str) or not argument or not isinstance(speaker, str):
            raise ValueError(f"invalid public turn: {path}")
        by_speaker[speaker].append(argument)
        blank_line_turns += bool(BLANK_LINE.search(argument))
    if len(by_speaker) != 2 or sorted(map(len, by_speaker.values())) != [3, 3]:
        raise ValueError(f"expected two speakers with three turns each: {path}")

    speaker_any_repeat = sum(len(set(arguments)) < 3 for arguments in by_speaker.values())
    speaker_all_three_identical = sum(len(set(arguments)) == 1 for arguments in by_speaker.values())
    return {
        "matched_question_id": pair["matched_question_id"],
        "family_id": pair["family"],
        "participant_model": pair["participant_model"],
        "generation_provenance": pair["generation_provenance"],
        "arm": arm,
        "episode_id": expected_id,
        "episode_path": str(path.relative_to(path.parents[3])),
        "episode_sha256": digest(path),
        "trajectory_any_within_speaker_exact_repeat": bool(speaker_any_repeat),
        "speakers_with_any_exact_repeat": speaker_any_repeat,
        "speakers_all_three_identical": speaker_all_three_identical,
        "blank_line_turns": blank_line_turns,
        "speaker_trajectories": 2,
        "turns": 6,
    }


def aggregate(rows: list[dict], *, participant_model: str, provenance: str) -> list[dict]:
    selected = [
        row for row in rows
        if (participant_model == "all" or row["participant_model"] == participant_model)
        and (provenance == "all" or row["generation_provenance"] == provenance)
    ]
    output = []
    for arm in ARMS:
        arm_rows = [row for row in selected if row["arm"] == arm]
        trajectories = len(arm_rows)
        speaker_trajectories = sum(row["speaker_trajectories"] for row in arm_rows)
        turns = sum(row["turns"] for row in arm_rows)
        any_repeat = sum(row["trajectory_any_within_speaker_exact_repeat"] for row in arm_rows)
        speaker_repeat = sum(row["speakers_with_any_exact_repeat"] for row in arm_rows)
        all_three = sum(row["speakers_all_three_identical"] for row in arm_rows)
        blank_line = sum(row["blank_line_turns"] for row in arm_rows)
        output.append({
            "arm": arm,
            "participant_model": participant_model,
            "generation_provenance": provenance,
            "trajectories": trajectories,
            "trajectories_any_within_speaker_exact_repeat": any_repeat,
            "trajectory_any_repeat_rate": any_repeat / trajectories if trajectories else None,
            "speaker_trajectories": speaker_trajectories,
            "speakers_with_any_exact_repeat": speaker_repeat,
            "speaker_any_repeat_rate": speaker_repeat / speaker_trajectories if speaker_trajectories else None,
            "speakers_all_three_identical": all_three,
            "speaker_all_three_identical_rate": all_three / speaker_trajectories if speaker_trajectories else None,
            "turns": turns,
            "blank_line_turns": blank_line,
            "blank_line_turn_rate": blank_line / turns if turns else None,
        })
    return output


def build_summary(rows: list[dict], pairs_path: Path) -> dict:
    participants = sorted({row["participant_model"] for row in rows})
    provenances = sorted({row["generation_provenance"] for row in rows})
    aggregates = []
    aggregates += aggregate(rows, participant_model="all", provenance="all")
    for participant in participants:
        aggregates += aggregate(rows, participant_model=participant, provenance="all")
    for provenance in provenances:
        aggregates += aggregate(rows, participant_model="all", provenance=provenance)
    for participant in participants:
        for provenance in provenances:
            aggregates += aggregate(rows, participant_model=participant, provenance=provenance)
    return {
        "schema": "grid5_environment_quality_v1",
        "pairs_sha256": digest(pairs_path),
        "definitions": {
            "trajectory_any_within_speaker_exact_repeat": (
                "True when either speaker has fewer than three distinct argument strings across its three turns. "
                "Equality is exact Python string equality; no whitespace, case, or punctuation normalization is applied."
            ),
            "speaker_all_three_identical": (
                "A speaker-trajectory whose three public argument strings are exactly identical. "
                "There are two speaker-trajectories per six-turn trajectory."
            ),
            "blank_line_turn": (
                "A public argument containing a newline, optional whitespace, and another newline. "
                "This is only a mechanical proxy for the requested two-paragraph format."
            ),
        },
        "population": {
            "pairs": len(rows) // 2,
            "trajectories": len(rows),
            "speaker_trajectories": sum(row["speaker_trajectories"] for row in rows),
            "turns": sum(row["turns"] for row in rows),
        },
        "aggregates": aggregates,
        "scope": (
            "Additive environment-quality audit over the canonical completed trajectories joined by "
            "judge_grid_v1/pairs.json. No outcome, probe score, or text was used to select rows."
        ),
    }


def lookup(summary: dict, arm: str, participant: str = "all", provenance: str = "all") -> dict:
    matches = [row for row in summary["aggregates"] if row["arm"] == arm
               and row["participant_model"] == participant
               and row["generation_provenance"] == provenance]
    if len(matches) != 1:
        raise ValueError("aggregate lookup is not unique")
    return matches[0]


def render(summary: dict, out: Path) -> None:
    colors = {"no_feedback": "#376596", "feedback": "#CC6633"}
    participants = ["all", "qwen/qwen3.5-27b", "qwen/qwen3.8-27b"]
    participant_labels = ["All", "Qwen3.5-27B", "Qwen3.8-27B"]
    provenances = ["new_iteration5", "reused_iteration3_and_4"]
    provenance_labels = ["New 142 questions", "Reused 24-question pilot"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.4))
    width = .36
    x = np.arange(len(participants))
    for offset, arm in zip((-.18, .18), ARMS):
        values = [100 * lookup(summary, arm, participant=p)["trajectory_any_repeat_rate"] for p in participants]
        bars = axes[0].bar(x + offset, values, width, label=ARM_LABELS[arm], color=colors[arm])
        for bar, participant in zip(bars, participants):
            row = lookup(summary, arm, participant=participant)
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                         f"{row['trajectories_any_within_speaker_exact_repeat']}/{row['trajectories']}",
                         ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(x, participant_labels)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Trajectories with an exact repeat (%)")
    axes[0].set_title("Either speaker repeats among three turns")
    axes[0].grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False, loc="lower left")

    metrics = [("speaker_all_three_identical_rate", "All 3 speaker turns\nidentical"),
               ("blank_line_turn_rate", "Turn contains a blank line\n(two-paragraph proxy)")]
    x = np.arange(len(metrics))
    for offset, arm in zip((-.18, .18), ARMS):
        row = lookup(summary, arm)
        values = [100 * row[key] for key, _ in metrics]
        bars = axes[1].bar(x + offset, values, width, color=colors[arm], label=ARM_LABELS[arm])
        counts = [(row["speakers_all_three_identical"], row["speaker_trajectories"]),
                  (row["blank_line_turns"], row["turns"])]
        for bar, (num, den) in zip(bars, counts):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                         f"{num}/{den}", ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(x, [label for _, label in metrics])
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Population share (%)")
    axes[1].set_title("Severity and formatting proxies")
    axes[1].grid(axis="y", alpha=.2)

    x = np.arange(len(provenances))
    for offset, arm in zip((-.18, .18), ARMS):
        values = [100 * lookup(summary, arm, provenance=p)["trajectory_any_repeat_rate"] for p in provenances]
        bars = axes[2].bar(x + offset, values, width, color=colors[arm], label=ARM_LABELS[arm])
        for bar, provenance in zip(bars, provenances):
            row = lookup(summary, arm, provenance=provenance)
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                         f"{row['trajectories_any_within_speaker_exact_repeat']}/{row['trajectories']}",
                         ha="center", va="bottom", fontsize=8)
    axes[2].set_xticks(x, provenance_labels)
    axes[2].set_ylim(0, 105)
    axes[2].set_ylabel("Trajectories with an exact repeat (%)")
    axes[2].set_title("Repetition by generation provenance")
    axes[2].grid(axis="y", alpha=.2)

    fig.suptitle("Probity iteration 5: public-turn repetition and format adherence", fontsize=15)
    fig.text(.02, .040,
             "Exact string equality only; repeated ideas with different wording are not counted. "
             "Blank lines are a mechanical paragraph proxy, not a semantic quality judgment.",
             fontsize=8, color="#444444")
    fig.text(.02, .015,
             "Arms are separately generated with one seed and unrandomized provider routing. "
             "Live feedback = prior-turn percentiles, speaker means, and an optional recommendation from the frozen adapted response-mean head.",
             fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, .09, 1, .93), w_pad=2.2)
    fig.savefig(out, dpi=180, metadata={"Software": "probity reports/build_grid5_environment_quality.py"})
    plt.close(fig)


def readme_text(summary: dict) -> str:
    no_feedback = lookup(summary, "no_feedback")
    feedback = lookup(summary, "feedback")
    return f"""# Iteration-5 environment-quality diagnostic

This additive diagnostic reads all {summary['population']['trajectories']} canonical public trajectories joined by
`runs/iteration5/judge_grid_v1/pairs.json`. It does not change the frozen grid,
probe scores, judge inputs, outcomes, or inclusion decisions.

## Measures

- **Trajectory with any exact repeat:** either speaker has fewer than three
  distinct public `argument` strings across its three turns.
- **Speaker all-three identical:** all three argument strings for one speaker
  are exactly equal. There are two speaker-trajectories per debate trajectory.
- **Blank-line turn:** the argument contains `\\n`, optional whitespace, and a
  second `\\n`. This is a mechanical proxy for the requested two-paragraph
  format, not a semantic assessment.

String equality is exact. The audit does not normalize whitespace, punctuation,
or case, and therefore does not count paraphrased repetition.

## Pooled result

- No-feedback A/C transcript arm: **{no_feedback['trajectories_any_within_speaker_exact_repeat']}/{no_feedback['trajectories']}**
  trajectories contain an exact within-speaker repeat; **{no_feedback['speakers_all_three_identical']}/{no_feedback['speaker_trajectories']}**
  speaker-trajectories repeat the same argument all three times; **{no_feedback['blank_line_turns']}/{no_feedback['turns']}**
  turns contain a blank line.
- Live-feedback B/D transcript arm: **{feedback['trajectories_any_within_speaker_exact_repeat']}/{feedback['trajectories']}**
  trajectories contain an exact within-speaker repeat; **{feedback['speakers_all_three_identical']}/{feedback['speaker_trajectories']}**
  speaker-trajectories repeat the same argument all three times; **{feedback['blank_line_turns']}/{feedback['turns']}**
  turns contain a blank line.

`summary.csv` gives pooled, participant-model, provenance, and model-by-provenance
breakdowns. `trajectory_rows.csv` retains one text-free diagnostic row per
trajectory. The PNG presents the principal pooled and stratified rates.

## Limitations

Exact repetition is a narrow lower bound on repetitive behavior. This audit does
not measure semantic redundancy, quotation validity, whether a rebuttal engages
the opponent, factuality, deception, or overall debate quality. A blank line can
occur without two good paragraphs, and two semantic paragraphs can be formatted
without a blank line. Comparisons between arms are descriptive: the public
trajectories were separately generated and API provider routing was not
randomized. The reused 24-question pilot and 142-question expansion were created
at different times and are shown separately rather than treated as repetitions.

## Reproduction

From the repository root:

```bash
python3 reports/build_grid5_environment_quality.py
```

The command requires a new output directory by default. Remove or redirect the
existing additive output directories only when intentionally rebuilding them.
`manifest.json` records the input, script, table, README, and figure hashes.
"""


def run(root: Path, pairs_path: Path, data_out: Path, figure_out: Path) -> dict:
    pairs_path = root / pairs_path if not pairs_path.is_absolute() else pairs_path
    data_out = root / data_out if not data_out.is_absolute() else data_out
    figure_out = root / figure_out if not figure_out.is_absolute() else figure_out
    if data_out.exists() or figure_out.exists():
        raise ValueError("new immutable output directories required")
    pairs = read_json(pairs_path)
    if not isinstance(pairs, list) or len(pairs) != 332:
        raise ValueError("completed iteration-5 pairs must contain 332 question/model units")

    rows = []
    for pair in pairs:
        for arm in ARMS:
            episode_id = pair["no_feedback_episode_id" if arm == "no_feedback" else "feedback_episode_id"]
            path = episode_path(root, pair["generation_provenance"], arm, episode_id)
            rows.append(inspect_episode(pair, arm, path))
    if len(rows) != 664 or len({row["episode_id"] for row in rows}) != 664:
        raise ValueError("expected 664 distinct canonical trajectories")

    summary = build_summary(rows, pairs_path)
    expected = {
        "no_feedback": (278, 137, 263),
        "feedback": (242, 90, 97),
    }
    for arm, values in expected.items():
        row = lookup(summary, arm)
        observed = (row["trajectories_any_within_speaker_exact_repeat"],
                    row["speakers_all_three_identical"], row["blank_line_turns"])
        if observed != values:
            raise ValueError(f"frozen pooled count mismatch for {arm}: {observed} != {values}")

    data_out.mkdir(parents=True)
    figure_out.mkdir(parents=True)
    summary_path = data_out / "summary.json"
    summary_csv = data_out / "summary.csv"
    row_csv = data_out / "trajectory_rows.csv"
    readme = data_out / "README.md"
    figure = figure_out / "environment_quality.png"
    write_json(summary_path, summary)
    write_csv(summary_csv, summary["aggregates"])
    write_csv(row_csv, rows)
    readme.write_text(readme_text(summary))
    render(summary, figure)

    manifest = {
        "schema": "grid5_environment_quality_manifest_v1",
        "script_sha256": digest(Path(__file__)),
        "inputs": {display_path(pairs_path, root): digest(pairs_path)},
        "outputs": {
            display_path(path, root): digest(path)
            for path in (summary_path, summary_csv, row_csv, readme, figure)
        },
        "verified_pooled_counts": {
            arm: {
                "trajectories_any_repeat": lookup(summary, arm)["trajectories_any_within_speaker_exact_repeat"],
                "trajectories": lookup(summary, arm)["trajectories"],
                "speakers_all_three_identical": lookup(summary, arm)["speakers_all_three_identical"],
                "speaker_trajectories": lookup(summary, arm)["speaker_trajectories"],
                "blank_line_turns": lookup(summary, arm)["blank_line_turns"],
                "turns": lookup(summary, arm)["turns"],
            }
            for arm in ARMS
        },
    }
    write_json(data_out / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--pairs", type=Path, default=Path("runs/iteration5/judge_grid_v1/pairs.json"))
    parser.add_argument("--data-out", type=Path,
                        default=Path("reports/data/iteration5/environment_quality_v1"))
    parser.add_argument("--figure-out", type=Path,
                        default=Path("reports/figures/iteration5/environment_quality_v1"))
    args = parser.parse_args()
    manifest = run(args.root.resolve(), args.pairs, args.data_out, args.figure_out)
    print(json.dumps(manifest["verified_pooled_counts"], indent=2))


if __name__ == "__main__":
    main()
