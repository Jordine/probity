"""Export a fixed development-only manual factual audit with exact coordinates.

The annotations are supplied explicitly; this module does not infer truth from
roles or select responses/units from probe scores. No paid calls are made.
"""

import argparse
from collections import Counter
import json
from pathlib import Path

from .local3_annotate_units import UNIT_RULES, load_inputs
from .preflight import sha256


def build(args):
    if args.out.exists():
        raise ValueError("immutable output directory required")
    manual = json.loads(args.annotations.read_text())
    rows, stories = load_inputs(args.dataset, [args.episodes], ["sentence"])
    episodes = {
        e["episode_id"]: e
        for e in map(
            json.loads, (p.read_text() for p in args.episodes.glob("episode_*.json"))
        )
    }
    selected = []
    for row in rows:
        episode = episodes[row["source_episode"]]
        first_wrong = next(
            i
            for i, t in enumerate(episode["turns"])
            if t["assigned_answer"] != episode["gold"]
        )
        if row["world"] == "original" and row["turn_index"] == first_wrong:
            selected.append(row)
    expected_keys = {
        f"{line}:{model}"
        for line in (90, 112, 128, 137)
        for model in ("qwen/qwen3.5-27b", "qwen/qwen3.8-27b")
    }
    if {
        f"{r['parent_source_line']}:{r['participant_model']}" for r in selected
    } != expected_keys:
        raise ValueError("fixed eight-response selection changed")
    if set(manual["responses"]) != expected_keys or len(selected) != 8:
        raise ValueError("manual coverage differs from deterministic selection")
    for row in selected:
        key = f"{row['parent_source_line']}:{row['participant_model']}"
        judgments = manual["responses"][key]
        if len(judgments) != len(row["units"]):
            raise ValueError("not every fixed unit has a manual decision")
        source = stories[row["source_problem"]]
        paragraphs = source.split("\n\n")
        anchors = manual["source_anchors"][str(row["parent_source_line"])]
        evidence_ids = sorted(
            {i for anchor in anchors for i, p in enumerate(paragraphs) if anchor in p}
        )
        if any(not any(anchor in p for p in paragraphs) for anchor in anchors):
            raise ValueError("exact evidence anchor absent from full source")
        row.update(
            source="manual_environment3_development_shift_audit",
            source_story=source,
            source_story_sha256=sha256(source),
            response_sha256=sha256(row["response"]),
            label_status="Single Codex agent manual review, not human ground truth or independent silver consensus; frozen-head development shift diagnostic only.",
            annotations=[],
        )
        for i, (unit, check) in enumerate(zip(row["units"], judgments)):
            verdict, reason = check
            if verdict not in ("supported", "contradicted", "unknown"):
                raise ValueError("unknown manual verdict")
            label = None if verdict == "unknown" else int(verdict == "contradicted")
            if row["response"][unit["start"] : unit["end"]] != unit["text"]:
                raise ValueError("raw response coordinate mismatch")
            annotation = {
                "unit_index": i,
                "unit": unit,
                "label": label,
                "verdict": verdict,
                "rationale": reason,
                "rejection_reasons": [] if label is not None else ["manual_unknown"],
                "evidence_paragraph_ids": evidence_ids,
                "evidence_paragraphs": [paragraphs[j] for j in evidence_ids],
            }
            row["annotations"].append(annotation)
            if label is not None:
                row["spans"].append(
                    {
                        **unit,
                        "label": label,
                        "unit_index": i,
                        "label_kind": "manual_fixed_unit_development_diagnostic",
                    }
                )
        if (
            "".join(u["display_text"] for u in row["units"])
            != episodes[row["source_episode"]]["turns"][row["turn_index"]]["argument"]
        ):
            raise ValueError("complete argument partition changed")
    args.out.mkdir(parents=True)
    corpus = args.out / "corpus.jsonl"
    corpus.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected)
    )
    summary = {
        "rows": len(selected),
        "parents": len({r["group"] for r in selected}),
        "units": sum(len(r["units"]) for r in selected),
        "verdicts": dict(
            Counter(a["verdict"] for r in selected for a in r["annotations"])
        ),
        "selection": "Original world, first wrong-advocate turn for every old development parent and both participant models; all fixed sentence units retained.",
        "annotation_policy": UNIT_RULES,
        "blinding": "No probe scores inspected. Annotator knew source and role; diagnostic labels are not blinded or human-adjudicated.",
        "use_limit": "No fitting, no tuning, no selection of additional turns to force class balance. If no supported units remain, AUROC and specificity are undefined; report coverage and contradicted-unit scores only.",
        "source_artifacts": {
            "dataset_sha256": sha256(args.dataset.read_text()),
            "episode_manifest_sha256": sha256(
                (args.episodes / "manifest.json").read_text()
            ),
            "annotations_sha256": sha256(args.annotations.read_text()),
            "script_sha256": sha256(Path(__file__).read_text()),
            "corpus_sha256": sha256(corpus.read_text()),
        },
        "granularity_caveat": "Fixed punctuation partition preserves all characters but splits 'Dr. ' and can join quoted sentences. Mixed factual units and ambiguous references remain unknown; no truth-selected snippets exported as labels.",
    }
    if summary["units"] != 35:
        raise ValueError("fixed selection no longer contains 35 units")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "annotation_policy"}, indent=2
        )
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    build(p.parse_args())


if __name__ == "__main__":
    main()
