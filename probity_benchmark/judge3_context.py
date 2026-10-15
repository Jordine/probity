"""Independently verify complete private-context diagnostic receipts."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from .judge3_final import audit_completion, file_hash, read_rows
from .preflight import load_problems


def compare_context(original, diagnostic, story):
    expected = [m for m in original if m["role"] != "system"]
    if (
        len(diagnostic) != len(expected) + 2
        or diagnostic[0]["role"] != "system"
        or diagnostic[-1]["role"] != "user"
        or diagnostic[1:-1] != expected
    ):
        raise ValueError("diagnostic changed the complete non-system context")
    if story not in "\n".join(m["content"] for m in diagnostic):
        raise ValueError("diagnostic lacks the complete authentic source")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("immutable new audit path required")
    problems, source_audit = load_problems(args.dataset)
    if source_audit["errors"]:
        raise ValueError("invalid source dataset")
    by_problem = {p.problem_id: p for p in problems}
    episodes = {
        e["episode_id"]: e
        for path in args.episodes.glob("episode_*.json")
        for e in [json.loads(path.read_text())]
    }
    generation = json.loads((args.episodes / "summary.json").read_text())
    diagnostic_summary = json.loads((args.diagnostic / "summary.json").read_text())
    if (
        generation["episodes"] != len(episodes)
        or generation["expected_episodes"] != len(episodes)
        or generation["errors"]
        or set(generation["finish_reasons"]) != {"stop"}
        or diagnostic_summary["errors"]
        or set(diagnostic_summary["finish_reasons"]) != {"stop"}
    ):
        raise ValueError("generation or diagnostic completion gate failed")
    original_rows = read_rows(args.episodes / "requests.jsonl")
    diagnostic_rows = read_rows(args.diagnostic / "requests.jsonl")
    originals = {r["request_sha256"]: r for r in original_rows if r.get("valid")}
    diagnostics = {
        (r["request_sha256"], r["tag"]["episode"], r["tag"]["speaker"]): r
        for r in diagnostic_rows
        if r.get("valid")
    }
    expected = {
        (eid, speaker)
        for eid, episode in episodes.items()
        for speaker in {turn["speaker"] for turn in episode["turns"]}
    }
    records = json.loads((args.diagnostic / "results.json").read_text())
    if (
        len(records) != len(expected)
        or {(r["episode_id"], r["speaker"]) for r in records} != expected
        or diagnostic_summary["completed"] != len(expected)
        or diagnostic_summary["expected"] != len(expected)
    ):
        raise ValueError("missing or duplicate contextual diagnostic result")
    reviewed = []
    for row in records:
        episode = episodes[row["episode_id"]]
        latest = {turn["speaker"]: turn for turn in episode["turns"]}
        turn = latest[row["speaker"]]
        if row["original_request_sha256"] != turn["request_sha256"]:
            raise ValueError("diagnostic does not reference the final speaker input")
        original = originals[turn["request_sha256"]]
        if (
            original["model"] != episode["model"]
            or original["tag"]["episode"] != episode["episode_id"]
            or original["tag"]["speaker"] != row["speaker"]
            or original["tag"]["round"] != turn["round"]
            or original["parsed"]["argument"] != turn["argument"]
            or original["parsed"]["answer"] != turn["answer"]
        ):
            raise ValueError("episode final turn differs from its generation receipt")
        diagnostic = diagnostics[
            (row["request_sha256"], row["episode_id"], row["speaker"])
        ]
        audit_completion(original)
        audit_completion(diagnostic)
        story = by_problem[episode["problem_id"]].story
        compare_context(
            original["request"]["messages"], diagnostic["request"]["messages"], story
        )
        correct = diagnostic["parsed"]["answer"] == episode["gold"]
        exact = diagnostic["parsed"]["evidence"] in story
        if (
            diagnostic["model"] != episode["model"]
            or row["gold"] != episode["gold"]
            or row["answer"] != diagnostic["parsed"]["answer"]
            or row["evidence"] != diagnostic["parsed"]["evidence"]
            or row["correct"] != correct
            or row["quote_exact"] != exact
        ):
            raise ValueError("diagnostic result disagrees with complete receipt/source")
        reviewed.append(
            {
                "episode_id": row["episode_id"],
                "speaker": row["speaker"],
                "original_request_sha256": turn["request_sha256"],
                "diagnostic_request_sha256": row["request_sha256"],
                "complete_non_system_context_preserved": True,
                "complete_source_present": True,
                "correct_answer": correct,
                "exact_source_evidence": exact,
            }
        )
    report = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": reviewed,
        "complete_contexts": len(reviewed),
        "correct_answers": sum(r["correct_answer"] for r in reviewed),
        "exact_source_evidence": sum(r["exact_source_evidence"] for r in reviewed),
        "diagnostic_finish_reasons": dict(
            Counter(r.get("finish_reason") for r in diagnostic_rows)
        ),
        "input_hashes": {
            str(path): file_hash(path)
            for path in [
                args.dataset,
                args.episodes / "requests.jsonl",
                args.episodes / "summary.json",
                args.diagnostic / "requests.jsonl",
                args.diagnostic / "results.json",
                args.diagnostic / "summary.json",
                Path(__file__),
            ]
        },
        "limit": "Instruction-switched check of complete input before the current public response, including source/plan/history. It does not include that response's later reasoning. Explicit private authentic answers certify retained answer access, not independent source reasoning or intent. Nonexact evidence strings remain unverified.",
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("rows", "input_hashes")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
