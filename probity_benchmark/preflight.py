"""Strict dataset loading, balanced baseline views, and complete audit receipts."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import re


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Problem:
    problem_id: str
    story: str
    question: str
    options: tuple[str, str]
    gold: str
    source_line: int

    def view(self, with_story: bool, swapped: bool) -> tuple[list[dict], str]:
        options = self.options[::-1] if swapped else self.options
        gold = {"A": "B", "B": "A"}[self.gold] if swapped else self.gold
        # Only these explicitly allowed fields enter a model prompt. The source
        # row contains answer keys and generation/evaluation metadata.
        body = f"Story:\n{self.story}\n\n" if with_story else ""
        body += f"Question:\n{self.question}\n\nA: {options[0]}\nB: {options[1]}"
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the two-option reading-comprehension question. "
                    "If no story is supplied, make your best choice from the question "
                    "and options alone. Return exactly A or B, without explanation."
                ),
            },
            {"role": "user", "content": body},
        ]
        return messages, gold


def load_problems(path: Path) -> tuple[list[Problem], dict]:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    problems = []
    errors = []
    nonempty = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        nonempty += 1
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("dataset row must be an object")
            required = (
                "synthetic_story",
                "question",
                "option_a",
                "option_b",
                "correct_answer",
            )
            if any(
                not isinstance(row.get(k), str) or not row[k].strip() for k in required
            ):
                raise ValueError("required text field missing or empty")
            gold = row["correct_answer"].strip().upper()
            if gold not in ("A", "B"):
                raise ValueError("gold label must be A or B")
            options = (row["option_a"], row["option_b"])
            if options[0].strip() == options[1].strip():
                raise ValueError("answer options are identical")
            fingerprint = sha256(
                json.dumps([row[k] for k in required], ensure_ascii=False)
            )
            problems.append(
                Problem(
                    fingerprint,
                    row["synthetic_story"],
                    row["question"],
                    options,
                    gold,
                    line_number,
                )
            )
        except (ValueError, TypeError, KeyError) as exc:
            errors.append({"line": line_number, "error": str(exc)})
    lengths = sorted(len(p.story) for p in problems)
    counts = Counter(p.problem_id for p in problems)
    story_counts = Counter(sha256(p.story) for p in problems)
    report = {
        "dataset_path": str(path.resolve()),
        "dataset_sha256": hashlib.sha256(raw).hexdigest(),
        "bytes_on_disk": path.stat().st_size,
        "bytes_read": len(raw),
        "nonempty_lines": nonempty,
        "valid_rows": len(problems),
        "errors": errors,
        "duplicate_problem_hashes": {k: v for k, v in counts.items() if v > 1},
        "repeated_story_hashes": {k: v for k, v in story_counts.items() if v > 1},
        "gold_counts": dict(Counter(p.gold for p in problems)),
        "story_characters": (
            {
                str(q): lengths[min(len(lengths) - 1, int((len(lengths) - 1) * q))]
                for q in (0, 0.5, 0.9, 0.95, 1)
            }
            if lengths
            else {}
        ),
        "token_counts": "Not measured by this character-level audit; provider token counts are recorded during API runs.",
        "selection_unit": "story hash; split by story before any probe fitting or tuning",
    }
    if len(raw) != path.stat().st_size or len(problems) + len(errors) != nonempty:
        raise ValueError("dataset read accounting failed")
    return problems, report


def select_pilot(problems: list[Problem], n: int, seed: int) -> list[Problem]:
    if not 1 <= n <= len(problems):
        raise ValueError("pilot size outside dataset size")
    # One problem per story prevents shared-story dependence in this first pilot.
    unique = {sha256(p.story): p for p in problems}
    if n > len(unique):
        raise ValueError("pilot requires more distinct stories than available")
    return random.Random(seed).sample(
        sorted(unique.values(), key=lambda p: p.problem_id), n
    )


def parse_answer(text: str) -> str:
    answer = text.strip()
    if not re.fullmatch("[AB]", answer):
        raise ValueError(
            "response must be exactly A or B; no fallback or answer guessing"
        )
    return answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset", type=Path, default=Path("data/quality_synthetic/stories.jsonl")
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    _, report = load_problems(args.dataset)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if (
        report["errors"]
        or report["duplicate_problem_hashes"]
        or not report["valid_rows"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
