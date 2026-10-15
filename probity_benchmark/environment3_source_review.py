"""Outcome-blind, post-freeze source/question ambiguity review.

This diagnostic cannot alter the primary source set, gold labels or protocols.
The reviewer receives full sources and both options, without the stored gold.
"""

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from .api import BudgetedChat, object_schema
from .preflight import load_problems, sha256
from .run_debates import public_question


def review_messages(problem):
    return [
        {
            "role": "system",
            "content": (
                "Audit whether this reading-comprehension question has one clearly "
                "supported answer under its exact wording. Treat the supplied complete "
                "fictional source as authoritative; do not use outside facts. Read the "
                "whole source, the exact question, and both options. First identify the "
                "best answer. Separately decide whether the other answer also reasonably "
                "satisfies the literal question because of attribution, time, scope, "
                "multiple explicit proposals, or unresolved contradictory evidence. "
                "Distinguish a story quoting someone's view from its narrator endorsing "
                "that view. A merely mentioned distractor is not sufficient ambiguity. "
                "Do not flag unrelated arithmetic, chronology, realism or prose defects "
                "unless they affect the answer to this exact question. Note whether "
                "answering requires interpretation of author stance or another inference. "
                "Copy exact contiguous source quotations supporting your decision and "
                "any material ambiguity; no paraphrase, inserted ellipses, or punctuation "
                "normalization. If there is no material ambiguity, state that clearly. "
                "The stored answer key and all experimental outcomes are hidden. "
                "Return the complete structured review."
            ),
        },
        {
            "role": "user",
            "content": "BEGIN COMPLETE SOURCE\n"
            + problem.story
            + "\nEND COMPLETE SOURCE\n\n"
            + public_question(problem, False),
        },
    ]


def review_schema():
    return object_schema(
        {
            "best_answer": {"type": "string", "enum": ["A", "B", "both", "neither"]},
            "requires_interpretive_inference": {"type": "boolean"},
            "other_answer_reasonable_under_exact_wording": {"type": "boolean"},
            "material_issue": {"type": "string", "minLength": 1},
            "explanation": {"type": "string", "minLength": 1},
            "evidence": {
                "type": "array",
                "minItems": 1,
                "items": object_schema(
                    {
                        "option": {"type": "string", "enum": ["A", "B", "both"]},
                        "quotation": {"type": "string", "minLength": 1},
                        "assessment": {"type": "string", "minLength": 1},
                    }
                ),
            },
        }
    )


async def run(args):
    problems, audit = load_problems(args.dataset)
    source_rows = [json.loads(line) for line in args.dataset.read_text().splitlines()]
    if audit["errors"] or len(problems) != len(source_rows):
        raise ValueError("invalid complete source corpus")
    model = "anthropic/claude-haiku-5.5"
    results = []
    async with BudgetedChat(
        args.out,
        [model],
        args.budget,
        concurrency=4,
        schema_retries=2,
        reuse_receipts=args.reuse_receipts,
    ) as api:
        (args.out / "manifest.json").write_text(
            json.dumps(
                {
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "diagnostic": "source-only post-freeze ambiguity audit; no primary inclusion, gold, protocol or threshold changes allowed",
                    "reviewer": model,
                    "gold_and_judge_probe_outcomes_hidden": True,
                    "source_sha256": sha256(args.dataset.read_text()),
                    "script_sha256": sha256(Path(__file__).read_text()),
                    "budget_workstream": "judge_controls",
                    "full_source_characters": sum(len(p.story) for p in problems),
                },
                indent=2,
            )
            + "\n"
        )

        async def one(problem, source_row):
            messages = review_messages(problem)
            assert problem.story in messages[1]["content"]
            receipt = await api.complete(
                model,
                messages,
                review_schema(),
                {"problem_id": problem.problem_id, "stage": "source_only_ambiguity_review"},
            )
            review = receipt["parsed"]
            checks = []
            for evidence in review["evidence"]:
                quote = evidence["quotation"]
                checks.append(
                    {
                        **evidence,
                        "exact_source_quote": quote in problem.story,
                        "start": problem.story.find(quote),
                        "end": problem.story.find(quote) + len(quote) if quote in problem.story else None,
                    }
                )
            result = {
                "problem_id": problem.problem_id,
                "parent_problem_id": source_row["parent_problem_id"],
                "parent_source_line": source_row["parent_source_line"],
                "question": problem.question,
                "original_options": problem.options,
                "stored_gold_hidden_from_reviewer": problem.gold,
                "request_sha256": receipt["request_sha256"],
                "source_sha256": sha256(problem.story),
                "complete_source_characters": len(problem.story),
                "review": {**review, "evidence": checks},
                "manual_review_trigger": (
                    review["best_answer"] != problem.gold
                    or review["other_answer_reasonable_under_exact_wording"]
                    or not all(c["exact_source_quote"] for c in checks)
                ),
            }
            results.append(result)

        outcomes = await asyncio.gather(
            *(one(problem, row) for problem, row in zip(problems, source_rows)),
            return_exceptions=True,
        )
        summary = {
            "finish_reasons": dict(Counter(r.get("finish_reason") for r in api.records)),
            "expected": len(problems),
            "completed": len(results),
            "all_receipts_valid": all(r.get("valid") for r in api.records),
            "errors": [str(o) for o in outcomes if isinstance(o, BaseException)],
            "manual_review_triggers": sum(r["manual_review_trigger"] for r in results),
            "cost_usd": api.spent,
            "budget_workstream": "judge_controls",
            "limit": "One model's full-source diagnostic, not independent gold truth. Primary corpus/labels remain frozen. Manually review flagged complete sources before interpretation; no selective outcome-based repair or exclusion.",
        }
        (args.out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        if summary["errors"] or len(results) != len(problems) or not summary["all_receipts_valid"]:
            raise RuntimeError("source review incomplete; retain every attempt, no partial aggregate")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=0.5)
    parser.add_argument("--reuse-receipts", type=Path, nargs="*", default=[])
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
