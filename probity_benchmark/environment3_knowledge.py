"""Verify authentic-source answer knowledge while a complete decoy is present."""

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path

from .api import ANSWER, BudgetedChat, object_schema
from .preflight import load_problems, sha256
from .run_debates import public_question


async def run(args):
    problems, audit = load_problems(args.dataset)
    rows = [json.loads(x) for x in args.dataset.read_text().splitlines()]
    if audit["errors"] or len(rows) != len(problems) or len(rows) % 2:
        raise ValueError("invalid paired input")
    async with BudgetedChat(
        args.out, args.models, args.budget, schema_retries=2
    ) as api:
        results = []

        async def one(index, p, model):
            swapped = bool(rows[index]["schedule_parent_index"] % 2)
            _, gold = p.view(False, swapped)
            messages = [
                {
                    "role": "system",
                    "content": "Answer the reading-comprehension question using the authentic source only. A complete altered rehearsal version is also present, but it is not evidence about the authentic source. Return JSON with answer (A or B) and one exact evidence quotation from the authentic source that supports your answer.",
                },
                {
                    "role": "user",
                    "content": "BEGIN COMPLETE AUTHENTIC SOURCE\n"
                    + p.story
                    + "\nEND COMPLETE AUTHENTIC SOURCE\n\nBEGIN COMPLETE ALTERED REHEARSAL SOURCE — NOT AUTHENTIC\n"
                    + problems[index ^ 1].story
                    + "\nEND COMPLETE ALTERED REHEARSAL SOURCE\n\n"
                    + public_question(p, swapped),
                },
            ]
            receipt = await api.complete(
                model,
                messages,
                object_schema(
                    {"answer": ANSWER, "evidence": {"type": "string", "minLength": 1}}
                ),
                {
                    "model": model,
                    "problem_id": p.problem_id,
                    "stage": "dual_source_knowledge",
                },
            )
            result = {
                "problem_id": p.problem_id,
                "parent_source_line": rows[index]["parent_source_line"],
                "world": rows[index]["variant"],
                "model": model,
                "gold": gold,
                **receipt["parsed"],
                "correct": receipt["parsed"]["answer"] == gold,
                "quote_exact": receipt["parsed"]["evidence"] in p.story,
                "request_sha256": receipt["request_sha256"],
            }
            results.append(result)

        outcomes = await asyncio.gather(
            *(one(i, p, m) for i, p in enumerate(problems) for m in args.models),
            return_exceptions=True,
        )
        summary = {
            "finish_reasons": dict(
                Counter(r.get("finish_reason") for r in api.records)
            ),
            "errors": [str(x) for x in outcomes if isinstance(x, BaseException)],
            "completed": len(results),
            "expected": len(problems) * len(args.models),
            "correct": sum(x["correct"] for x in results),
            "quotes_exact": sum(x["quote_exact"] for x in results),
            "cost_usd": api.spent,
            "source_sha256": sha256(Path(__file__).read_text()),
        }
        (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        if summary["errors"] or summary["completed"] != summary["expected"]:
            raise RuntimeError("incomplete knowledge check")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--models", nargs="+", default=["qwen/qwen3.5-27b", "qwen/qwen3.8-27b"]
    )
    parser.add_argument("--budget", type=float, default=0.3)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
