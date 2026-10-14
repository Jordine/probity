"""Independent full-story checks with complete verbatim evidence and no-story twins."""

import argparse
import asyncio
from collections import Counter, defaultdict
import json
from pathlib import Path

from .api import ANSWER, BudgetedChat, object_schema
from .preflight import load_problems, sha256
from .shortcut_diagnostics import MODELS


async def run(args):
    problems, audit = load_problems(args.dataset)
    source = [json.loads(x) for x in args.dataset.read_text().splitlines()]
    if audit["errors"] or len(source) != len(problems):
        raise ValueError("invalid paired-world dataset")
    async with BudgetedChat(
        args.out,
        MODELS,
        args.budget,
        schema_retries=2,
        reuse_receipts=args.reuse_receipts,
    ) as api:
        manifest = {
            "dataset_audit": audit,
            "runner_sha256": sha256(Path(__file__).read_text()),
            "criterion": "Report all worlds and models. Do not silently exclude disagreements. Evidence quotes must occur verbatim in the complete story.",
        }
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        results = []

        async def one(index, p, model, swapped, with_story):
            messages, gold = p.view(with_story, swapped, structured=True)
            if with_story:
                messages[0][
                    "content"
                ] = "Answer the reading-comprehension question from the complete supplied story. Return JSON with all four fields: answer (A or B), explanation (string), evidence_quotes (array of exact verbatim passages from the story), and ambiguous (boolean, always included). Do not invent or normalize quotation text. If the story does not uniquely determine the answer, set ambiguous=true and explain the ambiguity; otherwise set ambiguous=false."
                schema = object_schema(
                    {
                        "answer": ANSWER,
                        "explanation": {"type": "string"},
                        "evidence_quotes": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                            "minItems": 1,
                        },
                        "ambiguous": {"type": "boolean"},
                    }
                )
            else:
                schema = object_schema({"answer": ANSWER})
            tag = {
                "problem_id": p.problem_id,
                "parent_problem_id": source[index]["parent_problem_id"],
                "world": source[index]["variant"],
                "split": source[index]["split"],
                "swapped": swapped,
                "with_story": with_story,
                "gold": gold,
                "source_line": source[index]["parent_source_line"],
            }
            receipt = await api.complete(model, messages, schema, tag)
            obj = receipt["parsed"]
            result = {
                **tag,
                "model": model,
                **obj,
                "correct": obj["answer"] == gold,
                "quotes_exact": all(
                    q in p.story for q in obj.get("evidence_quotes", [])
                ),
                "request_sha256": receipt["request_sha256"],
            }
            results.append(result)
            with (args.out / "evaluations.jsonl").open("a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        jobs = [
            (i, p, m, swap, story)
            for i, p in enumerate(problems)
            for m in MODELS
            for swap in [False, True]
            for story in [True, False]
            if story or source[i]["variant"] == "original"
        ]
        outcomes = await asyncio.gather(
            *(one(*job) for job in jobs), return_exceptions=True
        )
        grouped = defaultdict(list)
        for r in results:
            grouped[(r["model"], r["with_story"])].append(r)
        summary = {
            "finish_reasons": dict(
                Counter(r.get("finish_reason", "request_error") for r in api.records)
            ),
            "planned_calls": len(jobs),
            "completed_calls": len(api.records),
            "accounted_cost_upper_usd": api.spent,
            "errors": [str(x) for x in outcomes if isinstance(x, BaseException)],
            "cells": [
                {
                    "model": model,
                    "with_story": story,
                    "rows": len(rows),
                    "accuracy_on_queried_worlds": sum(r["correct"] for r in rows)
                    / len(rows),
                    "ambiguous": sum(r.get("ambiguous", False) for r in rows),
                    "inexact_quote_rows": sum(not r["quotes_exact"] for r in rows),
                }
                for (model, story), rows in sorted(grouped.items())
            ],
            "no_story_paired_accuracy": 0.5,
            "paired_baseline_note": "No-story original-world predictions reused for their opposite-gold twin, so paired accuracy is exactly 0.5 by information symmetry.",
            "claim_limit": "Source validity diagnostic. Model agreement is not independently verified ground truth. Answer order and twins share a parent.",
        }
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        if summary["errors"]:
            raise RuntimeError("validation incomplete")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--budget", type=float, default=2)
    ap.add_argument("--reuse-receipts", type=Path, nargs="*", default=[])
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
