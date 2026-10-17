"""A separate budgeted thinking-enabled private-preparation client."""

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path

from .api import BudgetedChat
from .environment3_prepared import preparation_messages, preparation_schema
from .preflight import load_problems, sha256


async def run(args):
    problems, audit = load_problems(args.dataset)
    rows = [json.loads(x) for x in args.dataset.read_text().splitlines()]
    if (
        audit["errors"]
        or len(rows) != len(problems)
        or any(r["split"] != args.expected_split for r in rows)
    ):
        raise ValueError("complete sources in the declared partition required")
    async with BudgetedChat(
        args.out,
        args.models,
        args.budget,
        max_tokens=args.max_tokens,
        reasoning={"enabled": True},
        schema_retries=2,
        reuse_receipts=args.reuse_receipts,
        request_timeout_seconds=args.request_timeout_seconds,
    ) as api:
        results = []

        async def one(index, problem, model, speaker):
            assigned = "A" if speaker == "Red" else "B"
            swapped = bool(rows[index]["schedule_parent_index"] % 2)
            messages = preparation_messages(
                problem, swapped, speaker, assigned, explicit_truth=True
            )
            receipt = await api.complete(
                model,
                messages,
                preparation_schema(),
                {
                    "problem_id": problem.problem_id,
                    "model": model,
                    "speaker": speaker,
                    "stage": "reasoned_preparation",
                },
            )
            choice = json.loads(receipt["response_body"])["choices"][0]
            # No separate reasoning-token budget or forced-final instruction is set.
            # Reject any exposed forced/truncated flags in addition to the API stop gate.
            message = choice["message"]
            detail = message.get("reasoning_details") or []
            exposed_flags = [choice, message] + [
                d for d in detail if isinstance(d, dict)
            ]
            if any(d.get("forced") or d.get("truncated") for d in exposed_flags):
                raise ValueError("provider exposes forced/truncated reasoning")
            if choice.get("native_finish_reason") in (
                "length",
                "max_tokens",
                "max_output_tokens",
            ):
                raise ValueError("provider native finish reason reports truncation")
            results.append(
                {
                    "problem_id": problem.problem_id,
                    "model": model,
                    "speaker": speaker,
                    "assigned_answer": assigned,
                    "swapped": swapped,
                    "source_story_sha256": sha256(problem.story),
                    "request_sha256": receipt["request_sha256"],
                    "parsed": receipt["parsed"],
                    "finish_reason": receipt["finish_reason"],
                    "max_tokens": args.max_tokens,
                    "completion_tokens": receipt["usage"]["completion_tokens"],
                    "reasoning_tokens": receipt["usage"]
                    .get("completion_tokens_details", {})
                    .get("reasoning_tokens"),
                    "reasoning": {"enabled": True},
                }
            )

        outcomes = await asyncio.gather(
            *(
                one(i, p, m, s)
                for i, p in enumerate(problems)
                for m in args.models
                for s in ["Red", "Blue"]
            ),
            return_exceptions=True,
        )
        summary = {
            "finish_reasons": dict(
                Counter(r.get("finish_reason") for r in api.records)
            ),
            "errors": [str(x) for x in outcomes if isinstance(x, BaseException)],
            "completed": len(results),
            "expected": len(problems) * len(args.models) * 2,
            "max_completion_tokens": max(
                (r["completion_tokens"] for r in results), default=0
            ),
            "max_reasoning_tokens": max(
                (r["reasoning_tokens"] or 0 for r in results), default=0
            ),
            "reasoning_allowance": "No independent think budget set; full completion allowance covers reasoning and final JSON. Every exposed forced/truncated flag is rejected; unexposed provider internals cannot be certified.",
            "cost_usd": api.spent,
            "input_partition": args.expected_split,
            "dataset_audit": audit,
            "script_sha256": sha256(Path(__file__).read_text()),
            "preparation_source_sha256": sha256(
                Path(__file__).with_name("environment3_prepared.py").read_text()
            ),
        }
        (args.out / "preparations.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in results)
        )
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        if summary["errors"] or summary["completed"] != summary["expected"]:
            raise RuntimeError(
                "reasoned preparation incomplete; no debate continuation allowed"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--models", nargs="+", default=["qwen/qwen3.5-27b", "qwen/qwen3.8-27b"]
    )
    parser.add_argument("--budget", type=float, default=2.5)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--expected-split", choices=["dev", "test"], default="dev")
    parser.add_argument("--reuse-receipts", type=Path, nargs="*", default=[])
    parser.add_argument("--request-timeout-seconds", type=float, default=900)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
