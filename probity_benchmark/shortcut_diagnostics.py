"""Repeated complete no-story prompts and an explicitly options-only control."""

import argparse
import asyncio
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import re

from .api import ANSWER, BudgetedChat, object_schema
from .preflight import load_problems, sha256

MODELS = ["anthropic/claude-haiku-5.5", "qwen/qwen3.5-27b", "qwen/qwen3.8-27b"]


def interval(values):
    rng = random.Random(42)
    boot = sorted(
        sum(rng.choices(values, k=len(values))) / len(values) for _ in range(4000)
    )
    return [boot[99], boot[3899]]


def summarize(records, original, pmap):
    all_rows = list(original)
    for r in records:
        if r["valid"]:
            tag = r["tag"]
            all_rows.append(
                {
                    **tag,
                    "model": r["model"],
                    "answer": r["parsed"]["answer"],
                    "correct": r["parsed"]["answer"] == tag["gold"],
                }
            )
    cells = []
    for model in MODELS:
        for condition in ["no_story", "options_only"]:
            selected = [
                r
                for r in all_rows
                if r["model"] == model and r["condition"] == condition
            ]
            for subgroup in ["all", "short_options", "descriptive_options"]:
                by_story = defaultdict(list)
                for r in selected:
                    p = pmap[r["problem_id"]]
                    kind = (
                        "short_options"
                        if max(len(re.findall(r"\w+", x)) for x in p.options) <= 6
                        else "descriptive_options"
                    )
                    if subgroup == "all" or kind == subgroup:
                        by_story[p.problem_id].append(r)
                expected = 6 if condition == "no_story" else 2
                complete = {
                    pid: rows for pid, rows in by_story.items() if len(rows) == expected
                }
                if not complete:
                    continue
                values = [
                    sum(r["correct"] for r in rows) / len(rows)
                    for rows in complete.values()
                ]
                disagreement = []
                for rows in complete.values():
                    for swap in [False, True]:
                        answers = [r["answer"] for r in rows if r["swapped"] == swap]
                        if len(answers) > 1:
                            disagreement.append(len(set(answers)) > 1)
                cells.append(
                    {
                        "model": model,
                        "condition": condition,
                        "subgroup": subgroup,
                        "stories": len(complete),
                        "accuracy": sum(values) / len(values),
                        "story_bootstrap_95_ci": interval(values),
                        "repeat_disagreement_fraction": (
                            sum(disagreement) / len(disagreement)
                            if disagreement
                            else None
                        ),
                    }
                )
    return {
        "finish_reasons": dict(
            Counter(r.get("finish_reason", "request_error") for r in records)
        ),
        "new_requests": len(records),
        "valid": sum(r["valid"] for r in records),
        "cells": cells,
        "claim_limit": "Development diagnosis, with option-length subgroup chosen after inspecting the first pilot. Repeats and answer orders are not independent stories. Options-only is a deliberate information condition, not truncated context.",
    }


async def run(args):
    problems, audit = load_problems(args.dataset)
    if audit["errors"]:
        raise ValueError("dataset errors")
    pmap = {p.problem_id: p for p in problems}
    original = []
    for path in args.original:
        for r in map(json.loads, (path / "responses.jsonl").read_text().splitlines()):
            if not r["with_story"]:
                if not r["valid"] or r["finish_reason"] != "stop":
                    raise ValueError("original baseline incomplete")
                original.append(
                    {
                        "problem_id": r["problem_id"],
                        "model": r["model"],
                        "swapped": r["swapped"],
                        "gold": r["gold"],
                        "answer": r["answer"],
                        "correct": r["correct"],
                        "condition": "no_story",
                        "repeat": 0,
                    }
                )
    selected_ids = sorted({r["problem_id"] for r in original})
    if len(original) != len(selected_ids) * len(MODELS) * 2:
        raise ValueError("baseline cells incomplete or duplicated")
    async with BudgetedChat(args.out, MODELS, args.budget) as api:
        manifest = {
            "dataset_audit": audit,
            "selected_problem_ids": selected_ids,
            "source_sha256": sha256(Path(__file__).read_text()),
            "original_paths": [str(p.resolve()) for p in args.original],
            "new_repeats": 2,
            "subgroup_rule": "both options at most six regex word tokens",
            "subgroup_status": "exploratory rule fixed before these new calls",
        }
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

        async def one(pid, model, swap, condition, repeat):
            p = pmap[pid]
            messages, gold = p.view(False, swap, structured=True)
            if condition == "options_only":
                options = p.options[::-1] if swap else p.options
                messages = [
                    {
                        "role": "system",
                        "content": 'You are shown two possible answers to an unseen reading-comprehension question. Neither the story nor the question is available. Make your best guess about which option is correct. Return only a JSON object with one field, "answer", whose value is "A" or "B".',
                    },
                    {"role": "user", "content": f"A: {options[0]}\nB: {options[1]}"},
                ]
            return await api.complete(
                model,
                messages,
                object_schema({"answer": ANSWER}),
                {
                    "problem_id": pid,
                    "gold": gold,
                    "swapped": swap,
                    "condition": condition,
                    "repeat": repeat,
                },
                schema_name="reading_answer",
            )

        jobs = [
            (pid, m, swap, condition, repeat)
            for pid in selected_ids
            for m in MODELS
            for swap in [False, True]
            for condition, repeat in [
                ("no_story", 1),
                ("no_story", 2),
                ("options_only", 0),
            ]
        ]
        random.Random(20261015).shuffle(jobs)
        results = await asyncio.gather(
            *(one(*job) for job in jobs), return_exceptions=True
        )
        summary = summarize(api.records, original, pmap)
        summary.update(
            expected_new_requests=len(jobs),
            accounted_cost_upper_usd=api.spent,
            errors=[str(r) for r in results if isinstance(r, BaseException)],
        )
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
        if summary["errors"] or len(api.records) != len(jobs):
            raise RuntimeError("diagnostic incomplete; all in-flight receipts retained")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset", type=Path, default=Path("data/quality_synthetic/stories.jsonl")
    )
    ap.add_argument("--original", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--budget", type=float, default=1.5)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
