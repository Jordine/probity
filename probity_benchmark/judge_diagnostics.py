"""Judge-model comparison on identical, complete, story-blind debate transcripts."""

import argparse
import asyncio
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .api import ANSWER, BudgetedChat, object_schema
from .preflight import load_problems, sha256
from .run_debates import judge_messages


def grouped_accuracy(rows):
    by_parent = defaultdict(list)
    for row in rows:
        by_parent[row["parent_problem_id"]].append(row["correct"])
    values = np.array([np.mean(v) for v in by_parent.values()])
    rng = np.random.default_rng(20261015)
    boot = rng.choice(values, (4000, len(values)), replace=True).mean(1)
    return {
        "judgments": len(rows),
        "parent_stories": len(values),
        "accuracy": float(values.mean()),
        "parent_bootstrap_95_ci": np.quantile(boot, [0.025, 0.975]).tolist(),
    }


async def run(args):
    problems, audit = load_problems(args.dataset)
    if audit["errors"]:
        raise ValueError("invalid source")
    by_id = {p.problem_id: p for p in problems}
    episodes = [
        json.loads(p.read_text()) for p in sorted(args.episodes.glob("episode_*.json"))
    ]
    if not episodes or any(not e["complete"] for e in episodes):
        raise ValueError("incomplete episodes")
    if args.repeats > 1 and args.reuse_receipts:
        raise ValueError("independent repeat diagnostic cannot reuse prior answers")
    results = []
    async with BudgetedChat(
        args.out,
        args.models,
        args.budget,
        schema_retries=2,
        reuse_receipts=args.reuse_receipts,
    ) as api:
        (args.out / "manifest.json").write_text(
            json.dumps(
                {
                    "dataset_sha256": sha256(args.dataset.read_text()),
                    "judge_source_sha256": sha256(Path(__file__).read_text()),
                    "conditions": "same complete transcripts, no story or detector information",
                    "judges": args.models,
                    "repeats": args.repeats,
                    "unit": "parent story; twins, debater models, and repeats are dependent",
                },
                indent=2,
            )
            + "\n"
        )

        async def one(episode, model, repeat):
            receipt = await api.complete(
                model,
                judge_messages(
                    by_id[episode["problem_id"]], episode["swapped"], episode["turns"]
                ),
                object_schema({"answer": ANSWER, "explanation": {"type": "string"}}),
                {"episode": episode["episode_id"], "stage": "judge", "repeat": repeat},
            )
            row = {
                "episode_id": episode["episode_id"],
                "parent_problem_id": episode["parent_problem_id"],
                "split": episode["split"],
                "world": episode["world"],
                "participant_model": episode["model"],
                "judge_model": model,
                "repeat": repeat,
                "knowledge_correct": episode["knowledge_correct"],
                "gold": episode["gold"],
                **receipt["parsed"],
                "correct": receipt["parsed"]["answer"] == episode["gold"],
                "request_sha256": receipt["request_sha256"],
            }
            results.append(row)
            with (args.out / "evaluations.jsonl").open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        outcomes = await asyncio.gather(
            *(
                one(e, m, r)
                for e in episodes
                for m in args.models
                for r in range(args.repeats)
            ),
            return_exceptions=True,
        )
        cells = []
        for model in args.models:
            for participant in sorted({e["model"] for e in episodes}):
                for split in ["all", "dev", "test"]:
                    selected = [
                        r
                        for r in results
                        if r["judge_model"] == model
                        and r["participant_model"] == participant
                        and (split == "all" or r["split"] == split)
                    ]
                    if selected:
                        cells.append(
                            {
                                "judge": model,
                                "participant": participant,
                                "split": split,
                                **grouped_accuracy(selected),
                            }
                        )
        summary = {
            "finish_reasons": dict(
                Counter(r.get("finish_reason") for r in api.records)
            ),
            "expected_judgments": len(episodes) * len(args.models) * args.repeats,
            "completed_judgments": len(results),
            "cells": cells,
            "errors": [str(x) for x in outcomes if isinstance(x, BaseException)],
            "cost_upper_usd": api.spent,
            "claim_limit": "Development headroom diagnostic on dependent paired worlds; no probe uplift claim.",
        }
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        if summary["errors"] or len(results) != summary["expected_judgments"]:
            raise RuntimeError("judge comparison incomplete")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--episodes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--budget", type=float, default=1)
    ap.add_argument(
        "--models",
        nargs="+",
        default=["anthropic/claude-haiku-5.5", "qwen/qwen3.5-27b", "qwen/qwen3.8-27b"],
    )
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--reuse-receipts", type=Path, nargs="*", default=[])
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
