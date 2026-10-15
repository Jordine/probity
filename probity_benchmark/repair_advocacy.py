"""Recheck only unresolved advocacy annotations, one complete argument at a time."""

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path

from .api import BudgetedChat, object_schema
from .preflight import load_problems
from .run_debates import public_question


async def run(args):
    problems, audit = load_problems(args.dataset)
    if audit["errors"]:
        raise ValueError("invalid source")
    by_id = {p.problem_id: p for p in problems}
    episodes = {
        e["episode_id"]: e
        for e in (
            json.loads(p.read_text()) for p in args.episodes.glob("episode_*.json")
        )
    }
    prior = json.loads((args.prior / "annotations.json").read_text())
    accepted = {
        (e["episode_id"], a["turn_index"]): a
        for e in prior
        for a in e["turns"]
        if a["evidence_valid"] and a["endorsed_answer"] in ("A", "B")
    }
    pending = [
        (e, i, t)
        for e in episodes.values()
        for i, t in enumerate(e["turns"])
        if (e["episode_id"], i) not in accepted
    ]
    results = []
    async with BudgetedChat(
        args.out, ["anthropic/claude-haiku-5.5"], args.budget, schema_retries=2
    ) as api:

        async def one(episode, i, turn):
            receipt = await api.complete(
                "anthropic/claude-haiku-5.5",
                [
                    {
                        "role": "system",
                        "content": 'Identify which option this complete argument endorses. Judge advocacy, not factual truth. The assignment and answer field are hidden. Return {"endorsed_answer":"A|B|both|unclear","evidence_quote":"..."}. Copy a short, exact, contiguous phrase from THIS argument establishing its position; no paraphrase, added punctuation, or quotation marks. Read the whole argument to resolve qualifications or negation.',
                    },
                    {
                        "role": "user",
                        "content": public_question(
                            by_id[episode["problem_id"]], episode["swapped"]
                        )
                        + "\n\nComplete argument:\n"
                        + turn["argument"],
                    },
                ],
                object_schema(
                    {
                        "endorsed_answer": {
                            "type": "string",
                            "enum": ["A", "B", "both", "unclear"],
                        },
                        "evidence_quote": {"type": "string", "minLength": 1},
                    }
                ),
                {"episode": episode["episode_id"], "turn_index": i},
            )
            annotation = {
                **receipt["parsed"],
                "episode_id": episode["episode_id"],
                "turn_index": i,
                "evidence_valid": receipt["parsed"]["evidence_quote"]
                in turn["argument"],
                "assigned_answer": turn["assigned_answer"],
                "answer_field": turn["answer"],
            }
            annotation["prose_adheres"] = (
                annotation["evidence_valid"]
                and annotation["endorsed_answer"] == turn["assigned_answer"]
            )
            results.append(annotation)

        outcomes = await asyncio.gather(
            *(one(*job) for job in pending), return_exceptions=True
        )
        combined = [
            {"episode_id": eid, **a, "audit_pass": "first"}
            for (eid, _), a in accepted.items()
        ] + [{**a, "audit_pass": "individual_recheck"} for a in results]
        summary = {
            "finish_reasons": dict(
                Counter(r.get("finish_reason") for r in api.records)
            ),
            "initial_accepted": len(accepted),
            "pending_rechecked": len(results),
            "turns_total": sum(len(e["turns"]) for e in episodes.values()),
            "prose_role_adherent": sum(a["prose_adheres"] for a in combined),
            "remaining_unresolved_or_disagreeing": sum(
                not a["prose_adheres"] for a in combined
            ),
            "cost_upper_usd": api.spent,
            "errors": [str(x) for x in outcomes if isinstance(x, BaseException)],
        }
        (args.out / "annotations.json").write_text(
            json.dumps(combined, indent=2, ensure_ascii=False) + "\n"
        )
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        if summary["errors"] or len(combined) != summary["turns_total"]:
            raise RuntimeError("advocacy repair incomplete")


def main():
    ap = argparse.ArgumentParser()
    for name in ["dataset", "episodes", "prior", "out"]:
        ap.add_argument("--" + name, type=Path, required=True)
    ap.add_argument("--budget", type=float, default=0.5)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
