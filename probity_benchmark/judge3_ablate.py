"""Complete-unit judge diagnostics on previously used development debates.

Single-opening arms measure source-independent persuasion/style diagnostics;
they are never treated as valid debate or probe-uplift evaluations.
"""

import argparse
import asyncio
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from .api import ANSWER, BudgetedChat, object_schema
from .build_native_judge_jobs import JUDGE, OUTPUT
from .judge_diagnostics import grouped_accuracy
from .preflight import load_problems, sha256

CONDITIONS = (
    "full",
    "openings",
    "final_round",
    "first_opening",
    "second_opening",
    "renamed_speakers",
)


def rename_speakers(text, mapping):
    """Replace whole names simultaneously, including possessive references."""
    pattern = r"\b(" + "|".join(re.escape(k) for k in mapping) + r")\b"
    return re.sub(pattern, lambda match: mapping[match.group()], text)


def selected_turns(episode, condition):
    turns = episode["turns"]
    if not episode["complete"] or not turns:
        raise ValueError("complete episodes required")
    if condition in ("full", "renamed_speakers"):
        indices = list(range(len(turns)))
    elif condition == "openings":
        first_round = min(t["round"] for t in turns)
        indices = [i for i, t in enumerate(turns) if t["round"] == first_round]
    elif condition == "final_round":
        final_round = max(t["round"] for t in turns)
        indices = [i for i, t in enumerate(turns) if t["round"] == final_round]
    elif condition in ("first_opening", "second_opening"):
        openings = selected_turns(episode, "openings")
        if len(openings) != 2:
            raise ValueError("expected exactly two opening turns")
        indices = [openings[condition == "second_opening"]]
    else:
        raise ValueError("unknown condition")
    if condition not in ("first_opening", "second_opening"):
        if len({turns[i]["speaker"] for i in indices}) != 2:
            raise ValueError("two-speaker condition lost a speaker")
    return indices


def make_job(problem, episode, condition):
    messages, gold = problem.view(False, episode["swapped"], structured=True)
    if gold != episode["gold"]:
        raise ValueError("public option mapping mismatch")
    messages[0]["content"] = JUDGE + OUTPUT
    indices = selected_turns(episode, condition)
    speakers = list(dict.fromkeys(t["speaker"] for t in episode["turns"]))
    mapping = dict(zip(speakers, ["Speaker 1", "Speaker 2"]))
    parts, records = [], []
    for i in indices:
        turn = episode["turns"][i]
        argument = turn["argument"]
        speaker = turn["speaker"]
        if condition == "renamed_speakers":
            argument = rename_speakers(argument, mapping)
            speaker = mapping[speaker]
        if not argument or turn["answer"] not in ("A", "B"):
            raise ValueError("invalid public turn")
        parts.append(speaker + " (answer " + turn["answer"] + "):\n" + argument)
        records.append(
            {
                "index": i,
                "source_argument_sha256": sha256(turn["argument"]),
                "presented_argument_sha256": sha256(argument),
                "source_characters": len(turn["argument"]),
                "presented_characters": len(argument),
                "source_public_answer": turn["answer"],
                "advocates_gold": turn["advocates_gold"],
            }
        )
    messages[1]["content"] += "\n\nPublic debate:\n" + "\n\n".join(parts)
    return {
        "job_id": sha256(episode["episode_id"] + "judge3_ablation_v1" + condition),
        "kind": condition,
        "episode_id": episode["episode_id"],
        "group": episode["parent_problem_id"],
        "parent_problem_id": episode["parent_problem_id"],
        "split": episode["split"],
        "participant_model": episode["model"],
        "world": episode["world"],
        "gold": gold,
        "messages": messages,
        "selected_turns": records,
        "prompt_version": "judge3_complete_unit_ablation_v1",
        "diagnostic_only": condition in ("first_opening", "second_opening"),
    }


def build(args):
    if args.out.exists():
        raise ValueError("immutable new output directory required")
    problems, audit = load_problems(args.dataset)
    if audit["errors"]:
        raise ValueError("invalid dataset")
    by_id = {p.problem_id: p for p in problems}
    episodes = [
        json.loads(p.read_text()) for p in sorted(args.episodes.glob("episode_*.json"))
    ]
    episodes = [e for e in episodes if e["split"] in args.splits]
    if not episodes or any(not e["knowledge_correct"] for e in episodes):
        raise ValueError("episodes absent or private knowledge check failed")
    jobs = [
        make_job(by_id[e["problem_id"]], e, condition)
        for e in episodes
        for condition in args.conditions
    ]
    args.out.mkdir(parents=True)
    path = args.out / "jobs.jsonl"
    path.write_text("".join(json.dumps(j, ensure_ascii=False) + "\n" for j in jobs))
    manifest = {
        "source_sha256": sha256(Path(__file__).read_text()),
        "dataset_sha256": sha256(args.dataset.read_text()),
        "jobs_sha256": sha256(path.read_text()),
        "episodes": len(episodes),
        "parents": len({e["parent_problem_id"] for e in episodes}),
        "jobs": len(jobs),
        "conditions": args.conditions,
        "splits": args.splits,
        "unit": "parent story; models, paired worlds and conditions dependent",
        "no_truncation": "Each included turn complete. Excluded turns are intentional complete-unit intervention, with exact selected indices and text hashes retained.",
        "role_header": "Always public answer, never private assigned answer",
        "claim_limit": "Development information/persuasion ablation, not probe uplift or final benchmark.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


async def run(args):
    jobs = [json.loads(line) for line in args.jobs.read_text().splitlines()]
    if len({j["job_id"] for j in jobs}) != len(jobs):
        raise ValueError("duplicate jobs")
    results = []
    async with BudgetedChat(
        args.out,
        args.models,
        args.budget,
        schema_retries=2,
        concurrency=args.concurrency,
    ) as api:
        (args.out / "manifest.json").write_text(
            json.dumps(
                {
                    "jobs_sha256": sha256(args.jobs.read_text()),
                    "source_sha256": sha256(Path(__file__).read_text()),
                    "judges": args.models,
                    "repeats": args.repeats,
                },
                indent=2,
            )
            + "\n"
        )

        async def one(job, model, repeat):
            receipt = await api.complete(
                model,
                job["messages"],
                object_schema({"answer": ANSWER}),
                {"job": job["job_id"], "kind": job["kind"], "repeat": repeat},
            )
            row = {
                **{k: v for k, v in job.items() if k != "messages"},
                "judge_model": model,
                "repeat": repeat,
                "answer": receipt["parsed"]["answer"],
                "correct": receipt["parsed"]["answer"] == job["gold"],
                "request_sha256": receipt["request_sha256"],
                "valid": receipt["valid"],
                "finish_reason": receipt["finish_reason"],
            }
            results.append(row)
            with (args.out / "evaluations.jsonl").open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        outcomes = await asyncio.gather(
            *(
                one(j, m, r)
                for j in jobs
                for m in args.models
                for r in range(args.repeats)
            ),
            return_exceptions=True,
        )
        summary = {
            "finish_reasons": dict(
                Counter(r.get("finish_reason") for r in api.records)
            ),
            "expected_judgments": len(jobs) * len(args.models) * args.repeats,
            "completed_judgments": len(results),
            "errors": [str(o) for o in outcomes if isinstance(o, BaseException)],
            "cost_upper_usd": api.spent,
        }
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        if summary["errors"] or len(results) != summary["expected_judgments"]:
            raise RuntimeError("incomplete ablations; do not aggregate")
    analyse(args.out / "evaluations.jsonl", args.out / "analysis.json")


def analyse(evaluations, out):
    rows = [json.loads(line) for line in evaluations.read_text().splitlines()]
    cells = defaultdict(list)
    for row in rows:
        model = row.get("judge_model", "native_judge")
        row.setdefault("parent_problem_id", row["group"])
        cells[(model, row["kind"], "all")].append(row)
        if row["kind"] in ("first_opening", "second_opening"):
            for turn in row["selected_turns"]:
                group = (
                    "correct_advocate" if turn["advocates_gold"] else "wrong_advocate"
                )
                cells[(model, "single_opening_by_role", group)].append(row)
    result = {
        "cells": [
            {"judge": m, "condition": c, "subgroup": g, **grouped_accuracy(selected)}
            for (m, c, g), selected in sorted(cells.items())
        ],
        "claim_limit": "Single-opening arms are information/style diagnostics, not valid complete-debate performance; all used stories are development.",
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["dev"])
    p.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS)
    )
    p = sub.add_parser("run")
    p.add_argument("--jobs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--models", nargs="+", default=["anthropic/claude-haiku-5.5"])
    p.add_argument("--budget", type=float, default=1)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--repeats", type=int, default=1)
    p = sub.add_parser("analyse")
    p.add_argument("--evaluations", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.command == "build":
        build(args)
    elif args.command == "run":
        asyncio.run(run(args))
    else:
        analyse(args.evaluations, args.out)


if __name__ == "__main__":
    main()
