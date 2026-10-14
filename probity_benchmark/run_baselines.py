"""Paid API pilot with full prompts/responses, balanced answers, and a spend cap.

This is a black-box environment check, not a probe or white-box experiment.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
import urllib.request

import aiohttp

from .preflight import load_problems, parse_answer, select_pilot, sha256


def summarize(records: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in records:
        groups[(r["model"], r["with_story"])].append(r)
    output = []
    for (model, with_story), group in sorted(groups.items()):
        usable = [r for r in group if r.get("valid")]
        by_story = defaultdict(dict)
        for r in usable:
            if type(r.get("swapped")) is not bool:
                raise ValueError("valid records must identify answer order")
            orders = by_story[r["problem_id"]]
            if r["swapped"] in orders:
                raise ValueError(
                    "duplicate answer order in a story/model/baseline cell"
                )
            orders[r["swapped"]] = int(r["correct"])
        paired = [
            sum(v.values()) / 2 for v in by_story.values() if set(v) == {False, True}
        ]
        # Bootstrap stories, never treat answer-order variants as independent.
        rng = random.Random(42)
        boot = (
            sorted(
                sum(rng.choices(paired, k=len(paired))) / len(paired)
                for _ in range(2000)
            )
            if paired
            else []
        )
        output.append(
            {
                "model": model,
                "with_story": with_story,
                "finish_reasons": dict(
                    Counter(r.get("finish_reason", "request_error") for r in group)
                ),
                "attempted": len(group),
                "valid": len(usable),
                "complete_story_pairs": len(paired),
                "accuracy": sum(paired) / len(paired) if paired else None,
                "story_bootstrap_95_ci": [boot[49], boot[1949]] if boot else None,
                "invalid_records": len(group) - len(usable),
                "accounted_cost_upper_usd": sum(r.get("cost_usd", 0) for r in group),
                "completion_tokens_max": max(
                    (r.get("usage", {}).get("completion_tokens", 0) for r in group),
                    default=0,
                ),
                "prompt_tokens_max": max(
                    (r.get("usage", {}).get("prompt_tokens", 0) for r in group),
                    default=0,
                ),
            }
        )
    return {
        "cells": output,
        "accounted_cost_upper_usd": sum(r.get("cost_usd", 0) for r in records),
        "provider_reported_cost_usd": sum(
            r.get("cost_usd", 0)
            for r in records
            if r.get("cost_basis") == "provider_reported"
        ),
        "unpriced_request_count": sum(
            r.get("cost_basis") == "reservation_upper_bound" for r in records
        ),
        "claim_limit": "Development pilot only; API provider routes are not a pinned local probe model.",
    }


async def run(args):
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY unavailable")
    if args.out.exists():
        raise RuntimeError(
            "output directory already exists; never overwrite or silently rerun paid work"
        )
    problems, audit = load_problems(args.dataset)
    if audit["errors"] or audit["duplicate_problem_hashes"]:
        raise RuntimeError("dataset audit failed")
    selected = select_pilot(problems, args.n, args.seed)
    catalog = json.load(
        urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=30)
    )
    models = {m["id"]: m for m in catalog["data"]}
    missing = set(args.models) - set(models)
    if missing:
        raise ValueError(f"model identifiers unavailable: {sorted(missing)}")
    if len(set(args.models)) != len(args.models):
        raise ValueError("duplicate model identifiers would repeat paid requests")
    # Check every model against the longest selected full-story prompt before
    # opening a paid session. Answer swapping preserves this byte count.
    largest_prompt_bound = max(
        len(json.dumps(p.view(True, False)[0], ensure_ascii=False).encode()) + 1024
        for p in selected
    )
    for model in args.models:
        spec = models[model]
        context = spec["top_provider"].get("context_length") or spec["context_length"]
        if largest_prompt_bound + args.max_tokens > context:
            raise ValueError(f"complete pilot prompts exceed {model}'s context limit")
        rates = [
            float(spec["pricing"].get(k, 0))
            for k in ("prompt", "completion", "request")
        ]
        if any(not math.isfinite(rate) or rate < 0 for rate in rates):
            raise ValueError(f"invalid pricing for {model}")
    args.out.mkdir(parents=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_audit": audit,
        "models": {m: models[m] for m in args.models},
        "seed": args.seed,
        "selected_problem_ids": [p.problem_id for p in selected],
        "max_completion_tokens": args.max_tokens,
        "budget_usd": args.budget,
        "purpose": "environment validity pilot; no probes; no final-test claims",
        "reasoning": "disabled explicitly; hidden reasoning is not silently clipped",
    }
    manifest["git_commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    manifest["runner_sha256"] = sha256(Path(__file__).read_text())
    manifest["preflight_sha256"] = sha256(
        Path(__file__).with_name("preflight.py").read_text()
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records = []
    spent = 0.0
    reserved = 0.0
    stopped = False
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=240)
    async with aiohttp.ClientSession(
        timeout=timeout,
        headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]},
    ) as session:

        async def one(model, problem, with_story, swapped):
            nonlocal spent, reserved, stopped
            messages, gold = problem.view(with_story, swapped)
            price = models[model]["pricing"]
            input_rate = float(price["prompt"])
            output_rate = float(price["completion"])
            # Conservative byte bound plus template overhead; full text is sent.
            byte_bound = len(json.dumps(messages, ensure_ascii=False).encode()) + 1024
            upper = (
                byte_bound * input_rate
                + args.max_tokens * output_rate
                + float(price.get("request", 0))
            )
            context = (
                models[model]["top_provider"].get("context_length")
                or models[model]["context_length"]
            )
            if byte_bound + args.max_tokens > context:
                raise RuntimeError(
                    "conservative context bound exceeds provider limit; refusing truncation"
                )
            body = {
                "model": model,
                "messages": messages,
                "max_tokens": args.max_tokens,
                "reasoning": {"enabled": False},
                "provider": {
                    "require_parameters": True,
                    "max_price": {
                        "prompt": input_rate * 1_000_000,
                        "completion": output_rate * 1_000_000,
                        "request": float(price.get("request", 0)),
                    },
                },
                "usage": {"include": True},
            }
            # Recent Anthropic models reject temperature rather than ignore it.
            if "temperature" in models[model].get("supported_parameters", []):
                body["temperature"] = 0
            async with sem:
                async with lock:
                    if stopped or spent + reserved + upper > args.budget:
                        stopped = True
                        return
                    reserved += upper
                record = {
                    "model": model,
                    "problem_id": problem.problem_id,
                    "source_line": problem.source_line,
                    "with_story": with_story,
                    "swapped": swapped,
                    "gold": gold,
                    "request": body,
                    "request_sha256": sha256(json.dumps(body, sort_keys=True)),
                    "valid": False,
                    "reserved_usd": upper,
                }
                started = time.monotonic()
                try:
                    async with session.post(
                        "https://openrouter.ai/api/v1/chat/completions", json=body
                    ) as resp:
                        raw = await resp.text()
                        record["http_status"] = resp.status
                        record["response_body"] = raw
                        payload = json.loads(raw)
                        if resp.status != 200 or "error" in payload:
                            raise RuntimeError(f"provider returned HTTP {resp.status}")
                    usage = payload.get("usage", {})
                    record["usage"] = usage
                    if "prompt_tokens" not in usage or "completion_tokens" not in usage:
                        raise RuntimeError("missing usage accounting")
                    reported = usage.get("cost")
                    cost = (
                        float(reported)
                        if reported is not None
                        else usage["prompt_tokens"] * input_rate
                        + usage["completion_tokens"] * output_rate
                        + float(price.get("request", 0))
                    )
                    if not math.isfinite(cost) or cost < 0:
                        raise RuntimeError("provider returned invalid cost accounting")
                    record["cost_usd"] = cost
                    record["cost_basis"] = (
                        "provider_reported"
                        if reported is not None
                        else "catalog_estimate"
                    )
                    if cost > upper or usage["prompt_tokens"] > byte_bound:
                        raise RuntimeError(
                            "provider exceeded reserved cost or input token bound"
                        )
                    choice = payload["choices"][0]
                    finish = choice.get("finish_reason")
                    record["finish_reason"] = finish
                    if finish != "stop":
                        raise RuntimeError(
                            f"incomplete or nonstandard finish: {finish}"
                        )
                    if usage["completion_tokens"] >= args.max_tokens:
                        raise RuntimeError(
                            "completion hit token cap despite stop reason"
                        )
                    answer = parse_answer(choice["message"].get("content") or "")
                    record.update(answer=answer, correct=answer == gold, valid=True)
                except Exception as exc:
                    record["error"] = str(exc)
                record["duration_seconds"] = time.monotonic() - started
                async with lock:
                    reserved -= upper
                    # Conservatively charge unknown failures at their reservation.
                    record.setdefault("cost_usd", upper)
                    record.setdefault("cost_basis", "reservation_upper_bound")
                    spent += record["cost_usd"]
                    if not record["valid"]:
                        stopped = True
                    records.append(record)
                    with (args.out / "responses.jsonl").open("a") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    (args.out / "summary.json").write_text(
                        json.dumps(summarize(records), indent=2) + "\n"
                    )
                    print(
                        json.dumps(
                            {
                                "completed": len(records),
                                "model": model,
                                "with_story": with_story,
                                "valid": record["valid"],
                                "accounted_cost_upper_usd": round(spent, 5),
                                "error": record.get("error"),
                            }
                        ),
                        flush=True,
                    )

        jobs = [
            (model, p, story, swap)
            for p in selected
            for model in args.models
            for story in [False, True]
            for swap in [False, True]
        ]
        await asyncio.gather(*(one(*job) for job in jobs))
    summary = summarize(records)
    summary.update(
        planned_requests=len(jobs),
        completed_requests=len(records),
        stopped_early=stopped,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if stopped:
        raise SystemExit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset", type=Path, default=Path("data/quality_synthetic/stories.jsonl")
    )
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if (
        not math.isfinite(args.budget)
        or args.budget <= 0
        or args.concurrency < 1
        or args.max_tokens < 1
    ):
        ap.error("positive budget, concurrency, and token allowance required")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
