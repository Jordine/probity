"""Four old-development calls exercising the already-frozen annotation formats."""

import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path

import jsonschema

from .api import BudgetedChat, object_schema
from .evaluate_debate_transfer import argument_mapping
from .local3_annotate_units import ANNOTATION_REASONING, UNIT_RULES, display_units
from .local3_secondary_clauses import CLAUSE_RULES, complete_payload
from .local3_verify_twins import MIXED_CHECK, validate_checks
from .preflight import sha256


async def main():
    base = Path("runs/iteration3/local_probes")
    source = Path("runs/iteration3/environment/manual_transfer_v1/corpus.jsonl")
    out = base / "frozen_annotation_smoke_v1"
    lock = json.loads((base / "fresh_annotation_protocol_lock_v2.json").read_text())
    if (
        hashlib.sha256(Path(__file__).with_name("api.py").read_bytes()).hexdigest()
        != lock["api_source_sha256"]
    ):
        raise ValueError("frozen annotation API required")
    corpus = [json.loads(line) for line in source.read_text().splitlines()]
    if len(corpus) != 8 or {row["parent_source_line"] for row in corpus} != {
        90,
        112,
        128,
        137,
    }:
        raise ValueError("only the fixed previously-used development sample is allowed")
    problem = min(row["source_problem"] for row in corpus)
    rows = [row for row in corpus if row["source_problem"] == problem]
    if len(rows) != 2 or len({row["participant_model"] for row in rows}) != 2:
        raise ValueError("one complete two-model development family required")
    if len({row["source_story"] for row in rows}) != 1:
        raise ValueError("development source stories differ")
    paragraphs = rows[0]["source_story"].split("\n\n")
    results = []
    async with BudgetedChat(
        out,
        list(ANNOTATION_REASONING),
        1.0,
        max_tokens=16384,
        reasoning_by_model=ANNOTATION_REASONING,
        schema_retries=1,
        concurrency=1,
    ) as api:
        manifest = {
            "source_corpus_sha256": sha256(source.read_text()),
            "rows": [row["id"] for row in rows],
            "source_problem": problem,
            "parent_source_lines": [row["parent_source_line"] for row in rows],
            "protocol": "Exact frozen primary and secondary payloads on old development data; no prompt, schema, head or sample-selection changes.",
            "requested_reasoning_by_model": ANNOTATION_REASONING,
            "max_tokens": 16384,
            "reservation_cap_usd": 1.0,
            "script_sha256": sha256(Path(__file__).read_text()),
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for granularity in ["sentence", "clause"]:
            items = [
                (row, index, unit)
                for row in rows
                for index, unit in enumerate(
                    display_units(row["response"], [granularity])[1]
                )
            ]
            if len(items) > 48:
                raise ValueError("smoke must fit the frozen chunk size")
            if granularity == "sentence":
                claims = [
                    {
                        "index": i,
                        "text": unit["display_text"],
                        "COMPLETE_ARGUMENT": argument_mapping(row["response"])[0],
                    }
                    for i, (row, _, unit) in enumerate(items)
                ]
                payload = {
                    "SOURCE_STORY_PARAGRAPHS": [
                        {"id": i, "text": p} for i, p in enumerate(paragraphs)
                    ],
                    "EXACT_SUBSTRINGS_TO_ASSESS": claims,
                }
                rules = UNIT_RULES
            else:
                payload, claims = complete_payload(paragraphs, items)
                rules = CLAUSE_RULES

            def validate(obj):
                validate_checks(obj, len(items), len(paragraphs))
                if any(
                    c["exact_text"] != claims[c["index"]]["text"] for c in obj["checks"]
                ):
                    raise jsonschema.ValidationError(
                        "exact_text must copy indexed units including whitespace"
                    )

            for model in ANNOTATION_REASONING:
                receipt = await api.complete(
                    model,
                    [
                        {"role": "system", "content": rules},
                        {
                            "role": "user",
                            "content": json.dumps(payload, ensure_ascii=False),
                        },
                    ],
                    object_schema(
                        {
                            "checks": {
                                "type": "array",
                                "items": MIXED_CHECK,
                                "minItems": len(items),
                                "maxItems": len(items),
                            }
                        }
                    ),
                    {
                        "stage": "frozen_annotation_old_dev_smoke",
                        "granularity": granularity,
                        "source_problem": problem,
                    },
                    response_validator=validate,
                )
                raw = json.loads(receipt["response_body"])
                choice = raw["choices"][0]
                results.append(
                    {
                        "model": model,
                        "granularity": granularity,
                        "expected_units": len(items),
                        "completed_units": len(receipt["parsed"]["checks"]),
                        "units_with_trailing_whitespace": sum(
                            c["text"] != c["text"].rstrip() for c in claims
                        ),
                        "finish_reason": choice["finish_reason"],
                        "native_finish_reason": choice.get("native_finish_reason"),
                        "reported_reasoning_tokens": raw["usage"]
                        .get("completion_tokens_details", {})
                        .get("reasoning_tokens"),
                        "returned_reasoning_present": bool(
                            choice["message"].get("reasoning")
                        ),
                        "request_sha256": receipt["request_sha256"],
                    }
                )
        summary = {
            "calls_completed": len(results),
            "cells": results,
            "finish_reasons": dict(
                Counter(r.get("finish_reason") for r in api.records)
            ),
            "cost_upper_usd": api.spent,
            "fresh_outcomes_consumed": False,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
