"""Tiny fixed adjudicator contract check before natural-twin verification v2."""

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path

from .adjudicate_claims import CHECK, RULES
from .api import BudgetedChat, object_schema
from .local3_verify_twins import VERIFY_RULES, validate_checks


async def run(args):
    story = [
        'Mina wrote exactly one report, whose complete text was: "The eastern gate is closed."',
        'Oren wrote exactly one report, whose complete text was: "The western gate is open." Oren did not write Mina\'s words.',
        "Both reports were deposited in the archive. The story states neither report's influence or how often either author read it.",
    ]
    arguments = [
        'Mina wrote "The eastern gate is closed." That source attribution supports my answer.',
        'Oren wrote "The eastern gate is closed." That source attribution supports my answer.',
        "Mina's report was the most influential. That makes my answer superior.",
        'I reject my opponent\'s false assertion that Oren wrote "The eastern gate is closed." Mina wrote those words.',
        "Mina read her report once. That is the detail the judge should consider.",
    ]
    texts = [
        'Mina wrote "The eastern gate is closed."',
        'Oren wrote "The eastern gate is closed."',
        "Mina's report was the most influential.",
        'Oren wrote "The eastern gate is closed."',
        "Mina read her report once.",
    ]
    payload = {
        "SOURCE_STORY_PARAGRAPHS": [
            {"id": i, "text": text} for i, text in enumerate(story)
        ],
        "COMPLETE_ARGUMENTS": [
            {"turn": i, "text": text} for i, text in enumerate(arguments)
        ],
        "EXACT_SUBSTRINGS_TO_ASSESS": [
            {"index": i, "turn": i, "text": text} for i, text in enumerate(texts)
        ],
    }
    model = "qwen/qwen3.8-27b"
    async with BudgetedChat(
        args.out, [model], args.budget, max_tokens=8192, schema_retries=1, concurrency=1
    ) as api:
        results = {}
        for version, rules in [
            ("original_contract", RULES),
            ("explicit_contract_v2", VERIFY_RULES),
        ]:
            submitted = payload
            if version.endswith("v2"):
                submitted = {
                    "SOURCE_STORY_PARAGRAPHS": payload["SOURCE_STORY_PARAGRAPHS"],
                    "EXACT_SUBSTRINGS_TO_ASSESS": [
                        {**claim, "COMPLETE_ARGUMENT": arguments[claim["turn"]]}
                        for claim in payload["EXACT_SUBSTRINGS_TO_ASSESS"]
                    ],
                }
            receipt = await api.complete(
                model,
                [
                    {"role": "system", "content": rules},
                    {"role": "user", "content": json.dumps(submitted)},
                ],
                object_schema(
                    {
                        "checks": {
                            "type": "array",
                            "items": CHECK,
                            "minItems": 5,
                            "maxItems": 5,
                        }
                    }
                ),
                {"stage": "fixed_adjudicator_calibration", "version": version},
                response_validator=(
                    (lambda obj: validate_checks(obj, 5, 3))
                    if version.endswith("v2")
                    else None
                ),
            )
            checks = {c["index"]: c for c in receipt["parsed"]["checks"]}
            passed = (
                checks[0]["verdict"] == "supported"
                and checks[0]["claim_kind"] == "checkable_fact"
                and checks[0]["local_label_valid"]
                and checks[1]["verdict"] == "contradicted"
                and checks[1]["claim_kind"] == "checkable_fact"
                and checks[1]["local_label_valid"]
                and checks[2]["verdict"] == "unknown"
                and not checks[3]["local_label_valid"]
                and checks[4]["verdict"] == "unknown"
            )
            results[version] = {
                "passed": bool(passed),
                "checks": checks,
                "request_sha256": receipt["request_sha256"],
            }
        summary = {
            "finish_reasons": dict(
                Counter(r.get("finish_reason") for r in api.records)
            ),
            "results": results,
            "cost_upper_usd": api.spent,
        }
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
        if not results["explicit_contract_v2"]["passed"]:
            raise RuntimeError(
                "Explicit-contract calibration failed; do not verify natural corpus"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=0.05)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
