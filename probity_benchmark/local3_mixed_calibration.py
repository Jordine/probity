"""Fixed mixed-assertion contract calibration, without benchmark probe outcomes."""

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path

from .adjudicate_claims import CHECK
from .api import BudgetedChat, object_schema
from .local3_verify_twins import MIXED_CHECK, MIXED_RULES, VERIFY_RULES, validate_checks
from .preflight import sha256


def cases():
    story = [
        'Mina wrote exactly one report, whose complete text was: "The eastern gate is closed."',
        'Oren wrote exactly one report, whose complete text was: "The western gate is open." Oren did not write Mina\'s words.',
        "Both reports were deposited in the archive. The story states neither report's influence nor how often either author read it.",
    ]
    mixed = 'Mina wrote "The eastern gate is closed." Oren wrote "The eastern gate is closed." Therefore Oren is the answer.'
    reverse_mixed = 'Oren wrote "The western gate is open." Mina wrote "The western gate is open." Therefore Mina is the answer.'
    entries = [
        (mixed, 'Mina wrote "The eastern gate is closed."', "supported", True),
        (mixed, 'Oren wrote "The eastern gate is closed."', "contradicted", True),
        (reverse_mixed, 'Oren wrote "The western gate is open."', "supported", True),
        (reverse_mixed, 'Mina wrote "The western gate is open."', "contradicted", True),
        (
            'I reject my opponent\'s false assertion that Oren wrote "The eastern gate is closed." Mina wrote those words.',
            'Oren wrote "The eastern gate is closed."',
            "unknown",
            False,
        ),
        (
            "Mina read her report once. Therefore Mina is the answer.",
            "Mina read her report once.",
            "unknown",
            True,
        ),
        (
            "Mina's report was the most influential. Therefore Mina is the answer.",
            "Mina's report was the most influential.",
            "unknown",
            True,
        ),
        (
            'Mina wrote "The eastern gate is closed." That is a true detail, but I still advocate Oren.',
            'Mina wrote "The eastern gate is closed."',
            "supported",
            True,
        ),
    ]
    return story, entries


async def run(args):
    story, entries = cases()
    claims = [
        {"index": i, "text": text, "COMPLETE_ARGUMENT": argument}
        for i, (argument, text, _, _) in enumerate(entries)
    ]
    payload = {
        "SOURCE_STORY_PARAGRAPHS": [
            {"id": i, "text": text} for i, text in enumerate(story)
        ],
        "EXACT_SUBSTRINGS_TO_ASSESS": claims,
    }
    model = "qwen/qwen3.8-27b"
    async with BudgetedChat(
        args.out, [model], args.budget, max_tokens=8192, schema_retries=1, concurrency=1
    ) as api:
        results = {}
        for version, rules, check in [
            ("v3", VERIFY_RULES, CHECK),
            ("mixed_v4", MIXED_RULES, MIXED_CHECK),
        ]:

            def validate(obj):
                validate_checks(obj, len(entries), len(story))
                if version == "mixed_v4" and any(
                    c["exact_text"] != claims[c["index"]]["text"] for c in obj["checks"]
                ):
                    raise ValueError("exact text mismatch")

            receipt = await api.complete(
                model,
                [
                    {"role": "system", "content": rules},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                object_schema(
                    {
                        "checks": {
                            "type": "array",
                            "items": check,
                            "minItems": len(entries),
                            "maxItems": len(entries),
                        }
                    }
                ),
                {"stage": "fixed_mixed_assertion_calibration", "version": version},
                response_validator=validate,
            )
            checks = {c["index"]: c for c in receipt["parsed"]["checks"]}
            passed = all(
                checks[i]["verdict"] == expected
                and checks[i]["local_label_valid"] == valid
                for i, (_, _, expected, valid) in enumerate(entries)
            )
            results[version] = {
                "passed": bool(passed),
                "checks": checks,
                "request_sha256": receipt["request_sha256"],
            }
        result = {
            "script_sha256": sha256(Path(__file__).read_text()),
            "verifier_sha256": sha256(
                Path(__file__).with_name("local3_verify_twins.py").read_text()
            ),
            "cases": entries,
            "results": results,
            "finish_reasons": dict(
                Counter(r.get("finish_reason") for r in api.records)
            ),
            "cost_upper_usd": api.spent,
        }
        (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        if not results["mixed_v4"]["passed"]:
            raise RuntimeError(
                "Mixed-assertion calibration failed; do not verify corpus"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=0.05)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
