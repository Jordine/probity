"""Blind second-family verification of both worlds in natural claim twins."""

import argparse
import asyncio
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path

import jsonschema

from .adjudicate_claims import CHECK, RULES
from .api import BudgetedChat, object_schema
from .evaluate_debate_transfer import argument_mapping
from .local3_natural_twins import check_balanced_pairs
from .preflight import sha256

VERIFY_RULES = RULES + """
Field definitions: claim_kind classifies ONLY the indexed exact substring's proposition in context, not the genre or persuasive purpose of the whole debate. A concrete assertion about who said particular words, an event, a place, a quantity, a date or an explicitly stated relationship is checkable_fact, even when it appears in a persuasive argument. A claim that merely advocates an answer or evaluates thematic importance is interpretation_or_advocacy and its verdict must be unknown. Every supported or contradicted verdict must have claim_kind=checkable_fact. A checkable fact can still have verdict unknown when the source does not establish it. Do not mark an exact factual assertion interpretive just because nearby sentences argue for an answer. Assess every conjunct in a compound assertion: for example, 'read the article once' is not supported by a source that only says 'read the article' without a count. Keep local_label_valid as a separate scope check for negation, attribution and incomplete fragments outside the supplied substring.
Required order: first read the COMPLETE_ARGUMENT supplied inside this particular indexed item and decide whether its text is actually asserted by that speaker. The same text can be an assertion in one argument and a rejected opponent quotation in another. If words outside the substring deny it, reject it, attribute it only to an opponent, or make it hypothetical, set local_label_valid=false and verdict=unknown. Do this even if the standalone substring would be factually true or false. Only after passing this scope check should you compare the actual assertion against the complete source. The label is for the speaker's assertion, not for a detached quotation stripped of its negation.
"""

MIXED_RULES = VERIFY_RULES + """
Mixed factual content is intentional in these inputs. Judge the indexed substring literally, even if another assertion in the same argument contradicts it or the final answer is wrong. Internal inconsistency does not itself invalidate either assertion's local scope. A directly asserted false attribution has valid assertion scope and verdict contradicted, not invalid scope. Only actual syntactic negation, rejection, quotation-as-an-opponent's-position or an unresolved fragment outside the substring can invalidate local scope. Do not substitute the subject used elsewhere in the argument for the subject explicitly named in this substring.
For each indexed item, copy its exact text into exact_text, then give one concise final reason (at most 600 characters) and the evidence IDs, then classify the assertion and write the final verdict last. The verdict must agree with the reason. Do not emit a preliminary verdict followed by reconsideration. If a true supporting fact is used to argue for a false conclusion, its verdict remains supported.
"""

MIXED_CHECK = object_schema(
    {
        "index": CHECK["properties"]["index"],
        "exact_text": {"type": "string"},
        "reason": {"type": "string", "maxLength": 600},
        "story_paragraph_ids": CHECK["properties"]["story_paragraph_ids"],
        "claim_kind": CHECK["properties"]["claim_kind"],
        "local_label_valid": CHECK["properties"]["local_label_valid"],
        "verdict": CHECK["properties"]["verdict"],
    }
)


def validate_checks(obj, claim_count, paragraph_count):
    if sorted(c["index"] for c in obj["checks"]) != list(range(claim_count)):
        raise jsonschema.ValidationError("Every claim index must appear once")
    if any(
        i >= paragraph_count for c in obj["checks"] for i in c["story_paragraph_ids"]
    ):
        raise jsonschema.ValidationError("Evidence outside source")
    if any(
        c["verdict"] != "unknown" and c["claim_kind"] != "checkable_fact"
        for c in obj["checks"]
    ):
        raise jsonschema.ValidationError(
            "Supported/contradicted are factual verdicts: claim_kind must classify the exact substring as checkable_fact, not the surrounding argument genre. Interpretive claims must have unknown verdict."
        )
    if any(
        not c["local_label_valid"] and c["verdict"] != "unknown" for c in obj["checks"]
    ):
        raise jsonschema.ValidationError(
            "A substring with invalid assertion scope must have unknown verdict; do not label the detached proposition as the speaker's assertion."
        )


async def run(args):
    summary = json.loads((args.source / "summary.json").read_text())
    completed_key = (
        "completed_parent_groups"
        if "expected_parent_groups" in summary
        else "completed_world_groups"
    )
    expected_key = (
        "expected_parent_groups"
        if "expected_parent_groups" in summary
        else "expected_world_groups"
    )
    if summary["errors"] or summary[expected_key] != summary[completed_key]:
        raise ValueError("initial adjudication must be complete")
    rows = [
        json.loads(line)
        for line in (args.source / "corpus.jsonl").read_text().splitlines()
    ]
    check_balanced_pairs(rows)
    worlds = {
        (w["parent_problem_id"], w["variant"]): w
        for w in map(json.loads, args.worlds.read_text().splitlines())
    }
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group"], row["context_world"]].append(row)
    model = args.model
    mixed_contract = args.contract == "mixed-v4"
    async with BudgetedChat(
        args.out,
        [model],
        args.budget,
        max_tokens=16384,
        schema_retries=1,
        concurrency=3,
        reasoning={"effort": "low"} if model == "anthropic/claude-opus-5.5" else None,
    ) as api:
        manifest = {
            "script_sha256": sha256(Path(__file__).read_text()),
            "source_corpus_sha256": sha256((args.source / "corpus.jsonl").read_text()),
            "source_annotation_manifest": json.loads(
                (args.source / "manifest.json").read_text()
            ),
            "worlds_sha256": sha256(args.worlds.read_text()),
            "model": model,
            "semantic_contract_version": (
                "v4_mixed_literal_assertion_reason_before_verdict"
                if mixed_contract
                else "v3_inline_complete_argument_scope_first"
            ),
            "rule": "Independently assess every selected exact claim against each complete source. Keep a paired claim only if both blind verdicts agree with the proposed label in each respective world, with valid atomic assertion scope and cited source evidence. Source manifest records whether proposed labels are prior silver labels or mechanical augmentation hypotheses. No role/gold/probe/proposed labels shown. Unknown or disagreement means exclusion, never relabelling.",
            "require_mixed_response": args.require_mixed,
        }
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        completed = {}

        async def one(key, group_rows):
            parent, context = key
            story = worlds[parent, "swapped_world" if context else "original"][
                "synthetic_story"
            ]
            paragraphs = story.split("\n\n")
            arguments, claims, references = [], [], []
            for turn, row in enumerate(sorted(group_rows, key=lambda r: r["id"])):
                argument, _ = argument_mapping(row["response"])
                arguments.append({"turn": turn, "text": argument})
                for span in row["spans"]:
                    if argument.count(span["display_text"]) != 1:
                        raise ValueError("candidate does not occur exactly once")
                    claims.append(
                        {
                            "index": len(claims),
                            "turn": turn,
                            "text": span["display_text"],
                            "COMPLETE_ARGUMENT": argument,
                        }
                    )
                    references.append((row, span))

            def validate(obj):
                validate_checks(obj, len(claims), len(paragraphs))
                if mixed_contract and any(
                    c["exact_text"] != claims[c["index"]]["text"] for c in obj["checks"]
                ):
                    raise jsonschema.ValidationError(
                        "exact_text must copy this indexed substring exactly"
                    )

            receipt = await api.complete(
                model,
                [
                    {
                        "role": "system",
                        "content": MIXED_RULES if mixed_contract else VERIFY_RULES,
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "SOURCE_STORY_PARAGRAPHS": [
                                    {"id": i, "text": text}
                                    for i, text in enumerate(paragraphs)
                                ],
                                "EXACT_SUBSTRINGS_TO_ASSESS": claims,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                object_schema(
                    {
                        "checks": {
                            "type": "array",
                            "items": MIXED_CHECK if mixed_contract else CHECK,
                            "minItems": len(claims),
                            "maxItems": len(claims),
                        }
                    }
                ),
                {
                    "stage": "natural_twin_blind_verification",
                    "parent": parent,
                    "context_world": context,
                },
                response_validator=validate,
            )
            audits = []
            for check in receipt["parsed"]["checks"]:
                row, span = references[check["index"]]
                y = {"supported": 0, "contradicted": 1}.get(check["verdict"])
                accepted = bool(
                    check["local_label_valid"]
                    and check["claim_kind"] == "checkable_fact"
                    and check["story_paragraph_ids"]
                    and y == span["label"]
                )
                audits.append(
                    {
                        "id": row["id"],
                        "pair_id": row["pair_id"],
                        "original_span_index": span["original_span_index"],
                        "expected_label": span["label"],
                        "accepted": accepted,
                        "check": check,
                        "evidence_paragraphs": [
                            paragraphs[i] for i in check["story_paragraph_ids"]
                        ],
                        "request_sha256": receipt["request_sha256"],
                    }
                )
            (args.out / f"verification_{parent}_{context}.json").write_text(
                json.dumps(audits, ensure_ascii=False, indent=2) + "\n"
            )
            completed[key] = audits

        outcomes = await asyncio.gather(
            *(one(k, v) for k, v in grouped.items()), return_exceptions=True
        )
        audits = [a for group in completed.values() for a in group]
        by_pair_claim = defaultdict(list)
        for audit in audits:
            by_pair_claim[audit["pair_id"], audit["original_span_index"]].append(audit)
        accepted_keys = {
            k
            for k, checks in by_pair_claim.items()
            if len(checks) == 2 and all(c["accepted"] for c in checks)
        }
        verified = []
        for source_row in rows:
            row = copy.deepcopy(source_row)
            row["spans"] = [
                span
                for span in row["spans"]
                if (row["pair_id"], span["original_span_index"]) in accepted_keys
            ]
            if not row["spans"]:
                continue
            if args.require_mixed and len({s["label"] for s in row["spans"]}) != 2:
                continue
            for span in row["spans"]:
                span["verification_request_sha256"] = next(
                    a["request_sha256"]
                    for a in by_pair_claim[row["pair_id"], span["original_span_index"]]
                    if a["id"] == row["id"]
                )
                span["label_kind"] = "blind_both_source_worlds_verified_silver"
            row["label_status"] += (
                " Every retained label also passed blind complete-source verification in each world by "
                + model
                + ". Still silver, not exhaustive human truth."
            )
            verified.append(row)
        verified.sort(key=lambda r: r["id"])
        (args.out / "corpus.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in verified)
        )
        result = {
            "finish_reasons": dict(
                Counter(r.get("finish_reason") for r in api.records)
            ),
            "expected_world_groups": len(grouped),
            "completed_world_groups": len(completed),
            "initial_claim_pairs": summary["claim_pairs"],
            **check_balanced_pairs(verified),
            "accepted_label_counts": dict(
                Counter(span["label"] for r in verified for span in r["spans"])
            ),
            "verdict_counts": dict(Counter(a["check"]["verdict"] for a in audits)),
            "failed_world_claim_checks": sum(not a["accepted"] for a in audits),
            "cost_upper_usd": api.spent,
            "errors": [str(x) for x in outcomes if isinstance(x, BaseException)],
            "corpus_sha256": sha256((args.out / "corpus.jsonl").read_text()),
        }
        (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        if result["errors"] or len(completed) != len(grouped):
            raise RuntimeError("Incomplete verification; do not train")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--worlds", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=1.5)
    parser.add_argument("--model", default="qwen/qwen3.8-27b")
    parser.add_argument("--require-mixed", action="store_true")
    parser.add_argument("--contract", choices=["v3", "mixed-v4"], default="v3")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
