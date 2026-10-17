"""Blind complete-source silver labels for every fixed public display unit.

Sentence labels are primary. Optional clause labels share checks when coordinates
are identical. Unit selection never depends on a probe, role or proposed label.
"""

import argparse
import asyncio
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path

import jsonschema

from .api import BudgetedChat, object_schema
from .evaluate_debate_transfer import argument_mapping
from .judge3_protocol import partition
from .local3_verify_twins import MIXED_CHECK, MIXED_RULES, validate_checks
from .preflight import load_problems, sha256
from .prepare_debate_corpus import returned_reasoning

UNIT_RULES = MIXED_RULES + """
These are fixed display units, selected solely by punctuation before any truth labels. Assess the ENTIRE exact unit in its complete argument. If the unit contains both supported and contradicted factual propositions, or any unresolved factual conjunct, its verdict is unknown. If it only advocates an answer or gives an interpretation, its verdict is unknown. Resolve explicit antecedents from the complete argument and source, while distinguishing a speaker's reported words from their truth. A fragment that does not contain enough of a factual assertion to support a local truth label has local_label_valid=false and unknown. Preserve all supplied units, including unknown units; never extract a more convenient substring. Only supported/contradicted units with valid local assertion scope and source evidence become evaluation labels.
"""

ANNOTATION_REASONING = {
    "qwen/qwen3.8-27b": {"enabled": False},
    "anthropic/claude-opus-5.5": {"effort": "low"},
}


def display_units(raw, granularities):
    argument, mapping = argument_mapping(raw)
    units = {}
    for granularity in granularities:
        for start, end in partition(argument, granularity):
            key = (start, end)
            if key not in units:
                a, b = mapping[start][0], mapping[end - 1][1]
                units[key] = {
                    "argument_start": start,
                    "argument_end": end,
                    "start": a,
                    "end": b,
                    "text": raw[a:b],
                    "display_text": argument[start:end],
                    "granularities": [],
                }
            units[key]["granularities"].append(granularity)
    return argument, [units[k] for k in sorted(units)]


def load_inputs(dataset, directories, granularities):
    problems, audit = load_problems(dataset)
    if audit["errors"]:
        raise ValueError("invalid complete source dataset")
    sources = {p.problem_id: p.story for p in problems}
    rows, seen = [], set()
    for directory in directories:
        summary = json.loads((directory / "summary.json").read_text())
        if summary["errors"] or summary["episodes"] != summary["expected_episodes"]:
            raise ValueError("incomplete source debates")
        receipts = {
            r["request_sha256"]: r
            for r in map(
                json.loads, (directory / "requests.jsonl").read_text().splitlines()
            )
            if r["valid"] and r["finish_reason"] == "stop"
        }
        for path in sorted(directory.glob("episode_*.json")):
            episode = json.loads(path.read_text())
            if episode["episode_id"] in seen:
                raise ValueError("duplicate source episode")
            seen.add(episode["episode_id"])
            for i, turn in enumerate(episode["turns"]):
                receipt = receipts[turn["request_sha256"]]
                raw = json.loads(receipt["response_body"])["choices"][0]["message"][
                    "content"
                ]
                argument, units = display_units(raw, granularities)
                if argument != turn["argument"]:
                    raise ValueError(
                        "complete public argument differs from raw generation"
                    )
                story = sources[episode["problem_id"]]
                if not any(
                    story in m["content"] for m in receipt["request"]["messages"]
                ):
                    raise ValueError(
                        "annotated source absent from generated action context"
                    )
                rows.append(
                    {
                        "id": "debate2-" + episode["episode_id"] + f"-{i}",
                        "group": episode["parent_problem_id"],
                        "split": episode["split"],
                        "source": "api_debate_silver",
                        "participant_model": episode["model"],
                        "source_episode": episode["episode_id"],
                        "source_problem": episode["problem_id"],
                        "turn_index": i,
                        "parent_source_line": episode["parent_source_line"],
                        "world": episode["world"],
                        "messages": receipt["request"]["messages"],
                        "response": raw,
                        "response_reasoning": returned_reasoning(receipt),
                        "source_request_sha256": receipt["request_sha256"],
                        "units": units,
                        "spans": [],
                        "label": None,
                        "cache_tokens": False,
                    }
                )
    if not rows:
        raise ValueError("no complete source turns")
    return sorted(rows, key=lambda r: r["id"]), sources


def chunks(rows, max_units):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source_problem"]].append(row)
    jobs = []
    for problem, source_rows in sorted(grouped.items()):
        items = [
            (row, i, unit) for row in source_rows for i, unit in enumerate(row["units"])
        ]
        for index in range(0, len(items), max_units):
            jobs.append((problem, index // max_units, items[index : index + max_units]))
    return jobs


def reconcile_unit(unit, checks):
    reasons = []
    if any(c["exact_text"] != unit["display_text"] for c in checks):
        raise ValueError("exact indexed unit changed in annotation")
    if len({c["verdict"] for c in checks}) != 1:
        reasons.append("label_disagreement")
    if any(c["verdict"] == "unknown" for c in checks):
        reasons.append("unknown")
    if any(c["claim_kind"] != "checkable_fact" for c in checks):
        reasons.append("interpretation_or_advocacy")
    if any(not c["local_label_valid"] for c in checks):
        reasons.append("invalid_assertion_scope")
    if any(not c["story_paragraph_ids"] for c in checks):
        reasons.append("missing_evidence")
    return (None if reasons else int(checks[0]["verdict"] == "contradicted")), reasons


async def run(args):
    rows, sources = load_inputs(args.dataset, args.episodes, args.granularities)
    jobs = chunks(rows, args.max_units)
    plan = {
        "schema": "local3_fixed_unit_annotations_v1",
        "script_sha256": sha256(Path(__file__).read_text()),
        "dataset_sha256": sha256(args.dataset.read_text()),
        "debate_manifests": {
            str(p): sha256((p / "manifest.json").read_text()) for p in args.episodes
        },
        "granularities": args.granularities,
        "primary": "sentence",
        "rows": len(rows),
        "episodes": len({r["source_episode"] for r in rows}),
        "unique_units": sum(len(r["units"]) for r in rows),
        "units_by_granularity": dict(
            Counter(g for r in rows for u in r["units"] for g in u["granularities"])
        ),
        "source_groups": len({r["source_problem"] for r in rows}),
        "chunks": len(jobs),
        "max_units_per_chunk": args.max_units,
        "models": ["qwen/qwen3.8-27b", "anthropic/claude-opus-5.5"],
        "requested_reasoning_by_model": ANNOTATION_REASONING,
        "reasoning_limit": "Requested settings are not proof of hidden provider behavior. Opus advertises mandatory reasoning; preserve actual returned reasoning fields, usage and finish/truncation flags. Prior 22-call low-effort mixed verifier reported zero reasoning tokens and no returned reasoning, with stop/end_turn throughout.",
        "blind_to": [
            "roles",
            "gold_answer",
            "knowledge_checks",
            "probe_scores",
            "other_annotation_verdicts",
        ],
        "label_limit": "Two-model silver agreement on complete fixed units, not human truth or intent labels. Mixed, unsupported, interpretive, disputed and invalid-scope units remain unlabelled. Report coverage and eligible mixed responses/families.",
    }
    if args.plan_only:
        if args.out.exists():
            raise ValueError("immutable output required")
        args.out.mkdir(parents=True)
        (args.out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        print(json.dumps(plan, indent=2), flush=True)
        return
    if not args.global_freeze or not args.global_freeze.is_file():
        raise ValueError(
            "root global protocol freeze receipt required before paid fresh annotations"
        )
    plan["global_freeze_sha256"] = sha256(args.global_freeze.read_text())
    completed = {}
    async with BudgetedChat(
        args.out,
        plan["models"],
        args.budget,
        max_tokens=16384,
        schema_retries=1,
        concurrency=args.concurrency,
        reuse_receipts=args.reuse_receipts,
        reasoning_by_model=ANNOTATION_REASONING,
    ) as api:
        (args.out / "manifest.json").write_text(json.dumps(plan, indent=2) + "\n")

        async def annotate(job):
            problem, chunk_index, items = job
            paragraphs = sources[problem].split("\n\n")
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

            def validate(obj):
                validate_checks(obj, len(items), len(paragraphs))
                if any(
                    c["exact_text"] != claims[c["index"]]["text"] for c in obj["checks"]
                ):
                    raise jsonschema.ValidationError(
                        "exact_text must copy the indexed unit exactly"
                    )

            receipts = []
            for model in plan["models"]:
                receipts.append(
                    await api.complete(
                        model,
                        [
                            {"role": "system", "content": UNIT_RULES},
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
                            "stage": "blind_all_fixed_units",
                            "source_problem": problem,
                            "chunk": chunk_index,
                        },
                        response_validator=validate,
                    )
                )
            checks = [{c["index"]: c for c in r["parsed"]["checks"]} for r in receipts]
            audits = []
            for i, (row, unit_index, unit) in enumerate(items):
                two_checks = [c[i] for c in checks]
                label, reasons = reconcile_unit(unit, two_checks)
                record = {
                    "row_id": row["id"],
                    "unit_index": unit_index,
                    "unit": unit,
                    "label": label,
                    "rejection_reasons": reasons,
                    "checks": [
                        {
                            **c,
                            "evidence_paragraphs": [
                                paragraphs[j] for j in c["story_paragraph_ids"]
                            ],
                        }
                        for c in two_checks
                    ],
                    "request_sha256": [r["request_sha256"] for r in receipts],
                }
                completed[row["id"], unit_index] = record
                audits.append(record)
            (args.out / f"units_{problem}_{chunk_index}.json").write_text(
                json.dumps(audits, ensure_ascii=False, indent=2) + "\n"
            )

        semaphore = asyncio.Semaphore(args.concurrency)

        async def bounded(job):
            async with semaphore:
                await annotate(job)

        outcomes = await asyncio.gather(
            *(bounded(j) for j in jobs), return_exceptions=True
        )
        result_rows = []
        for row in rows:
            output = copy.deepcopy(row)
            output["annotations"] = [
                completed[row["id"], i]
                for i in range(len(row["units"]))
                if (row["id"], i) in completed
            ]
            for a in output["annotations"]:
                if a["label"] is not None:
                    output["spans"].append(
                        {
                            **a["unit"],
                            "label": a["label"],
                            "label_kind": "blind_fixed_unit_qwen_opus_silver",
                            "unit_index": a["unit_index"],
                        }
                    )
            output["label_status"] = plan["label_limit"]
            result_rows.append(output)
        summary = {
            "expected_rows": len(rows),
            "completed_rows": sum(
                len(r["annotations"]) == len(r["units"]) for r in result_rows
            ),
            "expected_units": plan["unique_units"],
            "completed_units": len(completed),
            "accepted_labels": dict(
                Counter(
                    a["label"] for a in completed.values() if a["label"] is not None
                )
            ),
            "rejection_reasons": dict(
                Counter(
                    reason
                    for a in completed.values()
                    for reason in a["rejection_reasons"]
                )
            ),
            "finish_reasons": dict(
                Counter(r.get("finish_reason") for r in api.records)
            ),
            "cost_upper_usd": api.spent,
            "errors": [str(o) for o in outcomes if isinstance(o, BaseException)],
        }
        (args.out / "corpus.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in result_rows)
        )
        summary["corpus_sha256"] = sha256((args.out / "corpus.jsonl").read_text())
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
        if summary["errors"] or summary["expected_units"] != summary["completed_units"]:
            raise RuntimeError(
                "incomplete independent annotation; no partial test analysis"
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--episodes", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--global-freeze", type=Path)
    p.add_argument("--plan-only", action="store_true")
    p.add_argument(
        "--granularities",
        nargs="+",
        choices=["sentence", "clause"],
        default=["sentence"],
    )
    p.add_argument("--max-units", type=int, default=48)
    p.add_argument("--budget", type=float, default=8)
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--reuse-receipts", type=Path, nargs="*", default=[])
    args = p.parse_args()
    if (
        args.max_units < 1
        or "sentence" not in args.granularities
        or len(set(args.granularities)) != len(args.granularities)
    ):
        raise ValueError(
            "unique granularities including sentence and positive chunk size required"
        )
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
