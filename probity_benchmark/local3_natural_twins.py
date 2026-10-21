"""Natural exact-response source twins, independently labelled after intervention.

Existing silver labels describe the original source. An independent blind
adjudication labels those exact spans under the complementary full story.
Only verified supported/contradicted flips enter the balanced paired corpus.
Complete response and non-source prompt bytes are preserved in both members.
"""

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
from .preflight import sha256


def candidates(corpus, worlds):
    world_index = {(w["parent_problem_id"], w["variant"]): w for w in worlds}
    by_parent = defaultdict(list)
    for row in corpus:
        # All these parents were already queried. Original splits are retained
        # for training/development selection; iteration-two test is not used.
        if row["split"] not in ["train", "dev"] or row["world"] != "original":
            continue
        source = world_index[row["group"], "original"]
        names = source["entity_mapping"]
        spans = [
            (i, span)
            for i, span in enumerate(row["spans"])
            if any(name in span["display_text"] for name in names)
            and len(span["display_text"].split()) >= 4
        ]
        if spans:
            by_parent[row["group"]].append((row, spans))
    return by_parent, world_index


def replace_source(messages, before, after):
    matches = sum(m["content"].count(before) for m in messages)
    if matches != 1:
        raise ValueError("source must occur exactly once in complete input")
    output = copy.deepcopy(messages)
    for message in output:
        if before in message["content"]:
            prefix, suffix = message["content"].split(before)
            message["content"] = prefix + after + suffix
            if message["content"].split(after) != [prefix, suffix]:
                raise ValueError("non-source prompt changed")
    return output


def check_balanced_pairs(rows):
    pairs = defaultdict(list)
    for row in rows:
        pairs[row["pair_id"]].append(row)
        for span in row["spans"]:
            if row["response"][span["start"] : span["end"]] != span["text"]:
                raise ValueError("span does not preserve raw response bytes")
    claims = 0
    for pair in pairs.values():
        if len(pair) != 2:
            raise ValueError("each response must have exactly two source contexts")
        a, b = sorted(pair, key=lambda r: r["context_world"])
        if a["response"] != b["response"] or a["group"] != b["group"]:
            raise ValueError("response or parent differs within pair")
        if len(a["spans"]) != len(b["spans"]):
            raise ValueError("paired span count differs")
        for sa, sb in zip(a["spans"], b["spans"]):
            if any(sa[k] != sb[k] for k in ["start", "end", "text"]):
                raise ValueError("paired claim text or position differs")
            if sa["label"] != 1 - sb["label"]:
                raise ValueError("paired claim labels must be complementary")
            claims += 1
    return {"response_pairs": len(pairs), "claim_pairs": claims, "rows": len(rows)}


async def run(args):
    if args.out.exists():
        raise ValueError("immutable output directory required")
    corpus = [json.loads(line) for line in args.corpus.read_text().splitlines()]
    worlds = [json.loads(line) for line in args.worlds.read_text().splitlines()]
    groups, index = candidates(corpus, worlds)
    model = "anthropic/claude-opus-5.5"
    if not groups:
        raise ValueError("no prior training/development candidate groups")
    async with BudgetedChat(
        args.out,
        [model],
        args.budget,
        max_tokens=32768,
        schema_retries=1,
        concurrency=2,
        reuse_receipts=args.reuse,
        reasoning={"effort": "low"},
    ) as api:
        manifest = {
            "script_sha256": sha256(Path(__file__).read_text()),
            "input_corpus_sha256": sha256(args.corpus.read_text()),
            "worlds_sha256": sha256(args.worlds.read_text()),
            "world_groups": len(groups),
            "candidate_responses": sum(len(v) for v in groups.values()),
            "candidate_claims": sum(
                len(spans) for values in groups.values() for _, spans in values
            ),
            "candidate_rule": "Prior train/dev parents, original-world responses, accepted silver exact spans of at least four whitespace words that contain a source-swap name. Every eligible span retained.",
            "selection": "Keep source-consistent original silver labels only when independent blind complementary-source adjudication returns a valid factual label of the opposite sign with source paragraph evidence.",
            "train_contract": "All complete responses paired with both source contexts. Only full story is replaced; system assignment, question, public history and response bytes stay unchanged. This is factual consistency under a causal source intervention, not generated deception or intent.",
            "blind_to": [
                "original_verdicts",
                "gold_answers",
                "participant_models",
                "role_assignments",
                "probe_scores",
            ],
            "model": model,
        }
        (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        completed = []

        async def one(parent, values):
            source = index[parent, "original"]
            swapped = index[parent, "swapped_world"]
            paragraphs = swapped["synthetic_story"].split("\n\n")
            arguments, claims, references = [], [], []
            for turn, (row, spans) in enumerate(
                sorted(values, key=lambda x: x[0]["id"])
            ):
                argument, _ = argument_mapping(row["response"])
                arguments.append({"turn": turn, "text": argument})
                for span_index, span in spans:
                    if argument.count(span["display_text"]) != 1:
                        raise ValueError("candidate not unique in full argument")
                    claims.append(
                        {
                            "index": len(claims),
                            "turn": turn,
                            "text": span["display_text"],
                        }
                    )
                    references.append((row, span_index, span))

            def validate(obj):
                if sorted(c["index"] for c in obj["checks"]) != list(
                    range(len(claims))
                ):
                    raise jsonschema.ValidationError(
                        "Each exact claim index must occur once"
                    )
                if any(
                    i >= len(paragraphs)
                    for c in obj["checks"]
                    for i in c["story_paragraph_ids"]
                ):
                    raise jsonschema.ValidationError(
                        "Evidence paragraph outside complete story"
                    )

            receipt = await api.complete(
                model,
                [
                    {"role": "system", "content": RULES},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "SOURCE_STORY_PARAGRAPHS": [
                                    {"id": i, "text": t}
                                    for i, t in enumerate(paragraphs)
                                ],
                                "COMPLETE_ARGUMENTS": arguments,
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
                            "items": CHECK,
                            "minItems": len(claims),
                            "maxItems": len(claims),
                        }
                    }
                ),
                {
                    "stage": "natural_source_twin_adjudication",
                    "parent": parent,
                    "claims": len(claims),
                },
                response_validator=validate,
            )
            accepted = defaultdict(list)
            audit = []
            for check in receipt["parsed"]["checks"]:
                row, span_index, span = references[check["index"]]
                verdict_label = {"supported": 0, "contradicted": 1}.get(
                    check["verdict"]
                )
                valid = bool(
                    check["local_label_valid"]
                    and check["claim_kind"] == "checkable_fact"
                    and check["story_paragraph_ids"]
                    and verdict_label is not None
                )
                flipped = valid and verdict_label == 1 - span["label"]
                record = {
                    "source_row_id": row["id"],
                    "original_span_index": span_index,
                    "original_span": span,
                    "complementary_adjudication": check,
                    "evidence_paragraphs": [
                        paragraphs[i] for i in check["story_paragraph_ids"]
                    ],
                    "accepted_flip": flipped,
                    "request_sha256": receipt["request_sha256"],
                }
                audit.append(record)
                if flipped:
                    accepted[row["id"]].append((span_index, span, record))
            output_rows = []
            for row, _ in sorted(values, key=lambda x: x[0]["id"]):
                selected = sorted(accepted[row["id"]], key=lambda item: item[0])
                if not selected:
                    continue
                for world in [0, 1]:
                    output = {
                        k: copy.deepcopy(row[k])
                        for k in [
                            "group",
                            "split",
                            "response",
                            "participant_model",
                            "parent_source_line",
                            "source_episode",
                            "turn_index",
                        ]
                    }
                    output.update(
                        {
                            "id": "local3-" + row["id"] + f"-context{world}",
                            "source": "natural_context_twins",
                            "pair_id": "local3-" + row["id"],
                            "source_row_id": row["id"],
                            "context_world": world,
                            "source_world": "swapped_world" if world else "original",
                            "messages": (
                                replace_source(
                                    row["messages"],
                                    source["synthetic_story"],
                                    swapped["synthetic_story"],
                                )
                                if world
                                else copy.deepcopy(row["messages"])
                            ),
                            "spans": [
                                {
                                    **copy.deepcopy(span),
                                    "label": span["label"] ^ world,
                                    "original_span_index": span_index,
                                    "label_kind": "opposite_context_independently_adjudicated_silver",
                                    "counterpart_request_sha256": receipt[
                                        "request_sha256"
                                    ],
                                }
                                for span_index, span, _ in selected
                            ],
                            "label": None,
                            "cache_tokens": False,
                            "label_status": "Silver factual consistency under paired source intervention. Identical response text, span positions, public history and role instructions; balanced true/false labels by independent opposite-source adjudication. Not generated intent.",
                        }
                    )
                    output_rows.append(output)
            check_balanced_pairs(output_rows)
            (args.out / f"parent_{parent}.json").write_text(
                json.dumps(
                    {"parent": parent, "audit": audit, "rows": output_rows},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )
            completed.append((parent, output_rows, audit))

        outcomes = await asyncio.gather(
            *(one(k, v) for k, v in groups.items()), return_exceptions=True
        )
        rows = sorted(
            (r for _, pair_rows, _ in completed for r in pair_rows),
            key=lambda r: r["id"],
        )
        summary = {
            "finish_reasons": dict(
                Counter(r.get("finish_reason") for r in api.records)
            ),
            "expected_parent_groups": len(groups),
            "completed_parent_groups": len(completed),
            "candidate_claims": manifest["candidate_claims"],
            **check_balanced_pairs(rows),
            "split_rows": dict(Counter(r["split"] for r in rows)),
            "accepted_label_counts": dict(
                Counter(s["label"] for r in rows for s in r["spans"])
            ),
            "cost_upper_usd": api.spent,
            "errors": [str(x) for x in outcomes if isinstance(x, BaseException)],
        }
        (args.out / "corpus.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        )
        summary["corpus_sha256"] = sha256((args.out / "corpus.jsonl").read_text())
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
        if summary["errors"] or len(completed) != len(groups):
            raise RuntimeError("Incomplete paired annotation; no partial training")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--worlds", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=5.0)
    parser.add_argument("--reuse", type=Path, nargs="*", default=[])
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
