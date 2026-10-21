"""Propose one local name-swap in complete arguments for blind verification.

The resulting response pairs contain both proposed true and false claims in
each world, so a broadcast response score cannot localize them. Proposed
labels are mechanical hypotheses, not accepted training labels.
"""

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path

from .counterfactuals import swap_entities
from .evaluate_debate_transfer import argument_mapping
from .local3_natural_twins import check_balanced_pairs
from .preflight import sha256


def nonoverlapping(spans):
    output = []
    for span in sorted(spans, key=lambda s: (s["argument_end"], s["argument_start"])):
        if not output or span["argument_start"] >= output[-1]["argument_end"]:
            output.append(span)
    return output


def mutate(row, selected, mapping):
    original_argument, _ = argument_mapping(row["response"])
    chosen = sorted(
        selected,
        key=lambda s: sha256(
            row["pair_id"] + str(s["original_span_index"]) + "local3-mixed-20261014"
        ),
    )[0]
    text = chosen["display_text"]
    replacement = swap_entities(text, mapping)
    if replacement == text or swap_entities(replacement, mapping) != text:
        raise ValueError("local name-swap must be nontrivial and reversible")
    encoded = (
        json.dumps(replacement, ensure_ascii=False).removeprefix('"').removesuffix('"')
    )
    response = (
        row["response"][: chosen["start"]] + encoded + row["response"][chosen["end"] :]
    )
    argument, offsets = argument_mapping(response)
    start, end = chosen["argument_start"], chosen["argument_end"]
    expected = original_argument[:start] + replacement + original_argument[end:]
    if argument != expected:
        raise ValueError("local edit changed other argument bytes")
    shift = len(replacement) - len(text)
    spans = []
    for old in selected:
        span = copy.deepcopy(old)
        altered = old["original_span_index"] == chosen["original_span_index"]
        display = replacement if altered else old["display_text"]
        a = old["argument_start"] + (shift if old["argument_start"] >= end else 0)
        b = a + len(display)
        if argument[a:b] != display or argument.count(display) != 1:
            raise ValueError("edited claim is not exact and unique")
        raw_start, raw_end = offsets[a][0], offsets[b - 1][1]
        span.update(
            start=raw_start,
            end=raw_end,
            text=response[raw_start:raw_end],
            display_text=display,
            argument_start=a,
            argument_end=b,
            label=old["label"] ^ int(altered),
            label_kind="mechanical_mixed_augmentation_requires_blind_verification",
        )
        spans.append(span)
    if len({s["label"] for s in spans}) != 2:
        raise ValueError("augmentation must have mixed truth labels")
    return (
        response,
        spans,
        {
            "kind": "single_claim_entity_swap",
            "changed_original_span_index": chosen["original_span_index"],
            "before": text,
            "after": replacement,
            "argument_sha256_before": sha256(original_argument),
            "argument_sha256_after": sha256(argument),
            "source_pair_id": row["pair_id"],
            "selection": "One response per prior parent and participant, lowest deterministic hash among responses with at least two nonoverlapping verified spans. Changed claim selected by deterministic hash before probe outcomes.",
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--worlds", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("immutable new output required")
    source = [
        json.loads(line)
        for line in (args.source / "corpus.jsonl").read_text().splitlines()
    ]
    by_id = {r["id"]: r for r in source}
    worlds = {
        (r["parent_problem_id"], r["variant"]): r
        for r in map(json.loads, args.worlds.read_text().splitlines())
    }
    grouped = defaultdict(list)
    for row in source:
        if row["context_world"] == 0 and len(nonoverlapping(row["spans"])) >= 2:
            grouped[row["group"], row["participant_model"]].append(row)
    rows, selections = [], []
    for key, eligible in sorted(grouped.items()):
        row = min(eligible, key=lambda r: sha256(r["id"] + "mixed-selection-20261014"))
        selected = nonoverlapping(row["spans"])
        mapping = worlds[row["group"], "original"]["entity_mapping"]
        response, spans, augmentation = mutate(row, selected, mapping)
        pair_id = "mixed-" + row["pair_id"]
        for context in [0, 1]:
            original = by_id[row["pair_id"] + f"-context{context}"]
            output = copy.deepcopy(original)
            output.update(
                id=pair_id + f"-context{context}",
                pair_id=pair_id,
                response=response,
                spans=[{**s, "label": s["label"] ^ context} for s in spans],
                augmentation=augmentation,
            )
            output["label_status"] = (
                "Proposed labels only: one verified claim name-swapped inside the complete original response. Both source worlds and every local label require blind independent verification before training. This is teacher-forced factual inconsistency augmentation, not generated deception."
            )
            rows.append(output)
        selections.append(
            {
                "group": key[0],
                "participant": key[1],
                "source_row_id": row["id"],
                "eligible_responses": len(eligible),
                "selected_claims": len(spans),
                **augmentation,
            }
        )
    summary = {
        "errors": [],
        "expected_parent_groups": len({r["group"] for r in rows}),
        "completed_parent_groups": len({r["group"] for r in rows}),
        **check_balanced_pairs(rows),
        "split_rows": dict(Counter(r["split"] for r in rows)),
        "all_responses_mixed": all(
            len({s["label"] for s in r["spans"]}) == 2 for r in rows
        ),
    }
    args.out.mkdir(parents=True)
    (args.out / "corpus.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    manifest = {
        "script_sha256": sha256(Path(__file__).read_text()),
        "source_corpus_sha256": sha256((args.source / "corpus.jsonl").read_text()),
        "corpus_sha256": sha256((args.out / "corpus.jsonl").read_text()),
        "selection_count": len(selections),
        "selections": selections,
        "label_status": "Mechanical hypotheses pending both-world independent annotation; not training labels.",
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
