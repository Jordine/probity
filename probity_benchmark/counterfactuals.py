"""Two manually inspected, reversible entity-swap controls; development only."""

import argparse
import json
from pathlib import Path
import re

from .preflight import load_problems, sha256

DATASET_SHA = "3c773e22849131571f5e88a774fe6001cb4bb41ca28576e00d3d634bc02b0863"
SWAPS = {
    128: {
        "Aldoria": "Jarvonia",
        "Jarvonia": "Aldoria",
        "Aldorian": "Jarvonian",
        "Jarvonian": "Aldorian",
    },
    137: {"CogniCorp": "NeuralDynamics", "NeuralDynamics": "CogniCorp"},
}


def swap_entities(text, mapping):
    pattern = r"\b(?:" + "|".join(re.escape(x) for x in mapping) + r")\b"
    return re.sub(pattern, lambda m: mapping[m.group()], text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset", type=Path, default=Path("data/quality_synthetic/stories.jsonl")
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError("refusing to overwrite a dataset")
    problems, audit = load_problems(args.dataset)
    if audit["dataset_sha256"] != DATASET_SHA or audit["errors"]:
        raise ValueError("these reviewed edits apply only to the pinned source dataset")
    rows = []
    for p in problems:
        if p.source_line not in SWAPS:
            continue
        mapping = SWAPS[p.source_line]
        changed = swap_entities(p.story, mapping)
        if changed == p.story or swap_entities(changed, mapping) != p.story:
            raise ValueError("entity swap is not complete and reversible")
        if swap_entities(p.question, mapping) != p.question:
            raise ValueError("question contains a renamed entity")
        if mapping[p.options[0]] != p.options[1]:
            raise ValueError("answer options do not follow the reviewed entity swap")
        for variant, story, gold in [
            ("original", p.story, p.gold),
            ("swapped_world", changed, {"A": "B", "B": "A"}[p.gold]),
        ]:
            rows.append(
                dict(
                    synthetic_story=story,
                    question=p.question,
                    option_a=p.options[0],
                    option_b=p.options[1],
                    correct_answer=gold,
                    parent_problem_id=p.problem_id,
                    parent_source_line=p.source_line,
                    variant=variant,
                    entity_mapping=mapping,
                    story_sha256=sha256(story),
                    review="Complete original story reviewed; systematic entity renaming preserves all relations. Development control, not a held-out benchmark.",
                )
            )
    if len(rows) != 2 * len(SWAPS):
        raise ValueError("missing reviewed source rows")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(
        json.dumps(
            {
                "rows": len(rows),
                "independent_parent_stories": len(SWAPS),
                "out": str(args.out),
                "bytes": args.out.stat().st_size,
                "no_story_contract": "Identical question/options across opposite-gold worlds; reuse one no-story prediction for exact pairing.",
            }
        )
    )


if __name__ == "__main__":
    main()
