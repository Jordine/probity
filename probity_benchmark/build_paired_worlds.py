"""Bijective entity renamings with source-group partitions fixed before new runs."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from .counterfactuals import DATASET_SHA, swap_entities
from .preflight import load_problems, sha256

# These are proper-name swaps, not numerical edits or semantic paraphrases.
# Full names and aliases move together. The full-story validation is separate.
SPECS = {
    5: ("train", [("Nora Price", "Lucy Stone"), ("Nora", "Lucy"), ("Price", "Stone")]),
    22: (
        "train",
        [("Paul Graham", "Yuki Newman"), ("Paul", "Yuki"), ("Graham", "Newman")],
    ),
    82: ("train", [("Crystalbrook", "Greenfield")]),
    98: ("train", [("TechLease", "AutomationPro")]),
    117: (
        "train",
        [
            ("Charles Fernandez", "Bianca Holmes"),
            ("Charles", "Bianca"),
            ("Fernandez", "Holmes"),
        ],
    ),
    134: ("train", [("Northfield", "Westfield")]),
    136: (
        "train",
        [("Wei Cooper", "Walter Ford"), ("Wei", "Walter"), ("Cooper", "Ford")],
    ),
    143: (
        "train",
        [
            ("Samuel Kennedy", "Elizabeth Henry"),
            ("Samuel", "Elizabeth"),
            ("Kennedy", "Henry"),
        ],
    ),
    90: ("dev", [("Larchmont", "Oakdale")]),
    112: ("dev", [("Isaac Hunt", "Peter Ford"), ("Isaac", "Peter"), ("Hunt", "Ford")]),
    128: ("dev", [("Aldoria", "Jarvonia"), ("Aldorian", "Jarvonian")]),
    137: ("dev", [("CogniCorp", "NeuralDynamics")]),
    125: ("test", [("Stephen", "Patel")]),
    132: ("test", [("Oakdale", "Brookhaven")]),
    139: (
        "test",
        [
            ("Elizabeth Obrien", "Megan Wilson"),
            ("Elizabeth", "Megan"),
            ("Obrien", "Wilson"),
        ],
    ),
    148: ("test", [("Forestdale", "Dalebrook")]),
}


def unmapped_name_forms(story, mapping):
    """Flag extended proper names, such as country adjectives, before swapping."""
    words = set(re.findall(r"\b\w+\b", story))
    return sorted(
        {
            word
            for name in mapping
            for word in words
            if word.startswith(name) and word != name and word not in mapping
        }
    )


def build(dataset, specs=None):
    specs = SPECS if specs is None else specs
    problems, audit = load_problems(dataset)
    if audit["dataset_sha256"] != DATASET_SHA or audit["errors"]:
        raise ValueError("source corpus differs from the selected dataset")
    rows = []
    for p in problems:
        if p.source_line not in specs:
            continue
        split, pairs = specs[p.source_line]
        mapping = {
            a: b for left, right in pairs for a, b in [(left, right), (right, left)]
        }
        mapping = dict(sorted(mapping.items(), key=lambda item: -len(item[0])))
        missing = unmapped_name_forms(p.story, mapping)
        if missing:
            raise ValueError(
                f"review missing name aliases at {p.source_line}: {missing}"
            )
        changed = swap_entities(p.story, mapping)
        if changed == p.story or swap_entities(changed, mapping) != p.story:
            raise ValueError(f"non-bijective story swap at {p.source_line}")
        if swap_entities(p.question, mapping) != p.question:
            raise ValueError(f"question changes under swap at {p.source_line}")
        if (
            swap_entities(p.options[0], mapping) != p.options[1]
            or swap_entities(p.options[1], mapping) != p.options[0]
        ):
            raise ValueError(f"options are not exchanged exactly at {p.source_line}")
        counts = {
            key: len(re.findall(r"\b" + re.escape(key) + r"\b", p.story))
            for key in mapping
        }
        for world, story, gold in [
            ("original", p.story, p.gold),
            ("swapped_world", changed, {"A": "B", "B": "A"}[p.gold]),
        ]:
            rows.append(
                {
                    "synthetic_story": story,
                    "question": p.question,
                    "option_a": p.options[0],
                    "option_b": p.options[1],
                    "correct_answer": gold,
                    "parent_problem_id": p.problem_id,
                    "parent_source_line": p.source_line,
                    "variant": world,
                    "split": split,
                    "entity_mapping": mapping,
                    "original_entity_counts": counts,
                    "story_sha256": sha256(story),
                    "validation_status": "Mechanical reversible renaming; source answer validity and ambiguity require independent full-story checks.",
                }
            )
    if len(rows) != 2 * len(specs):
        raise ValueError("missing selected parent stories")
    return rows, audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset", type=Path, default=Path("data/quality_synthetic/stories.jsonl")
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--specs", type=Path)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError("new dataset required")
    specs = (
        {int(k): v for k, v in json.loads(args.specs.read_text()).items()}
        if args.specs
        else SPECS
    )
    rows, audit = build(args.dataset, specs)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    receipt = {
        "rows": len(rows),
        "parents": len(specs),
        "split_world_counts": dict(Counter(r["split"] for r in rows)),
        "sha256": sha256(args.out.read_text()),
        "source_sha256": audit["dataset_sha256"],
        "source_bytes_read": audit["bytes_read"],
        "builder_sha256": sha256(Path(__file__).read_text()),
        "no_story_contract": "Reuse one prediction per parent/model/answer order for both opposite-gold worlds; exactly chance by construction, not a statistical estimate.",
        "split_policy": "Both worlds and all generations/annotations of a parent remain together. Test parents are locked before model calls; do not tune on their scores.",
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
