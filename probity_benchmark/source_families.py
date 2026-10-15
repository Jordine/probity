"""Identify source-story families before splitting rewritten question datasets."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata

from .preflight import load_problems


def normalize_family(name):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("A nonempty original source-story identifier is required")
    return " ".join(unicodedata.normalize("NFKC", name).casefold().split())


def family_id(name):
    return hashlib.sha256(
        ("quality-source-family-v1:" + normalize_family(name)).encode()
    ).hexdigest()


def build_index(dataset):
    problems, audit = load_problems(dataset)
    raw = dataset.read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if audit["errors"] or len(rows) != len(problems) or len(raw) != dataset.stat().st_size:
        raise ValueError("Source rows and problem identities do not align")
    by_parent, families = {}, defaultdict(lambda: {"parents": [], "source_lines": []})
    for problem, row in zip(problems, rows):
        name = normalize_family(row.get("story_name"))
        key = family_id(name)
        by_parent[problem.problem_id] = key
        families[key]["name"] = name
        families[key]["parents"].append(problem.problem_id)
        families[key]["source_lines"].append(problem.source_line)
    return {
        "dataset_sha256": audit["dataset_sha256"],
        "normalization": "NFKC,casefold,collapsewhitespace; family id sha256 of quality-source-family-v1:<normalized story_name>",
        "by_parent": by_parent,
        "families": dict(families),
    }


def validate_disjoint_splits(parent_splits, index):
    """Fail even when rewritten texts and question hashes are all distinct."""
    owners = defaultdict(set)
    for split, parents in parent_splits.items():
        for parent in parents:
            owners[index["by_parent"][parent]].add(split)
    overlap = {key: sorted(splits) for key, splits in owners.items() if len(splits) > 1}
    if overlap:
        raise ValueError("Source families cross splits: " + json.dumps(overlap, sort_keys=True))
    return {split: len({index["by_parent"][p] for p in parents}) for split, parents in parent_splits.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("New family-index output required")
    index = build_index(args.dataset)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps({"parents": len(index["by_parent"]), "families": len(index["families"])}))


if __name__ == "__main__":
    main()
