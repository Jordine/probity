"""Apply explicit, hash-guarded manual span exclusions without losing annotations."""

import argparse
import copy
import json
from pathlib import Path

from .preflight import sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--review", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError("new output required")
    review = json.loads(args.review.read_text())
    rows = list(map(json.loads, args.corpus.read_text().splitlines()))
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate row")
    applied = []
    for item in review["exclusions"]:
        row = by_id[item["row_id"]]
        annotation = next(
            a for a in row["annotations"] if a["index"] == item["annotation_index"]
        )
        if annotation["claim"]["text"] != item["expected_text"]:
            raise ValueError("review text mismatch")
        if "pre_manual_review_spans" not in row:
            row["pre_manual_review_spans"] = copy.deepcopy(row["spans"])
        removed = [
            s for s in row["spans"] if s["annotation_index"] == item["annotation_index"]
        ]
        if len(removed) != 1:
            raise ValueError("review must remove exactly one accepted span")
        row["spans"] = [
            s for s in row["spans"] if s["annotation_index"] != item["annotation_index"]
        ]
        annotation["manual_exclusion"] = item["reason"]
        applied.append(item)
    args.out.mkdir(parents=True)
    (args.out / "corpus.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    (args.out / "review_receipt.json").write_text(
        json.dumps(
            {
                "input_sha256": sha256(args.corpus.read_text()),
                "review_sha256": sha256(args.review.read_text()),
                "output_sha256": sha256((args.out / "corpus.jsonl").read_text()),
                "rows": len(rows),
                "applied": applied,
                "limit": "Targeted manual audit, not an exhaustive human relabelling. Remaining labels are independent-model silver labels. Test metrics were not consulted.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
