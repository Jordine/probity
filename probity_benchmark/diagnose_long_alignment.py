"""Inspect exact DYL prefixes without relaxing the strict alignment criterion."""

import argparse
import json
from pathlib import Path

from .qwen_extract import Runtime, serialize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError("new output file required")
    from probe_inference import load_probe_from_hub

    runtime = Runtime()
    probe = load_probe_from_hub("qwen3.5-27b/linear")
    rows = [json.loads(line) for line in args.corpus.read_text().splitlines()]
    longest = max(rows, key=lambda row: len(json.dumps(row["messages"])))
    encoded = serialize(runtime.tokenizer, longest["messages"], longest["response"])
    # Both forced replies tokenize as one word and a period. Cut at their
    # semantic boundary, retaining the entire preceding conversation.
    prefix = encoded["ids"][: encoded["dyl_position"] - 1]
    if (
        runtime.tokenizer.decode(
            encoded["ids"][encoded["dyl_position"] - 1 : encoded["dyl_position"] + 1]
        )
        != "Yes."
    ):
        raise ValueError("unexpected forced-answer tokenization")
    results = {"row_id": longest["id"], "arms": {}}
    for answer in ("No.", "Yes."):
        result = runtime.alignment_check(
            prefix_ids=prefix, response_text=answer, probe=probe
        )
        results["arms"][answer] = result
        print(json.dumps({"answer": answer, "result": result}), flush=True)
        args.out.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
