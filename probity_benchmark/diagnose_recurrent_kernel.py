"""Compare long-sequence parity before/after a model-local kernel substitution."""

import argparse
import json
from pathlib import Path

from .qwen_extract import Runtime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError("new receipt required")
    rows = [json.loads(x) for x in args.corpus.read_text().splitlines()]
    longest = max(rows, key=lambda r: len(json.dumps(r["messages"])))
    runtime = Runtime()
    result = {"standard": runtime.alignment_check(longest["messages"])}
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    patched = 0
    for layer in runtime.model.model.layers:
        if hasattr(layer, "linear_attn"):
            attention = layer.linear_attn
            attention.chunk_gated_delta_rule = attention.recurrent_gated_delta_rule
            patched += 1
    result["recurrent_for_prefill"] = runtime.alignment_check(longest["messages"])
    result["patched_linear_attention_layers"] = patched
    result["scope"] = (
        "Numerical diagnostic only. This modified execution is not used in the main feature cache."
    )
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
