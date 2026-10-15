"""Project verified span subsets from a fixed provisional pooled feature cache.

No model runs or token/input edits occur here. Complete source, response and
segment features are preserved; only exact pre-existing claim indices may be
selected. A new immutable cache records both source and projected hashes.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import torch

from .local3_natural_twins import check_balanced_pairs
from .preflight import sha256


def project(cache, row):
    old = cache["row"]
    for key in [
        "id",
        "pair_id",
        "group",
        "split",
        "participant_model",
        "context_world",
        "messages",
        "response",
    ]:
        if old[key] != row[key]:
            raise ValueError("verified projection changed fixed input field: " + key)
    lookup = {}
    for i, span in enumerate(old["spans"]):
        identity = span["original_span_index"]
        if identity in lookup:
            raise ValueError("duplicate source span identity")
        lookup[identity] = i
    indices = []
    for span in row["spans"]:
        index = lookup[span["original_span_index"]]
        previous = old["spans"][index]
        for key in ["start", "end", "text", "display_text", "label"]:
            if previous[key] != span[key]:
                raise ValueError(
                    "verified projection must only select unchanged claims: " + key
                )
        indices.append(index)
    if len(indices) != len(set(indices)) or not indices:
        raise ValueError("projection must retain a unique nonempty claim subset")
    output = dict(cache)
    output["row"] = row
    for key in ["span_mean", "span_last"]:
        output[key] = cache[key][:, indices].clone()
        if not torch.equal(output[key], cache[key][:, indices]):
            raise ValueError("pooled features changed during projection")
    output["span_lengths"] = [cache["span_lengths"][i] for i in indices]
    return output, indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("immutable new cache required")
    source_summary = json.loads((args.features / "summary.json").read_text())
    source_files = sorted(args.features.glob("*.pt"))
    if (
        source_summary["rows_expected"] != source_summary["rows_completed"]
        or len(source_files) != source_summary["rows_expected"]
    ):
        raise ValueError("source feature extraction incomplete")
    rows = [json.loads(line) for line in args.corpus.read_text().splitlines()]
    paired = check_balanced_pairs(rows)
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("empty or duplicate verified corpus")
    args.out.mkdir(parents=True)
    manifest = json.loads((args.features / "manifest.json").read_text())
    manifest.update(
        {
            "source_manifest_sha256": sha256(
                (args.features / "manifest.json").read_text()
            ),
            "source_feature_directory": str(args.features),
            "source_corpus_sha256": manifest["corpus_sha256"],
            "corpus_sha256": sha256(args.corpus.read_text()),
            "corpus_bytes": args.corpus.stat().st_size,
            "expected_rows": len(rows),
            "projection_sha256": sha256(Path(__file__).read_text()),
            "projection_contract": "Complete response/messages unchanged. Retain only exact existing spans with unchanged text, raw coordinates and labels. Select corresponding span mean/last tensors; all sentence/clause and response/DYL features unchanged. No additional model replay.",
        }
    )
    receipts = []
    for row in rows:
        source = args.features / (row["id"] + ".pt")
        cache = torch.load(source, weights_only=True, map_location="cpu")
        output, indices = project(cache, row)
        destination = args.out / source.name
        torch.save(output, destination)
        receipts.append(
            {
                "id": row["id"],
                "original_span_indices": indices,
                "original_label_span_ids": [
                    s["original_span_index"] for s in row["spans"]
                ],
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "bytes": destination.stat().st_size,
                "input_tokens": cache["input_tokens"],
                "response_tokens": cache["response_tokens"],
                "termination": "pooled_span_projection_without_replay",
            }
        )
    for name in ["alignment.json", "long_alignment.json"]:
        source = args.features / name
        if source.exists():
            shutil.copyfile(source, args.out / name)
    summary = {
        "termination_counts": dict(Counter(r["termination"] for r in receipts)),
        "rows_expected": len(rows),
        "rows_completed": len(receipts),
        "context_tokens_max": max(r["input_tokens"] for r in receipts),
        "bytes": sum(r["bytes"] for r in receipts),
        **paired,
        "scope": manifest["projection_contract"],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out / "receipts.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in receipts)
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
