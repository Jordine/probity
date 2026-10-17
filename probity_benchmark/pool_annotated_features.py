"""Join blind claim annotations to a frozen token cache without another model pass."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .extract_iteration import pool
from .preflight import sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError("new output directory required")
    base_summary = json.loads((args.features / "summary.json").read_text())
    if base_summary["rows_expected"] != base_summary["rows_completed"]:
        raise ValueError("base extraction incomplete")
    labels = {
        r["id"]: r for r in map(json.loads, args.annotations.read_text().splitlines())
    }
    args.out.mkdir(parents=True)
    manifest = json.loads((args.features / "manifest.json").read_text())
    manifest.update(
        base_manifest_sha256=sha256((args.features / "manifest.json").read_text()),
        annotations_sha256=sha256(args.annotations.read_text()),
        pooling_script_sha256=sha256(Path(__file__).read_text()),
        cache_complete_response_tokens=False,
        cache_sentence_features=False,
        token_cache_scope="Complete tokens used for pooling; original token cache retained on task GPU during training.",
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    receipts = []
    for path in sorted(args.features.glob("*.pt")):
        cache = torch.load(path, map_location="cpu", weights_only=True)
        row = labels[cache["id"]]
        for key in [
            "id",
            "group",
            "split",
            "messages",
            "response",
            "source_request_sha256",
        ]:
            if row[key] != cache["row"][key]:
                raise ValueError("annotation/cache mismatch: " + key)
        for span in row["spans"]:
            if row["response"][span["start"] : span["end"]] != span["text"]:
                raise ValueError("annotation interval differs from complete text")
        means, lasts, lengths = pool(
            cache["token_features"], cache["response_offsets"], row["spans"]
        )
        cache.update(
            row=row,
            span_mean=means,
            span_last=lasts,
            span_lengths=lengths,
            token_features=None,
            segment_mean=None,
            segment_last=None,
        )
        dest = args.out / path.name
        torch.save(cache, dest)
        receipts.append(
            {
                "id": row["id"],
                "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                "bytes": dest.stat().st_size,
            }
        )
    if len(receipts) != base_summary["rows_expected"]:
        raise ValueError("pooled row accounting mismatch")
    (args.out / "receipts.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in receipts)
    )
    summary = {
        **base_summary,
        "bytes": sum(r["bytes"] for r in receipts),
        "labelled_rows": len(receipts),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
