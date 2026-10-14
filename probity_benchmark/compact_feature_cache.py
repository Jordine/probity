"""Archive fit-ready features without duplicate sentence or full token tensors."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .preflight import sha256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError("new output required")
    manifest_text = (args.features / "manifest.json").read_text()
    manifest = json.loads(manifest_text)
    summary = json.loads((args.features / "summary.json").read_text())
    paths = sorted(args.features.glob("*.pt"))
    if (
        len(paths) != summary["rows_expected"]
        or len(paths) != summary["rows_completed"]
    ):
        raise ValueError("incomplete source cache")
    args.out.mkdir(parents=True)
    receipts = []
    for path in paths:
        row = torch.load(path, map_location="cpu", weights_only=True)
        removed = {}
        for key in ["token_features", "segment_mean", "segment_last"]:
            value = row.get(key)
            if isinstance(value, torch.Tensor):
                removed[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
            row[key] = None
        dest = args.out / path.name
        torch.save(row, dest)
        check = torch.load(dest, map_location="cpu", weights_only=True)
        for key, value in row.items():
            if isinstance(value, torch.Tensor):
                if not torch.equal(check[key], value):
                    raise ValueError("archived feature changed")
            elif check[key] != value:
                raise ValueError("archived metadata changed")
        receipts.append(
            {
                "id": row["row"]["id"],
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
                "bytes": dest.stat().st_size,
                "omitted_tensors": removed,
            }
        )
    manifest.update(
        source_manifest_sha256=sha256(manifest_text),
        cache_complete_response_tokens=False,
        cache_sentence_features=False,
        archive_scope="Exact retained span and whole-response/DYL features, complete original texts and offsets. Token and duplicate sentence tensors omitted; recompute from pinned model/corpus for token retraining.",
    )
    (args.out / "source_manifest.json").write_text(manifest_text)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out / "summary.json").write_text(
        json.dumps({**summary, "bytes": sum(r["bytes"] for r in receipts)}, indent=2)
        + "\n"
    )
    (args.out / "receipts.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in receipts)
    )
    print(
        json.dumps({"rows": len(receipts), "bytes": sum(r["bytes"] for r in receipts)})
    )


if __name__ == "__main__":
    main()
