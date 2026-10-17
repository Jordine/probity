"""Restore reference-only views from exact retained development feature caches.

The archive metadata and pooled tensors remain immutable. A separate view uses
the original manifest solely to reproduce the already-frozen normalizer hash.
Only the frozen development reference rows have complete sentence/clause pools;
this view is not a replacement for the original general-purpose feature cache.
"""

import hashlib
import json
from pathlib import Path


BASE = Path("runs/iteration3/local_probes")


def digest_file(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    all_receipts = []
    frozen_references = json.loads(
        (BASE / "references_local3_v1/sentence_references.json").read_text()
    )["references"]
    for short, participant in [
        ("qwen35", "qwen/qwen3.5-27b"),
        ("qwen38", "qwen/qwen3.8-27b"),
    ]:
        retained = BASE / f"retained_verified_features_{short}_v2"
        destination = BASE / f"verified_features_{short}_v2"
        if destination.exists():
            raise ValueError(f"immutable new reference view required: {destination}")
        retention_manifest = json.loads((retained / "manifest.json").read_text())
        source_bytes = (retained / "source_manifest.json").read_bytes()
        source_manifest = json.loads(source_bytes)
        if (
            hashlib.sha256(source_bytes).hexdigest()
            != retention_manifest["source_manifest_sha256"]
            or canonical_digest(source_manifest)
            != frozen_references[participant]["manifest_sha256"]
        ):
            raise ValueError("preserved source manifest differs from frozen reference")
        summary = json.loads((retained / "summary.json").read_text())
        files = sorted(retained.glob("*.pt"))
        receipts = [
            json.loads(line)
            for line in (retained / "receipts.jsonl").read_text().splitlines()
        ]
        expected = {record["id"]: record for record in receipts}
        if (
            len(files) != summary["rows_expected"]
            or summary["rows_expected"] != summary["rows_completed"]
            or {path.stem for path in files} != set(expected)
        ):
            raise ValueError("complete retained source coverage required")
        verified = []
        for path in files:
            record = expected[path.stem]
            if (
                digest_file(path) != record["archive_sha256"]
                or path.stat().st_size != record["bytes"]
            ):
                raise ValueError(f"retained tensor file changed: {path}")
            verified.append(
                {"id": path.stem, "sha256": record["archive_sha256"]}
            )
        destination.mkdir()
        for path in files:
            (destination / path.name).symlink_to(path.resolve())
        (destination / "manifest.json").write_bytes(source_bytes)
        (destination / "summary.json").write_bytes(
            (retained / "summary.json").read_bytes()
        )
        receipt = {
            "schema": "local3_frozen_reference_view_restore_v1",
            "participant": participant,
            "source_retained_directory": str(retained),
            "reference_view": str(destination),
            "scope": "Only reproducing frozen reference normalization. Complete sentence/clause tensors exist for the frozen reference rows; other rows retain exact span/response/DYL pools and original metadata, but may omit segment pools.",
            "retained_manifest_sha256": digest_file(retained / "manifest.json"),
            "original_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "canonical_original_manifest_sha256": canonical_digest(source_manifest),
            "frozen_reference_row_ids": frozen_references[participant]["rows"],
            "verified_tensor_files": verified,
            "retained_archive_unchanged": True,
        }
        (destination / "restoration_receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        all_receipts.append(
            {
                "participant": participant,
                "rows": len(files),
                "frozen_reference_rows": len(receipt["frozen_reference_row_ids"]),
                "view": str(destination),
                "receipt_sha256": digest_file(destination / "restoration_receipt.json"),
            }
        )
    print(json.dumps({"status": "restored_reference_views", "views": all_receipts}, indent=2))


if __name__ == "__main__":
    main()
