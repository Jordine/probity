"""Preserve complete fresh input corpora and original extraction diagnostics."""

import hashlib
import json
from pathlib import Path
import tarfile

BASE = Path("runs/iteration3")
FINAL = BASE / "final"
OUT = FINAL / "local_fresh_feature_provenance_v1.tar.gz"
RECEIPT = FINAL / "local_fresh_feature_provenance_receipt_v1.json"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    if OUT.exists() or RECEIPT.exists():
        raise ValueError("immutable provenance archive and receipt required")
    audit_path = FINAL / "local_feature_completion_audit_v1.json"
    audit = json.loads(audit_path.read_text())
    if audit["status"] != "all_complete_sequence_invariants_passed":
        raise ValueError("complete fresh feature audit required")
    proof_by_model = {model["participant"]: model for model in audit["models"]}
    files = [
        audit_path,
        BASE / "coordinator/feature_completion_policy_v1.json",
        BASE / "coordinator/fresh_features_ready_v1.json",
        FINAL / "global_freeze_v2.json",
    ]
    for short, participant in [
        ("qwen35", "qwen/qwen3.5-27b"),
        ("qwen38", "qwen/qwen3.8-27b"),
    ]:
        proof = proof_by_model[participant]
        directory = FINAL / f"features_{short}"
        corpus = FINAL / f"corpus_{short}.jsonl"
        if (
            digest(corpus) != proof["corpus_sha256"]
            or digest(directory / "manifest.json") != proof["manifest_sha256"]
            or digest(directory / "alignment.json")
            != proof["diagnostics"]["short"]["sha256"]
            or digest(directory / "long_alignment.json")
            != proof["diagnostics"]["long"]["sha256"]
        ):
            raise ValueError("audited input corpus or original diagnostics changed")
        files.append(corpus)
        files.extend(
            directory / name
            for name in [
                "alignment.json",
                "long_alignment.json",
                "manifest.json",
                "summary.json",
                "receipts.jsonl",
            ]
        )
    files.sort(key=str)
    records = {
        str(path): {"bytes": path.stat().st_size, "sha256": digest(path)}
        for path in files
    }
    with tarfile.open(OUT, "x:gz", compresslevel=6) as archive:
        for path in files:
            archive.add(path, arcname=str(path), recursive=False)
    verified = {}
    with tarfile.open(OUT, "r:gz") as archive:
        for member in archive:
            if (
                not member.isfile()
                or member.name not in records
                or member.name in verified
            ):
                raise ValueError("unexpected archive member")
            value, size = hashlib.sha256(), 0
            handle = archive.extractfile(member)
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                value.update(block)
                size += len(block)
            verified[member.name] = {"bytes": size, "sha256": value.hexdigest()}
    if verified != records:
        raise ValueError("lossless provenance roundtrip differs from original files")
    result = {
        "schema": "local3_lossless_fresh_provenance_archive_v1",
        "status": "all_original_file_bytes_and_hashes_verified",
        "script_sha256": digest(__file__),
        "feature_completion_audit_sha256": digest(audit_path),
        "archive": str(OUT),
        "archive_bytes": OUT.stat().st_size,
        "archive_sha256": digest(OUT),
        "member_count": len(records),
        "uncompressed_member_bytes": sum(
            record["bytes"] for record in records.values()
        ),
        "members": records,
        "original_files_deleted_or_modified": False,
        "scope": "Exact canonical corpus JSONL, both original numerical diagnostic files per model, original extraction manifests/summaries/receipts, completion policy, ready receipt, global freeze and detailed feature audit. No feature tensor copies or lossy text changes. Retain GPU originals until independent transfer proof is complete.",
    }
    RECEIPT.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "members"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
