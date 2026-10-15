"""Require exact replay of every frozen development reference and text control."""

import hashlib
import json
from pathlib import Path

BASE = Path("runs/iteration3/local_probes")
OUT = BASE / "restored_reference_identity_audit_v1.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise ValueError("immutable reference replay receipt required")
    records = []
    failures = []
    for prefix in ["local3", "local2"]:
        for granularity in ["sentence", "clause"]:
            frozen_path = BASE / f"references_{prefix}_v1/{granularity}_references.json"
            replay_path = (
                BASE
                / f"restored_{prefix}_reference_replay_v1/{granularity}_references.json"
            )
            frozen, replay = [
                json.loads(path.read_text()) for path in [frozen_path, replay_path]
            ]
            equality = {
                key: frozen[key] == replay[key]
                for key in ["head_sha256", "reference_sha256", "references"]
            }
            model_deltas = {}
            for model, original in frozen["references"].items():
                actual = replay["references"].get(model, {})
                old_scores, new_scores = original["scores"], actual.get("scores", [])
                paired = len(old_scores) == len(new_scores)
                model_deltas[model] = {
                    "frozen_scores": len(old_scores),
                    "replayed_scores": len(new_scores),
                    "maximum_absolute_score_difference": (
                        max(abs(a - b) for a, b in zip(old_scores, new_scores))
                        if paired
                        else None
                    ),
                    "nonidentical_scores": (
                        sum(a != b for a, b in zip(old_scores, new_scores))
                        if paired
                        else None
                    ),
                }
            record = {
                "frozen_path": str(frozen_path),
                "frozen_file_sha256": digest(frozen_path),
                "replayed_path": str(replay_path),
                "replayed_file_sha256": digest(replay_path),
                "exact_identity_fields": equality,
                "models": model_deltas,
            }
            records.append(record)
            if not all(equality.values()):
                failures.append(record)
    text_control_path = BASE / "text_control_provenance_v1.json"
    text_controls = json.loads(text_control_path.read_text())["artifacts"]
    for record in text_controls:
        if digest(record["path"]) != record["sha256"]:
            failures.append({"text_control_changed": record["path"]})
    result = {
        "schema": "local3_restored_reference_identity_audit_v1",
        "status": (
            "exact_frozen_reference_replay_passed"
            if not failures
            else "mismatch_stop_before_export"
        ),
        "verification_source_sha256": digest(__file__),
        "text_control_provenance_sha256": digest(text_control_path),
        "reference_bundles": records,
        "text_controls": text_controls,
        "failures": failures,
        "scope": "Exact frozen arrays remain authoritative. No tolerance, recentering, normalization replacement or new reference selection is allowed. A numerical mismatch stops export for root review and preserves both original and replayed arrays.",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if failures:
        raise ValueError(
            "restored references differ; preserved originals and stopped before export"
        )


if __name__ == "__main__":
    main()
