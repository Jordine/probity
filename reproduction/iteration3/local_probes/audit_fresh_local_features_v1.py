"""Read-only CPU audit of fresh GPU caches before frozen local export.

No model weights or truth labels are loaded. Actual complete-sequence invariants
are gated separately from cached/padded numerical comparison diagnostics.
"""

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from probity_benchmark.extract_iteration import intervals
from probity_benchmark.local3_annotate_units import load_inputs
from probity_benchmark.qwen_extract import LAYERS, overlapping_tokens, serialize

BASE = Path("runs/iteration3")
POLICY = BASE / "coordinator/feature_completion_policy_v1.json"
POLICY_SHA = "7a97045af9d22b3dd1b675f86407842706d628f50f780583552e47ff9c5ee1de"
OUT = BASE / "final/local_feature_completion_audit_v1.json"


def digest_file(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def text_digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def diagnostic_record(path, require_pass):
    record = json.loads(path.read_text())
    if record["criterion"] != {"cosine_min": 0.999, "relative_rms_max": 0.03}:
        raise ValueError("diagnostic criteria changed")
    checks = record["checks"]
    if len(checks) != 12 or {(row["layer"], row["mode"]) for row in checks} != {
        (layer, mode) for layer in LAYERS for mode in ["incremental", "left_padding"]
    }:
        raise ValueError("incomplete mode/layer diagnostic")
    if any(
        not math.isfinite(row[field])
        for row in checks
        for field in ["cosine_min", "relative_rms_max"]
    ):
        raise ValueError("nonfinite diagnostic")
    failed = [
        row
        for row in checks
        if row["cosine_min"] < 0.999 or row["relative_rms_max"] > 0.03
    ]
    if record["passed"] != (not failed) or (require_pass and failed):
        raise ValueError("short alignment gate failed or diagnostic flag inconsistent")
    return {
        "path": str(path),
        "sha256": digest_file(path),
        "original_diagnostic": record,
        "failed_checks": failed,
        "scope": "Numerical comparison with cached incremental decoding and three-token masked left-padding. Generic ten-token synthetic continuation; not an all-token real-response audit. Long-mode failure is reported, not silently used as a new exclusion criterion.",
    }


def main():
    if OUT.exists():
        raise ValueError("immutable fresh feature audit required")
    if digest_file(POLICY) != POLICY_SHA:
        raise ValueError("explicit methodological policy changed")
    policy = json.loads(POLICY.read_text())
    freeze_path = Path(policy["global_freeze"])
    if digest_file(freeze_path) != policy["global_freeze_sha256"]:
        raise ValueError("policy global freeze changed")
    freeze = json.loads(freeze_path.read_text())
    for name, expected in freeze["source_hashes"].items():
        if digest_file(Path(name)) != expected:
            raise ValueError(f"frozen experimental source changed: {name}")
    handoff_path = BASE / "local_probes/final_handoff_v3/manifest.json"
    if digest_file(handoff_path) != freeze["artifacts_sha256"][str(handoff_path)]:
        raise ValueError("frozen handoff changed")
    handoff = json.loads(handoff_path.read_text())
    import probity_benchmark.local3_final_runner as runner

    runner.HANDOFF = handoff_path.parent
    episodes, _ = runner.verify(handoff)
    source_rows, _ = load_inputs(
        Path(handoff["dataset"]), [Path(handoff["episodes"])], ["sentence", "clause"]
    )
    expected_rows = {row["id"]: row for row in source_rows}
    if len(episodes) != 48 or len(expected_rows) != 288:
        raise ValueError("complete source row coverage required")
    specs = json.loads((handoff_path.parent / "all_unit_score_spec.json").read_text())[
        "models"
    ]
    torch.set_num_threads(4)
    result = {
        "schema": "local3_fresh_feature_completion_audit_v1",
        "policy": str(POLICY),
        "policy_sha256": POLICY_SHA,
        "global_freeze_sha256": digest_file(freeze_path),
        "dataset_sha256": digest_file(Path(handoff["dataset"])),
        "audit_source_sha256": digest_file(Path(__file__)),
        "model_inference_performed": False,
        "truth_annotations_loaded": False,
        "models": [],
        "status": "all_complete_sequence_invariants_passed",
    }
    for spec in specs:
        participant = spec["participant"]
        if len(spec["evaluation_features"]) != 1:
            raise ValueError("one complete fresh feature cache per model required")
        directory = Path(spec["evaluation_features"][0])
        short = "qwen35" if participant == "qwen/qwen3.5-27b" else "qwen38"
        corpus = BASE / f"final/corpus_{short}.jsonl"
        corpus_rows = [json.loads(line) for line in corpus.read_text().splitlines()]
        corpus_map = {row["id"]: row for row in corpus_rows}
        model_expected = {
            key: row
            for key, row in expected_rows.items()
            if row["participant_model"] == participant
        }
        if (
            len(corpus_rows) != 144
            or len(corpus_map) != 144
            or set(corpus_map) != set(model_expected)
        ):
            raise ValueError("complete canonical corpus identity mismatch")
        manifest = json.loads((directory / "manifest.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        receipts = [
            json.loads(line)
            for line in (directory / "receipts.jsonl").read_text().splitlines()
        ]
        receipt_map = {row["id"]: row for row in receipts}
        cache_paths = sorted(directory.glob("*.pt"))
        if (
            len(receipts) != 144
            or len(receipt_map) != 144
            or set(receipt_map) != set(corpus_map)
            or len(cache_paths) != 144
            or {path.stem for path in cache_paths} != set(corpus_map)
        ):
            raise ValueError(
                "cache/receipt rows do not cover the exact complete corpus"
            )
        if (
            summary["rows_expected"] != 144
            or summary["rows_completed"] != 144
            or summary["termination_counts"]
            != {"teacher_forced_complete_sequence": 144}
        ):
            raise ValueError("extraction is not complete")
        if (
            manifest["expected_rows"] != 144
            or manifest["corpus_sha256"] != digest_file(corpus)
            or manifest["corpus_bytes"] != corpus.stat().st_size
        ):
            raise ValueError("feature/corpus byte identity mismatch")
        if (
            manifest["layers"] != list(LAYERS)
            or manifest["dtype"] != "bfloat16"
            or manifest["allow_bf16_reduced_precision_reduction"] is not False
            or manifest["cache_complete_response_tokens"] is not False
        ):
            raise ValueError("frozen extraction precision/layer/cache contract changed")
        for field, module in [
            ("extractor_sha256", "extract_iteration.py"),
            ("runtime_sha256", "qwen_extract.py"),
        ]:
            if (
                manifest[field]
                != freeze["source_hashes"]["probity_benchmark/" + module]
            ):
                raise ValueError("feature runtime source provenance changed")
        reference_manifest = json.loads(
            (Path(spec["reference_features"]) / "manifest.json").read_text()
        )
        for field in [
            "model",
            "model_revision",
            "layers",
            "dtype",
            "versions",
            "published_weights_revision",
            "chat_template_sha256",
        ]:
            if manifest[field] != reference_manifest[field]:
                raise ValueError(
                    f"fresh runtime differs from frozen reference contract: {field}"
                )
        for library, version in manifest["versions"].items():
            if importlib.metadata.version(library) != version:
                raise ValueError(
                    f"audit runtime differs from extraction runtime: {library}"
                )
        head_hashes = {}
        for field in ["head", "old_head", "response_head"]:
            path = Path(spec[field])
            head_hashes[field] = digest_file(path)
            if head_hashes[field] != freeze["artifacts_sha256"][str(path)]:
                raise ValueError("frozen model head changed")
            contract = torch.load(path, weights_only=True, map_location="cpu")[
                "__contract__"
            ]
            if any(
                contract[field] != manifest[field]
                for field in ["model", "model_revision", "layers"]
            ):
                raise ValueError("head/extraction contract mismatch")
        tokenizer = AutoTokenizer.from_pretrained(
            manifest["model"],
            revision=manifest["model_revision"],
            local_files_only=True,
        )
        config = AutoConfig.from_pretrained(
            manifest["model"],
            revision=manifest["model_revision"],
            local_files_only=True,
        ).get_text_config()
        if (
            config.hidden_size != 5120
            or config.num_hidden_layers != 64
            or text_digest(tokenizer.chat_template) != manifest["chat_template_sha256"]
        ):
            raise ValueError("pinned tokenizer/model shape contract mismatch")
        diagnostics = {
            "short": diagnostic_record(directory / "alignment.json", True),
            "long": diagnostic_record(directory / "long_alignment.json", False),
        }
        row_receipts = []
        for path in cache_paths:
            receipt = receipt_map[path.stem]
            if (
                path.stat().st_size != receipt["bytes"]
                or digest_file(path) != receipt["sha256"]
            ):
                raise ValueError("immutable feature file changed")
            cache = torch.load(path, weights_only=True, map_location="cpu")
            row = corpus_map[path.stem]
            original = model_expected[path.stem]
            if cache["row"] != row or cache["id"] != row["id"]:
                raise ValueError("feature row differs from canonical complete input")
            for field in [
                "group",
                "split",
                "participant_model",
                "source_episode",
                "turn_index",
                "parent_source_line",
                "world",
                "messages",
                "response",
                "response_reasoning",
                "source_request_sha256",
            ]:
                if row[field] != original[field]:
                    raise ValueError(f"feature/source-generation bytes differ: {field}")
            if (
                row["spans"]
                or row["label"] is not None
                or cache["span_mean"] is not None
                or cache["span_last"] is not None
                or cache["token_features"] is not None
            ):
                raise ValueError(
                    "unexpected label/token-feature input in frozen fresh extraction"
                )
            arms = {
                prefill: serialize(
                    tokenizer,
                    row["messages"],
                    row["response"],
                    prefill=prefill,
                    reasoning=row.get("response_reasoning"),
                )
                for prefill in ["Yes.", "No."]
            }
            yes, no = arms["Yes."], arms["No."]
            if (
                max(len(arm["ids"]) for arm in arms.values())
                > config.max_position_embeddings
            ):
                raise ValueError("complete serialized context exceeds model limit")
            if (
                yes["original_tokens"] != no["original_tokens"]
                or yes["ids"][: yes["original_tokens"]]
                != no["ids"][: no["original_tokens"]]
                or yes["response_positions"] != no["response_positions"]
            ):
                raise ValueError(
                    "forced DYL arms changed the original complete action context"
                )
            for label, arm in [("yes", yes), ("no", no)]:
                if cache[f"serialized_{label}_sha256"] != text_digest(arm["text"]):
                    raise ValueError("cached serialized context changed")
            offsets = [
                (
                    yes["offsets"][i][0] - yes["response_start"],
                    yes["offsets"][i][1] - yes["response_start"],
                )
                for i in yes["response_positions"]
            ]
            if (
                cache["response_offsets"] != offsets
                or cache["input_tokens"] != len(yes["ids"])
                or cache["response_tokens"] != len(offsets)
                or receipt["input_tokens"] != len(yes["ids"])
                or receipt["response_tokens"] != len(offsets)
            ):
                raise ValueError("complete token/offset/count mismatch")
            if (
                cache["returned_reasoning_range"] != yes["returned_reasoning_range"]
                or cache["returned_reasoning_sha256"]
                != yes["returned_reasoning_sha256"]
            ):
                raise ValueError("returned reasoning serialization changed")
            segments, _ = intervals(row)
            if cache["segments"] != segments or cache["segment_lengths"] != [
                len(overlapping_tokens(offsets, unit["start"], unit["end"]))
                for unit in segments
            ]:
                raise ValueError("fixed full-text segment/token mapping mismatch")
            tensor_shapes = {
                "response_mean": ((6, 5120), torch.float32),
                "argument_mean": ((6, 5120), torch.float32),
                "dyl_yes": ((6, 5120), torch.bfloat16),
                "dyl_no": ((6, 5120), torch.bfloat16),
                "segment_mean": ((6, len(segments), 5120), torch.float32),
                "segment_last": ((6, len(segments), 5120), torch.float32),
            }
            for field, (shape, dtype) in tensor_shapes.items():
                value = cache[field]
                if (
                    not isinstance(value, torch.Tensor)
                    or tuple(value.shape) != shape
                    or value.dtype != dtype
                    or not bool(torch.isfinite(value).all())
                ):
                    raise ValueError(
                        f"nonfinite or mismatched complete pooled features: {field}"
                    )
            causality = cache["dyl_causality_relative_rms"]
            if len(causality) != 6 or any(
                not math.isfinite(value) or value > 0.03 for value in causality
            ):
                raise ValueError("frozen DYL causality invariant failed")
            for field in ["fibs_yes", "fibs_no", "fibs_contrast"]:
                value = cache[field]
                if manifest["published_probe_available"] != (value is not None) or (
                    value is not None and not math.isfinite(value)
                ):
                    raise ValueError("published probe coverage/finite-score mismatch")
            row_receipts.append(
                {
                    "id": row["id"],
                    "cache_sha256": receipt["sha256"],
                    "input_tokens": len(yes["ids"]),
                    "response_tokens": len(offsets),
                    "sentence_segments": sum(
                        unit["kind"] == "sentence" for unit in segments
                    ),
                    "clause_segments": sum(
                        unit["kind"] == "phrase" for unit in segments
                    ),
                    "dyl_causality_relative_rms": causality,
                    "serialized_yes_sha256": cache["serialized_yes_sha256"],
                    "serialized_no_sha256": cache["serialized_no_sha256"],
                }
            )
        if summary["context_tokens_max"] != max(
            row["input_tokens"] for row in row_receipts
        ) or summary["bytes"] != sum(row["bytes"] for row in receipts):
            raise ValueError("summary does not match complete cache")
        result["models"].append(
            {
                "participant": participant,
                "manifest_sha256": digest_file(directory / "manifest.json"),
                "corpus_sha256": digest_file(corpus),
                "head_sha256": head_hashes,
                "context_limit": config.max_position_embeddings,
                "input_tokens_range": [
                    min(row["input_tokens"] for row in row_receipts),
                    max(row["input_tokens"] for row in row_receipts),
                ],
                "response_tokens_range": [
                    min(row["response_tokens"] for row in row_receipts),
                    max(row["response_tokens"] for row in row_receipts),
                ],
                "maximum_dyl_causality_relative_rms": max(
                    value
                    for row in row_receipts
                    for value in row["dyl_causality_relative_rms"]
                ),
                "diagnostics": diagnostics,
                "rows": row_receipts,
            }
        )
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "artifact": str(OUT),
                "status": result["status"],
                "models": [
                    {
                        "participant": model["participant"],
                        "rows": len(model["rows"]),
                        "input_tokens_range": model["input_tokens_range"],
                        "response_tokens_range": model["response_tokens_range"],
                        "maximum_dyl_causality_relative_rms": model[
                            "maximum_dyl_causality_relative_rms"
                        ],
                        "long_mode_failed_checks": model["diagnostics"]["long"][
                            "failed_checks"
                        ],
                    }
                    for model in result["models"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
