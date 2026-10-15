"""Prospective operational adapter retaining authoritative frozen normalizers.

Every original experimental module remains unchanged. Reference calculations
must exactly reproduce the independently retained same-machine replay; only
those original development reference return values use the frozen artifact.
Fresh evaluation scores always use the original scorer without substitution.
An explicit hash-bound root release is required before export.
"""

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch

from probity_benchmark import local3_bundle, local3_final_runner
from probity_benchmark.evaluate_debate_transfer import argument_mapping
from probity_benchmark.judge3_protocol import partition

BASE = Path("runs/iteration3")
LOCAL = BASE / "local_probes"
FINAL = BASE / "final"
HANDOFF = LOCAL / "final_handoff_v3"
RECEIPT = FINAL / "local_frozen_reference_authority_receipt_v1.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strip_scores(reference):
    return {key: value for key, value in reference.items() if key != "scores"}


class ReferenceBinder:
    def __init__(self, prefix, specs, original, feature_audit):
        self.prefix, self.original = prefix, original
        self.entries, self.used = {}, Counter()
        self.fresh = {
            (model["participant"], row["id"])
            for model in feature_audit["models"]
            for row in model["rows"]
        }
        self.fresh_calls = Counter()
        self.proofs = []
        for granularity in ["sentence", "clause"]:
            frozen_path = (
                LOCAL / f"references_{prefix}_v1/{granularity}_references.json"
            )
            replay_path = (
                LOCAL
                / f"restored_{prefix}_reference_replay_v1/{granularity}_references.json"
            )
            frozen = json.loads(frozen_path.read_text())
            replay = json.loads(replay_path.read_text())
            if {
                key: value
                for key, value in frozen.items()
                if key not in ("references", "reference_sha256")
            } != {
                key: value
                for key, value in replay.items()
                if key not in ("references", "reference_sha256")
            }:
                raise ValueError("non-score reference bundle fields changed")
            for spec in specs:
                model = spec["participant"]
                a, b = frozen["references"][model], replay["references"][model]
                if strip_scores(a) != strip_scores(b) or len(a["scores"]) != len(
                    b["scores"]
                ):
                    raise ValueError("reference population or source manifest changed")
                if digest(spec["head"]) != frozen["head_artifacts"][model]:
                    raise ValueError("frozen reference head changed")
                offset = 0
                for identity in a["rows"]:
                    path = Path(spec["reference_features"]) / (identity + ".pt")
                    cache = torch.load(path, weights_only=True, map_location="cpu")
                    row = cache["row"]
                    if (
                        row["id"] != identity
                        or row["split"] != "dev"
                        or row["context_world"] != 0
                        or row["participant_model"] != model
                    ):
                        raise ValueError("reference row identity changed")
                    argument = argument_mapping(row["response"])[0]
                    count = len(partition(argument, granularity))
                    key = model, identity, granularity
                    if key in self.entries:
                        raise ValueError("duplicate reference row")
                    self.entries[key] = {
                        "frozen": a["scores"][offset : offset + count],
                        "replayed": b["scores"][offset : offset + count],
                        "argument": argument,
                        "spans": [
                            list(span) for span in partition(argument, granularity)
                        ],
                    }
                    offset += count
                if offset != len(a["scores"]):
                    raise ValueError(
                        "complete original reference segment coverage changed"
                    )
            self.proofs.append(
                {
                    "granularity": granularity,
                    "frozen_path": str(frozen_path),
                    "frozen_file_sha256": digest(frozen_path),
                    "replayed_path": str(replay_path),
                    "replayed_file_sha256": digest(replay_path),
                    "authoritative_reference_sha256": frozen["reference_sha256"],
                }
            )

    def score(self, cache, head, layers, granularity):
        argument, spans, scores = self.original(cache, head, layers, granularity)
        row = cache["row"]
        key = row["participant_model"], row["id"], granularity
        if key in self.entries:
            expected = self.entries[key]
            if row["split"] != "dev" or row.get("context_world") != 0:
                raise ValueError("only original development references may be supplied")
            if (
                argument != expected["argument"]
                or spans != expected["spans"]
                or scores != expected["replayed"]
            ):
                raise ValueError(
                    "recomputed reference differs from preserved same-machine replay"
                )
            self.used[key] += 1
            return argument, spans, list(expected["frozen"])
        if (row["participant_model"], row["id"]) not in self.fresh:
            raise ValueError("unexpected non-reference response in frozen export")
        self.fresh_calls[key] += 1
        return argument, spans, scores

    def finish(self):
        expected_fresh = {
            (model, identity, granularity)
            for model, identity in self.fresh
            for granularity in ["sentence", "clause"]
        }
        if self.used != Counter(
            {key: 1 for key in self.entries}
        ) or self.fresh_calls != Counter({key: 1 for key in expected_fresh}):
            raise ValueError("complete reference/fresh call coverage is not exact")
        return {
            "head_family": self.prefix,
            "reference_row_granularity_calls": sum(self.used.values()),
            "fresh_row_granularity_calls": sum(self.fresh_calls.values()),
            "fresh_scores_substituted": 0,
            "same_machine_reference_values_exactly_reproduced": True,
            "normalizers": self.proofs,
        }


def verify_release(path):
    release = json.loads(path.read_text())
    if (
        release.get("schema") != "local3_frozen_reference_authority_release_v1"
        or release.get("status") != "released"
        or release.get("adapter_sha256") != digest(__file__)
        or release.get("frozen_arrays_authoritative") is not True
        or release.get("fresh_score_substitution_permitted") is not False
    ):
        raise ValueError("explicit matching root reference-authority release required")
    required = {
        str(FINAL / "global_freeze_v2.json"),
        str(FINAL / "local_feature_completion_audit_v1.json"),
        str(FINAL / "root_feature_review_v1.json"),
        str(LOCAL / "restored_reference_identity_audit_v1.json"),
        str(LOCAL / "reference_roundoff_scope_diagnosis_v1.json"),
    }
    required.update(
        str(LOCAL / f"{directory}/{granularity}_references.json")
        for directory in [
            "references_local3_v1",
            "references_local2_v1",
            "restored_local3_reference_replay_v1",
            "restored_local2_reference_replay_v1",
        ]
        for granularity in ["sentence", "clause"]
    )
    inputs = release.get("input_sha256", {})
    if not required.issubset(inputs):
        raise ValueError(
            "root release must bind every frozen and replayed reference artifact"
        )
    for name, expected in inputs.items():
        if digest(name) != expected:
            raise ValueError(f"released reference-authority input changed: {name}")
    freeze = json.loads((FINAL / "global_freeze_v2.json").read_text())
    for name, expected in freeze["source_hashes"].items():
        if digest(name) != expected:
            raise ValueError(f"frozen experiment source changed: {name}")
    audit = json.loads((FINAL / "local_feature_completion_audit_v1.json").read_text())
    if audit["status"] != "all_complete_sequence_invariants_passed":
        raise ValueError("complete feature audit required")
    return release, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    args = parser.parse_args()
    release, feature_audit = verify_release(args.release)
    if RECEIPT.exists():
        raise ValueError("immutable operational export receipt required")
    handoff = json.loads((HANDOFF / "manifest.json").read_text())
    original_call = local3_final_runner.call
    proof = []

    def call(module, *arguments):
        if module != "probity_benchmark.local3_bundle":
            return original_call(module, *arguments)
        values = [str(argument) for argument in arguments]
        spec_path = Path(values[values.index("--spec") + 1])
        matches = [
            prefix
            for prefix in ["local3", "local2"]
            if spec_path == HANDOFF / f"{prefix}_bundle_spec.json"
        ]
        if len(matches) != 1:
            raise ValueError("unexpected frozen bundle specification")
        prefix = matches[0]
        if values != [
            "--spec",
            str(spec_path),
            "--episodes",
            handoff["episodes"],
            "--out",
            handoff[f"{prefix}_bundle_out"],
            "--detector-prefix",
            "local3" if prefix == "local3" else "local2_frozen",
        ]:
            raise ValueError("frozen bundle invocation changed")
        binder = ReferenceBinder(
            prefix,
            json.loads(spec_path.read_text())["models"],
            local3_bundle.segment_scores,
            feature_audit,
        )
        original_score, original_argv = local3_bundle.segment_scores, sys.argv
        local3_bundle.segment_scores = binder.score
        try:
            sys.argv = [module, *values]
            local3_bundle.main()
            proof.append(binder.finish())
        finally:
            local3_bundle.segment_scores, sys.argv = original_score, original_argv

    adapter_path = LOCAL / "run_final_local_v3.py"
    spec = importlib.util.spec_from_file_location(
        "local3_frozen_path_adapter", adapter_path
    )
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    original_argv = sys.argv
    local3_final_runner.call = call
    try:
        sys.argv = [str(adapter_path), "export"]
        adapter.main()
    finally:
        sys.argv = original_argv
        local3_final_runner.call = original_call
    if [item["head_family"] for item in proof] != ["local3", "local2"]:
        raise ValueError("both frozen head families must be exported exactly once")
    result = {
        "schema": "local3_frozen_reference_authority_receipt_v1",
        "status": "original_frozen_arrays_preserved_and_fresh_scorer_unchanged",
        "adapter_sha256": digest(__file__),
        "root_release_path": str(args.release),
        "root_release_sha256": digest(args.release),
        "input_sha256": release["input_sha256"],
        "head_families": proof,
        "frozen_export_receipt_sha256": digest(FINAL / "local_export_receipt_v1.json"),
        "scope": "Original frozen normalizers remain byte-exact in exported reference fields. Reference calculations exactly match preserved same-machine replay; only development-reference return values are replaced by their pre-existing frozen values. No fresh score/label use in reference selection or values; every fresh scorer call is unchanged. Original failure audit and all replayed arrays are preserved.",
    }
    RECEIPT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
