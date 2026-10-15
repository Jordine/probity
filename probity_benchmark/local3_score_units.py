"""Freeze all-unit local predictions and controls before annotation is joined."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from .local3_annotate_units import display_units
from .local3_bundle import read_cache
from .preflight import sha256


def score_cache(cache, saved, old, response, text_control):
    row = cache["row"]
    argument, units = display_units(row["response"], ["sentence", "clause"])
    layers = saved["__contract__"]["layers"]
    indices = {}
    for i, segment in enumerate(cache["segments"]):
        key = segment["start"], segment["end"]
        if key in indices:
            previous = indices[key]
            for name in ["segment_mean", "segment_last"]:
                if not torch.equal(cache[name][:, previous], cache[name][:, i]):
                    raise ValueError(
                        "identical text units have different pooled features"
                    )
        indices[key] = i
    text_values = text_control["classifier"].decision_function(
        text_control["vectorizer"].transform([u["display_text"] for u in units])
    )
    response_head = response[response["__contract__"]["selected_feature"]]
    if response_head["feature"] != "response_mean":
        raise ValueError("response control must use complete-response mean features")
    broadcast = float(
        cache["response_mean"][layers.index(response_head["layer"])].double()
        @ response_head["weight"].double()
        + response_head["bias"]
    )
    head_set = {
        "local3": saved["local3"],
        "random_direction": saved["random_direction"],
        "frozen_local2": old[old["__contract__"]["selected_track"]],
    }
    for j, unit in enumerate(units):
        index = indices[unit["start"], unit["end"]]
        if cache["segments"][index]["text"] != unit["display_text"]:
            raise ValueError("scored unit differs from complete public argument")
        scores = {}
        for name, head in head_set.items():
            feature_name = {"span_mean": "segment_mean", "span_last": "segment_last"}[
                head["feature"]
            ]
            values = cache[feature_name][layers.index(head["layer"]), index].double()
            scores[name] = float(values @ head["weight"].double() + head["bias"])
        position = unit["start"] / len(row["response"])
        scores.update(
            position=position,
            negative_position=-position,
            length=float(cache["segment_lengths"][index]),
            negative_length=-float(cache["segment_lengths"][index]),
            fixed_word_tfidf=float(text_values[j]),
            response_broadcast=broadcast,
        )
        if cache.get("fibs_contrast") is not None:
            scores["published_dyl_broadcast"] = float(cache["fibs_contrast"])
        if not all(np.isfinite(v) for v in scores.values()):
            raise ValueError("nonfinite all-unit score")
        unit["scores"] = scores
    return {
        "id": row["id"],
        "group": row["group"],
        "participant_model": row["participant_model"],
        "source_episode": row["source_episode"],
        "turn_index": row["turn_index"],
        "response_sha256": sha256(row["response"]),
        "argument_sha256": sha256(argument),
        "units": units,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("immutable prediction output required")
    args.out.mkdir(parents=True)
    torch.set_num_threads(4)
    specs = json.loads(args.spec.read_text())["models"]
    manifest = {
        "script_sha256": sha256(Path(__file__).read_text()),
        "spec": specs,
        "models": {},
        "blinding": "No truth annotations supplied. Every fixed sentence/clause scored before labels are joined.",
    }
    rows, seen = [], set()
    for spec in specs:
        artifacts = {
            key: Path(spec[key])
            for key in ["head", "old_head", "response_head", "text_control"]
        }
        saved, old, response = [
            torch.load(artifacts[k], weights_only=True, map_location="cpu")
            for k in ["head", "old_head", "response_head"]
        ]
        text_control = joblib.load(artifacts["text_control"])
        contract = saved["__contract__"]
        for other in [
            old["__contract__"],
            response["__contract__"],
            text_control["contract"],
        ]:
            if any(
                other[k] != contract[k] for k in ["model", "model_revision", "layers"]
            ):
                raise ValueError("frozen control/head model contract mismatch")
        metadata = {
            "artifact_sha256": {
                k: hashlib.sha256(v.read_bytes()).hexdigest()
                for k, v in artifacts.items()
            },
            "thresholds": {},
            "threshold_reference_artifacts": {},
            "threshold_definition": "90th percentile (NumPy linear interpolation) of the same frozen all-sentence old-development reference for the corresponding model/head. This is an unlabeled score quantile, not calibrated 10% false-positive rate.",
            "feature_manifests": [],
        }
        for name, key, artifact in [
            ("local3", "local_reference_bundle", "head"),
            ("frozen_local2", "old_reference_bundle", "old_head"),
        ]:
            reference_path = Path(spec[key])
            reference = json.loads(reference_path.read_text())
            if (
                reference["granularity"] != "sentence"
                or reference["schema"] != "judge3_reference_bundle_v1"
            ):
                raise ValueError("frozen all-sentence development references required")
            if (
                reference["head_artifacts"][spec["participant"]]
                != metadata["artifact_sha256"][artifact]
            ):
                raise ValueError("reference threshold/head artifact mismatch")
            if reference["reference_sha256"] != sha256(
                json.dumps(reference["references"], sort_keys=True)
            ):
                raise ValueError("frozen reference array hash mismatch")
            values = reference["references"][spec["participant"]]
            if values["split"] != "dev":
                raise ValueError(
                    "only old development references can define thresholds"
                )
            metadata["thresholds"][name] = float(
                np.quantile(values["scores"], 0.9, method="linear")
            )
            metadata["threshold_reference_artifacts"][name] = hashlib.sha256(
                reference_path.read_bytes()
            ).hexdigest()
        for directory in spec["evaluation_features"]:
            files, feature_manifest = read_cache(Path(directory))
            if any(
                feature_manifest[k] != contract[k]
                for k in ["model", "model_revision", "layers"]
            ):
                raise ValueError("complete feature cache/head contract mismatch")
            metadata["feature_manifests"].append(feature_manifest)
            for path in files:
                cache = torch.load(path, weights_only=True, map_location="cpu")
                if cache["row"].get("participant_model") != spec["participant"]:
                    continue
                row = score_cache(cache, saved, old, response, text_control)
                if row["id"] in seen:
                    raise ValueError("duplicate scored response")
                seen.add(row["id"])
                rows.append(row)
        manifest["models"][spec["participant"]] = metadata
    if not rows:
        raise ValueError("no complete all-unit predictions")
    rows.sort(key=lambda r: r["id"])
    (args.out / "scores.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {
        "rows": len(rows),
        "episodes": len({r["source_episode"] for r in rows}),
        "units": sum(len(r["units"]) for r in rows),
        "scores_sha256": sha256((args.out / "scores.jsonl").read_text()),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
