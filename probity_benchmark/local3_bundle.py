"""Export fixed all-text local score bundles for the iteration-three judge."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .evaluate_debate_transfer import argument_mapping
from .judge3_protocol import partition, validate_bundle
from .preflight import sha256


def read_cache(directory):
    summary = json.loads((directory / "summary.json").read_text())
    files = sorted(directory.glob("*.pt"))
    if (
        summary["rows_expected"] != summary["rows_completed"]
        or len(files) != summary["rows_expected"]
    ):
        raise ValueError("complete feature cache required")
    return files, json.loads((directory / "manifest.json").read_text())


def segment_scores(cache, head, layers, granularity):
    feature_name = {"span_mean": "segment_mean", "span_last": "segment_last"}.get(
        head["feature"]
    )
    if feature_name is None:
        raise ValueError("this export requires a frozen local mean/last head")
    argument, mapping = argument_mapping(cache["row"]["response"])
    intervals = partition(argument, granularity)
    kind = "phrase" if granularity == "clause" else "sentence"
    indices = [i for i, s in enumerate(cache["segments"]) if s["kind"] == kind]
    if len(indices) != len(intervals) or cache.get(feature_name) is None:
        raise ValueError("all fixed sentence/clause features must be retained")
    layer_index = layers.index(head["layer"])
    scores = []
    for index, (start, end) in zip(indices, intervals):
        segment = cache["segments"][index]
        if segment["text"] != argument[start:end]:
            raise ValueError("segment differs from complete public argument")
        if (segment["start"], segment["end"]) != (
            mapping[start][0],
            mapping[end - 1][1],
        ):
            raise ValueError("segment/raw JSON mapping mismatch")
        feature = cache[feature_name][layer_index, index].double()
        value = feature @ head["weight"].double() + head["bias"]
        if not torch.isfinite(value):
            raise ValueError("nonfinite local signal")
        scores.append(float(value))
    return argument, [list(s) for s in intervals], scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, nargs="+", default=[])
    parser.add_argument("--references-only", action="store_true")
    parser.add_argument("--detector-prefix", default="local3")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("immutable new output directory required")
    specs = json.loads(args.spec.read_text())["models"]
    episodes = {}
    for directory in args.episodes:
        for path in sorted(directory.glob("episode_*.json")):
            episode = json.loads(path.read_text())
            if episode["episode_id"] in episodes:
                raise ValueError("duplicate episode across directories")
            episodes[episode["episode_id"]] = episode
    if not episodes and not args.references_only:
        raise ValueError("no complete episodes")
    args.out.mkdir(parents=True)
    head_hashes = {}
    for granularity in ["sentence", "clause"]:
        bundle = {
            "schema": (
                "judge3_reference_bundle_v1"
                if args.references_only
                else "judge3_score_bundle_v1"
            ),
            "detector": args.detector_prefix + "_" + granularity,
            "orientation": "higher_more_unreliable",
            "granularity": granularity,
            "target": "local_factual_contradiction",
            "references": {},
            "rows": [],
            "reference_population": "All fixed text-only segments from original-source dev response rows in the verified natural-twin cache, matching participant model. Response rows were selected because at least one name-bearing claim passed source-flip verification; individual reference segments are never selected by labels. No fresh test rows or judged-episode outcomes used.",
            "limitation": "Raw linear scores and development percentiles are not factual probabilities; a display unit can mix true, false and unlabelled text. Complete all-turn/all-unit display, without annotation masks or selected peaks.",
        }
        for spec in specs:
            head_path = Path(spec["head"])
            saved = torch.load(head_path, weights_only=True, map_location="cpu")
            key = spec.get("head_key", "local3")
            head = saved[key]
            contract = saved["__contract__"]
            participant = spec["participant"]
            head_hashes[participant] = hashlib.sha256(
                head_path.read_bytes()
            ).hexdigest()
            reference_files, reference_manifest = read_cache(
                Path(spec["reference_features"])
            )
            for field in ["model", "model_revision", "layers"]:
                if contract[field] != reference_manifest[field]:
                    raise ValueError("reference feature/head contract mismatch")
            reference_scores, reference_parents, reference_rows = [], set(), []
            for path in reference_files:
                cache = torch.load(path, weights_only=True, map_location="cpu")
                row = cache["row"]
                if (
                    row["split"] != "dev"
                    or row.get("context_world") != 0
                    or row.get("participant_model") != participant
                ):
                    continue
                _, _, scores = segment_scores(
                    cache, head, contract["layers"], granularity
                )
                reference_scores.extend(scores)
                reference_parents.add(row["group"])
                reference_rows.append(row["id"])
            if not reference_scores:
                raise ValueError("empty development-only normalizer")
            bundle["references"][participant] = {
                "split": "dev",
                "parents": sorted(reference_parents),
                "scores": reference_scores,
                "rows": reference_rows,
                "manifest_sha256": sha256(
                    json.dumps(reference_manifest, sort_keys=True)
                ),
            }
            if args.references_only:
                continue
            for directory_name in spec["evaluation_features"]:
                files, manifest = read_cache(Path(directory_name))
                if any(
                    manifest[k] != contract[k]
                    for k in ["model", "model_revision", "layers"]
                ):
                    raise ValueError("evaluation feature/head contract mismatch")
                for path in files:
                    cache = torch.load(path, weights_only=True, map_location="cpu")
                    row = cache["row"]
                    if row.get("participant_model") != participant:
                        continue
                    episode_id = row["source_episode"]
                    if episode_id not in episodes:
                        raise ValueError("feature row lacks requested complete episode")
                    argument, spans, scores = segment_scores(
                        cache, head, contract["layers"], granularity
                    )
                    bundle["rows"].append(
                        {
                            "episode_id": episode_id,
                            "turn_index": row["turn_index"],
                            "argument_sha256": sha256(argument),
                            "spans": spans,
                            "scores": scores,
                        }
                    )
        bundle["head_artifacts"] = head_hashes
        bundle["head_sha256"] = sha256(json.dumps(head_hashes, sort_keys=True))
        bundle["reference_sha256"] = sha256(
            json.dumps(bundle["references"], sort_keys=True)
        )
        bundle["rows"].sort(key=lambda r: (r["episode_id"], r["turn_index"]))
        covered = set()
        if not args.references_only:
            _, covered = validate_bundle(bundle, episodes)
        else:
            bundle["scope"] = (
                "Frozen development-only normalization arrays and artifact hashes. No evaluation rows or performance measurement."
            )
        destination = args.out / (
            granularity + ("_references" if args.references_only else "") + ".json"
        )
        destination.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "granularity": granularity,
                    "references_only": args.references_only,
                    "head_sha256": bundle["head_sha256"],
                    "reference_sha256": bundle["reference_sha256"],
                    "episodes": len(covered),
                    "turns": len(bundle["rows"]),
                    "scored_units": sum(len(r["scores"]) for r in bundle["rows"]),
                    "sha256": sha256(destination.read_text()),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
