"""Export frozen model-matched response detectors into the common judge format."""

import argparse
import hashlib
import json
from pathlib import Path

from .judge3_protocol import validate_bundle
from .preflight import sha256

DETECTORS = (
    "adapted_primary",
    "apollo_response_mean",
    "apollo_dyl_difference",
    "published_fibs_dyl",
)


def load_scores(root, short, split=None):
    adapted = root / ("fitted_response_" + short)
    apollo = root / ("response_probes_" + short)
    selection = json.loads((adapted / "results.json").read_text())["selected_feature"]
    rows = {}
    for line in (adapted / "turn_predictions.jsonl").read_text().splitlines():
        row = json.loads(line)
        if split is not None and row["split"] != split:
            continue
        prefix = "debate2-" + row["episode_id"] + "-"
        if not row["id"].startswith(prefix) or not row["id"][len(prefix) :].isdigit():
            raise ValueError("unrecognized exact turn identity")
        turn_index = int(row["id"][len(prefix) :])
        if row.get("turn_index", turn_index) != turn_index:
            raise ValueError("turn identity disagreement")
        key = (row["episode_id"], turn_index)
        if key in rows:
            raise ValueError("duplicate adapted score")
        rows[key] = {
            "group": row["group"],
            "split": row["split"],
            "scores": {"adapted_primary": row["scores"][selection]},
        }
    seen = set()
    for line in (apollo / "turn_scores.jsonl").read_text().splitlines():
        row = json.loads(line)
        if split is not None and row["split"] != split:
            continue
        key = (row["episode_id"], row["turn_index"])
        if key in seen or key not in rows:
            raise ValueError("duplicate or unmatched frozen score")
        seen.add(key)
        if any(row[k] != rows[key][k] for k in ("group", "split")):
            raise ValueError("score provenance mismatch")
        rows[key]["scores"].update(row["scores"])
    if seen != set(rows):
        raise ValueError("missing frozen scores")
    return rows


def artifact_contract(root, short, head_root):
    adapted = json.loads(
        (root / ("fitted_response_" + short) / "results.json").read_text()
    )
    apollo = json.loads(
        (root / ("response_probes_" + short) / "results.json").read_text()
    )
    result = {}
    for category, folder, actual in (
        ("adapted", "fitted_response_", adapted.get("source_heads_sha256")),
        ("apollo", "fitted_apollo_", apollo["heads_sha256"]),
    ):
        path = head_root / (folder + short) / "heads.pt"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual is not None and actual != digest:
            raise ValueError("scored head differs from frozen artifact")
        result[category] = digest
    for key in ("model", "model_revision", "layers"):
        if adapted["manifest"][key] != apollo["manifest"][key]:
            raise ValueError("adapted/Apollo model contract mismatch")
    result["model_contract"] = {
        key: adapted["manifest"][key] for key in ("model", "model_revision", "layers")
    }
    result["published_weights_revision"] = apollo["manifest"].get(
        "published_weights_revision"
    )
    return result


def build(evaluation_root, reference_root, head_root, episode_dir, out):
    if out.exists():
        raise ValueError("immutable new output required")
    episodes = {}
    for path in sorted(episode_dir.glob("episode_*.json")):
        row = json.loads(path.read_text())
        if row["episode_id"] in episodes:
            raise ValueError("duplicate episode")
        episodes[row["episode_id"]] = row
    if not episodes:
        raise ValueError("no episodes")
    all_scores, references, artifacts = {}, {}, {}
    for short, participant in (
        ("qwen35", "qwen/qwen3.5-27b"),
        ("qwen38", "qwen/qwen3.8-27b"),
    ):
        all_scores[participant] = load_scores(evaluation_root, short)
        references[participant] = load_scores(reference_root, short, split="dev")
        artifacts[participant] = artifact_contract(evaluation_root, short, head_root)
        reference_artifacts = artifact_contract(reference_root, short, head_root)
        if artifacts[participant] != reference_artifacts:
            raise ValueError("development/test detector contract mismatch")
        expected = {
            (e["episode_id"], i)
            for e in episodes.values()
            if e["model"] == participant
            for i in range(len(e["turns"]))
        }
        if set(all_scores[participant]) != expected:
            raise ValueError("evaluation scores do not cover exact episode set")
    out.mkdir(parents=True)
    for detector in DETECTORS:
        models = [
            m
            for m in all_scores
            if detector != "published_fibs_dyl" or m == "qwen/qwen3.5-27b"
        ]
        bundle = {
            "schema": "judge3_score_bundle_v1",
            "detector": detector,
            "orientation": "higher_more_unreliable",
            "granularity": "response",
            "target": "wrong_answer_advocacy",
            "participant_models": models,
            "references": {},
            "rows": [],
            "head_artifacts": {m: artifacts[m] for m in models},
            "reference_population": "All old iteration-2 development debate turns for the matching participant; fixed before iteration-3 outcomes.",
            "limitation": "Wrong-answer advocacy proxy, not local factual correctness or deceptive intent. Newly fitted Apollo-data heads are distinct from collaborators' original checkpoints. Published FIBS is model-matched Qwen3.5 only.",
        }
        for model in models:
            reference = [
                row for row in references[model].values() if row["split"] == "dev"
            ]
            if not reference:
                raise ValueError("empty dev reference")
            bundle["references"][model] = {
                "split": "dev",
                "parents": sorted({row["group"] for row in reference}),
                "scores": [row["scores"][detector] for row in reference],
            }
            for (episode_id, index), row in sorted(all_scores[model].items()):
                episode = episodes[episode_id]
                if row["group"] != episode["parent_problem_id"]:
                    raise ValueError("score/episode parent mismatch")
                argument = episode["turns"][index]["argument"]
                bundle["rows"].append(
                    {
                        "episode_id": episode_id,
                        "turn_index": index,
                        "argument_sha256": sha256(argument),
                        "spans": [[0, len(argument)]],
                        "scores": [row["scores"][detector]],
                    }
                )
        bundle["head_sha256"] = sha256(
            json.dumps(bundle["head_artifacts"], sort_keys=True)
        )
        bundle["reference_sha256"] = sha256(
            json.dumps(bundle["references"], sort_keys=True)
        )
        _, covered = validate_bundle(bundle, episodes)
        if covered != {k for k, e in episodes.items() if e["model"] in models}:
            raise ValueError("missing detector episode")
        path = out / (detector + ".json")
        path.write_text(json.dumps(bundle, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "detector": detector,
                    "episodes": len(covered),
                    "turns": len(bundle["rows"]),
                    "sha256": sha256(path.read_text()),
                }
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--head-root", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build(
        args.evaluation_root,
        args.reference_root,
        args.head_root,
        args.episodes,
        args.out,
    )


if __name__ == "__main__":
    main()
