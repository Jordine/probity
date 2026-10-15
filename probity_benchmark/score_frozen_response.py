"""Apply frozen, model-matched response heads to fresh episodes without fitting."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .preflight import sha256
from .qwen_extract import LAYERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--heads", type=Path, required=True)
    ap.add_argument("--episodes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError("new output required")
    manifest = json.loads((args.features / "manifest.json").read_text())
    summary = json.loads((args.features / "summary.json").read_text())
    paths = sorted(args.features.glob("*.pt"))
    if (
        len(paths) != summary["rows_expected"]
        or len(paths) != summary["rows_completed"]
    ):
        raise ValueError("incomplete features")
    heads = torch.load(args.heads, map_location="cpu", weights_only=True)
    for key in ["model", "model_revision", "layers"]:
        if heads["__contract__"][key] != manifest[key]:
            raise ValueError("model/head mismatch")
    episodes = {
        e["episode_id"]: e
        for e in (
            json.loads(p.read_text()) for p in args.episodes.glob("episode_*.json")
        )
    }
    output = []
    for path in paths:
        c = torch.load(path, map_location="cpu", weights_only=True)
        r = c["row"]
        e = episodes[r["source_episode"]]
        t = e["turns"][r["turn_index"]]
        if r["source_request_sha256"] != t["request_sha256"]:
            raise ValueError("replay/episode mismatch")
        scores = {}
        for name, head in heads.items():
            if name.startswith("__"):
                continue
            features = (
                c["dyl_no"].float() - c["dyl_yes"].float()
                if name == "dyl_difference"
                else c[name].float()
            )
            scores[name] = float(
                features[LAYERS.index(head["layer"])].double() @ head["weight"].double()
                + head["bias"]
            )
        output.append(
            {
                "id": r["id"],
                "group": r["group"],
                "split": r["split"],
                "episode_id": e["episode_id"],
                "turn_index": r["turn_index"],
                "y": int(t["assigned_answer"] != e["gold"]),
                "knowledge_correct": e["knowledge_correct"],
                "scores": scores,
            }
        )
    args.out.mkdir(parents=True)
    (args.out / "turn_predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in output)
    )
    (args.out / "results.json").write_text(
        json.dumps(
            {
                "trained_here": False,
                "selected_feature": heads["__contract__"]["selected_feature"],
                "source_heads_sha256": hashlib.sha256(
                    args.heads.read_bytes()
                ).hexdigest(),
                "source_model_contract": heads["__contract__"],
                "manifest": manifest,
                "script_sha256": sha256(Path(__file__).read_text()),
                "rows": len(output),
                "selection": "Head and feature selected on earlier training/development stories, frozen before fresh calls. No fitting, calibration or threshold changes on these rows.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"rows": len(output), "trained_here": False}))


if __name__ == "__main__":
    main()
