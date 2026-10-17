"""Matched-model Apollo response and DYL heads from shared complete features."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from .fit_probe_pilots import linear_fit, paired_auc
from .preflight import sha256
from .qwen_extract import LAYERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError("new output directory required")
    summary = json.loads((args.features / "summary.json").read_text())
    files = sorted(args.features.glob("*.pt"))
    if (
        summary["rows_expected"] != summary["rows_completed"]
        or len(files) != summary["rows_expected"]
    ):
        raise ValueError("incomplete Apollo feature cache")
    rows = [torch.load(p, map_location="cpu", weights_only=True) for p in files]
    if any(r["row"]["source"] != "apollo_roleplaying_offpolicy" for r in rows):
        raise ValueError("Apollo-only corpus required")
    y = np.array([r["row"]["label"] for r in rows])
    groups = np.array([r["row"]["group"] for r in rows])
    splits = np.array([r["row"]["split"] for r in rows])
    train, dev, test = (splits == name for name in ["train", "dev", "test"])
    for group in set(groups):
        if len(set(splits[groups == group])) != 1:
            raise ValueError("Apollo group leakage")
    features = {
        "response_mean": np.stack([r["response_mean"].float().numpy() for r in rows]),
        "dyl_no": np.stack([r["dyl_no"].float().numpy() for r in rows]),
        "dyl_difference": np.stack(
            [(r["dyl_no"].float() - r["dyl_yes"].float()).numpy() for r in rows]
        ),
    }
    manifest = json.loads((args.features / "manifest.json").read_text())
    result = {
        "extraction_manifest": manifest,
        "trainer_sha256": sha256(Path(__file__).read_text()),
        "claim_limit": "Supplied Apollo labels; scenario-group split. Dataset familiarity and overlap with published FIBS training are not ruled out. This is not a local-span or model-organism test.",
        "candidates": [],
        "heads": {},
    }
    heads = {}
    args.out.mkdir(parents=True)
    for feature, matrix in features.items():
        candidates = []
        for li, layer in enumerate(LAYERS):
            x = matrix[:, li]
            for c in [0.001, 0.01, 0.1, 1.0]:
                w, b = linear_fit(x[train], y[train], c)
                record = {
                    "feature": feature,
                    "layer": layer,
                    "C": c,
                    "dev_auroc": float(roc_auc_score(y[dev], x[dev] @ w + b)),
                }
                candidates.append((record, w, b))
                result["candidates"].append(record)
        chosen, w, b = max(candidates, key=lambda c: (c[0]["dev_auroc"], -c[0]["C"]))
        values = matrix[:, LAYERS.index(chosen["layer"])] @ w + b
        heads[feature] = {**chosen, "weight": torch.tensor(w), "bias": b}
        result["heads"][feature] = {
            "selected": chosen,
            "test": paired_auc(y[test], values[test], groups[test]),
        }
        print(json.dumps({"feature": feature, **result["heads"][feature]}), flush=True)
    if manifest["published_probe_available"]:
        values = np.array([r["fibs_contrast"] for r in rows])
        result["published_fibs"] = paired_auc(y[test], values[test], groups[test])
    heads["__contract__"] = {
        k: manifest[k] for k in ["model", "model_revision", "layers"]
    }
    torch.save(heads, args.out / "heads.pt")
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
