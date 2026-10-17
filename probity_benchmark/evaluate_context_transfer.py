"""Test frozen natural-debate heads on identical-text, opposite-context facts."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .fit_local_iteration import metrics
from .preflight import sha256
from .qwen_extract import LAYERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--span-heads", type=Path, required=True)
    ap.add_argument("--token-heads", type=Path, required=True)
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
        raise ValueError("incomplete controlled cache")
    span = torch.load(args.span_heads, map_location="cpu", weights_only=True)
    token = torch.load(args.token_heads, map_location="cpu", weights_only=True)
    for heads in [span, token]:
        if any(
            heads["__contract__"][k] != manifest[k]
            for k in ["model", "model_revision", "layers"]
        ):
            raise ValueError("head/model mismatch")
    if token["mean_bce"]["pooling"] != "mean":
        raise ValueError("mean token head required for pooled-feature equivalence")
    heads = {
        "natural_span_selected": span[span["__contract__"]["selected_track"]],
        "natural_token_mean": {**token["mean_bce"], "feature": "span_mean"},
        "controlled_selected": span["relational_frozen_on_controlled_dev"],
    }
    rows = []
    for path in paths:
        c = torch.load(path, map_location="cpu", weights_only=True)
        r = c["row"]
        if r["split"] != "test":
            continue
        for i, s in enumerate(r["spans"]):
            values = {
                name: float(
                    c[h["feature"]][LAYERS.index(h["layer"]), i].double()
                    @ h["weight"].double()
                    + h["bias"]
                )
                for name, h in heads.items()
            }
            rows.append(
                {
                    "id": r["id"],
                    "group": r["group"],
                    "y": s["label"],
                    "pair_key": sha256(json.dumps([r["group"], r["response"], i])),
                    "scores": values,
                }
            )
    output = {
        "model": manifest["model"],
        "selection": "No fitting or selection in this diagnostic; natural heads and controlled head were already selected on their recorded development criteria.",
        "scope": "Identical response text and position under complementary factual contexts. Factual consistency, not generated deceptive intent. Controlled templates are shared across splits.",
        "artifacts": {
            "features_manifest_sha256": sha256(
                (args.features / "manifest.json").read_text()
            ),
            "span_heads_sha256": hashlib.sha256(
                args.span_heads.read_bytes()
            ).hexdigest(),
            "token_heads_sha256": hashlib.sha256(
                args.token_heads.read_bytes()
            ).hexdigest(),
        },
        "heads": {},
    }
    pairs = defaultdict(list)
    for row in rows:
        pairs[row["pair_key"]].append(row)
    if any(
        len(pair) != 2 or {r["y"] for r in pair} != {0, 1} for pair in pairs.values()
    ):
        raise ValueError("identical-text context twins must have complementary labels")
    y, units, groups = (np.array([r[k] for r in rows]) for k in ["y", "id", "group"])
    for name, h in heads.items():
        scores = np.array([r["scores"][name] for r in rows])
        flips = defaultdict(list)
        for pair in pairs.values():
            false, true = sorted(pair, key=lambda r: r["y"], reverse=True)
            difference = false["scores"][name] - true["scores"][name]
            flips[false["group"]].append(
                0.5 if difference == 0 else float(difference > 0)
            )
        values = np.array([np.mean(v) for v in flips.values()])
        rng = np.random.default_rng(2026101423)
        boot = values[rng.integers(len(values), size=(4000, len(values)))].mean(1)
        output["heads"][name] = {
            "feature": h["feature"],
            "layer": h["layer"],
            "metrics": metrics(y, scores, units, groups),
            "identical_text_context_flip": {
                "pairs": len(pairs),
                "parents": len(values),
                "accuracy": float(values.mean()),
                "parent_bootstrap_95_ci": np.quantile(boot, [0.025, 0.975]).tolist(),
            },
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
