"""Retain every preregistered candidate's predictions without reselection.

The original comparison retained every candidate's metrics and each track's best
predictions. This deterministic replay exports all 32 vectors/heads, verifies the
chosen vector against the already frozen artifact, and never changes that head.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .local3_fit import fit_head, load
from .preflight import sha256
from .qwen_extract import LAYERS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fitted", type=Path, required=True)
    p.add_argument("--natural", type=Path, required=True)
    p.add_argument("--twins", type=Path, nargs="+", required=True)
    p.add_argument("--natural-exclusions", type=Path, required=True)
    p.add_argument("--family-audit", type=Path, required=True)
    p.add_argument("--participant", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise ValueError("immutable retention export required")
    results = json.loads((args.fitted / "results.json").read_text())
    if results["script_sha256"] != sha256(
        Path(__file__).with_name("local3_fit.py").read_text()
    ):
        raise ValueError("replay must use exactly the frozen trainer source")
    if results["family_audit_sha256"] != sha256(args.family_audit.read_text()):
        raise ValueError("family grouping changed")
    for key, path in [("natural", args.natural)] + [
        (f"twins_{i}", d) for i, d in enumerate(args.twins)
    ]:
        if (
            json.loads((path / "manifest.json").read_text())
            != results["manifests"][key]
        ):
            raise ValueError("candidate replay input manifest changed")
    torch.set_num_threads(4)
    natural = load(
        args.natural, args.participant, json.loads(args.natural_exclusions.read_text())
    )
    twins = [r for path in args.twins for r in load(path)]
    rows = natural + twins
    family_by_parent = {
        r["problem_id"]: family
        for family, members in json.loads(args.family_audit.read_text())[
            "families"
        ].items()
        for r in members
    }
    for row in rows:
        row["group"] = "story-family-" + sha256(family_by_parent[row["group"]])
    domain = np.asarray([False] * len(natural) + [True] * len(twins))
    train = np.asarray([r["split"] == "train" for r in rows])
    x = np.stack([r["span_mean"] for r in rows])
    y = np.asarray([r["y"] for r in rows])
    previous = {
        (r["id"], r["span_index"]): r["scores"]["selected_local3"]
        for r in map(
            json.loads, (args.fitted / "predictions.jsonl").read_text().splitlines()
        )
    }
    saved = torch.load(args.fitted / "heads.pt", weights_only=True, map_location="cpu")
    predictions, heads = {}, {}
    selected_error = None
    for candidate in results["candidates"]:
        values = x[:, LAYERS.index(candidate["layer"])]
        weight, bias = fit_head(
            values,
            y,
            rows,
            train & ~domain,
            train & domain,
            candidate["track"],
            candidate["C"],
        )
        name = f"{candidate['track']}:L{candidate['layer']}:C{candidate['C']}"
        scores = values @ weight + bias
        predictions[name] = scores
        heads[name] = {**candidate, "weight": torch.tensor(weight), "bias": bias}
        if candidate == results["selected"]:
            expected = np.asarray([previous[r["id"], r["span_index"]] for r in rows])
            selected_error = float(np.max(np.abs(expected - scores)))
            if not np.allclose(
                expected, scores, rtol=1e-12, atol=1e-12
            ) or not np.allclose(
                weight, saved["local3"]["weight"].numpy(), rtol=1e-12, atol=1e-12
            ):
                raise ValueError(
                    "deterministic export changed the frozen selected candidate"
                )
    if selected_error is None:
        raise ValueError("frozen candidate missing from bounded grid")
    args.out.mkdir(parents=True)
    torch.save(heads, args.out / "candidate_heads.pt")
    (args.out / "candidate_predictions.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "id": r["id"],
                    "span_index": r["span_index"],
                    "y": r["y"],
                    "scores": {
                        name: float(values[i]) for name, values in predictions.items()
                    },
                }
            )
            + "\n"
            for i, r in enumerate(rows)
        )
    )
    audit = {
        "purpose": "Retention only, no selection or change to any frozen artifact",
        "rows": len(rows),
        "candidate_count": len(heads),
        "selected_max_absolute_error": selected_error,
        "frozen_heads_sha256": hashlib.sha256(
            (args.fitted / "heads.pt").read_bytes()
        ).hexdigest(),
        "trainer_sha256": results["script_sha256"],
        "exporter_sha256": sha256(Path(__file__).read_text()),
    }
    (args.out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
