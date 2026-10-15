"""Join frozen all-unit predictions with independently completed truth labels."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from .fit_local_iteration import metrics, unit_aucs
from .preflight import sha256


def paired_delta(y, main, control, ids, families):
    ours = unit_aucs(y, main, ids, families)
    other = unit_aucs(y, control, ids, families)
    if len(ours) != len(other):
        raise ValueError(
            "paired local comparison has different eligible response counts"
        )
    grouped = defaultdict(list)
    for a, b in zip(ours, other):
        if a[0] != b[0]:
            raise ValueError("paired response families differ")
        grouped[a[0]].append(a[1] - b[1])
    if not grouped:
        return {"families": 0, "mean": None, "bootstrap_95_ci": None}
    values = np.asarray([np.mean(v) for _, v in sorted(grouped.items())])
    rng = np.random.default_rng(2026101432)
    boot = values[rng.integers(0, len(values), (4000, len(values)))].mean(1)
    return {
        "families": len(values),
        "mean": float(values.mean()),
        "bootstrap_95_ci": np.quantile(boot, [0.025, 0.975]).tolist(),
        "per_family": {k: float(np.mean(v)) for k, v in grouped.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--family-map", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("immutable test evaluation output required")
    annotation_summary = json.loads((args.annotations / "summary.json").read_text())
    if (
        annotation_summary["errors"]
        or annotation_summary["expected_rows"] != annotation_summary["completed_rows"]
        or annotation_summary["expected_units"] != annotation_summary["completed_units"]
    ):
        raise ValueError("complete annotations required; no partial outcome analysis")
    predictions = {
        r["id"]: r
        for r in map(
            json.loads, (args.scores / "scores.jsonl").read_text().splitlines()
        )
    }
    annotations = [
        json.loads(line)
        for line in (args.annotations / "corpus.jsonl").read_text().splitlines()
    ]
    if len(predictions) != len(annotations) or set(predictions) != {
        r["id"] for r in annotations
    }:
        raise ValueError(
            "predictions and annotations must cover identical complete responses"
        )
    families = json.loads(args.family_map.read_text())["by_parent"]
    manifest = json.loads((args.scores / "manifest.json").read_text())
    rows = []
    for row in annotations:
        predicted = predictions[row["id"]]
        if (
            predicted["response_sha256"] != sha256(row["response"])
            or predicted["group"] != row["group"]
            or predicted["participant_model"] != row["participant_model"]
        ):
            raise ValueError("scored and annotated response bytes or identity differ")
        lookup = {
            (u["argument_start"], u["argument_end"]): u for u in predicted["units"]
        }
        for annotated in row["annotations"]:
            unit = annotated["unit"]
            scored = lookup[unit["argument_start"], unit["argument_end"]]
            for key in ["start", "end", "text", "display_text"]:
                if scored[key] != unit[key]:
                    raise ValueError("scored unit differs from exact annotated unit")
            for granularity in unit["granularities"]:
                if granularity not in scored["granularities"]:
                    raise ValueError("scored unit granularity mismatch")
                rows.append(
                    {
                        "id": row["id"],
                        "source_parent_id": row["group"],
                        "family": families[row["group"]],
                        "model": row["participant_model"],
                        "granularity": granularity,
                        "text": unit["display_text"],
                        "argument_start": unit["argument_start"],
                        "argument_end": unit["argument_end"],
                        "y": annotated["label"],
                        "rejection_reasons": annotated["rejection_reasons"],
                        "scores": scored["scores"],
                    }
                )
    result = {
        "script_sha256": sha256(Path(__file__).read_text()),
        "score_manifest": manifest,
        "annotation_manifest": json.loads(
            (args.annotations / "manifest.json").read_text()
        ),
        "annotation_summary": annotation_summary,
        "family_map_sha256": sha256(args.family_map.read_text()),
        "independence_unit": "Original normalized story family. Macro means first average mixed responses within family; bootstrap resamples families.",
        "threshold_limit": "Frozen 90th percentile of unlabeled all-sentence old-development reference scores for each model/head. Actual fresh precision, recall and false-positive rate are measured on independently accepted labels; this threshold does not target 10% FPR. Unknown units have unmeasured truth.",
        "cells": {},
    }
    for model in sorted({r["model"] for r in rows}):
        result["cells"][model] = {}
        for granularity in sorted(
            {r["granularity"] for r in rows if r["model"] == model}
        ):
            all_units = [
                r
                for r in rows
                if r["model"] == model and r["granularity"] == granularity
            ]
            known = [r for r in all_units if r["y"] is not None]
            cell = {
                "coverage": {
                    "all_units": len(all_units),
                    "accepted_units": len(known),
                    "fraction": len(known) / len(all_units),
                    "all_responses": len({r["id"] for r in all_units}),
                    "all_families": len({r["family"] for r in all_units}),
                    "rejection_reasons": dict(
                        Counter(
                            reason
                            for r in all_units
                            for reason in r["rejection_reasons"]
                        )
                    ),
                },
                "metrics": {},
                "paired_local3_minus_control": {},
            }
            if known:
                y = np.asarray([r["y"] for r in known])
                ids = np.asarray([r["id"] for r in known])
                groups = np.asarray([r["family"] for r in known])
                tracks = sorted(known[0]["scores"])
                if any(sorted(r["scores"]) != tracks for r in known):
                    raise ValueError("incomplete score tracks within model")
                for name in tracks:
                    score = np.asarray([r["scores"][name] for r in known])
                    threshold = manifest["models"][model]["thresholds"].get(name)
                    cell["metrics"][name] = metrics(y, score, ids, groups, threshold)
                    if threshold is not None:
                        flags = score >= threshold
                        cell["metrics"][name]["flagged_labelled_units"] = int(
                            flags.sum()
                        )
                        cell["metrics"][name]["flag_precision"] = (
                            float(y[flags].mean()) if flags.any() else None
                        )
                        cell["metrics"][name]["flagged_all_units"] = sum(
                            r["scores"][name] >= threshold for r in all_units
                        )
                    if name != "local3":
                        primary = np.asarray([r["scores"]["local3"] for r in known])
                        cell["paired_local3_minus_control"][name] = paired_delta(
                            y, primary, score, ids, groups
                        )
                    if (
                        name.endswith("broadcast")
                        and cell["metrics"][name]["mixed_responses"]
                    ):
                        if not np.isclose(
                            cell["metrics"][name]["within_response_auroc"][
                                "parent_macro_mean"
                            ],
                            0.5,
                        ):
                            raise ValueError(
                                "response broadcast spuriously localized a claim"
                            )
            result["cells"][model][granularity] = cell
    args.out.mkdir(parents=True)
    (args.out / "predictions_with_labels.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cells": result["cells"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
