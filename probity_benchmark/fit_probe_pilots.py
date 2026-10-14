"""Fit all small heads from one cache; select on dev, report grouped diagnostics."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch

from .qwen_extract import LAYERS


def grouped_ci(values, groups, seed=42):
    by_group = defaultdict(list)
    for value, group in zip(values, groups):
        by_group[group].append(float(value))
    group_means = np.array([np.mean(v) for v in by_group.values()])
    rng = np.random.default_rng(seed)
    boots = rng.choice(group_means, (2000, len(group_means)), replace=True).mean(1)
    return {
        "mean": float(group_means.mean()),
        "group_bootstrap_95_ci": np.quantile(boots, [0.025, 0.975]).tolist(),
        "groups": len(group_means),
    }


def local_metrics(scores, y, transcript, groups):
    units = sorted(set(transcript))
    aurocs, auprcs, top_one, unit_groups = [], [], [], []
    for unit in units:
        mask = transcript == unit
        target, values = y[mask], scores[mask]
        if set(target) != {0, 1}:
            raise ValueError(
                "localization requires mixed truth and falsehood within each response"
            )
        aurocs.append(roc_auc_score(target, values))
        auprcs.append(average_precision_score(target, values))
        best = values == values.max()
        top_one.append(float(target[best].mean()))
        unit_groups.append(groups[mask][0])
    return {
        "within_response_auroc": grouped_ci(aurocs, unit_groups),
        "within_response_auprc": grouped_ci(auprcs, unit_groups),
        "top_span_false_precision": grouped_ci(top_one, unit_groups),
        "pooled_span_auroc": float(roc_auc_score(y, scores)),
        "responses": len(units),
        "spans": len(y),
    }


def paired_auc(y, scores, groups):
    rng = np.random.default_rng(42)
    unique = sorted(set(groups))
    indices = [np.flatnonzero(groups == group) for group in unique]
    boots = []
    for _ in range(2000):
        ix = np.concatenate(
            [indices[i] for i in rng.integers(0, len(indices), len(indices))]
        )
        if len(set(y[ix])) != 2:
            raise ValueError("each source group must include both labels")
        boots.append(roc_auc_score(y[ix], scores[ix]))
    return {
        "auroc": float(roc_auc_score(y, scores)),
        "group_bootstrap_95_ci": np.quantile(boots, [0.025, 0.975]).tolist(),
        "groups": len(unique),
        "rows": len(y),
    }


def linear_fit(x, y, c):
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(
        C=c, solver="liblinear", dual=True, max_iter=2000, random_state=42
    )
    clf.fit(scaler.transform(x), y)
    if clf.n_iter_.max() >= clf.max_iter:
        raise ValueError("linear head failed to converge")
    # Fold standardization into a portable raw-activation head.
    weight = clf.coef_[0] / scaler.scale_
    bias = clf.intercept_[0] - np.dot(weight, scaler.mean_)
    return weight, float(bias)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError("new output directory required")
    summary = json.loads((args.features / "summary.json").read_text())
    if summary["rows_expected"] != summary["rows_completed"]:
        raise ValueError("extraction incomplete")
    alignment = json.loads((args.features / "alignment.json").read_text())
    if not alignment["passed"]:
        raise ValueError("alignment gate failed")
    args.out.mkdir(parents=True)
    local, apollo, pure = [], [], []
    files = sorted(args.features.glob("*.pt"))
    if len(files) != summary["rows_expected"]:
        raise ValueError("cache file count mismatch")
    for path in files:
        cache = torch.load(path, map_location="cpu", weights_only=True)
        r = cache["row"]
        if r["source"] == "apollo_roleplaying_offpolicy":
            apollo.append(
                {
                    "id": r["id"],
                    "group": r["group"],
                    "split": r["split"],
                    "y": r["label"],
                    "response_mean": cache["response_mean"].float().numpy(),
                    "dyl_features": cache["dyl_features"].float().numpy(),
                    "fibs": cache["fibs_score"],
                    "length": len(cache["response_positions"]),
                }
            )
        elif r["variant"] in ["truth", "false"]:
            pure.append(
                {
                    "group": r["group"],
                    "split": r["split"],
                    "y": r["label"],
                    "fibs": cache["fibs_score"],
                }
            )
        else:
            for i, span in enumerate(r["spans"]):
                local.append(
                    {
                        "id": r["id"],
                        "group": r["group"],
                        "split": r["split"],
                        "y": span["label"],
                        "span_index": i,
                        "span_mean": cache["span_mean"][:, i].numpy(),
                        "span_last": cache["span_last"][:, i].numpy(),
                        "length": len(cache["span_positions"][i]),
                    }
                )
    for collection in [local, apollo, pure]:
        partitions = defaultdict(set)
        for r in collection:
            partitions[r["group"]].add(r["split"])
        if any(len(x) != 1 for x in partitions.values()):
            raise ValueError("group leakage")
    result = {
        "extraction_completion": summary,
        "claim_limit": "Development probe pilot. Atomic labels measure teacher-forced factual consistency; Apollo labels are supplied and not independently verified. No downstream uplift claim.",
        "local_candidates": [],
        "apollo_candidates": [],
    }
    saved_heads = {}
    for name, collection, features in [
        ("local", local, ["span_mean", "span_last"]),
        ("apollo", apollo, ["response_mean", "dyl_features"]),
    ]:
        y = np.array([r["y"] for r in collection])
        splits = np.array([r["split"] for r in collection])
        transcripts = np.array([r["id"] for r in collection])
        groups = np.array([r["group"] for r in collection])
        train, dev, test = (splits == x for x in ["train", "dev", "test"])
        candidates = []
        for feature in features:
            stacked = np.stack([r[feature] for r in collection])
            for li, layer in enumerate(LAYERS):
                x = stacked[:, li]
                for c in [0.001, 0.01, 0.1, 1.0]:
                    weight, bias = linear_fit(x[train], y[train], c)
                    dev_scores = x[dev] @ weight + bias
                    if name == "local":
                        # Selection uses only the per-response ranking on dev.
                        units = sorted(set(transcripts[dev]))
                        value = float(
                            np.mean(
                                [
                                    roc_auc_score(
                                        y[dev][transcripts[dev] == unit],
                                        dev_scores[transcripts[dev] == unit],
                                    )
                                    for unit in units
                                ]
                            )
                        )
                    else:
                        value = float(roc_auc_score(y[dev], dev_scores))
                    record = {
                        "feature": feature,
                        "layer": layer,
                        "C": c,
                        "dev_auroc": value,
                    }
                    candidates.append((record, weight, bias))
                    result[name + "_candidates"].append(record)
                    print(json.dumps({"track": name, **record}), flush=True)
        best, weight, bias = max(
            candidates, key=lambda x: (x[0]["dev_auroc"], -x[0]["C"])
        )
        li = LAYERS.index(best["layer"])
        x = np.stack([r[best["feature"]][li] for r in collection])
        scores = x[test] @ weight + bias
        saved_heads[name] = {"weight": torch.from_numpy(weight), "bias": bias, **best}
        if name == "local":
            metrics = local_metrics(scores, y[test], transcripts[test], groups[test])
            length = np.array([r["length"] for r in collection])[test]
            position = np.array([r["span_index"] for r in collection])[test]
            rng = np.random.default_rng(42)
            random_scores = x[test] @ rng.normal(size=x.shape[1])
            controls = {
                "length": local_metrics(
                    length, y[test], transcripts[test], groups[test]
                ),
                "position": local_metrics(
                    position, y[test], transcripts[test], groups[test]
                ),
                "random_weight": local_metrics(
                    random_scores, y[test], transcripts[test], groups[test]
                ),
            }
        else:
            metrics = paired_auc(y[test], scores, groups[test])
            fibs = np.array([r["fibs"] for r in collection])[test]
            length = np.array([r["length"] for r in collection])[test]
            controls = {
                "published_fibs_dyl": paired_auc(y[test], fibs, groups[test]),
                "length": paired_auc(y[test], length, groups[test]),
            }
        result[name] = {
            "selected_on_dev": best,
            "test": metrics,
            "controls": controls,
            "train_rows": int(train.sum()),
            "dev_rows": int(dev.sum()),
            "test_rows": int(test.sum()),
        }
        if name == "apollo":
            # Report the best DYL and response-mean baseline separately as well.
            by_feature = {}
            for feature in features:
                chosen, w, b = max(
                    (c for c in candidates if c[0]["feature"] == feature),
                    key=lambda x: (x[0]["dev_auroc"], -x[0]["C"]),
                )
                li = LAYERS.index(chosen["layer"])
                values = np.stack([r[feature][li] for r in collection])[test] @ w + b
                by_feature[feature] = {
                    "selected_on_dev": chosen,
                    "test": paired_auc(y[test], values, groups[test]),
                }
                saved_heads["apollo_" + feature] = {
                    "weight": torch.from_numpy(w),
                    "bias": b,
                    **chosen,
                }
            result["apollo"]["by_feature"] = by_feature
    pure_test = [r for r in pure if r["split"] == "test"]
    result["published_fibs_atomic_pure"] = paired_auc(
        np.array([r["y"] for r in pure_test]),
        np.array([r["fibs"] for r in pure_test]),
        np.array([r["group"] for r in pure_test]),
    )
    torch.save(saved_heads, args.out / "heads.pt")
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if not k.endswith("_candidates")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
