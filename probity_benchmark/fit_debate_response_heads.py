"""Adapt response and DYL heads on training-story advocacy labels only."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch

from .evaluate_response_probes import grouped_auc
from .fit_probe_pilots import linear_fit
from .preflight import sha256
from .qwen_extract import LAYERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--episodes", type=Path, required=True)
    ap.add_argument("--exclude-rows", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError("new output required")
    manifest = json.loads((args.features / "manifest.json").read_text())
    summary = json.loads((args.features / "summary.json").read_text())
    paths = sorted(args.features.glob("*.pt"))
    if len(paths) != summary["rows_expected"] or summary["rows_completed"] != len(
        paths
    ):
        raise ValueError("feature cache incomplete")
    episodes = {
        e["episode_id"]: e
        for e in (
            json.loads(p.read_text()) for p in args.episodes.glob("episode_*.json")
        )
    }
    excluded = (
        set(json.loads(args.exclude_rows.read_text())) if args.exclude_rows else set()
    )
    rows = []
    for path in paths:
        c = torch.load(path, map_location="cpu", weights_only=True)
        r = c["row"]
        e = episodes[r["source_episode"]]
        t = e["turns"][r["turn_index"]]
        if r["source_request_sha256"] != t["request_sha256"]:
            raise ValueError("episode/features mismatch")
        if not e["knowledge_correct"] or r["id"] in excluded:
            continue
        rows.append(
            {
                "id": r["id"],
                "group": r["group"],
                "split": r["split"],
                "episode_id": e["episode_id"],
                "y": int(t["assigned_answer"] != e["gold"]),
                "text": t["argument"],
                "response_mean": c["response_mean"].float().numpy(),
                "argument_mean": c["argument_mean"].float().numpy(),
                "dyl_no": c["dyl_no"].float().numpy(),
                "dyl_difference": (c["dyl_no"].float() - c["dyl_yes"].float()).numpy(),
            }
        )
    y = np.array([r["y"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    splits = np.array([r["split"] for r in rows])
    train, dev, test = (splits == s for s in ["train", "dev", "test"])
    if any(len(set(splits[groups == g])) != 1 for g in set(groups)):
        raise ValueError("parent group crosses splits")
    result = {
        "manifest": manifest,
        "trainer_sha256": sha256(Path(__file__).read_text()),
        "target": "Incorrect assigned advocacy after correct private knowledge check; known prose mismatches excluded. Whole-response operational proxy, not local factual labels or proof of intent.",
        "selection": "Training on eight parent stories; all hyperparameters chosen on four development parents. Four test parents untouched by fitting or selection. Features are offline API-text replay, not live API hidden states.",
        "train_rows": int(train.sum()),
        "dev_rows": int(dev.sum()),
        "test_rows": int(test.sum()),
        "candidates": [],
        "heads": {},
        "controls": {},
    }
    heads, scores = {}, {}
    args.out.mkdir(parents=True)
    for feature in ["response_mean", "argument_mean", "dyl_no", "dyl_difference"]:
        matrix = np.stack([r[feature] for r in rows])
        candidates = []
        for li, layer in enumerate(LAYERS):
            for c in [0.001, 0.01, 0.1, 1.0]:
                w, b = linear_fit(matrix[train, li], y[train], c)
                record = {
                    "feature": feature,
                    "layer": layer,
                    "C": c,
                    "dev_auroc": float(roc_auc_score(y[dev], matrix[dev, li] @ w + b)),
                }
                candidates.append((record, w, b))
                result["candidates"].append(record)
        chosen, w, b = max(candidates, key=lambda x: (x[0]["dev_auroc"], -x[0]["C"]))
        values = matrix[:, LAYERS.index(chosen["layer"])] @ w + b
        scores[feature] = values
        heads[feature] = {**chosen, "weight": torch.tensor(w), "bias": b}
        result["heads"][feature] = {
            "selected": chosen,
            "dev": grouped_auc(y[dev], values[dev], groups[dev]),
            "test": grouped_auc(y[test], values[test], groups[test]),
        }
        print(json.dumps({"feature": feature, **result["heads"][feature]}), flush=True)
    text = np.array([r["text"] for r in rows])
    candidates = []
    for analyzer, grams in [("word", (1, 2)), ("char_wb", (3, 5))]:
        vectorizer = TfidfVectorizer(
            analyzer=analyzer, ngram_range=grams, min_df=2, max_features=20000
        )
        vectorizer.fit(text[train])
        x = vectorizer.transform(text)
        for c in [0.01, 0.1, 1.0, 10.0]:
            clf = LogisticRegression(
                C=c, solver="liblinear", dual=True, max_iter=2000, random_state=42
            )
            clf.fit(x[train], y[train])
            if clf.n_iter_.max() >= clf.max_iter:
                raise ValueError("text control did not converge")
            values = clf.decision_function(x)
            candidates.append(
                (float(roc_auc_score(y[dev], values[dev])), analyzer, c, values)
            )
    _, analyzer, c, values = max(candidates, key=lambda x: x[0])
    scores["text_only"] = values
    result["controls"]["text_only"] = {
        "selected": {"analyzer": analyzer, "C": c},
        "dev": grouped_auc(y[dev], values[dev], groups[dev]),
        "test": grouped_auc(y[test], values[test], groups[test]),
    }
    result["selected_feature"] = max(
        result["heads"], key=lambda name: result["heads"][name]["selected"]["dev_auroc"]
    )
    heads["__contract__"] = {
        k: manifest[k] for k in ["model", "model_revision", "layers"]
    }
    heads["__contract__"]["selected_feature"] = result["selected_feature"]
    torch.save(heads, args.out / "heads.pt")
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out / "turn_predictions.jsonl").write_text(
        "".join(
            json.dumps(
                {k: r[k] for k in ["id", "group", "split", "episode_id", "y"]}
                | {"scores": {name: float(v[i]) for name, v in scores.items()}}
            )
            + "\n"
            for i, r in enumerate(rows)
        )
    )


if __name__ == "__main__":
    main()
