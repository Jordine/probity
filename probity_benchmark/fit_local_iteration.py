"""Grouped local-probe comparison using one activation cache and frozen test groups."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F

from .fit_probe_pilots import linear_fit
from .preflight import sha256
from .qwen_extract import LAYERS


def unit_aucs(y, score, units, groups):
    output = []
    for unit in sorted(set(units)):
        ix = units == unit
        if len(set(y[ix])) == 2:
            output.append(
                (
                    groups[ix][0],
                    float(roc_auc_score(y[ix], score[ix])),
                    float(average_precision_score(y[ix], score[ix])),
                    float(y[ix][score[ix] == score[ix].max()].mean()),
                )
            )
    return output


def dev_value(y, score, units, groups):
    values = defaultdict(list)
    for group, auc, _, _ in unit_aucs(y, score, units, groups):
        values[group].append(auc)
    if not values:
        raise ValueError("no mixed-label development responses")
    return float(np.mean([np.mean(v) for v in values.values()]))


def metrics(y, scores, units, groups, threshold=None):
    mixed = unit_aucs(y, scores, units, groups)
    by_group = defaultdict(list)
    for group, auc, ap, top in mixed:
        by_group[group].append((auc, ap, top))
    result = {
        "claims": len(y),
        "false_claims": int(y.sum()),
        "parent_groups": len(set(groups)),
        "mixed_responses": len(mixed),
        "mixed_parent_groups": len(by_group),
        "pooled_auroc": float(roc_auc_score(y, scores)) if len(set(y)) == 2 else None,
    }
    if by_group:
        values = np.array([np.mean(v, axis=0) for v in by_group.values()])
        rng = np.random.default_rng(20261015)
        boot = values[rng.integers(0, len(values), (4000, len(values)))].mean(1)
        for i, name in enumerate(
            [
                "within_response_auroc",
                "within_response_auprc",
                "top_claim_false_precision",
            ]
        ):
            result[name] = {
                "parent_macro_mean": float(values[:, i].mean()),
                "parent_bootstrap_95_ci": np.quantile(
                    boot[:, i], [0.025, 0.975]
                ).tolist(),
            }
    if threshold is not None:
        flags = scores >= threshold
        result["threshold"] = float(threshold)
        result["false_claim_recall"] = (
            float(flags[y == 1].mean()) if np.any(y == 1) else None
        )
        result["true_claim_false_positive_rate"] = (
            float(flags[y == 0].mean()) if np.any(y == 0) else None
        )
    return result


def threshold_at_fpr(y, scores, fpr=0.1):
    negative = scores[y == 0]
    if not len(negative):
        raise ValueError("no development negatives for threshold")
    choices = np.r_[np.unique(negative), np.nextafter(negative.max(), np.inf)]
    return float(next(t for t in choices if np.mean(negative >= t) <= fpr))


def pair_indices(y, units):
    pairs = []
    for unit in sorted(set(units)):
        pos = np.flatnonzero((units == unit) & (y == 1))
        neg = np.flatnonzero((units == unit) & (y == 0))
        pairs.extend((p, n) for p in pos for n in neg)
    if not pairs:
        raise ValueError("ranking requires mixed training responses")
    return np.array(pairs)


def rank_fit(x, y, units, c):
    scaler = StandardScaler().fit(x)
    z = scaler.transform(x)
    pairs = pair_indices(y, units)
    delta = z[pairs[:, 0]] - z[pairs[:, 1]]
    clf = LogisticRegression(
        C=c,
        fit_intercept=False,
        dual=True,
        solver="liblinear",
        max_iter=2000,
        random_state=42,
    )
    clf.fit(
        np.concatenate([delta, -delta]),
        np.r_[np.ones(len(delta)), np.zeros(len(delta))],
    )
    if clf.n_iter_.max() >= clf.max_iter:
        raise ValueError("ranking head did not converge")
    weight = clf.coef_[0] / scaler.scale_
    return weight, float(-weight @ scaler.mean_)


def annealed_fit(x, y, units, objective, lr):
    torch.manual_seed(42)
    scaler = StandardScaler().fit(x)
    z = torch.tensor(scaler.transform(x), dtype=torch.float32)
    target = torch.tensor(y, dtype=torch.float32)
    pairs = torch.tensor(pair_indices(y, units), dtype=torch.long)
    model = torch.nn.Linear(x.shape[1], 1)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    bags = [
        (
            torch.tensor(np.flatnonzero((units == u) & (y == 1))),
            torch.tensor(np.flatnonzero((units == u) & (y == 0))),
        )
        for u in sorted(set(units))
    ]
    for epoch in range(100):
        opt.zero_grad()
        logits = model(z).squeeze(1)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        if objective == "annealed_ranking":
            specialized = F.softplus(
                -(logits[pairs[:, 0]] - logits[pairs[:, 1]])
            ).mean()
        elif objective == "annealed_bag":
            losses = []
            for pos, neg in bags:
                if len(pos):
                    # Smooth max among verified false claims; all verified true
                    # claims remain negatives. Unlabelled text is never used.
                    maximum = 0.25 * (
                        torch.logsumexp(logits[pos] / 0.25, 0) - np.log(len(pos))
                    )
                    losses.append(F.softplus(-maximum))
                if len(neg):
                    losses.append(F.softplus(logits[neg]).mean())
            specialized = torch.stack(losses).mean()
        else:
            raise ValueError(objective)
        alpha = max(0.0, min(1.0, (epoch - 30) / 69))
        loss = (
            (1 - alpha) * bce + alpha * specialized + 0.01 * model.weight.square().sum()
        )
        if not torch.isfinite(loss):
            raise ValueError("nonfinite annealed objective")
        loss.backward()
        opt.step()
    w = model.weight.detach().numpy()[0] / scaler.scale_
    b = float(model.bias.detach()[0] - w @ scaler.mean_)
    return w, b


def load_claims(directory, participant=None):
    summary = json.loads((directory / "summary.json").read_text())
    if summary["rows_expected"] != summary["rows_completed"]:
        raise ValueError("incomplete feature cache")
    output = []
    files = sorted(directory.glob("*.pt"))
    if len(files) != summary["rows_expected"]:
        raise ValueError("cache count mismatch")
    for path in files:
        cache = torch.load(path, map_location="cpu", weights_only=True)
        row = cache["row"]
        if participant and row.get("participant_model") != participant:
            continue
        for i, span in enumerate(row["spans"]):
            output.append(
                {
                    "id": row["id"],
                    "group": row["group"],
                    "split": row["split"],
                    "source": row["source"],
                    "y": span["label"],
                    "span_index": i,
                    "span_mean": cache["span_mean"][:, i].numpy(),
                    "span_last": cache["span_last"][:, i].numpy(),
                    "length": cache["span_lengths"][i],
                    "position": span["start"] / len(row["response"]),
                    "text": span.get("display_text", span["text"]),
                    "fibs": cache["fibs_contrast"],
                }
            )
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", type=Path, required=True)
    ap.add_argument("--debate", type=Path, required=True)
    ap.add_argument("--old-heads", type=Path)
    ap.add_argument("--participant", default="qwen/qwen3.5-27b")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError("new output directory required")
    args.out.mkdir(parents=True)
    torch.set_num_threads(4)
    synthetic = load_claims(args.synthetic)
    debate = load_claims(args.debate, args.participant)
    rows = synthetic + debate
    y = np.array([r["y"] for r in rows])
    units = np.array([r["id"] for r in rows])
    groups = np.array([r["group"] for r in rows])
    split = np.array([r["split"] for r in rows])
    domain = np.array([r["source"] == "api_debate_silver" for r in rows])
    for group in set(groups):
        if len(set(split[groups == group])) != 1:
            raise ValueError("parent group crosses splits")
    dev = (split == "dev") & domain
    test = (split == "test") & domain
    controlled_dev = (split == "dev") & ~domain
    controlled_test = (split == "test") & ~domain
    matrices = {f: np.stack([r[f] for r in rows]) for f in ["span_mean", "span_last"]}
    result = {
        "trainer_sha256": sha256(Path(__file__).read_text()),
        "synthetic_manifest": json.loads(
            (args.synthetic / "manifest.json").read_text()
        ),
        "debate_manifest": json.loads((args.debate / "manifest.json").read_text()),
        "claim_limit": "Four held-out parent stories at most; silver claim labels. Model/prompt/API replay shift remains. Test labels not used for model selection. Synthetic templates share families across splits.",
        "candidates": [],
        "tracks": {},
    }
    for key in ["model", "model_revision", "layers"]:
        if result["synthetic_manifest"][key] != result["debate_manifest"][key]:
            raise ValueError("synthetic/debate model contract mismatch")
    heads = {}
    for track in ["relational_only", "debate_only", "combined"]:
        train = (split == "train") & (
            ~domain
            if track == "relational_only"
            else domain if track == "debate_only" else np.ones(len(rows), dtype=bool)
        )
        candidates = []
        for feature, matrix in matrices.items():
            for li, layer in enumerate(LAYERS):
                x = matrix[:, li]
                settings = [
                    (objective, c)
                    for objective in ["bce", "pairwise"]
                    for c in [0.001, 0.01, 0.1]
                ]
                if (
                    track == "debate_only"
                    and feature == "span_mean"
                    and layer in (27, 35, 43)
                ):
                    settings += [
                        (objective, lr)
                        for objective in ["annealed_ranking", "annealed_bag"]
                        for lr in [0.003, 0.01]
                    ]
                for objective, parameter in settings:
                    if objective == "bce":
                        w, b = linear_fit(x[train], y[train], parameter)
                    elif objective == "pairwise":
                        w, b = rank_fit(x[train], y[train], units[train], parameter)
                    else:
                        w, b = annealed_fit(
                            x[train], y[train], units[train], objective, parameter
                        )
                    values = x[dev] @ w + b
                    record = {
                        "track": track,
                        "feature": feature,
                        "layer": layer,
                        "objective": objective,
                        "parameter": parameter,
                        "dev_parent_macro_local_auroc": dev_value(
                            y[dev], values, units[dev], groups[dev]
                        ),
                    }
                    if track == "relational_only":
                        record["controlled_dev_parent_macro_local_auroc"] = dev_value(
                            y[controlled_dev],
                            x[controlled_dev] @ w + b,
                            units[controlled_dev],
                            groups[controlled_dev],
                        )
                    result["candidates"].append(record)
                    candidates.append((record, w, b))
                    print(json.dumps(record), flush=True)
        if track == "relational_only":
            frozen, fw, fb = max(
                candidates,
                key=lambda c: (
                    c[0]["controlled_dev_parent_macro_local_auroc"],
                    -c[0]["parameter"],
                ),
            )
            fx = matrices[frozen["feature"]][:, LAYERS.index(frozen["layer"])]
            fv = fx @ fw + fb
            ft = threshold_at_fpr(y[controlled_dev], fv[controlled_dev])
            heads["relational_frozen_on_controlled_dev"] = {
                **frozen,
                "weight": torch.tensor(fw),
                "bias": fb,
                "threshold_dev_fpr10": ft,
            }
            result["controlled_frozen_head"] = {
                "selected": frozen,
                "controlled_test": metrics(
                    y[controlled_test],
                    fv[controlled_test],
                    units[controlled_test],
                    groups[controlled_test],
                    ft,
                ),
                "natural_dev": metrics(y[dev], fv[dev], units[dev], groups[dev], ft),
                "natural_test": metrics(
                    y[test], fv[test], units[test], groups[test], ft
                ),
            }
        selected, w, b = max(
            candidates,
            key=lambda c: (c[0]["dev_parent_macro_local_auroc"], -c[0]["parameter"]),
        )
        x = matrices[selected["feature"]][:, LAYERS.index(selected["layer"])]
        values = x @ w + b
        threshold = threshold_at_fpr(y[dev], values[dev])
        heads[track] = {
            **selected,
            "weight": torch.tensor(w),
            "bias": b,
            "threshold_dev_fpr10": threshold,
        }
        result["tracks"][track] = {
            "selected": selected,
            "train_claims": int(train.sum()),
            "train_groups": len(set(groups[train])),
            "dev": metrics(y[dev], values[dev], units[dev], groups[dev], threshold),
            "test": metrics(
                y[test], values[test], units[test], groups[test], threshold
            ),
        }
    best_track = max(
        ["relational_only", "debate_only", "combined"],
        key=lambda t: heads[t]["dev_parent_macro_local_auroc"],
    )
    result["selected_track"] = best_track
    old = (
        torch.load(args.old_heads, map_location="cpu", weights_only=True)
        if args.old_heads
        else {}
    )
    contract = old.get("__contract__")
    new_contract = result["debate_manifest"]
    if contract and any(
        contract[k] != new_contract[k] for k in ["model", "model_revision"]
    ):
        raise ValueError("old head/model mismatch")
    result["controls"] = {}
    score_sets = {}
    text_train = (split == "train") & domain
    text = np.array([r["text"] for r in rows])
    text_candidates = []
    for analyzer, grams in [("word", (1, 2)), ("char_wb", (3, 5))]:
        vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=grams,
            min_df=2,
            max_features=20000,
            sublinear_tf=True,
        )
        vectorizer.fit(text[text_train])
        matrix = vectorizer.transform(text)
        for c in [0.01, 0.1, 1.0, 10.0]:
            classifier = LogisticRegression(
                C=c, solver="liblinear", dual=True, max_iter=2000, random_state=42
            )
            classifier.fit(matrix[text_train], y[text_train])
            if classifier.n_iter_.max() >= classifier.max_iter:
                raise ValueError("text control did not converge")
            values = classifier.decision_function(matrix)
            text_candidates.append(
                (
                    dev_value(y[dev], values[dev], units[dev], groups[dev]),
                    {"analyzer": analyzer, "C": c},
                    values,
                )
            )
    text_best = max(text_candidates, key=lambda x: x[0])
    result["text_only_control_selection"] = text_best[1]
    score_sets["text_only_tfidf"] = text_best[2]
    for name in ["local", "apollo_response_mean", "apollo_dyl_features"]:
        if name not in old:
            continue
        if name == "apollo_dyl_features":
            continue  # A DYL head is never applied to argument-token features.
        head = old[name]
        feature = head["feature"] if name == "local" else "span_mean"
        x = matrices[feature][:, LAYERS.index(head["layer"])]
        values = x @ head["weight"].numpy() + head["bias"]
        score_sets["old_" + name] = values
    for feature in ["position", "length"]:
        values = np.array([r[feature] for r in rows])
        sign = max(
            [1, -1],
            key=lambda s: dev_value(y[dev], values[dev] * s, units[dev], groups[dev]),
        )
        score_sets[feature + "_dev_selected_direction"] = values * sign
    chosen = heads[best_track]
    x = matrices[chosen["feature"]][:, LAYERS.index(chosen["layer"])]
    rng = np.random.default_rng(20261015)
    random_candidates = [x @ rng.normal(size=x.shape[1]) for _ in range(96)]
    random_best = max(
        random_candidates,
        key=lambda v: dev_value(y[dev], v[dev], units[dev], groups[dev]),
    )
    score_sets["best_of_96_random_directions_on_dev"] = random_best
    # A whole-response score cannot localize within a response: expected local
    # AUROC is 0.5. It is still useful as a separate response-level detector.
    if all(r["fibs"] is not None for r in rows):
        score_sets["published_dyl_broadcast_to_claims"] = np.array(
            [r["fibs"] for r in rows]
        )
    for name, values in score_sets.items():
        threshold = threshold_at_fpr(y[dev], values[dev])
        result["controls"][name] = {
            "dev": metrics(y[dev], values[dev], units[dev], groups[dev], threshold),
            "test": metrics(
                y[test], values[test], units[test], groups[test], threshold
            ),
        }
    for name, head in heads.items():
        score_sets[name] = (
            matrices[head["feature"]][:, LAYERS.index(head["layer"])]
            @ head["weight"].numpy()
            + head["bias"]
        )
    predictions = []
    for i, row in enumerate(rows):
        if domain[i]:
            predictions.append(
                {k: row[k] for k in ["id", "group", "split", "span_index", "y", "text"]}
                | {"scores": {n: float(v[i]) for n, v in score_sets.items()}}
            )
    (args.out / "claim_predictions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in predictions)
    )
    heads["__contract__"] = {
        k: new_contract[k] for k in ["model", "model_revision", "layers"]
    }
    heads["__contract__"]["selected_track"] = best_track
    torch.save(heads, args.out / "heads.pt")
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ["candidates", "synthetic_manifest", "debate_manifest"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
