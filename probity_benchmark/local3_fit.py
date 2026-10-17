"""Bounded source-pair local-head comparison from pooled activation caches.

Fit once per layer from cached full-context activations. Primary development
selection maximizes the minimum of natural within-response AUROC, exact-text
source-pair ordering accuracy and mixed-twin within-response AUROC; ties favor
natural AUROC then smaller C.
No new final-holdout outcomes are consumed by this module.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch

from .fit_local_iteration import metrics, threshold_at_fpr
from .preflight import sha256
from .qwen_extract import LAYERS

GRID_LAYERS = [27, 35, 43, 51]
GRID_C = [0.001, 0.01]
TRACKS = ["natural_bce", "twins_bce", "twins_pairwise", "mixed_pairwise"]


def load(directory, participant=None, exclusions=(), response_head=None):
    summary = json.loads((directory / "summary.json").read_text())
    files = sorted(directory.glob("*.pt"))
    if not (summary["rows_expected"] == summary["rows_completed"] == len(files)):
        raise ValueError("incomplete pooled feature cache")
    rows = []
    for path in files:
        cache = torch.load(path, weights_only=True, map_location="cpu")
        row = cache["row"]
        if participant and row.get("participant_model") != participant:
            continue
        response_score = None
        if response_head is not None:
            head, layers = response_head
            if head["feature"] != "response_mean":
                raise ValueError("broadcast control requires frozen response-mean head")
            response_score = float(
                cache["response_mean"][layers.index(head["layer"])].double()
                @ head["weight"].double()
                + head["bias"]
            )
        for i, span in enumerate(row["spans"]):
            excluded = [
                e
                for e in exclusions
                if e["source_row_id"] == row["id"] and e["span_index"] == i
            ]
            if excluded:
                if len(excluded) != 1 or excluded[0]["expected_text"] != span.get(
                    "display_text", span["text"]
                ):
                    raise ValueError("guarded natural-label exclusion does not match")
                continue
            rows.append(
                {
                    "id": row["id"],
                    "group": row["group"],
                    "split": row["split"],
                    "span_index": i,
                    "y": span["label"],
                    "text": span.get("display_text", span["text"]),
                    "position": span["start"] / len(row["response"]),
                    "pair_id": row.get("pair_id"),
                    "context_world": row.get("context_world"),
                    "source_row_id": row.get("source_row_id", row["id"]),
                    "original_span_index": span.get("original_span_index", i),
                    "is_mixed_augmentation": row.get("augmentation", {}).get("kind")
                    == "single_claim_entity_swap",
                    "response_broadcast": response_score,
                    "published_dyl_broadcast": cache.get("fibs_contrast"),
                    "span_mean": cache["span_mean"][:, i].float().numpy(),
                    "span_last": cache["span_last"][:, i].float().numpy(),
                }
            )
    return rows


def source_pairs(rows, mask):
    by_pair = defaultdict(list)
    for i, row in enumerate(rows):
        if mask[i] and row["pair_id"] is not None:
            by_pair[row["pair_id"], row["original_span_index"]].append(i)
    pairs = []
    for indices in by_pair.values():
        if len(indices) != 2:
            raise ValueError("source twin lacks one member")
        a, b = indices
        if (
            rows[a]["text"] != rows[b]["text"]
            or rows[a]["position"] != rows[b]["position"]
        ):
            raise ValueError("paired text or position changed")
        if rows[a]["y"] == rows[b]["y"] or rows[a]["group"] != rows[b]["group"]:
            raise ValueError("source-pair labels/groups invalid")
        pairs.append((a, b) if rows[a]["y"] else (b, a))
    if not pairs:
        raise ValueError("no source pairs")
    return np.asarray(pairs, dtype=int)


def within_pairs(rows, mask):
    by_unit = defaultdict(list)
    for i, row in enumerate(rows):
        if mask[i]:
            by_unit[row["id"]].append(i)
    pairs = []
    for indices in by_unit.values():
        positive = [i for i in indices if rows[i]["y"]]
        negative = [i for i in indices if not rows[i]["y"]]
        pairs.extend((p, n) for p in positive for n in negative)
    if not pairs:
        raise ValueError("no mixed natural training responses")
    return np.asarray(pairs, dtype=int)


def parent_weights(groups):
    unique, counts = np.unique(groups, return_counts=True)
    weight = dict(zip(unique, 1 / counts))
    output = np.asarray([weight[g] for g in groups])
    return output / output.mean()


def fit_head(x, y, rows, natural_train, twins_train, track, c):
    train = (
        natural_train
        if track == "natural_bce"
        else (
            twins_train
            if track in ["twins_bce", "twins_pairwise"]
            else natural_train | twins_train
        )
    )
    scaler = StandardScaler().fit(x[train])
    z = scaler.transform(x)
    if track.endswith("bce"):
        features = z[train]
        target = y[train]
        # Equal parent weight and equal class weight within each parent.
        cells = np.asarray(
            [r["group"] + ":" + str(r["y"]) for r, use in zip(rows, train) if use]
        )
        weights = parent_weights(cells)
        intercept = True
    else:
        twin_pairs = source_pairs(rows, twins_train)
        pair_sets = [twin_pairs]
        if track == "mixed_pairwise":
            pair_sets.append(within_pairs(rows, natural_train))
            mixed_train = twins_train & np.asarray(
                [r.get("is_mixed_augmentation", False) for r in rows]
            )
            if mixed_train.any():
                pair_sets.append(within_pairs(rows, mixed_train))
        deltas, targets, all_weights = [], [], []
        for pairs in pair_sets:
            delta = z[pairs[:, 0]] - z[pairs[:, 1]]
            weights = parent_weights(
                np.asarray([rows[i]["group"] for i in pairs[:, 0]])
            )
            # Each pair family has equal total loss, regardless of pair count.
            weights = weights / weights.sum()
            deltas.extend([delta, -delta])
            targets.extend([np.ones(len(delta)), np.zeros(len(delta))])
            all_weights.extend([weights, weights])
        features = np.concatenate(deltas)
        target = np.concatenate(targets)
        weights = np.concatenate(all_weights)
        weights = weights / weights.mean()
        intercept = False
    classifier = LogisticRegression(
        C=c,
        solver="liblinear",
        dual=True,
        fit_intercept=intercept,
        max_iter=3000,
        random_state=20261014,
    )
    classifier.fit(features, target, sample_weight=weights)
    if classifier.n_iter_.max() >= classifier.max_iter:
        raise ValueError("head failed to converge")
    weight = classifier.coef_[0] / scaler.scale_
    bias = float((classifier.intercept_[0] if intercept else 0) - weight @ scaler.mean_)
    return weight, bias


def paired_metrics(rows, scores, mask):
    pairs = source_pairs(rows, mask)
    by_parent = defaultdict(list)
    for pos, neg in pairs:
        by_parent[rows[pos]["group"]].append(
            float(scores[pos] > scores[neg]) + 0.5 * float(scores[pos] == scores[neg])
        )
    means = np.asarray([np.mean(v) for _, v in sorted(by_parent.items())])
    rng = np.random.default_rng(2026101431)
    boot = means[rng.integers(0, len(means), size=(4000, len(means)))].mean(1)
    return {
        "pairs": len(pairs),
        "parent_groups": len(by_parent),
        "parent_macro_order_accuracy": float(means.mean()),
        "parent_bootstrap_95_ci": np.quantile(boot, [0.025, 0.975]).tolist(),
        "per_parent": {
            g: {"pairs": len(v), "mean": float(np.mean(v))}
            for g, v in by_parent.items()
        },
        "text_position_control": "Exactly 0.5 by paired identical response/span positions and complementary labels.",
    }


def evaluate(rows, scores, natural_mask, twins_mask, threshold=None):
    y = np.asarray([r["y"] for r in rows])
    ids = np.asarray([r["id"] for r in rows])
    groups = np.asarray([r["group"] for r in rows])
    result = {
        "natural": metrics(
            y[natural_mask],
            scores[natural_mask],
            ids[natural_mask],
            groups[natural_mask],
            threshold,
        ),
        "source_twins": paired_metrics(rows, scores, twins_mask),
    }
    mixed = twins_mask & np.asarray(
        [r.get("is_mixed_augmentation", False) for r in rows]
    )
    if mixed.any():
        result["mixed_source_twins"] = metrics(
            y[mixed], scores[mixed], ids[mixed], groups[mixed], threshold
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--natural", type=Path, required=True)
    parser.add_argument("--twins", type=Path, nargs="+", required=True)
    parser.add_argument("--old-heads", type=Path, required=True)
    parser.add_argument("--old-predictions", type=Path, required=True)
    parser.add_argument("--response-heads", type=Path)
    parser.add_argument("--natural-exclusions", type=Path)
    parser.add_argument("--family-audit", type=Path, required=True)
    parser.add_argument("--participant", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("immutable output required")
    args.out.mkdir(parents=True)
    torch.set_num_threads(4)
    exclusions = (
        json.loads(args.natural_exclusions.read_text())
        if args.natural_exclusions
        else []
    )
    response_head = None
    if args.response_heads:
        response_saved = torch.load(
            args.response_heads, weights_only=True, map_location="cpu"
        )
        response_head = (
            response_saved[response_saved["__contract__"]["selected_feature"]],
            response_saved["__contract__"]["layers"],
        )
    natural = load(args.natural, args.participant, exclusions, response_head)
    twins = [
        r
        for directory in args.twins
        for r in load(directory, response_head=response_head)
    ]
    rows = natural + twins
    family_audit = json.loads(args.family_audit.read_text())
    family_by_parent = {
        row["problem_id"]: family
        for family, members in family_audit["families"].items()
        for row in members
    }
    for row in rows:
        parent = row["group"]
        if parent not in family_by_parent:
            raise ValueError("source parent missing from story-family audit")
        row["source_parent_id"] = parent
        row["story_family"] = family_by_parent[parent]
        row["group"] = "story-family-" + sha256(row["story_family"])
    domain = np.asarray([False] * len(natural) + [True] * len(twins))
    splits = np.asarray([r["split"] for r in rows])
    y = np.asarray([r["y"] for r in rows])
    x = np.stack([r["span_mean"] for r in rows])
    natural_train, twins_train = (splits == "train") & ~domain, (
        splits == "train"
    ) & domain
    natural_dev, twins_dev = (splits == "dev") & ~domain, (splits == "dev") & domain
    augmented = np.asarray([r["is_mixed_augmentation"] for r in rows])
    if not (augmented & twins_train).any() or not (augmented & twins_dev).any():
        raise ValueError(
            "mixed-response twins are required in both training and development"
        )
    for group in {r["group"] for r in rows}:
        if len({r["split"] for r in rows if r["group"] == group}) != 1:
            raise ValueError("parent group crosses train/development partition")
    manifests = {
        name: json.loads((directory / "manifest.json").read_text())
        for name, directory in [("natural", args.natural)]
        + [(f"twins_{i}", directory) for i, directory in enumerate(args.twins)]
    }
    for key in ["model", "model_revision", "layers"]:
        if any(
            manifests["natural"][key] != manifest[key]
            for manifest in manifests.values()
        ):
            raise ValueError("model/layer contract mismatch")
        if (
            args.response_heads
            and response_saved["__contract__"][key] != manifests["natural"][key]
        ):
            raise ValueError("frozen response-control model contract mismatch")
    old = torch.load(args.old_heads, weights_only=True, map_location="cpu")
    if any(
        old["__contract__"][k] != manifests["natural"][k]
        for k in ["model", "model_revision"]
    ):
        raise ValueError("old head contract mismatch")
    result = {
        "script_sha256": sha256(Path(__file__).read_text()),
        "natural_label_exclusions": exclusions,
        "frozen_control_artifacts": {
            "local_head_sha256": hashlib.sha256(
                args.old_heads.read_bytes()
            ).hexdigest(),
            "text_predictions_sha256": sha256(args.old_predictions.read_text()),
            "response_head_sha256": (
                hashlib.sha256(args.response_heads.read_bytes()).hexdigest()
                if args.response_heads
                else None
            ),
        },
        "family_audit_sha256": sha256(args.family_audit.read_text()),
        "independence_unit": "Normalized original story_name family. All fitting weights, candidate aggregates and bootstrap intervals group by family; source-parent IDs are retained separately.",
        "manifests": manifests,
        "candidate_grid": {
            "layers": GRID_LAYERS,
            "C": GRID_C,
            "tracks": TRACKS,
            "pooling": "span_mean",
        },
        "selection_rule": "Maximize minimum of natural-dev family-macro within-response AUROC, exact-text source-twin-dev family-macro ordering accuracy, and mixed-source-twin-dev family-macro within-response AUROC; ties natural AUROC then smaller C. All this data is prior-pilot development. No fresh test outcomes read.",
        "mixed_pairwise_loss": "Equal total weight for exact-source pair ranking, natural within-response ranking, and augmented mixed-response ranking; equal family weight inside each family of pairs.",
        "label_limit": "Sparse silver natural spans and selected source-intervention flips; intent not labelled. Source twins preserve complete original response, source position and non-source prompt, but can disagree with unreplaced historical statements.",
        "candidates": [],
        "controls": {},
    }
    candidates = []
    for track in TRACKS:
        for layer in GRID_LAYERS:
            values = x[:, LAYERS.index(layer)]
            for c in GRID_C:
                w, b = fit_head(values, y, rows, natural_train, twins_train, track, c)
                scores = values @ w + b
                evaluated = evaluate(rows, scores, natural_dev, twins_dev)
                natural_auc = evaluated["natural"]["within_response_auroc"][
                    "parent_macro_mean"
                ]
                twin_accuracy = evaluated["source_twins"]["parent_macro_order_accuracy"]
                mixed_auc = evaluated["mixed_source_twins"]["within_response_auroc"][
                    "parent_macro_mean"
                ]
                record = {
                    "track": track,
                    "layer": layer,
                    "C": c,
                    "feature": "span_mean",
                    "selection_value": min(natural_auc, twin_accuracy, mixed_auc),
                    "natural_dev_auc": natural_auc,
                    "twin_dev_pair_accuracy": twin_accuracy,
                    "mixed_twin_dev_local_auc": mixed_auc,
                }
                result["candidates"].append(record)
                candidates.append((record, w, b, scores))
                print(json.dumps(record), flush=True)
    best = max(
        candidates,
        key=lambda item: (
            item[0]["selection_value"],
            item[0]["natural_dev_auc"],
            -item[0]["C"],
        ),
    )
    record, w, b, scores = best
    threshold = threshold_at_fpr(y[natural_dev], scores[natural_dev])
    result["selected"] = record
    result["selected_development"] = evaluate(
        rows, scores, natural_dev, twins_dev, threshold
    )
    # Old iteration-two test remains labelled development evidence. Its scores
    # are reported only after the new head is selected and are never optimized.
    prior_test = (splits == "test") & ~domain
    ids = np.asarray([r["id"] for r in rows])
    groups = np.asarray([r["group"] for r in rows])
    result["prior_iteration2_test_diagnostic"] = metrics(
        y[prior_test],
        scores[prior_test],
        ids[prior_test],
        groups[prior_test],
        threshold,
    )
    score_sets = {"selected_local3": scores}
    text_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=20000,
        sublinear_tf=True,
    )
    texts = np.asarray([r["text"] for r in rows])
    text_matrix = text_vectorizer.fit_transform(texts[natural_train])
    text_classifier = LogisticRegression(
        C=1.0, solver="liblinear", dual=True, max_iter=3000, random_state=20261014
    )
    text_classifier.fit(text_matrix, y[natural_train])
    if text_classifier.n_iter_.max() >= text_classifier.max_iter:
        raise ValueError("fixed text control failed to converge")
    fixed_text_scores = text_classifier.decision_function(
        text_vectorizer.transform(texts)
    )
    score_sets["fixed_word_tfidf"] = fixed_text_scores
    result["controls"]["fixed_word_tfidf"] = {
        "definition": "Word (1,2)-gram TFIDF, min_df=1, max_features=20000, sublinear_tf, logistic C=1.0; fit only natural training claims, with no hyperparameter selection. Portable model saved for fresh fixed-unit evaluation.",
        "development": evaluate(rows, fixed_text_scores, natural_dev, twins_dev),
        "prior_test": metrics(
            y[prior_test],
            fixed_text_scores[prior_test],
            ids[prior_test],
            groups[prior_test],
        ),
    }
    old_predictions = {
        (r["id"], r["span_index"]): r
        for r in map(json.loads, args.old_predictions.read_text().splitlines())
    }
    text_values = []
    for row in rows:
        key = (row["source_row_id"], row["original_span_index"])
        # Twin caches include both participant models, while a model-specific
        # old text control only covered its own participant. Only exact
        # matching known rows are scored; report the matched population.
        prediction = old_predictions.get(key)
        if prediction and prediction["text"] != row["text"]:
            if row["is_mixed_augmentation"]:
                prediction = None
            else:
                raise ValueError("frozen text control span does not match")
        text_values.append(
            prediction["scores"]["text_only_tfidf"] if prediction else np.nan
        )
    text_values = np.asarray(text_values)
    text_known = np.isfinite(text_values)
    result["controls"]["frozen_iteration2_text_only"] = {
        "population": "Old model-specific TFIDF scores on exact original span IDs; both members of every matched source pair share the same score. Other-participant source pairs are excluded from this text control only.",
        "predictions_sha256": sha256(args.old_predictions.read_text()),
        "development": evaluate(
            rows,
            text_values,
            natural_dev & text_known,
            twins_dev & text_known,
        ),
        "prior_test": metrics(
            y[prior_test],
            text_values[prior_test],
            ids[prior_test],
            groups[prior_test],
        ),
    }
    for track in TRACKS:
        chosen = max(
            (item for item in candidates if item[0]["track"] == track),
            key=lambda item: (
                item[0]["selection_value"],
                item[0]["natural_dev_auc"],
                -item[0]["C"],
            ),
        )
        score_sets[track] = chosen[3]
        result.setdefault("track_results", {})[track] = {
            "selected": chosen[0],
            "development": evaluate(rows, chosen[3], natural_dev, twins_dev),
            "prior_test": metrics(
                y[prior_test],
                chosen[3][prior_test],
                ids[prior_test],
                groups[prior_test],
            ),
        }
    old_selected = old["__contract__"]["selected_track"]
    head = old[old_selected]
    old_values = (
        np.stack([r[head["feature"]][LAYERS.index(head["layer"])] for r in rows])
        @ head["weight"].numpy()
        + head["bias"]
    )
    score_sets["frozen_iteration2_selected"] = old_values
    result["controls"]["frozen_iteration2_selected"] = {
        "track": old_selected,
        "development": evaluate(rows, old_values, natural_dev, twins_dev),
        "prior_test": metrics(
            y[prior_test], old_values[prior_test], ids[prior_test], groups[prior_test]
        ),
    }
    for field in ["response_broadcast", "published_dyl_broadcast"]:
        if all(r[field] is not None for r in rows):
            broadcast = np.asarray([r[field] for r in rows])
            score_sets[field] = broadcast
            result["controls"][field] = {
                "definition": "One complete-response score copied to every claim; within-response AUROC must be exactly 0.5. Source-pair accuracy alone does not establish localization.",
                "development": evaluate(rows, broadcast, natural_dev, twins_dev),
                "prior_test": metrics(
                    y[prior_test],
                    broadcast[prior_test],
                    ids[prior_test],
                    groups[prior_test],
                ),
            }
            for population in ["natural", "mixed_source_twins"]:
                value = result["controls"][field]["development"][population][
                    "within_response_auroc"
                ]["parent_macro_mean"]
                if value is not None and not np.isclose(value, 0.5):
                    raise ValueError("response broadcast spuriously localizes claims")
    for sign, name in [(1, "position"), (-1, "negative_position")]:
        values = sign * np.asarray([r["position"] for r in rows])
        score_sets[name] = values
        result["controls"][name] = evaluate(rows, values, natural_dev, twins_dev)
    rng = np.random.default_rng(2026101431)
    random_candidates = []
    selected_x = x[:, LAYERS.index(record["layer"])]
    for _ in range(32):
        random_weight = rng.normal(size=selected_x.shape[1])
        random_values = selected_x @ random_weight
        evaluated = evaluate(rows, random_values, natural_dev, twins_dev)
        a = evaluated["natural"]["within_response_auroc"]["parent_macro_mean"]
        t = evaluated["source_twins"]["parent_macro_order_accuracy"]
        m = evaluated["mixed_source_twins"]["within_response_auroc"][
            "parent_macro_mean"
        ]
        random_candidates.append(
            (min(a, t, m), random_values, evaluated, random_weight)
        )
    _, random_values, random_eval, random_weight = max(
        random_candidates, key=lambda r: r[0]
    )
    score_sets["best_of_32_random_directions"] = random_values
    result["controls"]["best_of_32_random_directions"] = {
        "development": random_eval,
        "prior_test": metrics(
            y[prior_test],
            random_values[prior_test],
            ids[prior_test],
            groups[prior_test],
        ),
    }
    heads = {
        "local3": {
            **record,
            "weight": torch.tensor(w),
            "bias": b,
            "threshold_dev_fpr10": threshold,
        },
        "random_direction": {
            "feature": "span_mean",
            "layer": record["layer"],
            "weight": torch.tensor(random_weight),
            "bias": 0.0,
            "selection": "Best of 32 seeded random directions by same three-metric development maximin.",
        },
        "__contract__": {
            k: manifests["natural"][k] for k in ["model", "model_revision", "layers"]
        },
    }
    heads["__contract__"].update(
        {
            "selected_track": "local3",
            "trainer_sha256": result["script_sha256"],
            "fresh_holdout_used": False,
        }
    )
    torch.save(heads, args.out / "heads.pt")
    joblib.dump(
        {
            "vectorizer": text_vectorizer,
            "classifier": text_classifier,
            "contract": heads["__contract__"],
        },
        args.out / "text_control.joblib",
    )
    (args.out / "predictions.jsonl").write_text(
        "".join(
            json.dumps(
                {k: v for k, v in row.items() if k not in ["span_mean", "span_last"]}
                | {"scores": {name: float(s[i]) for name, s in score_sets.items()}}
            )
            + "\n"
            for i, row in enumerate(rows)
        )
    )
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "selected": record,
                "development": result["selected_development"],
                "prior_test_diagnostic": result["prior_iteration2_test_diagnostic"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
