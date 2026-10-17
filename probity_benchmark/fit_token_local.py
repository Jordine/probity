"""Train token MIL/ranking heads together from cached, verified claim bags."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F

from .fit_local_iteration import dev_value, metrics, threshold_at_fpr
from .preflight import sha256
from .qwen_extract import LAYERS, overlapping_tokens


def bag_losses(token_logits, lengths, targets, weights, units, objectives, epoch):
    """Only labelled claim tokens enter a loss; there is no background negative."""
    if not lengths or min(lengths) <= 0 or sum(lengths) != len(token_logits):
        raise ValueError("token bag accounting mismatch")
    sizes = torch.tensor(lengths, device=token_logits.device)
    mean = torch.segment_reduce(token_logits, "mean", lengths=sizes)
    maximum = torch.segment_reduce(token_logits, "max", lengths=sizes)
    minimum = torch.segment_reduce(token_logits, "min", lengths=sizes)
    neg_tokens = torch.segment_reduce(F.softplus(token_logits), "mean", lengths=sizes)
    unit_labels = list(zip(units, targets.detach().cpu().tolist()))
    alpha = 0.8 * max(0.0, min(1.0, (epoch - 15) / 44))
    losses = []
    for h, objective in enumerate(objectives):
        bce = F.binary_cross_entropy_with_logits(mean[:, h], targets, reduction="none")
        if objective == "mean_bce":
            specialized = bce
        elif objective == "legacy_max_min":
            specialized = torch.where(
                targets.bool(), F.softplus(-maximum[:, h]), F.softplus(minimum[:, h])
            )
        elif objective == "max_all_negative":
            specialized = torch.where(
                targets.bool(), F.softplus(-maximum[:, h]), neg_tokens[:, h]
            )
        elif objective == "within_response_rank":
            pairs = []
            for unit in set(units):
                pos = [
                    i
                    for i, (u, label) in enumerate(unit_labels)
                    if u == unit and label == 1
                ]
                neg = [
                    i
                    for i, (u, label) in enumerate(unit_labels)
                    if u == unit and label == 0
                ]
                if pos and neg:
                    pairs.append(
                        F.softplus(1 - (maximum[pos, h, None] - maximum[neg, h])).mean()
                    )
            ranking = (
                torch.stack(pairs).mean() if pairs else token_logits[:, h].sum() * 0
            )
            losses.append((1 - alpha) * (bce * weights).mean() + alpha * ranking)
            continue
        else:
            raise ValueError(objective)
        losses.append(((1 - alpha) * bce + alpha * specialized).mul(weights).mean())
    return torch.stack(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=Path, required=True)
    ap.add_argument("--pooled", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError("new output directory required")
    if not torch.cuda.is_available():
        raise ValueError("run token-head training on the task GPU")
    args.out.mkdir(parents=True)
    torch.set_num_threads(4)
    for directory in [args.tokens, args.pooled]:
        summary = json.loads((directory / "summary.json").read_text())
        if (
            summary["rows_expected"] != summary["rows_completed"]
            or len(list(directory.glob("*.pt"))) != summary["rows_expected"]
        ):
            raise ValueError("incomplete token or pooled feature cache")
    rows, bags = [], []
    for path in sorted(args.pooled.glob("*.pt")):
        cache = torch.load(path, map_location="cpu", weights_only=True)
        row = cache["row"]
        rows.append(row)
        for index, span in enumerate(row["spans"]):
            positions = overlapping_tokens(
                cache["response_offsets"], span["start"], span["end"]
            )
            bags.append(
                {
                    "id": row["id"],
                    "group": row["group"],
                    "split": row["split"],
                    "span_index": index,
                    "label": span["label"],
                    "positions": positions,
                    "mean": cache["span_mean"][:, index].numpy(),
                }
            )
    y = np.array([b["label"] for b in bags])
    units = np.array([b["id"] for b in bags])
    groups = np.array([b["group"] for b in bags])
    splits = np.array([b["split"] for b in bags])
    train, dev, test = (splits == name for name in ["train", "dev", "test"])
    for group in set(groups):
        if len(set(splits[groups == group])) != 1:
            raise ValueError("parent group crosses splits")
    if len(set(y[train])) != 2:
        raise ValueError("need both verified claim classes")
    by_row = {
        row["id"]: [i for i, bag in enumerate(bags) if bag["id"] == row["id"]]
        for row in rows
    }
    train_units = sorted(set(units[train]))
    settings = [
        (objective, lr)
        for objective in [
            "mean_bce",
            "legacy_max_min",
            "max_all_negative",
            "within_response_rank",
        ]
        for lr in [0.003, 0.01]
    ]
    candidates = []
    manifests = {
        k: json.loads((p / "manifest.json").read_text())
        for k, p in [("tokens", args.tokens), ("pooled", args.pooled)]
    }
    if manifests["pooled"]["base_manifest_sha256"] != sha256(
        (args.tokens / "manifest.json").read_text()
    ):
        raise ValueError("pooled features came from a different token cache")
    result = {
        "trainer_sha256": sha256(Path(__file__).read_text()),
        "manifests": manifests,
        "objectives": "Class-balanced mean BCE; legacy max-positive/min-negative control; annealed max-positive/all-known-negative; annealed within-response ranking. All eight heads per layer share one token-feature load and training pass.",
        "epochs": 60,
        "annealing": "15 epoch mean-BCE warmup, then linear to 80% specialized objective by epoch 59",
        "claim_limit": "False claim spans are weak bags, not tokenwise lie labels. Unlabelled text never becomes negative. Four held-out parent stories at most; silver annotations.",
        "candidates": [],
        "selected_per_objective": {},
    }
    for li, layer in enumerate(LAYERS):
        scaler = StandardScaler().fit(np.stack([b["mean"][li] for b in bags])[train])
        tensors = [None] * len(bags)
        for row in rows:
            ix = by_row[row["id"]]
            if not ix:
                continue
            raw = torch.load(
                args.tokens / (row["id"] + ".pt"), map_location="cpu", weights_only=True
            )
            if any(raw["row"][k] != row[k] for k in ["id", "messages", "response"]):
                raise ValueError("token/label cache mismatch")
            for i in ix:
                x = raw["token_features"][li, bags[i]["positions"]].float().numpy()
                tensors[i] = torch.tensor(
                    scaler.transform(x), device="cuda", dtype=torch.float32
                )
        torch.manual_seed(42)
        weight = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(5120, device="cuda")) for _ in settings]
        )
        bias = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros((), device="cuda")) for _ in settings]
        )
        optimizer = torch.optim.AdamW(
            [
                {"params": [w, b], "lr": lr}
                for w, b, (_, lr) in zip(weight, bias, settings)
            ],
            weight_decay=0.001,
        )
        class_weights = {
            label: len(y[train]) / (2 * int((y[train] == label).sum()))
            for label in [0, 1]
        }
        for epoch in range(60):
            order = list(train_units)
            random.Random(42 + epoch).shuffle(order)
            for start in range(0, len(order), 4):
                batch_units = order[start : start + 4]
                ix = [i for unit in batch_units for i in by_row[unit]]
                x = torch.cat([tensors[i] for i in ix])
                targets = torch.tensor(y[ix], device="cuda", dtype=torch.float32)
                sample_weights = torch.tensor(
                    [class_weights[int(y[i])] for i in ix], device="cuda"
                )
                logits = x @ torch.stack(list(weight)).T + torch.stack(list(bias))
                losses = bag_losses(
                    logits,
                    [len(tensors[i]) for i in ix],
                    targets,
                    sample_weights,
                    [units[i] for i in ix],
                    [s[0] for s in settings],
                    epoch,
                )
                regularizer = 0.01 * torch.stack([w.square().sum() for w in weight])
                loss = (losses + regularizer).sum()
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite token-head loss")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        all_scores = [[] for _ in settings]
        with torch.no_grad():
            for x in tensors:
                logits = x @ torch.stack(list(weight)).T + torch.stack(list(bias))
                for h, (objective, _) in enumerate(settings):
                    value = (
                        logits[:, h].mean()
                        if objective == "mean_bce"
                        else logits[:, h].max()
                    )
                    all_scores[h].append(float(value))
        for h, (objective, lr) in enumerate(settings):
            scores = np.array(all_scores[h])
            value = dev_value(y[dev], scores[dev], units[dev], groups[dev])
            w = weight[h].detach().cpu().numpy() / scaler.scale_
            b = float(bias[h].detach().cpu() - w @ scaler.mean_)
            record = {
                "layer": layer,
                "objective": objective,
                "lr": lr,
                "dev_parent_macro_local_auroc": value,
                "pooling": "mean" if objective == "mean_bce" else "max",
            }
            result["candidates"].append(record)
            candidates.append((record, w, b, scores))
            print(json.dumps(record), flush=True)
        del tensors, weight, bias, optimizer
        torch.cuda.empty_cache()
    selected = max(candidates, key=lambda c: c[0]["dev_parent_macro_local_auroc"])
    result["selected"] = selected[0]
    heads = {}
    score_sets = {}
    for objective in [
        "mean_bce",
        "legacy_max_min",
        "max_all_negative",
        "within_response_rank",
    ]:
        chosen, w, b, scores = max(
            (c for c in candidates if c[0]["objective"] == objective),
            key=lambda c: c[0]["dev_parent_macro_local_auroc"],
        )
        threshold = threshold_at_fpr(y[dev], scores[dev])
        heads[objective] = {
            **chosen,
            "weight": torch.tensor(w),
            "bias": b,
            "threshold_dev_fpr10": threshold,
        }
        result["selected_per_objective"][objective] = {
            "selected": chosen,
            "dev": metrics(y[dev], scores[dev], units[dev], groups[dev], threshold),
            "test": metrics(
                y[test], scores[test], units[test], groups[test], threshold
            ),
        }
        score_sets[objective] = scores
    predictions = [
        {k: bag[k] for k in ["id", "group", "split", "span_index", "label"]}
        | {"scores": {name: float(scores[i]) for name, scores in score_sets.items()}}
        for i, bag in enumerate(bags)
    ]
    (args.out / "claim_predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in predictions)
    )
    heads["__contract__"] = {
        k: manifests["tokens"][k] for k in ["model", "model_revision", "layers"]
    }
    heads["__contract__"]["selected_objective"] = selected[0]["objective"]
    torch.save(heads, args.out / "heads.pt")
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ["candidates", "manifests"]},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
