"""Position-matched audit of previously evaluated silver local-claim predictions.

All iteration-two partitions are development evidence for this new diagnostic.
No fitted head or new held-out selection is made here.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

from .preflight import sha256


def parent_macro(pairs):
    by_parent = defaultdict(list)
    for parent, value in pairs:
        by_parent[parent].append(value)
    return {
        "pairs": len(pairs),
        "parents": len(by_parent),
        "parent_macro_pair_accuracy": (
            sum(sum(v) / len(v) for v in by_parent.values()) / len(by_parent)
            if by_parent
            else None
        ),
    }


def audit(rows, prediction_rows, split=None):
    indexed = {(r["id"], i): (r, s) for r in rows for i, s in enumerate(r["spans"])}
    groups = defaultdict(list)
    for prediction in prediction_rows:
        row, span = indexed[prediction["id"], prediction["span_index"]]
        if prediction["y"] != span["label"]:
            raise ValueError("prediction label does not match canonical corpus")
        if split and row["split"] != split:
            continue
        groups[row["id"]].append(
            {
                "row": row,
                "span": span,
                "prediction": prediction,
                "position": span["start"] / len(row["response"]),
            }
        )
    all_rows = [r for values in groups.values() for r in values]
    output = {
        "claims": len(all_rows),
        "false_claims": sum(r["span"]["label"] for r in all_rows),
        "responses": len(groups),
        "parents": len({r["row"]["group"] for r in all_rows}),
        "false_by_advocacy": {},
        "pair_strata": {},
        "definition": "Within-response positive-negative ranking; each parent has equal total weight. Close pairs have normalized raw-response start-position distance at most the specified bound. This is a post-outcome diagnostic, not a new holdout.",
    }
    for advocacy in [False, True]:
        subset = [
            r for r in all_rows if r["row"]["assigned_advocacy_correct"] is advocacy
        ]
        output["false_by_advocacy"][str(advocacy)] = {
            "claims": len(subset),
            "false_claims": sum(r["span"]["label"] for r in subset),
        }
    score_names = sorted(all_rows[0]["prediction"]["scores"]) if all_rows else []
    for limit in [None, 0.2, 0.1]:
        label = "all" if limit is None else f"position_distance_le_{limit}"
        by_score = {
            name: [] for name in score_names + ["position", "negative_position"]
        }
        for values in groups.values():
            false = [r for r in values if r["span"]["label"] == 1]
            true = [r for r in values if r["span"]["label"] == 0]
            for pos in false:
                for neg in true:
                    if (
                        limit is not None
                        and abs(pos["position"] - neg["position"]) > limit
                    ):
                        continue
                    for name in by_score:
                        if name == "position":
                            a, b = pos["position"], neg["position"]
                        elif name == "negative_position":
                            a, b = -pos["position"], -neg["position"]
                        else:
                            a = pos["prediction"]["scores"][name]
                            b = neg["prediction"]["scores"][name]
                        by_score[name].append(
                            (pos["row"]["group"], float(a > b) + 0.5 * (a == b))
                        )
        output["pair_strata"][label] = {
            name: parent_macro(pairs) for name, pairs in by_score.items()
        }
    output["automatic_lexical_diagnostics"] = {}
    patterns = {
        "attribution_words": r"\b(?:said|says|told|asks?|asked|repl(?:y|ies|ied)|explains?|explained|states?|stated|narrator|speaker|responds?|responded)\b",
        "explicit_quotation": r"[\"“”‘’]",
    }
    for category, pattern in patterns.items():
        subset = [
            r
            for r in all_rows
            if re.search(
                pattern, r["span"].get("display_text", r["span"]["text"]), re.I
            )
        ]
        output["automatic_lexical_diagnostics"][category] = {
            "claims": len(subset),
            "false_claims": sum(r["span"]["label"] for r in subset),
            "limitation": "Regex description of wording; not a manually verified claim taxonomy.",
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("immutable new output required")
    corpus = args.root / "adjudicated_reviewed/corpus.jsonl"
    rows = [json.loads(line) for line in corpus.read_text().splitlines()]
    result = {
        "corpus_sha256": sha256(corpus.read_text()),
        "rows": len(rows),
        "models": {},
    }
    for model in ["qwen35", "qwen38"]:
        path = args.root / f"fitted_local_{model}_adjudicated/claim_predictions.jsonl"
        predictions = [json.loads(line) for line in path.read_text().splitlines()]
        result["models"][model] = {"predictions_sha256": sha256(path.read_text())}
        for split in [None, "train", "dev", "test"]:
            result["models"][model][split or "all_prior_development"] = audit(
                rows, predictions, split
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.out),
                "corpus_rows": len(rows),
                "output_sha256": sha256(args.out.read_text()),
            }
        )
    )


if __name__ == "__main__":
    main()
