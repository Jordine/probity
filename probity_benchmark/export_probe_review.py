"""Export full arguments, token scores and annotation-independent local displays."""

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import torch

from .evaluate_debate_transfer import argument_mapping, segments
from .preflight import load_problems, sha256
from .qwen_extract import LAYERS, overlapping_tokens


def interval_scores(
    raw_offsets, token_values, mapping, start, end, mean_pool, token_pool
):
    raw_start, raw_end = mapping[start][0], mapping[end - 1][1]
    ix = overlapping_tokens(raw_offsets, raw_start, raw_end)
    if not ix:
        raise ValueError("display interval has no response tokens")
    a, b = token_values["pooled"][ix], token_values["token"][ix]
    return {
        "start": start,
        "end": end,
        "raw_start": raw_start,
        "raw_end": raw_end,
        "token_indices": ix,
        "pooled_head": float(a.mean() if mean_pool else a[-1]),
        "token_head": float(b.mean() if token_pool == "mean" else b.max()),
        "token_mean": float(b.mean()),
        "token_max": float(b.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=Path, required=True)
    ap.add_argument("--pooled-heads", type=Path, required=True)
    ap.add_argument("--token-heads", type=Path, required=True)
    ap.add_argument("--response-heads", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--episodes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError("new output required")
    torch.set_num_threads(4)
    manifest = json.loads((args.tokens / "manifest.json").read_text())
    summary = json.loads((args.tokens / "summary.json").read_text())
    paths = sorted(args.tokens.glob("*.pt"))
    if (
        len(paths) != summary["rows_completed"]
        or len(paths) != summary["rows_expected"]
    ):
        raise ValueError("incomplete token cache")
    sets = {
        name: torch.load(path, map_location="cpu", weights_only=True)
        for name, path in [
            ("pooled", args.pooled_heads),
            ("token", args.token_heads),
            ("response", args.response_heads),
        ]
    }
    for heads in sets.values():
        for k in ["model", "model_revision", "layers"]:
            if heads["__contract__"][k] != manifest[k]:
                raise ValueError("head/model mismatch")
    chosen = {
        "pooled": sets["pooled"]["__contract__"]["selected_track"],
        "token": sets["token"]["__contract__"]["selected_objective"],
        "response": sets["response"]["__contract__"]["selected_feature"],
    }
    local = {name: sets[name][chosen[name]] for name in ["pooled", "token"]}
    fitted_predictions = {}
    for name, path in [("pooled", args.pooled_heads), ("token", args.token_heads)]:
        predictions = list(
            map(
                json.loads,
                (path.parent / "claim_predictions.jsonl").read_text().splitlines(),
            )
        )
        fitted_predictions[name] = {
            (r["id"], r["span_index"]): r["scores"][chosen[name]] for r in predictions
        }
    claim_checks, maximum_score_error = 0, 0.0
    labels = {
        r["id"]: r for r in map(json.loads, args.annotations.read_text().splitlines())
    }
    episodes = {
        e["episode_id"]: e
        for e in (
            json.loads(p.read_text()) for p in args.episodes.glob("episode_*.json")
        )
    }
    problems, audit = load_problems(args.dataset)
    if audit["errors"]:
        raise ValueError("source dataset invalid")
    problems = {p.problem_id: p for p in problems}
    args.out.mkdir(parents=True)
    records, pages = [], []
    for path in paths:
        cache = torch.load(path, map_location="cpu", weights_only=True)
        r = cache["row"]
        annotation = labels[r["id"]]
        if any(r[k] != annotation[k] for k in ["id", "messages", "response"]):
            raise ValueError("annotation/token replay mismatch")
        argument, mapping = argument_mapping(r["response"])
        raw_offsets = cache["response_offsets"]
        token_values = {}
        for name, h in local.items():
            token_values[name] = (
                cache["token_features"][LAYERS.index(h["layer"])].double().numpy()
                @ h["weight"].double().numpy()
                + h["bias"]
            )
            if (
                len(token_values[name]) != len(raw_offsets)
                or not np.isfinite(token_values[name]).all()
            ):
                raise ValueError("token score accounting or finiteness")

        reviewed_claims = []
        for i, span in enumerate(annotation["spans"]):
            positions = overlapping_tokens(raw_offsets, span["start"], span["end"])
            a, b = token_values["pooled"][positions], token_values["token"][positions]
            values = {
                "pooled": float(
                    a.mean() if local["pooled"]["feature"] == "span_mean" else a[-1]
                ),
                "token": float(
                    b.mean() if local["token"]["pooling"] == "mean" else b.max()
                ),
            }
            for name, value in values.items():
                expected = fitted_predictions[name][(r["id"], i)]
                error = abs(value - expected)
                maximum_score_error = max(maximum_score_error, error)
                if error > 1e-3 * max(1.0, abs(expected)):
                    raise ValueError(
                        "displayed claim score differs from fitted prediction"
                    )
                claim_checks += 1
            reviewed_claims.append({**span, "raw_scores": values})

        def intervals(clauses=False):
            output = []
            for start, end in segments(argument, clauses):
                value = interval_scores(
                    raw_offsets,
                    token_values,
                    mapping,
                    start,
                    end,
                    local["pooled"]["feature"] == "span_mean",
                    local["token"]["pooling"],
                )
                output.append({**value, "text": argument[start:end]})
            if "".join(x["text"] for x in output) != argument:
                raise ValueError("display omitted text")
            return output

        sentence, clause = intervals(), intervals(True)
        whole = interval_scores(
            raw_offsets,
            token_values,
            mapping,
            0,
            len(argument),
            local["pooled"]["feature"] == "span_mean",
            local["token"]["pooling"],
        )
        response_scores = {}
        for name, h in sets["response"].items():
            if name.startswith("__"):
                continue
            x = (
                cache["dyl_no"].float() - cache["dyl_yes"].float()
                if name == "dyl_difference"
                else cache[name].float()
            )
            response_scores["adapted_" + name] = float(
                x[LAYERS.index(h["layer"])].double() @ h["weight"].double() + h["bias"]
            )
        if cache["fibs_contrast"] is not None:
            response_scores["published_fibs_dyl"] = float(cache["fibs_contrast"])
        token_rows = [
            {
                "index": i,
                "raw_start": int(a),
                "raw_end": int(b),
                "text": r["response"][a:b],
                **{n: float(v[i]) for n, v in token_values.items()},
            }
            for i, (a, b) in enumerate(raw_offsets)
        ]
        # Every 8-token window inside the argument is exported, with no label-based selection.
        arg_ix = overlapping_tokens(raw_offsets, mapping[0][0], mapping[-1][1])
        windows = []
        width = min(8, len(arg_ix))
        for start in range(len(arg_ix) - width + 1):
            ix = arg_ix[start : start + width]
            a, b = max(raw_offsets[ix[0]][0], mapping[0][0]), min(
                raw_offsets[ix[-1]][1], mapping[-1][1]
            )
            windows.append(
                {
                    "raw_start": int(a),
                    "raw_end": int(b),
                    "text": r["response"][a:b],
                    "token_mean": float(token_values["token"][ix].mean()),
                    "token_max": float(token_values["token"][ix].max()),
                }
            )
        e = episodes[r["source_episode"]]
        record = {
            "id": r["id"],
            "group": r["group"],
            "split": r["split"],
            "episode_id": r["source_episode"],
            "turn_index": r["turn_index"],
            "participant_model": r["participant_model"],
            "raw_response": r["response"],
            "argument": argument,
            "sentence_scores": sentence,
            "clause_scores": clause,
            "whole_argument_local_scores": whole,
            "whole_response_scores": response_scores,
            "token_scores": token_rows,
            "eight_token_windows": windows,
            "reviewed_claims": reviewed_claims,
            "display_characters": sum(len(s["text"]) for s in sentence),
            "raw_response_sha256": sha256(r["response"]),
        }
        records.append(record)
        low, high = np.quantile([s["token_head"] for s in sentence], [0.1, 0.9])
        shaded = []
        for s in sentence:
            intensity = float(
                np.clip((s["token_head"] - low) / max(high - low, 1e-12), 0, 1)
            )
            color = f"rgba(215,75,65,{0.06 + 0.38 * intensity:.3f})"
            tooltip = html.escape(
                json.dumps(
                    {
                        k: s[k]
                        for k in ["token_head", "pooled_head", "raw_start", "raw_end"]
                    }
                ),
                quote=True,
            )
            shaded.append(
                f'<span style="background:{color}" title="{tooltip}">{html.escape(s["text"])}</span>'
            )
        claim_rows = "".join(
            f'<tr><td>{html.escape(s.get("display_text",s["text"]))}</td><td>{"False" if s["label"] else "True"} (silver)</td><td>{html.escape(json.dumps(s["raw_scores"]))}</td></tr>'
            for s in reviewed_claims
        )
        pages.append(
            f'<section id="{html.escape(r["id"])}"><h2>{html.escape(r["id"])}</h2><p>{html.escape(r["split"])} · {html.escape(r["participant_model"])} · speaker {html.escape(e["turns"][r["turn_index"]]["speaker"])}</p><div class="argument">{"".join(shaded)}</div><details><summary>Separate whole-response scores</summary><pre>{html.escape(json.dumps(response_scores,indent=2))}</pre></details><details><summary>Full JSON completion</summary><pre>{html.escape(r["response"])}</pre></details><details><summary>Reviewed claims; all other text is unlabelled</summary><table>{claim_rows}</table></details></section>'
        )
    with (args.out / "all_turn_scores.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    worlds = sorted({episodes[r["episode_id"]]["problem_id"] for r in records})
    sources = "".join(
        f"<details><summary>Complete source {pid}</summary><pre>{html.escape(problems[pid].story)}</pre></details>"
        for pid in worlds
    )
    document = (
        '<!doctype html><html><meta charset="utf-8"><title>Probity complete probe review</title><style>body{max-width:1000px;margin:auto;padding:2rem;font:16px/1.55 system-ui}pre,.argument{white-space:pre-wrap;overflow-wrap:anywhere}section{margin:3rem 0}td{padding:.5rem;border-bottom:1px solid #ddd}details{margin:1rem 0}</style><h1>Complete probe review</h1><p>All arguments are shown in full. Sentence shading is relative token-head score within each response, not a factual verdict or calibrated probability. Sentence, clause, whole-argument and every eight-token window were computed independently of reviewed claim selection. Whole-response and DYL scores remain separate; they cannot localize individual lies. Scores use native-model replay of API text. Claim labels are silver, not exhaustive human ground truth.</p>'
        + sources
        + "".join(pages)
        + "</html>"
    )
    (args.out / "review.html").write_text(document)
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "model_contract": {
                    k: manifest[k] for k in ["model", "model_revision", "layers"]
                },
                "selected_heads": chosen,
                "rows_expected": len(paths),
                "rows_exported": len(records),
                "script_sha256": sha256(Path(__file__).read_text()),
                "annotations_sha256": sha256(args.annotations.read_text()),
                "artifacts": {
                    p.name: {
                        "bytes": p.stat().st_size,
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    }
                    for p in args.out.iterdir()
                    if p.is_file()
                },
                "checks": {
                    "all_response_token_scores": True,
                    "all_argument_characters_displayed": True,
                    "all_sentence_and_clause_partitions_exact": True,
                    "no_dyl_broadcast_to_tokens": True,
                    "fitted_claim_score_comparisons": claim_checks,
                    "maximum_claim_score_absolute_error": maximum_score_error,
                    "claim_score_tolerance": "1e-3 times max(1, absolute fitted score), for float32 pooling/folded-head arithmetic",
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {"rows": len(records), "artifacts": [p.name for p in args.out.iterdir()]}
        )
    )


if __name__ == "__main__":
    main()
