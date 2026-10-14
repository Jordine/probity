"""Complete, unpadded offline features with both DYL arms in one model lifetime."""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import torch

from .evaluate_debate_transfer import argument_mapping, segments
from .preflight import sha256
from .qwen_extract import (
    LAYERS,
    MODEL,
    REVISION,
    Runtime,
    overlapping_tokens,
    serialize,
)


def intervals(row):
    if row["source"] != "api_debate_silver":
        return [], None
    argument, mapping = argument_mapping(row["response"])
    result = []
    for kind, clauses in [("sentence", False), ("phrase", True)]:
        for a, b in segments(argument, clauses=clauses):
            result.append(
                {
                    "kind": kind,
                    "text": argument[a:b],
                    "start": mapping[a][0],
                    "end": mapping[b - 1][1],
                }
            )
    return result, (mapping[0][0], mapping[-1][1])


def pool(response_acts, offsets, spans):
    means, lasts, lengths = [], [], []
    for span in spans:
        pos = overlapping_tokens(offsets, span["start"], span["end"])
        means.append(response_acts[:, pos].float().mean(1))
        lasts.append(response_acts[:, pos[-1]].float())
        lengths.append(len(pos))
    return (
        torch.stack(means, 1) if means else None,
        torch.stack(lasts, 1) if lasts else None,
        lengths,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cache-tokens", action="store_true")
    ap.add_argument("--model", choices=["qwen3.5", "qwen3.8"], default="qwen3.5")
    args = ap.parse_args()
    raw = args.corpus.read_bytes()
    rows = [json.loads(x) for x in raw.decode().splitlines()]
    if len(raw) != args.corpus.stat().st_size or len({r["id"] for r in rows}) != len(
        rows
    ):
        raise ValueError("corpus accounting failed")
    if args.out.exists() and not args.resume:
        raise ValueError("new output directory required")
    args.out.mkdir(parents=True, exist_ok=True)
    model_id, revision = (
        (MODEL, REVISION)
        if args.model == "qwen3.5"
        else ("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
    )
    runtime = Runtime(model_id, revision)
    from probe_inference import load_probe_from_hub
    from probe_inference.load import WEIGHTS_REVISION

    probe = (
        load_probe_from_hub("qwen3.5-27b/linear") if args.model == "qwen3.5" else None
    )
    if probe is not None and (probe.used_layers != LAYERS or probe.d_model != 5120):
        raise ValueError("published probe contract mismatch")
    manifest = {
        "model": model_id,
        "model_revision": revision,
        "layers": LAYERS,
        "dtype": "bfloat16",
        "corpus_sha256": hashlib.sha256(raw).hexdigest(),
        "corpus_bytes": len(raw),
        "expected_rows": len(rows),
        "extractor_sha256": sha256(Path(__file__).read_text()),
        "runtime_sha256": sha256(
            Path(__file__).with_name("qwen_extract.py").read_text()
        ),
        "published_weights_revision": WEIGHTS_REVISION if probe is not None else None,
        "published_probe_available": probe is not None,
        "gpu": torch.cuda.get_device_name(0),
        "allow_bf16_reduced_precision_reduction": False,
        "cache_complete_response_tokens": args.cache_tokens,
        "chat_template_sha256": sha256(runtime.tokenizer.chat_template),
        "dyl_contrast": "No. score minus Yes. score, read at period before EOT",
        "scope": "Unpadded teacher-forced API-text replay. No live-generation or padded-execution equivalence claim. All contexts and responses complete.",
        "pooling": "Span mean and last overlapping token. Response mean and argument mean exclude no response text; argument mean excludes JSON scaffolding. Phrase units are punctuation-derived, not guaranteed assertions.",
        "versions": {
            n: importlib.metadata.version(n)
            for n in [
                "torch",
                "transformers",
                "flash-linear-attention",
                "causal-conv1d",
            ]
        },
    }
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("resume contract changed")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    alignment = runtime.alignment_check()
    (args.out / "alignment.json").write_text(json.dumps(alignment, indent=2) + "\n")
    if not alignment["passed"]:
        raise ValueError("short-context token alignment failed")
    longest = max(
        rows, key=lambda r: len(json.dumps(r["messages"])) + len(r["response"])
    )
    if len(json.dumps(longest["messages"])) > 4096:
        long_check = runtime.alignment_check(longest["messages"])
        (args.out / "long_alignment.json").write_text(
            json.dumps(long_check, indent=2) + "\n"
        )
    started = time.monotonic()
    for row in rows:
        dest = args.out / (row["id"] + ".pt")
        if dest.exists():
            saved = torch.load(dest, map_location="cpu", weights_only=True)
            if saved["row"] != row:
                raise ValueError("resume row differs")
            continue
        yes = serialize(runtime.tokenizer, row["messages"], row["response"], "Yes.")
        no = serialize(runtime.tokenizer, row["messages"], row["response"], "No.")
        if yes["ids"][: yes["followup_start"]] != no["ids"][: no["followup_start"]]:
            raise ValueError("DYL arm changed original action context")
        acts, _ = runtime.forward(yes["ids"])
        pos = yes["response_positions"]
        response = torch.stack([acts[layer][pos] for layer in LAYERS])
        offsets = [
            (
                yes["offsets"][i][0] - yes["response_start"],
                yes["offsets"][i][1] - yes["response_start"],
            )
            for i in pos
        ]
        for span in row["spans"]:
            if row["response"][span["start"] : span["end"]] != span["text"]:
                raise ValueError("labelled span differs from complete response")
        span_mean, span_last, span_lengths = pool(response, offsets, row["spans"])
        displayed, arg_range = intervals(row)
        segment_mean, segment_last, segment_lengths = pool(response, offsets, displayed)
        argument_mean = (
            response.float().mean(1)
            if arg_range is None
            else response[:, overlapping_tokens(offsets, *arg_range)].float().mean(1)
        )
        mask = torch.zeros(len(yes["ids"]), dtype=torch.bool)
        mask[yes["dyl_position"]] = True
        yes_score = probe.score(acts, mask) if probe is not None else None
        yes_feature = torch.stack(
            [acts[layer][yes["dyl_position"]] for layer in LAYERS]
        )
        del acts
        no_acts, _ = runtime.forward(no["ids"])
        mask = torch.zeros(len(no["ids"]), dtype=torch.bool)
        mask[no["dyl_position"]] = True
        no_score = probe.score(no_acts, mask) if probe is not None else None
        no_feature = torch.stack(
            [no_acts[layer][no["dyl_position"]] for layer in LAYERS]
        )
        # Causality check: changing a future forced reply must not materially
        # change the original action features. Shapes can alter bf16 kernels.
        arm_delta = []
        for li, layer in enumerate(LAYERS):
            other = no_acts[layer][no["response_positions"]].float()
            ref = response[li].float()
            arm_delta.append(
                float((other - ref).square().mean().sqrt() / ref.square().mean().sqrt())
            )
        if max(arm_delta) > 0.03:
            raise ValueError(
                "future DYL arm materially changed original action features"
            )
        del no_acts
        cache = {
            "id": row["id"],
            "row": row,
            "input_tokens": len(yes["ids"]),
            "response_tokens": len(pos),
            "response_offsets": offsets,
            "response_mean": response.float().mean(1),
            "token_features": response if args.cache_tokens else None,
            "argument_mean": argument_mean,
            "span_mean": span_mean,
            "span_last": span_last,
            "span_lengths": span_lengths,
            "segments": displayed,
            "segment_mean": segment_mean,
            "segment_last": segment_last,
            "segment_lengths": segment_lengths,
            "dyl_yes": yes_feature,
            "dyl_no": no_feature,
            "fibs_yes": yes_score,
            "fibs_no": no_score,
            "fibs_contrast": no_score - yes_score if probe is not None else None,
            "dyl_causality_relative_rms": arm_delta,
            "serialized_yes_sha256": sha256(yes["text"]),
            "serialized_no_sha256": sha256(no["text"]),
        }
        tmp = dest.with_suffix(".tmp")
        torch.save(cache, tmp)
        tmp.replace(dest)
        receipt = {
            "id": row["id"],
            "input_tokens": len(yes["ids"]),
            "response_tokens": len(pos),
            "spans": len(row["spans"]),
            "termination": "teacher_forced_complete_sequence",
            "elapsed_seconds": time.monotonic() - started,
            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            "bytes": dest.stat().st_size,
        }
        with (args.out / "receipts.jsonl").open("a") as f:
            f.write(json.dumps(receipt) + "\n")
        print(json.dumps(receipt), flush=True)
    receipts = [
        json.loads(x) for x in (args.out / "receipts.jsonl").read_text().splitlines()
    ]
    if len(receipts) != len(rows) or {r["id"] for r in receipts} != {
        r["id"] for r in rows
    }:
        raise ValueError("incomplete extraction or duplicate receipts")
    summary = {
        "termination_counts": dict(Counter(r["termination"] for r in receipts)),
        "rows_expected": len(rows),
        "rows_completed": len(receipts),
        "context_tokens_max": max(r["input_tokens"] for r in receipts),
        "bytes": sum(r["bytes"] for r in receipts),
        "scope": manifest["scope"],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
