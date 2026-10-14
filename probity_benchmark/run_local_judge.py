"""Complete native Qwen judge generations, with reusable exact-prompt receipts."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import jsonschema
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .api import ANSWER, object_schema, strict_json
from .preflight import sha256
from .qwen_extract import mask_linear_padding


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--model-spec", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reuse", type=Path, nargs="*", default=[])
    ap.add_argument("--answer-only", action="store_true")
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError("new output required")
    spec = json.loads(args.model_spec.read_text())
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"])
    model = AutoModelForCausalLM.from_pretrained(
        spec["model"],
        revision=spec["revision"],
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
    ).eval()
    padding_handles = [
        layer.linear_attn.register_forward_pre_hook(
            mask_linear_padding, with_kwargs=True
        )
        for layer in model.model.layers
        if hasattr(layer, "linear_attn")
    ]
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    jobs = [json.loads(x) for x in args.jobs.read_text().splitlines()]
    if len({j["job_id"] for j in jobs}) != len(jobs):
        raise ValueError("duplicate job identity")
    contract = {
        "model": spec["model"],
        "revision": spec["revision"],
        "dtype": "bfloat16",
        "backend": "transformers native generate, unpadded, greedy, thinking disabled",
        "template_sha256": sha256(tokenizer.chat_template),
        "max_new_tokens": 8192,
        "runner_sha256": sha256(Path(__file__).read_text()),
        "answer_only": args.answer_only,
    }
    reusable = {}
    for root in args.reuse:
        if json.loads((root / "manifest.json").read_text())["contract"] != contract:
            raise ValueError("judge reuse contract mismatch")
        for r in map(json.loads, (root / "evaluations.jsonl").read_text().splitlines()):
            if r["valid"]:
                reusable[r["request_sha256"]] = r
    args.out.mkdir(parents=True)
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "contract": contract,
                "jobs_sha256": sha256(args.jobs.read_text()),
                "jobs_expected": len(jobs),
                "context_limit": model.config.get_text_config().max_position_embeddings,
                "claim_limit": "Native bf16 judge; not the same backend or constrained decoding as the API judges. Full prompts and complete generated JSON retained.",
            },
            indent=2,
        )
        + "\n"
    )
    results = []
    for job in jobs:
        text = tokenizer.apply_chat_template(
            job["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        if (
            tokenizer.decode(
                ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            != text
        ):
            raise ValueError("judge prompt failed exact tokenizer roundtrip")
        if len(ids) + 8192 > model.config.get_text_config().max_position_embeddings:
            raise ValueError("full judge context exceeds allowance")
        key = sha256(json.dumps({"contract": contract, "text": text}, sort_keys=True))
        metadata = {k: v for k, v in job.items() if k != "messages"}
        if key in reusable:
            old = reusable[key]
            result = {
                **old,
                **metadata,
                "correct": old["parsed"]["answer"] == job["gold"],
                "reused": True,
            }
        else:
            start = time.monotonic()
            with torch.inference_mode():
                output = model.generate(
                    input_ids=torch.tensor([ids], device=model.device),
                    attention_mask=torch.ones(
                        (1, len(ids)), device=model.device, dtype=torch.long
                    ),
                    max_new_tokens=8192,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            completion = output[0, len(ids) :].tolist()
            if output[0, : len(ids)].tolist() != ids:
                raise ValueError("generation changed input tokens")
            raw = tokenizer.decode(
                completion, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            finish = (
                "stop"
                if completion and completion[-1] in eos and len(completion) < 8192
                else "length"
            )
            result = {
                **metadata,
                "request_sha256": key,
                "input_tokens": len(ids),
                "completion_tokens": len(completion),
                "finish_reason": finish,
                "raw": raw,
                "valid": False,
                "duration_seconds": time.monotonic() - start,
                "reused": False,
            }
            try:
                if finish != "stop":
                    raise ValueError("incomplete judge generation")
                parsed = strict_json(raw)
                jsonschema.validate(
                    parsed,
                    object_schema(
                        {"answer": ANSWER}
                        if args.answer_only
                        else {"answer": ANSWER, "explanation": {"type": "string"}}
                    ),
                )
                result.update(
                    parsed=parsed, valid=True, correct=parsed["answer"] == job["gold"]
                )
            except Exception as exc:
                result["error"] = str(exc)
        results.append(result)
        with (args.out / "evaluations.jsonl").open("a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        reusable[key] = result
        print(
            json.dumps({k: v for k, v in result.items() if k not in ["raw", "parsed"]}),
            flush=True,
        )
        if not result["valid"]:
            raise ValueError("invalid judge completion; stopped, retained full receipt")
    summary = {
        "finish_reasons": dict(Counter(r["finish_reason"] for r in results)),
        "expected": len(jobs),
        "completed": len(results),
        "valid": sum(r["valid"] for r in results),
        "max_input_tokens": max(r["input_tokens"] for r in results),
        "max_completion_tokens": max(r["completion_tokens"] for r in results),
        "cells": [],
    }
    for kind in sorted({r["kind"] for r in results}):
        for participant in sorted(
            {r.get("participant_model", "") for r in results if r["kind"] == kind}
        ):
            for split in ["all", "dev", "test"]:
                selected = [
                    r
                    for r in results
                    if r["kind"] == kind
                    and r.get("participant_model", "") == participant
                    and (split == "all" or r["split"] == split)
                ]
                if selected:
                    summary["cells"].append(
                        {
                            "kind": kind,
                            "participant": participant,
                            "split": split,
                            "rows": len(selected),
                            "correct": sum(r["correct"] for r in selected),
                            "parent_groups": len({r["group"] for r in selected}),
                            "accuracy": sum(r["correct"] for r in selected)
                            / len(selected),
                        }
                    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
