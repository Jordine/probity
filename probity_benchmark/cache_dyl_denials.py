"""Cache the No. arm once and contrast it with saved Yes. scores for DYL."""

import argparse
import json
from pathlib import Path

import torch

from .preflight import sha256
from .qwen_extract import FOLLOWUP, LAYERS, MODEL, REVISION, Runtime, serialize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError("new output directory required")
    base = json.loads((args.features / "manifest.json").read_text())
    summary = json.loads((args.features / "summary.json").read_text())
    if (
        base["model"] != MODEL
        or base["model_revision"] != REVISION
        or base["dyl_prefill"] != "Yes."
        or base["dyl_question"] != FOLLOWUP
    ):
        raise ValueError("denial arm must match the saved action context and model")
    if summary["rows_completed"] != summary["rows_expected"]:
        raise ValueError("source extraction incomplete")
    args.out.mkdir(parents=True)
    runtime = Runtime()
    from probe_inference import load_probe_from_hub

    probe = load_probe_from_hub("qwen3.5-27b/linear")
    manifest = {
        "base_manifest_sha256": sha256((args.features / "manifest.json").read_text()),
        "source_sha256": sha256(Path(__file__).read_text()),
        "question": FOLLOWUP,
        "prefill": "No.",
        "read": "period before final EOT",
        "contrast": "score(No.) minus score(Yes.)",
        "score_target": "higher contrast means the preceding action is more likely false-labelled",
        "derivation": "The FIBS label is falsehood in the final completion. No. falsely denies a preceding lie; Yes. admits it. Paper section 4.2 and appendix L.1.",
        "expected_rows": summary["rows_expected"],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    completed = 0
    for path in sorted(args.features.glob("*.pt")):
        cache = torch.load(path, map_location="cpu", weights_only=True)
        row = cache["row"]
        s = serialize(
            runtime.tokenizer, row["messages"], row["response"], prefill="No."
        )
        yes_position = cache["dyl_position"]
        no_position = s["dyl_position"]
        if s["ids"][: no_position - 1] != cache["input_ids"][: yes_position - 1]:
            raise ValueError("No./Yes. arms differ before their forced answer")
        acts, _ = runtime.forward(s["ids"])
        mask = torch.zeros(len(s["ids"]), dtype=torch.bool)
        mask[no_position] = True
        no_score = probe.score(acts, mask)
        record = {
            "id": row["id"],
            "base_serialized_sha256": cache["serialized_text_sha256"],
            "dyl_features": torch.stack([acts[layer][no_position] for layer in LAYERS]),
            "input_ids": s["ids"],
            "dyl_position": no_position,
            "prefill": "No.",
            "fibs_no": no_score,
            "fibs_yes": cache["fibs_score"],
            "fibs_contrast": no_score - cache["fibs_score"],
        }
        torch.save(record, args.out / path.name)
        completed += 1
        print(
            json.dumps(
                {
                    "completed": completed,
                    "id": row["id"],
                    "fibs_no": no_score,
                    "fibs_yes": cache["fibs_score"],
                    "fibs_contrast": record["fibs_contrast"],
                }
            ),
            flush=True,
        )
    if completed != summary["rows_expected"]:
        raise ValueError("denial cache incomplete")
    (args.out / "summary.json").write_text(
        json.dumps(
            {
                "completed": completed,
                "expected": summary["rows_expected"],
                "termination": "complete_teacher_forced_sequences",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
