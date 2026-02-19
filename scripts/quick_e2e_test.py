"""
Quick end-to-end test: Train simple probes on NTML → eval on ai_liar + BashBench + APPS.

All inline, no probe format dependencies. Uses sklearn LogReg with forward hooks.

Usage:
    cd /root/probity
    PYTHONPATH=/root/probity python scripts/quick_e2e_test.py \
        --model_name Qwen/Qwen2.5-14B-Instruct \
        --ntml_path data/NTML-datasets/5T2L_300_tagged_direct_liar/5T2L_300samples_shuffled_explicit_direct_liar_prefix.json \
        --ai_liar_path data/deception_detection/ai_liar.jsonl \
        --bashbench_jsonl bashbench_test/bashbench_generated.jsonl \
        --apps_jsonl apps_test/apps_generated.jsonl \
        --layers 12 24 36 \
        --max_ntml 50 --max_eval 50
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Activation collection via hooks
# ============================================================

class ActivationCollector:
    def __init__(self, model, tokenizer, layers):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers
        self.captured = {}
        self.hooks = []

        for l in layers:
            self.hooks.append(
                model.model.layers[l].register_forward_hook(self._make_hook(l))
            )

    def _make_hook(self, l):
        def hook(module, input, out):
            self.captured[l] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook

    def get_activations(self, text, max_length=1536):
        enc = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True,
                             truncation=True, max_length=max_length)
        offsets = enc["offset_mapping"][0]
        inputs = {k: v.to(self.model.device) for k, v in enc.items() if k != "offset_mapping"}
        self.captured.clear()
        with torch.no_grad():
            self.model(**inputs)
        hidden = {l: self.captured[l][0].float().cpu() for l in self.layers}
        return hidden, offsets

    def cleanup(self):
        for h in self.hooks:
            h.remove()


def find_response_tokens(text, response_text, offsets):
    start = text.rfind(response_text)
    if start == -1:
        start = text.rfind(response_text[:200])
    if start == -1:
        return []
    end = start + len(response_text)
    return [i for i, (s, e) in enumerate(offsets.tolist()) if s >= start and e <= end and e > s]


def find_assistant_tokens(text, offsets, model_type="qwen"):
    """Find final assistant message tokens."""
    if model_type == "qwen":
        marker = "<|im_start|>assistant"
        end_marker = "<|im_end|>"
    else:
        marker = "<|start_header_id|>assistant<|end_header_id|>"
        end_marker = "<|eot_id|>"

    last = text.rfind(marker)
    if last == -1:
        return []
    content_start = last + len(marker)
    # Skip newline after marker
    while content_start < len(text) and text[content_start] in '\n':
        content_start += 1
    end_pos = text.find(end_marker, content_start)
    content_end = end_pos if end_pos != -1 else len(text)

    return [i for i, (s, e) in enumerate(offsets.tolist())
            if s >= content_start and e <= content_end and e > s]


# ============================================================
# Phase 1: Collect NTML activations + train probes
# ============================================================

def train_probes_on_ntml(collector, ntml_data, layers, max_samples=50):
    """Train simple LogReg probes on NTML lie vs truth activations."""
    print(f"\n{'='*60}")
    print(f"PHASE 1: Training probes on {min(len(ntml_data), max_samples)} NTML samples")
    print(f"{'='*60}")

    # Collect mean activations for lie and truth versions
    lie_acts = {l: [] for l in layers}
    truth_acts = {l: [] for l in layers}
    n_used = 0

    for sample in tqdm(ntml_data[:max_samples], desc="Collecting NTML activations"):
        response_text = sample["lie_version"]["model"]

        # Lie version
        lie_msgs = [
            {"role": "system", "content": sample["lie_version"]["system"]},
            {"role": "user", "content": sample["lie_version"]["user"]},
            {"role": "assistant", "content": response_text},
        ]
        lie_text = collector.tokenizer.apply_chat_template(lie_msgs, tokenize=False)
        lie_hidden, lie_offsets = collector.get_activations(lie_text)
        lie_resp = find_response_tokens(lie_text, response_text, lie_offsets)

        # Truth version
        truth_msgs = [
            {"role": "system", "content": sample["truth_version"]["system"]},
            {"role": "user", "content": sample["truth_version"]["user"]},
            {"role": "assistant", "content": response_text},
        ]
        truth_text = collector.tokenizer.apply_chat_template(truth_msgs, tokenize=False)
        truth_hidden, truth_offsets = collector.get_activations(truth_text)
        truth_resp = find_response_tokens(truth_text, response_text, truth_offsets)

        if len(lie_resp) < 5 or len(truth_resp) < 5:
            continue

        for l in layers:
            # Mean activation over response tokens
            lie_acts[l].append(lie_hidden[l][lie_resp].mean(dim=0).numpy())
            truth_acts[l].append(truth_hidden[l][truth_resp].mean(dim=0).numpy())

        n_used += 1

    print(f"Used {n_used} samples for training")

    # Train probes per layer
    probes = {}
    for l in layers:
        X = np.vstack(lie_acts[l] + truth_acts[l])
        y = np.array([1] * len(lie_acts[l]) + [0] * len(truth_acts[l]))

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=42)
        clf.fit(X_s, y)

        train_auroc = roc_auc_score(y, clf.predict_proba(X_s)[:, 1])
        probes[l] = {"scaler": scaler, "clf": clf, "train_auroc": train_auroc}
        print(f"  Layer {l}: train AUROC = {train_auroc:.4f}")

    return probes


# ============================================================
# Phase 2: Evaluate on any JSONL dataset
# ============================================================

def evaluate_on_jsonl(collector, probes, layers, jsonl_path, dataset_name, max_samples=50,
                      model_type="qwen"):
    """Evaluate probes on a JSONL dataset with honest/deceptive labels."""
    print(f"\n{'='*60}")
    print(f"EVALUATING: {dataset_name} ({jsonl_path})")
    print(f"{'='*60}")

    # Load samples
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            label = d.get("label") or d.get("metadata", {}).get("label")
            if label not in ("honest", "deceptive"):
                continue
            msgs = d.get("messages") or d.get("conversations", {}).get("main", {}).get("messages", [])
            if not msgs:
                continue
            samples.append({"messages": msgs, "label": label})
            if max_samples and len(samples) >= max_samples:
                break

    print(f"Loaded {len(samples)} samples ({sum(1 for s in samples if s['label']=='deceptive')} deceptive)")

    # Collect activations and score
    scores = {l: [] for l in layers}
    labels = []

    for sample in tqdm(samples, desc=f"Evaluating {dataset_name}"):
        labels.append(1 if sample["label"] == "deceptive" else 0)

        text = collector.tokenizer.apply_chat_template(sample["messages"], tokenize=False)
        hidden, offsets = collector.get_activations(text)
        resp_tokens = find_assistant_tokens(text, offsets, model_type)

        if not resp_tokens:
            for l in layers:
                scores[l].append(0.5)
            continue

        for l in layers:
            acts = hidden[l][resp_tokens].mean(dim=0).numpy().reshape(1, -1)
            acts_s = probes[l]["scaler"].transform(acts)
            score = probes[l]["clf"].predict_proba(acts_s)[0, 1]
            scores[l].append(float(score))

    labels = np.array(labels)
    results = {}

    for l in layers:
        s = np.array(scores[l])
        if len(np.unique(labels)) < 2:
            print(f"  Layer {l}: SKIP (single class)")
            continue
        auroc = roc_auc_score(labels, s)
        results[l] = auroc
        print(f"  Layer {l}: sample AUROC = {auroc:.4f}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--ntml_path", type=str, required=True)
    parser.add_argument("--ai_liar_path", type=str, default=None)
    parser.add_argument("--bashbench_jsonl", type=str, default=None)
    parser.add_argument("--apps_jsonl", type=str, default=None)
    parser.add_argument("--layers", type=int, nargs="+", default=[12, 24, 36])
    parser.add_argument("--max_ntml", type=int, default=50)
    parser.add_argument("--max_eval", type=int, default=50)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    model_type = "qwen" if "qwen" in args.model_name.lower() else "llama"

    # Load model
    print(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    model.eval()
    print(f"Model on {model.device}")

    collector = ActivationCollector(model, tokenizer, args.layers)

    # Load NTML data
    with open(args.ntml_path) as f:
        ntml = json.load(f)
    if isinstance(ntml, dict) and "data" in ntml:
        ntml = ntml["data"]

    # Phase 1: Train
    probes = train_probes_on_ntml(collector, ntml, args.layers, args.max_ntml)

    # Phase 2: Evaluate on each dataset
    all_results = {}

    if args.ai_liar_path and Path(args.ai_liar_path).exists():
        all_results["ai_liar"] = evaluate_on_jsonl(
            collector, probes, args.layers, args.ai_liar_path, "ai_liar",
            args.max_eval, model_type)

    if args.bashbench_jsonl and Path(args.bashbench_jsonl).exists():
        all_results["bashbench"] = evaluate_on_jsonl(
            collector, probes, args.layers, args.bashbench_jsonl, "BashBench",
            args.max_eval, model_type)

    if args.apps_jsonl and Path(args.apps_jsonl).exists():
        all_results["apps"] = evaluate_on_jsonl(
            collector, probes, args.layers, args.apps_jsonl, "APPS",
            args.max_eval, model_type)

    collector.cleanup()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Sample-Level Detection AUROC")
    print(f"{'='*60}")
    print(f"{'Dataset':<15} " + " ".join(f"L{l:<6}" for l in args.layers))
    print("-" * 60)
    for dataset, results in all_results.items():
        row = f"{dataset:<15} "
        for l in args.layers:
            if l in results:
                row += f"{results[l]:<8.4f}"
            else:
                row += f"{'N/A':<8}"
        print(row)

    # Save
    out = {"model": args.model_name, "layers": args.layers, "results": {}}
    for ds, res in all_results.items():
        out["results"][ds] = {f"L{l}": v for l, v in res.items()}
    for l in args.layers:
        out[f"L{l}_train_auroc"] = probes[l]["train_auroc"]

    with open("e2e_test_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to e2e_test_results.json")


if __name__ == "__main__":
    main()
