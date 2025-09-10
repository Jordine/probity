#!/usr/bin/env python3
# probity/black_to_white_evaluations/run_b2w_evaluation.py
#
# Probe-only evaluation script used for the Black-to-White boosts project.
# This version guarantees that every monitor name contains exactly ONE
# "_l" delimiter (the layer suffix) so no changes are required in the B2W
# analysis code.
#
# Author: OpenAI (o3 patch, 2025-09-10)

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# --------------------------------------------------------------------------- #
#  Path setup                                                                 #
# --------------------------------------------------------------------------- #
def setup_paths(b2w_repo_path: Path):
    """Add probity and black_to_white_boosts repos to PYTHONPATH."""
    probity_path = Path.cwd()
    b2w_src_path = b2w_repo_path / "src"

    if not (probity_path / "probity").exists():
        raise FileNotFoundError(
            f"Could not find 'probity' directory in cwd: {probity_path}"
        )
    if not b2w_src_path.exists():
        raise FileNotFoundError(
            f"Could not find 'src' directory in B2W repo: {b2w_src_path}"
        )

    if str(probity_path) not in sys.path:
        sys.path.insert(0, str(probity_path))
    if str(b2w_src_path) not in sys.path:
        sys.path.insert(0, str(b2w_src_path))

    print("Successfully added project paths:")
    print(f"  - Probity: {probity_path}")
    print(f"  - B2W Src: {b2w_src_path}")


# --------------------------------------------------------------------------- #
#  Probe loading                                                              #
# --------------------------------------------------------------------------- #
def _rebuild_probe_from_dict(probe_dict: dict, device: str) -> "BaseProbe":
    from probity.training.configs import get_probe_class

    probe_type_name = probe_dict.get("probe_type")
    if not probe_type_name:
        raise ValueError("Saved probe dict is missing 'probe_type'.")
    probe_type_key = probe_type_name.replace("Probe", "").lower()

    probe_cls = get_probe_class(probe_type_key)
    probe = probe_cls(probe_dict["config"])
    probe.load_state_dict(probe_dict["state_dict"])
    probe.to(device).eval()
    return probe


def load_probes(probes_path: Path, device: str) -> Dict[str, "BaseProbe"]:
    """
    Load probes and return a dict with keys of the form
    'probity-<probe_type>_l<layer>'.  The hyphen guarantees that '_l' occurs
    exactly once, avoiding downstream parsing issues.
    """
    from probity.probes.base import BaseProbe

    probes: Dict[str, BaseProbe] = {}

    if probes_path.is_file() and probes_path.suffix == ".pt":
        probe_files = [probes_path]
    elif probes_path.is_dir():
        probe_files = list(probes_path.glob("**/*.pt"))
    else:
        raise ValueError(f"Invalid probes path: {probes_path}")

    print(f"Found {len(probe_files)} probe files to load.")

    for path in probe_files:
        try:
            probe_type_raw = path.parent.name          # e.g. "logistic"
            probe_type = probe_type_raw.replace("logistic", "logreg")

            stem = path.stem.replace("_probe", "")      # layer_15 → layer15
            layer_number = "".join(filter(str.isdigit, stem))
            if not layer_number:
                raise RuntimeError(
                    f"Could not extract layer number from '{path.name}'."
                )

            #   >>> probity-logreg_l15   (ONLY one '_l')
            probe_key = f"probity-{probe_type}_l{layer_number}"

            saved = torch.load(path, map_location=device)
            if isinstance(saved, BaseProbe):
                probe: BaseProbe = saved
            elif isinstance(saved, dict):
                probe = _rebuild_probe_from_dict(saved, device)
            else:
                raise TypeError(
                    f"{path}: unexpected object type {type(saved)}."
                )

            probe.to(device).eval()
            probes[probe_key] = probe
            print(f"  - Loaded probe: {probe_key}")

        except Exception as err:
            print(f"Warning: could not load {path}: {err}")

    return probes


# --------------------------------------------------------------------------- #
#  Helper: locate assistant span                                              #
# --------------------------------------------------------------------------- #
def find_assistant_token_indices_robust(
    formatted_text: str, tokenizer: "PreTrainedTokenizer"
) -> Optional[Tuple[int, int]]:
    """
    Return (start_idx, end_idx) of assistant response tokens in `formatted_text`.
    """
    assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"
    eot_marker = "<|eot_id|>"

    assistant_start_char = formatted_text.rfind(assistant_marker)
    if assistant_start_char == -1:
        return None

    content_start_char = assistant_start_char + len(assistant_marker)
    while (
        content_start_char < len(formatted_text)
        and formatted_text[content_start_char].isspace()
    ):
        content_start_char += 1

    content_end_char = formatted_text.find(eot_marker, content_start_char)
    if content_end_char == -1:
        content_end_char = len(formatted_text)

    enc = tokenizer(formatted_text, add_special_tokens=False)
    start_tok = enc.char_to_token(content_start_char)
    end_tok = enc.char_to_token(content_end_char - 1)
    if start_tok is None or end_tok is None:
        return None
    return start_tok, end_tok + 1  # inclusive → exclusive


# --------------------------------------------------------------------------- #
#  Phase 1 – scoring                                                          #
# --------------------------------------------------------------------------- #
def run_scoring_phase(args, probes):
    """Score a raw dataset with the supplied probes."""
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer

    print("\n--- Phase 1: Scoring Datasets ---")

    print(f"Loading model: {args.model_name}")
    model = HookedTransformer.from_pretrained_no_processing(
        args.model_name, device=args.device, torch_dtype=torch.bfloat16
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    required_layers = {p.config.hook_layer for p in probes.values()}
    hook_pts = [f"blocks.{l}.hook_resid_pre" for l in required_layers]
    print(f"Extracting activations from layers: {sorted(required_layers)}")

    raw_path = Path(args.raw_dataset_path)
    ann_dir = args.output_dir / "annotated_datasets"
    ann_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing dataset: {raw_path.name}")
    output_path = ann_dir / raw_path.name

    with open(raw_path, "r") as f_in, open(output_path, "w") as f_out:
        lines = [ln for ln in f_in.readlines() if ln.strip()]

        for i in tqdm(range(0, len(lines), args.batch_size),
                      desc=f"Scoring {raw_path.stem}"):
            batch_json = [json.loads(ln) for ln in lines[i:i + args.batch_size]]

            # ------ build chat templates -------------------------------- #
            texts = []
            for sample in batch_json:
                if "messages" in sample:                     # ChatML format
                    messages = sample["messages"]
                else:                                        # Alpaca format
                    sys_txt = "You are a helpful assistant."
                    instr = sample.get("instruction", "")
                    inp = sample.get("input", "")
                    out = sample.get("output", "")
                    user_txt = f"{instr}\n{inp}".strip()
                    messages = [
                        {"role": "system", "content": sys_txt},
                        {"role": "user", "content": user_txt},
                        {"role": "assistant", "content": out},
                    ]
                texts.append(
                    tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                )

            # ------ forward pass + cache -------------------------------- #
            toks = model.to_tokens(texts, prepend_bos=False).to(args.device)
            with torch.no_grad():
                _, cache = model.run_with_cache(toks, names_filter=hook_pts)

            # ------ per-sample probe scoring ---------------------------- #
            for j, sample in enumerate(batch_json):
                span = find_assistant_token_indices_robust(texts[j], tokenizer)
                if span is None:
                    start_idx, end_idx = 0, 0        # empty slice ⇒ score 0.0
                    print(f"Warning: assistant span not found "
                          f"(sample {sample.get('sample_index', 'N/A')}).")
                else:
                    start_idx, end_idx = span

                monitor_results: List[Dict] = []
                for key, probe in probes.items():
                    layer = probe.config.hook_layer
                    hook = f"blocks.{layer}.hook_resid_pre"
                    acts = cache[hook][j][start_idx:end_idx]

                    if acts.shape[0] == 0:
                        mean_score = 0.0
                    else:
                        with torch.no_grad():
                            mean_score = float(probe.encode(acts)
                                               .mean()
                                               .cpu()
                                               .item())

                    monitor_results.append(
                        {
                            "monitor": {
                                "name": f"mean_{key}",
                                "method": "mean",
                                "probe": key,
                                "model": "no_model",
                            },
                            "result": {
                                "suspiciousness": mean_score,
                                "reasoning": {},
                            },
                            "timestamp": "2025-01-01T00:00:00Z",
                        }
                    )

                out_record = {
                    "metadata": {
                        "dataset": sample.get("dataset", raw_path.stem),
                        "sample_index": sample.get("sample_index", i + j),
                        "label": sample.get("label", "unknown"),
                    },
                    "conversations": {
                        "main": {"messages": sample.get("messages", [])}
                    },
                    "monitor_results": monitor_results,
                }
                f_out.write(json.dumps(out_record) + "\n")

    del model
    torch.cuda.empty_cache()
    print("\nScoring phase complete.")
    return ann_dir


# --------------------------------------------------------------------------- #
#  Phase 2 – analysis                                                         #
# --------------------------------------------------------------------------- #
def run_analysis_phase(args, ann_dir: Path):
    """Convert JSONL → CSV and run Black-to-White analysis + plots."""
    from black_to_white_boosts.analysis.results_to_csv import (
        create_monitor_scores_csv,
    )
    from black_to_white_boosts.analysis.auroc import create_performance_table

    print("\n--- Phase 2: Analyzing Scores ---")

    if not any(p.is_file() and p.stat().st_size for p in ann_dir.glob("*.jsonl")):
        print(f"Error: annotated files in {ann_dir} are empty.")
        return

    csv_path = args.output_dir / "monitor_scores.csv"
    create_monitor_scores_csv(ann_dir, csv_path)
    print(f"Monitor scores saved to {csv_path}")

    df = pd.read_csv(csv_path)

    # ------------------------------------------------------------------ #
    print("\n--- AUROC Performance Table ---")
    txt_table = create_performance_table(df, metric="auroc")
    print(txt_table)
    (args.output_dir / "auroc_table.txt").write_text(txt_table)

    latex_path = args.output_dir / "auroc_table.tex"
    create_performance_table(df, metric="auroc", output_path=latex_path, latex=True)
    print(f"LaTeX performance table saved to {latex_path}")

    # ------------------------------------------------------------------ #
    print("\n--- Generating Plots ---")
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    meta_cols = ["dataset", "sample_index", "label"]
    monitor_cols = [c for c in df.columns if c not in meta_cols]
    datasets = sorted(
        [d for d in df["dataset"].unique() if "alpaca" not in str(d).lower()]
    )
    if not (monitor_cols and datasets):
        print("Nothing to plot.")
        return

    aurocs = {m: [] for m in monitor_cols}
    for monitor in monitor_cols:
        for dataset in datasets:
            sub = df[df["dataset"] == dataset]
            labels = (sub["label"] == "deceptive").astype(int)
            scores = sub[monitor].astype(float).values
            if len(np.unique(labels)) < 2:
                auroc = 0.5
            else:
                from sklearn.metrics import roc_auc_score

                auroc = roc_auc_score(labels, scores)
            aurocs[monitor].append(auroc)

    n_monitors, n_datasets = len(monitor_cols), len(datasets)
    x = np.arange(n_datasets)
    width = 0.8 / n_monitors

    fig, ax = plt.subplots(figsize=(max(8, n_datasets * 1.5), 6))
    for i, (monitor, vals) in enumerate(aurocs.items()):
        ax.bar(x + width * (i - (n_monitors - 1) / 2), vals, width,
               label=monitor.replace("mean_", ""))
    ax.set_ylabel("AUROC")
    ax.set_title("Probe Performance Across Datasets")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.axhline(0.5, color="grey", linestyle="--")
    ax.legend(title="Probes", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.6)
    fig.tight_layout()
    plot_path = plot_dir / "probe_performance_overview.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Performance plot saved to {plot_path}")


# --------------------------------------------------------------------------- #
#  CLI                                                                        #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Run probe-only Black-to-White evaluation."
    )
    parser.add_argument("--raw-dataset-path", required=True)
    parser.add_argument("--probes-path", required=True)
    parser.add_argument("--b2w-repo-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", default="./b2w_evaluation_results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--skip-scoring", action="store_true")
    args = parser.parse_args()

    args.b2w_repo_path = Path(args.b2w_repo_path).resolve()
    args.output_dir = Path(args.output_dir).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    setup_paths(args.b2w_repo_path)

    if args.skip_scoring:
        ann_dir = args.output_dir / "annotated_datasets"
        if not any(ann_dir.glob("*.jsonl")):
            print(f"No annotated data found in {ann_dir}")
            return
        print(f"Skipping scoring phase. Using {ann_dir}")
    else:
        probes = load_probes(Path(args.probes_path), args.device)
        if not probes:
            print("No probes loaded. Aborting.")
            return
        ann_dir = run_scoring_phase(args, probes)

    run_analysis_phase(args, ann_dir)
    print("\n--- Evaluation Complete ---")
    print(f"All results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()