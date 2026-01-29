#!/usr/bin/env python3
"""
Evaluate Phase 1 sweep probes with nested directory structure.

Expected probe structure:
  probes/merged/{dataset}/{probe_variant}/layer_{N}_probe.pt

Example:
  probes/merged/5T2L_direct_liar/attention_h1_t1.0/layer_22_probe.pt
"""
import argparse
import json
import subprocess
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


# Probe variants from sweep_config.json
PROBE_VARIANTS = [
    "attention_h1_t1.0",
    "attention_h4_t1.0",
    "attention_h8_t1.0",
    "attention_h4_t0.5",
    "attention_h4_t2.0",
    "sklearn_logistic_C0.1",
    "sklearn_logistic_C1.0",
    "sklearn_logistic_C10.0",
    "logistic_default",
    "meandiff_default",
]

LAYERS = [22, 36, 44]

DATASETS = [
    "5T2L_direct_liar",
    "5T2L_roleplay",
    "5T2L_instructional",
    "5T2L_conditional",
    "5T2L_neutral",
    "5T2L_neutral_identity",
    "10T1L_direct_liar",
    "10T1L_roleplay",
    "10T1L_instructional",
    "10T1L_conditional",
    "10T1L_neutral",
    "10T1L_neutral_identity",
]


def find_available_probes(probe_dir: Path) -> Dict[str, Dict[str, List[int]]]:
    """
    Scan probe directory and return available probes.
    Returns: {dataset: {variant: [layers]}}
    """
    available = {}

    for dataset_dir in probe_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        available[dataset] = {}

        for variant_dir in dataset_dir.iterdir():
            if not variant_dir.is_dir():
                continue
            variant = variant_dir.name

            layers = []
            for probe_file in variant_dir.glob("layer_*_probe.pt"):
                layer = int(probe_file.stem.split("_")[1])
                layers.append(layer)

            if layers:
                available[dataset][variant] = sorted(layers)

    return available


def evaluate_single_probe(
    probe_path: Path,
    eval_datasets: List[str],
    labeled_dir: Path,
    results_dir: Path,
    model_name: str,
    layer: int,
    batch_size: int = 4
) -> Dict:
    """
    Evaluate a single probe on evaluation datasets.
    Returns metrics dict.
    """
    # Create temp dir with expected structure: probe_type/layer_N_probe.pt
    temp_dir = Path(f"/tmp/probe_eval_{layer}")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    # Determine probe type from path (e.g., attention_h1_t1.0 -> attention)
    variant_name = probe_path.parent.name
    base_type = variant_name.split("_")[0]  # attention, sklearn, logistic, meandiff
    if base_type == "sklearn":
        base_type = "sklearn_logistic"

    # Copy probe to temp dir
    temp_probe_dir = temp_dir / base_type
    temp_probe_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(probe_path, temp_probe_dir / probe_path.name)

    # Run evaluation
    cmd = [
        sys.executable, "scripts/probe_eval_deception_datasets.py",
        "--model_name", model_name,
        "--probe_dir", str(temp_dir),
        "--datasets", *eval_datasets,
        "--labeled_dir", str(labeled_dir),
        "--results_dir", str(results_dir),
        "--batch_size", str(batch_size),
        "--device", "cuda"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[-500:]}")
            return None
    except Exception as e:
        print(f"  Exception: {e}")
        return None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Load results
    metrics = {}
    for eval_ds in eval_datasets:
        results_file = results_dir / f"{eval_ds}_results.json"
        if results_file.exists():
            with open(results_file) as f:
                metrics[eval_ds] = json.load(f)

    return metrics


def run_sweep_evaluation(
    probe_dir: Path,
    eval_datasets: List[str],
    labeled_dir: Path,
    results_dir: Path,
    model_name: str,
    batch_size: int = 4
):
    """Run evaluation on all probes in sweep."""

    print("=" * 60)
    print("Phase 1 Sweep Evaluation")
    print("=" * 60)

    # Find available probes
    available = find_available_probes(probe_dir)

    total_probes = sum(
        len(layers)
        for variants in available.values()
        for layers in variants.values()
    )
    print(f"Found {len(available)} datasets, {total_probes} total probes")

    # Results storage
    all_results = []

    probe_count = 0
    for dataset in sorted(available.keys()):
        print(f"\n>>> Dataset: {dataset}")

        for variant in sorted(available[dataset].keys()):
            layers = available[dataset][variant]

            for layer in layers:
                probe_count += 1
                probe_path = probe_dir / dataset / variant / f"layer_{layer}_probe.pt"

                print(f"  [{probe_count}/{total_probes}] {variant} layer {layer}...", end=" ", flush=True)

                # Create temp results dir
                temp_results = results_dir / "temp"
                temp_results.mkdir(parents=True, exist_ok=True)

                metrics = evaluate_single_probe(
                    probe_path=probe_path,
                    eval_datasets=eval_datasets,
                    labeled_dir=labeled_dir,
                    results_dir=temp_results,
                    model_name=model_name,
                    layer=layer,
                    batch_size=batch_size
                )

                if metrics:
                    # Extract key metrics
                    for eval_ds, ds_metrics in metrics.items():
                        result = {
                            "dataset": dataset,
                            "mix": dataset.split("_")[0],  # 5T2L or 10T1L
                            "style": "_".join(dataset.split("_")[1:]),  # direct_liar, etc.
                            "variant": variant,
                            "probe_type": variant.split("_")[0],
                            "layer": layer,
                            "eval_dataset": eval_ds,
                        }

                        # Add metrics
                        if isinstance(ds_metrics, dict):
                            if "aggregated" in ds_metrics:
                                agg = ds_metrics["aggregated"]
                                result["sample_auroc"] = agg.get("auroc", None)
                                result["sample_accuracy"] = agg.get("accuracy", None)
                                result["sample_f1"] = agg.get("f1", None)
                            if "token_metrics" in ds_metrics:
                                tm = ds_metrics["token_metrics"]
                                result["token_auroc"] = tm.get("auroc", None)
                                result["token_auprc"] = tm.get("auprc", None)

                        all_results.append(result)
                    print("OK")
                else:
                    print("FAILED")

                # Cleanup temp
                shutil.rmtree(temp_results, ignore_errors=True)

    # Save results
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    with open(results_dir / "sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Save as CSV for easy analysis
    df = pd.DataFrame(all_results)
    df.to_csv(results_dir / "sweep_results.csv", index=False)

    print(f"\n{'=' * 60}")
    print(f"Evaluation complete!")
    print(f"Results saved to: {results_dir}")
    print(f"  - sweep_results.json")
    print(f"  - sweep_results.csv")
    print(f"{'=' * 60}")

    # Print summary
    if all_results:
        df = pd.DataFrame(all_results)
        print("\n=== SUMMARY ===")
        print(f"\nBy dataset mix (mean sample AUROC on ai_liar):")
        print(df[df["eval_dataset"] == "ai_liar"].groupby("mix")["sample_auroc"].mean())
        print(f"\nBy instruction style (mean sample AUROC on ai_liar):")
        print(df[df["eval_dataset"] == "ai_liar"].groupby("style")["sample_auroc"].mean())
        print(f"\nBy probe type (mean sample AUROC on ai_liar):")
        print(df[df["eval_dataset"] == "ai_liar"].groupby("probe_type")["sample_auroc"].mean())


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 1 sweep probes")
    parser.add_argument("--probe_dir", type=Path, required=True,
                        help="Directory containing probes (merged/)")
    parser.add_argument("--eval_datasets", nargs="+",
                        default=["ai_liar", "sandbagging_v2__wmdp_mmlu"],
                        help="Evaluation datasets")
    parser.add_argument("--labeled_dir", type=Path,
                        default=Path("data/deception_detection"),
                        help="Directory with labeled .jsonl files")
    parser.add_argument("--results_dir", type=Path, required=True,
                        help="Output directory for results")
    parser.add_argument("--model_name", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()

    run_sweep_evaluation(
        probe_dir=args.probe_dir,
        eval_datasets=args.eval_datasets,
        labeled_dir=args.labeled_dir,
        results_dir=args.results_dir,
        model_name=args.model_name,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
