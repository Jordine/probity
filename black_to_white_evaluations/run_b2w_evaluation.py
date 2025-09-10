# probity/black_to_white_evaluations/run_b2w_evaluation.py

import argparse
import json
import torch
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# --- Dynamically add repositories to Python path ---
def setup_paths(b2w_repo_path: Path):
    """Add probity and black_to_white_boosts repos to the Python path."""
    probity_path = Path.cwd()
    # The path to add is the 'src' directory, which contains the package
    b2w_src_path = b2w_repo_path / "src"

    if not probity_path.exists() or not (probity_path / "probity").exists():
        raise FileNotFoundError(f"Could not find 'probity' directory in current path: {probity_path}")
    if not b2w_src_path.exists():
        raise FileNotFoundError(f"Could not find 'src' directory in black_to_white_boosts repo: {b2w_src_path}")

    # Add paths only if they are not already present
    if str(probity_path) not in sys.path:
        sys.path.insert(0, str(probity_path))
    # Correctly add the 'src' directory to the path
    if str(b2w_src_path) not in sys.path:
        sys.path.insert(0, str(b2w_src_path))
    
    print("Successfully added project paths:")
    print(f"  - Probity: {probity_path}")
    print(f"  - B2W Src: {b2w_src_path}")

def _rebuild_probe_from_dict(probe_dict: dict, device: str) -> 'BaseProbe':
    """Helper to reconstruct a probe object from a saved dictionary."""
    from probity.training.configs import get_probe_class

    probe_type_name = probe_dict.get("probe_type")
    if not probe_type_name:
        raise ValueError("Saved probe dictionary is missing 'probe_type' key.")

    probe_type_key = probe_type_name.replace("Probe", "").lower()
    
    probe_cls = get_probe_class(probe_type_key)
    config = probe_dict["config"]
    
    probe = probe_cls(config)
    probe.load_state_dict(probe_dict["state_dict"])
    probe.to(device)
    probe.eval()
    return probe

def load_probes(probes_path: Path, device: str) -> Dict[str, 'BaseProbe']:
    """Loads a single probe or all probes from a directory using the correct method."""
    from probity.probes.base import BaseProbe

    probes = {}
    if probes_path.is_file() and probes_path.suffix == '.pt':
        probe_files = [probes_path]
    elif probes_path.is_dir():
        probe_files = list(probes_path.glob("**/*.pt"))
    else:
        raise ValueError(f"Invalid probes path: {probes_path}")

    print(f"Found {len(probe_files)} probe files to load.")
    for probe_path in probe_files:
        try:
            probe_type = probe_path.parent.name
            layer_name = probe_path.stem.replace('_probe', '')
            probe_key = f"probity_{probe_type}_{layer_name}"
            
            saved_object = torch.load(probe_path, map_location=device)
            
            if isinstance(saved_object, BaseProbe):
                probe = saved_object
                probe.to(device)
            elif isinstance(saved_object, dict):
                probe = _rebuild_probe_from_dict(saved_object, device)
            else:
                raise TypeError(f"Saved object in {probe_path} is of unexpected type: {type(saved_object)}")

            probe.eval()
            probes[probe_key] = probe
            print(f"  - Loaded probe: {probe_key}")
        except Exception as e:
            print(f"Warning: Could not load probe {probe_path}. Error: {e}")
    return probes

def run_scoring_phase(args, probes):
    """Phase 1: Score B2W datasets with trained probes."""
    from transformer_lens import HookedTransformer
    # We will treat the data as raw dictionaries, so we only need the sample class for formatting text
    from black_to_white_boosts.base.sample import DeceptionDetectionSample

    print("\n--- Phase 1: Scoring Datasets ---")
    
    print(f"Loading model: {args.model_name}")
    model = HookedTransformer.from_pretrained_no_processing(args.model_name, device=args.device)
    model.eval()

    required_layers = {p.config.hook_layer for p in probes.values()}
    hook_points = [f"blocks.{layer}.hook_resid_pre" for layer in required_layers]
    print(f"Extracting activations from layers: {sorted(list(required_layers))}")

    b2w_datasets_dir = args.b2w_repo_path / "datasets" / "clean_binary"
    annotated_data_dir = args.output_dir / "annotated_datasets"
    annotated_data_dir.mkdir(parents=True, exist_ok=True)

    dataset_files = list(b2w_datasets_dir.glob("*.jsonl"))
    print(f"Found {len(dataset_files)} datasets to process in {b2w_datasets_dir}")

    for dataset_path in dataset_files:
        if not dataset_path.is_file(): continue
        print(f"\nProcessing dataset: {dataset_path.name}")
        output_path = annotated_data_dir / dataset_path.name
        
        with open(dataset_path, 'r') as f_in, open(output_path, 'w') as f_out:
            lines = f_in.readlines()
            for i in tqdm(range(0, len(lines), args.batch_size), desc=f"Scoring {dataset_path.stem}"):
                batch_lines = lines[i:i+args.batch_size]
                batch_data = [json.loads(line) for line in batch_lines]
                
                # Use the B2W class just for its text formatting utility
                texts = [DeceptionDetectionSample.from_dict(d).conversation("main").format_public_text() for d in batch_data]
                tokens = model.to_tokens(texts, prepend_bos=True).to(args.device)
                
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, names_filter=hook_points)

                for j, sample_data in enumerate(batch_data):
                    # === THIS IS THE CORRECTED LOGIC ===
                    # Treat sample_data as a plain dictionary, not a class instance
                    if "monitor_results" not in sample_data:
                        sample_data["monitor_results"] = []

                    for probe_key, probe in probes.items():
                        layer = probe.config.hook_layer
                        hook_point = f"blocks.{layer}.hook_resid_pre"
                        
                        activations = cache[hook_point][j]
                        num_tokens = len(sample_data["conversations"]["main"]["tokens"])
                        activations = activations[:num_tokens]

                        with torch.no_grad():
                            scores = probe.encode(activations).cpu().numpy()
                        
                        # Create the new monitor result as a dictionary
                        new_result = {
                            "monitor": {"name": f"mean_{probe_key}", "method": "mean", "probe": probe_key, "model": "no_model"},
                            "result": {"suspiciousness": float(np.mean(scores)), "reasoning": {}},
                            "timestamp": "2025-01-01T00:00:00Z"
                        }
                        
                        # Append the dictionary to the list
                        sample_data["monitor_results"].append(new_result)
                    
                    # Write the modified dictionary back to the file
                    f_out.write(json.dumps(sample_data) + '\n')
    
    del model
    torch.cuda.empty_cache()
    print("\nScoring phase complete.")
    return annotated_data_dir

def run_analysis_phase(args, annotated_data_dir):
    """Phase 2: Analyze scores and generate plots/tables."""
    from black_to_white_boosts.analysis.results_to_csv import create_monitor_scores_csv
    from black_to_white_boosts.analysis.auroc import create_performance_table

    print("\n--- Phase 2: Analyzing Scores ---")

    # Check if annotated data exists before proceeding
    if not any(annotated_data_dir.glob("*.jsonl")):
        print(f"Error: No annotated dataset files found in {annotated_data_dir}.")
        print("The scoring phase may have failed to produce output. Aborting analysis.")
        return

    csv_path = args.output_dir / "monitor_scores.csv"
    create_monitor_scores_csv(annotated_data_dir, csv_path)
    print(f"Monitor scores saved to {csv_path}")

    df = pd.read_csv(csv_path)
    
    print("\n--- AUROC Performance Table ---")
    text_table = create_performance_table(df, metric='auroc')
    print(text_table)
    with open(args.output_dir / "auroc_table.txt", "w") as f:
        f.write(text_table)

    latex_path = args.output_dir / "auroc_table.tex"
    create_performance_table(df, metric='auroc', output_path=latex_path, latex=True)
    print(f"LaTeX performance table saved to {latex_path}")

    print("\n--- Generating Plots ---")
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    
    metadata_cols = ['dataset', 'sample_index', 'label']
    monitor_cols = [c for c in df.columns if c not in metadata_cols]
    datasets = sorted([d for d in df['dataset'].unique() if 'alpaca' not in d])

    if not monitor_cols:
        print("No monitor columns found in CSV. Cannot generate plots.")
        return

    aurocs = {}
    for monitor in monitor_cols:
        aurocs[monitor] = []
        for dataset in datasets:
            dataset_df = df[df['dataset'] == dataset]
            if len(dataset_df) == 0:
                aurocs[monitor].append(0.5)
                continue
            labels = (dataset_df['label'] == 'deceptive').astype(int)
            scores = dataset_df[monitor].values
            
            if len(np.unique(labels)) < 2:
                auroc = 0.5
            else:
                from sklearn.metrics import roc_auc_score
                auroc = roc_auc_score(labels, scores)
            aurocs[monitor].append(auroc)

    n_monitors = len(monitor_cols)
    n_datasets = len(datasets)
    fig, ax = plt.subplots(figsize=(max(8, n_datasets * 1.5), 6))
    
    x = np.arange(n_datasets)
    width = 0.8 / n_monitors if n_monitors > 0 else 0.8
    
    for i, (monitor, values) in enumerate(aurocs.items()):
        offset = width * (i - (n_monitors - 1) / 2)
        ax.bar(x + offset, values, width, label=monitor.replace("mean_probity_", ""))

    ax.set_ylabel('AUROC')
    ax.set_title('Probe Performance Across Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.axhline(y=0.5, color='grey', linestyle='--')
    ax.legend(title="Probes", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plot_path = plot_dir / "probe_performance_overview.png"
    plt.savefig(plot_path)
    print(f"Performance plot saved to {plot_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Run probe-only evaluation using the Black-to-White framework.")
    parser.add_argument('--probes-path', type=str, required=True, help="Path to a single trained probe (.pt) or a directory of probes.")
    parser.add_argument('--b2w-repo-path', type=str, required=True, help="Path to the cloned 'black_to_white_boosts' repository.")
    parser.add_argument('--model-name', type=str, required=True, help="Name of the model used for activations (e.g., 'meta-llama/Llama-3.1-8B-Instruct').")
    parser.add_argument('--output-dir', type=str, default="./b2w_evaluation_results", help="Directory to save all evaluation outputs.")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch-size', type=int, default=8, help="Batch size for scoring.")
    parser.add_argument('--skip-scoring', action='store_true', help="Skip the scoring phase and run analysis on existing annotated datasets.")
    args = parser.parse_args()

    args.b2w_repo_path = Path(args.b2w_repo_path).resolve()
    args.output_dir = Path(args.output_dir).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_paths(args.b2w_repo_path)
    
    if not args.skip_scoring:
        probes = load_probes(Path(args.probes_path), args.device)
        if not probes:
            print("No probes were loaded. Aborting.")
            return
        annotated_data_dir = run_scoring_phase(args, probes)
    else:
        annotated_data_dir = args.output_dir / "annotated_datasets"
        if not annotated_data_dir.exists():
            print(f"Error: --skip-scoring was used, but annotated data not found in {annotated_data_dir}")
            return
        print(f"Skipping scoring. Using existing data from: {annotated_data_dir}")

    run_analysis_phase(args, annotated_data_dir)
    
    print("\n--- Evaluation Complete ---")
    print(f"All results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()
    print("test")