import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import torch

def load_results(results_dir: Path, probe_type: str, layer: int) -> Dict:
    """Load evaluation results for a specific probe."""
    results_path = results_dir / 'aggregated' / probe_type / f'layer_{layer}' / 'metrics_all_thresholds.json'
    
    # Also need to load the raw scores - check if there's a samples file
    samples_path = results_dir / 'aggregated' / probe_type / f'layer_{layer}' / 'all_samples.json'
    
    if not results_path.exists():
        return None
        
    with open(results_path, 'r') as f:
        return json.load(f)

def create_distribution_plots(token_scores: List[float], token_labels: List[int],
                            sample_scores: List[float], sample_labels: List[int],
                            probe_name: str, thresholds: Dict[str, float],
                            save_path: Path):
    """Create distribution plots for probe scores."""
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Convert to numpy arrays
    token_scores = np.array(token_scores)
    token_labels = np.array(token_labels)
    sample_scores = np.array(sample_scores)
    sample_labels = np.array(sample_labels)
    
    # A) Token-level distribution
    ax = axes[0]
    
    # Separate scores by label
    honest_tokens = token_scores[token_labels == 0]
    deceptive_tokens = token_scores[token_labels == 1]
    
    # Create overlapping histograms
    bins = np.linspace(min(token_scores), max(token_scores), 50)
    
    ax.hist(honest_tokens, bins=bins, alpha=0.6, color='green', 
            label=f'Honest (n={len(honest_tokens)})', density=True, edgecolor='darkgreen')
    ax.hist(deceptive_tokens, bins=bins, alpha=0.6, color='red',
            label=f'Deceptive (n={len(deceptive_tokens)})', density=True, edgecolor='darkred')
    
    # Add threshold lines
    colors = {'default_0.5': 'black', 'train_auroc': 'blue', 'fpr_1pct': 'purple', 'fpr_5pct': 'orange'}
    for thresh_name, thresh_value in thresholds.items():
        if thresh_value != float('inf') and thresh_value != float('-inf'):
            ax.axvline(thresh_value, linestyle='--', alpha=0.7, 
                      color=colors.get(thresh_name, 'gray'),
                      label=f'{thresh_name}: {thresh_value:.3f}')
    
    ax.set_xlabel('Activation Score')
    ax.set_ylabel('Density')
    ax.set_title(f'{probe_name} - Token-Level Score Distribution')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    ax.text(0.02, 0.98, f'Honest mean: {np.mean(honest_tokens):.3f}\n'
                        f'Deceptive mean: {np.mean(deceptive_tokens):.3f}\n'
                        f'Separation: {np.mean(deceptive_tokens) - np.mean(honest_tokens):.3f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # B) Sample-level distribution  
    ax = axes[1]
    
    # Separate scores by label
    honest_samples = sample_scores[sample_labels == 0]
    deceptive_samples = sample_scores[sample_labels == 1]
    
    # Create overlapping histograms
    bins = np.linspace(min(sample_scores), max(sample_scores), 30)
    
    ax.hist(honest_samples, bins=bins, alpha=0.6, color='green',
            label=f'Honest (n={len(honest_samples)})', density=True, edgecolor='darkgreen')
    ax.hist(deceptive_samples, bins=bins, alpha=0.6, color='red',
            label=f'Deceptive (n={len(deceptive_samples)})', density=True, edgecolor='darkred')
    
    # Add threshold lines
    for thresh_name, thresh_value in thresholds.items():
        if thresh_value != float('inf') and thresh_value != float('-inf'):
            ax.axvline(thresh_value, linestyle='--', alpha=0.7,
                      color=colors.get(thresh_name, 'gray'),
                      label=f'{thresh_name}: {thresh_value:.3f}')
    
    ax.set_xlabel('Mean Activation Score')
    ax.set_ylabel('Density')
    ax.set_title(f'{probe_name} - Sample-Level Score Distribution')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    ax.text(0.02, 0.98, f'Honest mean: {np.mean(honest_samples):.3f}\n'
                        f'Deceptive mean: {np.mean(deceptive_samples):.3f}\n'
                        f'Separation: {np.mean(deceptive_samples) - np.mean(honest_samples):.3f}\n'
                        f'AUROC: {calculate_auroc(sample_scores, sample_labels):.3f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved distribution plot to {save_path}")

def calculate_auroc(scores, labels):
    """Calculate AUROC."""
    from sklearn.metrics import roc_auc_score
    try:
        return roc_auc_score(labels, scores)
    except:
        return 0.5

def extract_scores_from_evaluation(eval_dir: Path, probe_type: str, layer: int):
    """Extract token and sample scores from saved evaluation results."""
    
    # This would need to be saved during evaluation - for now, return dummy data
    # You'll need to modify the evaluation script to save raw scores
    
    # Check for visualization HTML which contains token details
    viz_path = eval_dir / 'aggregated' / probe_type / f'layer_{layer}' / 'token_visualization.html'
    
    # For now, return None - you'll need to save these during evaluation
    return None, None, None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--probe_type', type=str, default='logistic')
    parser.add_argument('--layer', type=int, default=10)
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    # Load results
    results = load_results(results_dir, args.probe_type, args.layer)
    if not results:
        print(f"No results found for {args.probe_type} layer {args.layer}")
        exit(1)
    
    # Get thresholds
    thresholds = results.get('thresholds_evaluated', {'default_0.5': 0.5})
    
    # NOTE: You need to modify the evaluation script to save raw scores
    # For now, this is a template
    print("NOTE: You need to modify probe_eval_deception_datasets.py to save raw scores!")
    print("Add this to the results dict in evaluate_on_assistant_tokens:")
    print("  'all_token_scores': all_token_scores_list,")
    print("  'all_token_labels': all_token_labels_list,")
    print("  'sample_scores': all_scores,")
    print("  'sample_labels': all_labels")