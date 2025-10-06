import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from sklearn.metrics import roc_auc_score, roc_curve

def load_results_with_scores(results_dir: Path, probe_type: str, layer: int) -> Tuple[Dict, Optional[Dict]]:
    """Load evaluation results and raw scores for a specific probe."""
    # Load metrics
    metrics_path = results_dir / 'aggregated' / probe_type / f'layer_{layer}' / 'metrics_all_thresholds.json'
    if not metrics_path.exists():
        return None, None
        
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Try to load raw scores from a separate file (if saved)
    scores_path = results_dir / 'aggregated' / probe_type / f'layer_{layer}' / 'raw_scores.json'
    if scores_path.exists():
        with open(scores_path, 'r') as f:
            scores = json.load(f)
    else:
        scores = None
    
    return metrics, scores

def extract_scores_from_token_details(results_dir: Path, probe_type: str, layer: int) -> Optional[Dict]:
    """Extract scores from token visualization HTML or saved JSON files."""
    # First, try to find a saved all_samples.json file
    samples_path = results_dir / 'aggregated' / probe_type / f'layer_{layer}' / 'all_samples.json'
    
    if samples_path.exists():
        with open(samples_path, 'r') as f:
            samples = json.load(f)
        
        # Extract token and sample scores
        token_scores = []
        token_labels = []
        sample_scores = []
        sample_labels = []
        
        for sample in samples:
            # Sample-level
            sample_scores.append(sample['mean_score'])
            sample_labels.append(sample['true_label'])
            
            # Token-level (if available)
            if 'token_scores' in sample:
                token_scores.extend(sample['token_scores'])
                token_labels.extend([sample['true_label']] * len(sample['token_scores']))
        
        return {
            'token_scores': token_scores,
            'token_labels': token_labels,
            'sample_scores': sample_scores,
            'sample_labels': sample_labels
        }
    
    return None

def create_distribution_plots(token_scores: List[float], token_labels: List[int],
                            sample_scores: List[float], sample_labels: List[int],
                            probe_name: str, thresholds: Dict[str, float],
                            save_path: Path):
    """Create distribution plots for probe scores."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Convert to numpy arrays
    token_scores = np.array(token_scores)
    token_labels = np.array(token_labels)
    sample_scores = np.array(sample_scores)
    sample_labels = np.array(sample_labels)
    
    # A) Token-level distribution
    ax = axes[0, 0]
    
    # Separate scores by label
    honest_tokens = token_scores[token_labels == 0]
    deceptive_tokens = token_scores[token_labels == 1]
    
    # Create overlapping histograms
    bins = np.linspace(min(token_scores), max(token_scores), 50)
    
    ax.hist(honest_tokens, bins=bins, alpha=0.6, color='green', 
            label=f'Honest (n={len(honest_tokens):,})', density=True, edgecolor='darkgreen')
    ax.hist(deceptive_tokens, bins=bins, alpha=0.6, color='red',
            label=f'Deceptive (n={len(deceptive_tokens):,})', density=True, edgecolor='darkred')
    
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
    ax = axes[0, 1]
    
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
                      color=colors.get(thresh_name, 'gray'))
    
    ax.set_xlabel('Mean Activation Score (per sample)')
    ax.set_ylabel('Density')
    ax.set_title(f'{probe_name} - Sample-Level Score Distribution')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Calculate AUROC
    auroc = roc_auc_score(sample_labels, sample_scores)
    
    # Add statistics
    ax.text(0.02, 0.98, f'Honest mean: {np.mean(honest_samples):.3f}\n'
                        f'Deceptive mean: {np.mean(deceptive_samples):.3f}\n'
                        f'Separation: {np.mean(deceptive_samples) - np.mean(honest_samples):.3f}\n'
                        f'AUROC: {auroc:.3f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # C) Violin plot for sample scores
    ax = axes[1, 0]
    
    # Prepare data for violin plot
    plot_data = []
    plot_labels = []
    for score, label in zip(sample_scores, sample_labels):
        plot_data.append(score)
        plot_labels.append('Deceptive' if label == 1 else 'Honest')
    
    # Create violin plot
    parts = ax.violinplot([honest_samples, deceptive_samples], positions=[0, 1], 
                          showmeans=True, showmedians=True, showextrema=True)
    
    # Color the violins
    parts['bodies'][0].set_facecolor('green')
    parts['bodies'][0].set_alpha(0.6)
    parts['bodies'][1].set_facecolor('red')
    parts['bodies'][1].set_alpha(0.6)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Honest', 'Deceptive'])
    ax.set_ylabel('Score')
    ax.set_title('Sample Score Distribution (Violin Plot)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add threshold lines
    for thresh_name, thresh_value in thresholds.items():
        if thresh_value != float('inf') and thresh_value != float('-inf'):
            ax.axhline(thresh_value, linestyle='--', alpha=0.7,
                      color=colors.get(thresh_name, 'gray'),
                      label=f'{thresh_name}: {thresh_value:.3f}')
    ax.legend(loc='best')
    
    # D) ROC Curve
    ax = axes[1, 1]
    
    fpr, tpr, roc_thresholds = roc_curve(sample_labels, sample_scores)
    ax.plot(fpr, tpr, color='blue', linewidth=2, label=f'ROC (AUC = {auroc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    
    # Mark the threshold points
    for thresh_name, thresh_value in thresholds.items():
        if thresh_value != float('inf') and thresh_value != float('-inf'):
            # Find closest threshold in ROC curve
            idx = np.argmin(np.abs(roc_thresholds - thresh_value))
            ax.plot(fpr[idx], tpr[idx], 'o', markersize=8, 
                   color=colors.get(thresh_name, 'gray'),
                   label=f'{thresh_name}: FPR={fpr[idx]:.3f}, TPR={tpr[idx]:.3f}')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'{probe_name} Score Distributions and Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved distribution plot to {save_path}")
    
    # Print summary statistics
    print(f"\n{probe_name} Statistics:")
    print(f"  Token-level:")
    print(f"    Honest: mean={np.mean(honest_tokens):.3f}, std={np.std(honest_tokens):.3f}")
    print(f"    Deceptive: mean={np.mean(deceptive_tokens):.3f}, std={np.std(deceptive_tokens):.3f}")
    print(f"    Separation: {np.mean(deceptive_tokens) - np.mean(honest_tokens):.3f}")
    print(f"  Sample-level:")
    print(f"    Honest: mean={np.mean(honest_samples):.3f}, std={np.std(honest_samples):.3f}")
    print(f"    Deceptive: mean={np.mean(deceptive_samples):.3f}, std={np.std(deceptive_samples):.3f}")
    print(f"    Separation: {np.mean(deceptive_samples) - np.mean(honest_samples):.3f}")
    print(f"    AUROC: {auroc:.3f}")

def save_raw_scores(results_dir: Path, probe_type: str, layer: int, 
                   token_scores: List[float], token_labels: List[int],
                   sample_scores: List[float], sample_labels: List[int]):
    """Save raw scores for future analysis."""
    scores_dir = results_dir / 'aggregated' / probe_type / f'layer_{layer}'
    scores_dir.mkdir(parents=True, exist_ok=True)
    
    scores_data = {
        'token_scores': token_scores,
        'token_labels': token_labels,
        'sample_scores': sample_scores,
        'sample_labels': sample_labels
    }
    
    with open(scores_dir / 'raw_scores.json', 'w') as f:
        json.dump(scores_data, f)
    
    print(f"Saved raw scores to {scores_dir / 'raw_scores.json'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory with evaluation results')
    parser.add_argument('--probe_types', nargs='+', default=['logistic'],
                       help='Probe types to visualize')
    parser.add_argument('--layers', nargs='+', type=int, default=[10],
                       help='Layers to visualize')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: results_dir/visualizations)')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for probe_type in args.probe_types:
        for layer in args.layers:
            print(f"\nProcessing {probe_type} layer {layer}...")
            
            # Load results
            metrics, raw_scores = load_results_with_scores(results_dir, probe_type, layer)
            
            if not metrics:
                print(f"  No results found for {probe_type} layer {layer}")
                continue
            
            # Get thresholds
            thresholds = metrics.get('thresholds_evaluated', {'default_0.5': 0.5})
            
            # Try to get scores
            if raw_scores:
                token_scores = raw_scores['token_scores']
                token_labels = raw_scores['token_labels']
                sample_scores = raw_scores['sample_scores']
                sample_labels = raw_scores['sample_labels']
            else:
                # Try to extract from saved samples
                scores_data = extract_scores_from_token_details(results_dir, probe_type, layer)
                
                if not scores_data:
                    print(f"  No score data available. Re-run evaluation with score saving enabled.")
                    continue
                
                token_scores = scores_data['token_scores']
                token_labels = scores_data['token_labels']
                sample_scores = scores_data['sample_scores']
                sample_labels = scores_data['sample_labels']
                
                # Save for future use
                if token_scores:
                    save_raw_scores(results_dir, probe_type, layer,
                                  token_scores, token_labels,
                                  sample_scores, sample_labels)
            
            # Create visualization
            if token_scores and sample_scores:
                save_path = output_dir / f'{probe_type}_layer_{layer}_distributions.png'
                create_distribution_plots(
                    token_scores, token_labels,
                    sample_scores, sample_labels,
                    f'{probe_type.upper()} Layer {layer}',
                    thresholds,
                    save_path
                )
            else:
                print(f"  Insufficient data for visualization")