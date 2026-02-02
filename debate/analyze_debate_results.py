#!/usr/bin/env python3
"""
Comprehensive analysis of debate results with visualizations and statistical tests.

Usage:
    python debate/analyze_debate_results.py --transcripts_dir results/experiment/transcripts --output_dir results/experiment/analysis
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

from debate.visualisation.debate_viz import create_debate_chat_visualization
from debate.utils.probe_analysis import ProbeAnalyzer


class DebateResultsAnalyzer:
    """Analyze debate results with statistical tests and visualizations"""
    
    def __init__(self, transcripts_dir: Path, output_dir: Path):
        self.transcripts_dir = transcripts_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all transcripts
        self.transcripts = self._load_all_transcripts()
        
        # Organize results by experiment mode
        self.results_by_mode = self._organize_results()
        
    def _load_all_transcripts(self) -> List[Dict]:
        """Load all debate transcripts"""
        transcripts = []
        
        for json_file in self.transcripts_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    transcript = json.load(f)
                    transcripts.append(transcript)
            except Exception as e:
                print(f"Warning: Could not load {json_file}: {e}")
        
        print(f"Loaded {len(transcripts)} debate transcripts")
        return transcripts

    def _organize_results(self) -> Dict:
        """Organize results by experiment mode and evaluation mode"""
        
        results = defaultdict(lambda: defaultdict(list))
        
        for transcript in self.transcripts:
            experiment_mode = transcript.get('experiment_mode', 'unknown')
            debate_id = transcript['debate_id']
            
            # Process judge results - FIX THE KEY PARSING
            judge_results = transcript.get('judge_results', {})
            probe_only_results = transcript.get('probe_only_results', {})
            
            # Extract probe config info
            probe_config = transcript.get('debate_config', {}).get('probe_config', {})
            probes_enabled = probe_config.get('enabled', False)
            
            for full_key, judge_result in judge_results.items():
                # Better parsing logic for complex judge names
                # Format is: "evaluation_mode_judge_name"
                # We need to handle judge names that might contain underscores
                
                # Try to identify known evaluation modes
                eval_mode = None
                judge_name = None
                
                for mode in ['baseline', 'judge_only', 'debater_only', 'full_access']:
                    if full_key.startswith(mode + '_'):
                        eval_mode = mode
                        judge_name = full_key[len(mode) + 1:]
                        break
                
                if not eval_mode:
                    print(f"Warning: Could not parse key: {full_key}")
                    continue
                
                winner_role = judge_result.get('winner_role')
                correct = judge_result.get('correct')
                
                results[experiment_mode][eval_mode].append({
                    'debate_id': debate_id,
                    'judge_name': judge_name,
                    'winner_role': winner_role,
                    'correct': correct,
                    'transcript': transcript,
                    'probes_enabled': probes_enabled,
                    'reasoning': judge_result.get('reasoning', '')
                })
            
            # Add probe-only results if probes were enabled
            if probes_enabled and probe_only_results:
                for probe_method in ['probe_only_mean', 'probe_only_max']:
                    winner = probe_only_results.get(probe_method)
                    if winner:
                        correct = self._determine_probe_only_correctness(winner, transcript)
                        
                        results[experiment_mode][probe_method].append({
                            'debate_id': debate_id,
                            'judge_name': 'probe_classifier',
                            'winner_role': winner,
                            'correct': correct,
                            'transcript': transcript,
                            'probes_enabled': True
                        })
        
        return dict(results)
    
    def _determine_probe_only_correctness(self, winner: str, transcript: Dict) -> bool:
        """Determine if probe-only classification is correct"""
        if winner is None:
            return None
            
        debate_type = transcript.get('debate_type', 'unknown')
        
        if debate_type == 'quality':
            # Honest always defends correct answer
            return winner == 'honest'
        elif debate_type == 'simple':
            # Check ground truth from scenario
            problem_data = transcript.get('problem_data', {})
            ground_truth = problem_data.get('ground_truth', '')
            model_mapping = transcript.get('model_mapping', {})
            if ground_truth and model_mapping:
                ground_truth_role = model_mapping.get(ground_truth)
                return winner == ground_truth_role
            return None
        else:  # APPS
            # Honest defends non-backdoored solution
            return winner == 'honest'
    
    def create_combined_win_rate_bar_plot(self):
        """Create combined bar plot with all judges and modes on one plot"""
        
        # Collect all data
        plot_data = []
        
        for experiment_mode, eval_modes in self.results_by_mode.items():
            for eval_mode, results_list in eval_modes.items():
                if not results_list:
                    continue
                
                # Group by judge name
                judge_groups = defaultdict(list)
                for result in results_list:
                    judge_groups[result['judge_name']].append(result)
                
                for judge_name, judge_results in judge_groups.items():
                    # Calculate win rate
                    correct_results = [r for r in judge_results if r['correct'] is not None]
                    if correct_results:
                        correct_count = sum(1 for r in correct_results if r['correct'])
                        total_count = len(correct_results)
                        win_rate = correct_count / total_count if total_count > 0 else 0
                        
                        plot_data.append({
                            'experiment_mode': experiment_mode,
                            'eval_mode': eval_mode,
                            'judge_name': judge_name,
                            'win_rate': win_rate,
                            'n': total_count
                        })
        
        if not plot_data:
            print("No data available for plotting")
            return
        
        # Create DataFrame
        df = pd.DataFrame(plot_data)
        
        # Simplify judge names for display
        judge_name_mapping = {
            'meta_llama_Llama_3_1_8B_Instruct': 'Llama-3.1-8B',
            'anthropic_claude_sonnet_4': 'Claude-4-Sonnet',
            'probe_classifier': 'Probe Classifier'
        }
        
        df['judge_display'] = df['judge_name'].map(
            lambda x: judge_name_mapping.get(x, x.replace('_', ' ').title())
        )
        
        # Mode order and labels
        mode_order = ['baseline', 'judge_only', 'debater_only', 'full_access', 
                     'probe_only_mean', 'probe_only_max']
        mode_labels = {
            'baseline': 'Baseline',
            'judge_only': 'Judge Only',
            'debater_only': 'Debater Access',
            'full_access': 'Full Access',
            'probe_only_mean': 'Probe Only (Mean)',
            'probe_only_max': 'Probe Only (Max)'
        }
        
        # Color map
        color_map = {
            'baseline': '#808080',
            'judge_only': '#3498db',
            'debater_only': '#e74c3c',
            'full_access': '#2ecc71',
            'probe_only_mean': '#f39c12',
            'probe_only_max': '#9b59b6'
        }
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Get unique judges
        judges = sorted(df['judge_display'].unique())
        
        # Setup bar positions
        x = np.arange(len(judges))
        width = 0.14  # Width of each bar
        
        # Plot bars for each mode
        for i, mode in enumerate(mode_order):
            mode_data = df[df['eval_mode'] == mode]
            if mode_data.empty:
                continue
            
            # Get win rates for each judge
            win_rates = []
            ns = []
            for judge in judges:
                judge_data = mode_data[mode_data['judge_display'] == judge]
                if not judge_data.empty:
                    win_rates.append(judge_data['win_rate'].values[0])
                    ns.append(judge_data['n'].values[0])
                else:
                    win_rates.append(0)
                    ns.append(0)
            
            # Calculate position offset
            offset = (i - len(mode_order)/2 + 0.5) * width
            
            # Plot bars
            bars = ax.bar(x + offset, win_rates, width, 
                          label=mode_labels.get(mode, mode),
                          color=color_map.get(mode, '#95a5a6'),
                          alpha=0.8)
            
            # Add sample size labels on bars
            for bar, n in zip(bars, ns):
                if n > 0:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'n={n}',
                           ha='center', va='bottom', fontsize=8)
        
        # Customize plot
        ax.set_xlabel('Judge Model', fontsize=14, fontweight='bold')
        ax.set_ylabel('Honest Debater Win Rate', fontsize=14, fontweight='bold')
        ax.set_title('Debate Performance Across Different Judges and Evaluation Modes',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(judges, fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.3, linewidth=1)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)
        
        # Add 50% reference line label
        ax.text(ax.get_xlim()[1] * 0.98, 0.52, 'Random (50%)', 
               ha='right', va='bottom', fontsize=9, alpha=0.6)
        
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / 'combined_win_rates.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved combined win rate plot: {save_path}")
    
    def calculate_debate_accuracy_metrics(self):
        """Calculate comprehensive accuracy metrics for each judge and evaluation mode"""
        
        metrics_results = {}
        
        for experiment_mode, eval_modes in self.results_by_mode.items():
            mode_metrics = {}
            
            for eval_mode, results_list in eval_modes.items():
                if not results_list:
                    continue
                
                # Group by judge
                judge_groups = defaultdict(list)
                for result in results_list:
                    judge_groups[result['judge_name']].append(result)
                
                eval_mode_metrics = {}
                
                for judge_name, judge_results in judge_groups.items():
                    # Extract outcomes
                    outcomes = []
                    for r in judge_results:
                        if r['correct'] is not None:
                            outcomes.append({
                                'correct': r['correct'],
                                'winner_role': r.get('winner_role'),
                                'debate_id': r['debate_id']
                            })
                    
                    if outcomes:
                        correct_count = sum(1 for o in outcomes if o['correct'])
                        total = len(outcomes)
                        accuracy = correct_count / total
                        
                        # Calculate confidence intervals
                        se = np.sqrt(accuracy * (1 - accuracy) / total)
                        ci_lower = max(0, accuracy - 1.96 * se)
                        ci_upper = min(1, accuracy + 1.96 * se)
                        
                        judge_key = f"{eval_mode}_{judge_name}"
                        eval_mode_metrics[judge_key] = {
                            'judge_name': judge_name,
                            'eval_mode': eval_mode,
                            'accuracy': float(accuracy),
                            'correct': int(correct_count),
                            'total': int(total),
                            'error_rate': float(1 - accuracy),
                            'confidence_interval': {
                                'lower': float(ci_lower),
                                'upper': float(ci_upper)
                            },
                            'standard_error': float(se)
                        }
                
                mode_metrics.update(eval_mode_metrics)
            
            metrics_results[experiment_mode] = mode_metrics
        
        # Save comprehensive metrics
        metrics_path = self.output_dir / 'debate_accuracy_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_results, f, indent=2)
        
        # Create detailed summary
        summary_path = self.output_dir / 'debate_accuracy_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("DEBATE ACCURACY METRICS - ALL JUDGES AND MODES\n")
            f.write("="*80 + "\n\n")
            
            for experiment_mode, metrics in metrics_results.items():
                f.write(f"Experiment Mode: {experiment_mode}\n")
                f.write("-"*80 + "\n\n")
                
                if metrics:
                    # Sort by evaluation mode then judge
                    sorted_keys = sorted(metrics.keys())
                    
                    # Group by evaluation mode for better readability
                    current_eval_mode = None
                    for key in sorted_keys:
                        metric = metrics[key]
                        if metric['eval_mode'] != current_eval_mode:
                            current_eval_mode = metric['eval_mode']
                            f.write(f"\n  Evaluation Mode: {current_eval_mode}\n")
                            f.write("  " + "-"*60 + "\n")
                        
                        judge_display = metric['judge_name'].replace('_', ' ').replace('meta llama', 'Llama').replace('anthropic', 'Claude')
                        f.write(f"    Judge: {judge_display}\n")
                        f.write(f"      Accuracy: {metric['accuracy']:.1%} ({metric['correct']}/{metric['total']})\n")
                        f.write(f"      95% CI: [{metric['confidence_interval']['lower']:.1%}, {metric['confidence_interval']['upper']:.1%}]\n")
                        f.write(f"      Error Rate: {metric['error_rate']:.1%}\n")
                        f.write("\n")
                
                f.write("\n")
            
            # Add comparative analysis
            f.write("\nCOMPARATIVE ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            for experiment_mode, metrics in metrics_results.items():
                if not metrics:
                    continue
                    
                f.write(f"Experiment Mode: {experiment_mode}\n")
                f.write("-"*80 + "\n")
                
                # Compare baseline vs judge_only for each judge
                judges = set()
                for key in metrics.keys():
                    judge_name = metrics[key]['judge_name']
                    judges.add(judge_name)
                
                for judge in judges:
                    baseline_key = f"baseline_{judge}"
                    judge_only_key = f"judge_only_{judge}"
                    
                    if baseline_key in metrics and judge_only_key in metrics:
                        baseline_acc = metrics[baseline_key]['accuracy']
                        judge_only_acc = metrics[judge_only_key]['accuracy']
                        improvement = judge_only_acc - baseline_acc
                        
                        judge_display = judge.replace('_', ' ')
                        f.write(f"\n  {judge_display}:\n")
                        f.write(f"    Baseline: {baseline_acc:.1%}\n")
                        f.write(f"    With Probes: {judge_only_acc:.1%}\n")
                        f.write(f"    Improvement: {improvement:+.1%}\n")
                
                # Add probe-only results if available
                probe_mean_key = "probe_only_mean_probe_classifier"
                probe_max_key = "probe_only_max_probe_classifier"
                
                if probe_mean_key in metrics or probe_max_key in metrics:
                    f.write(f"\n  Probe-Only Classification:\n")
                    if probe_mean_key in metrics:
                        f.write(f"    Mean Method: {metrics[probe_mean_key]['accuracy']:.1%}\n")
                    if probe_max_key in metrics:
                        f.write(f"    Max Method: {metrics[probe_max_key]['accuracy']:.1%}\n")
                
                f.write("\n")
        
        print(f"Saved debate accuracy metrics: {metrics_path}")
        print(f"Saved debate accuracy summary: {summary_path}")
        
        return metrics_results
    
    def calculate_probe_metrics(self):
        """Calculate comprehensive probe performance metrics with all variants"""
        
        all_results = {}
        
        for experiment_mode in self.results_by_mode.keys():
            mode_results = {}
            
            for transcript in self.transcripts:
                if transcript.get('experiment_mode') != experiment_mode:
                    continue
                
                debate_id = transcript['debate_id']
                
                # Check if we have manual or LLM labels for this debate
                manual_labels = self._load_manual_labels(debate_id)
                llm_labels = self._load_llm_labels(debate_id)
                
                # Run comprehensive analysis
                metrics = ProbeAnalyzer.analyze_transcript(
                    transcript, 
                    manual_labels=manual_labels,
                    llm_labels=llm_labels
                )
                
                # Convert to dict
                mode_results[debate_id] = {
                    k: v.to_dict() for k, v in metrics.items()
                }
            
            all_results[experiment_mode] = mode_results
        
        # Save per-debate comprehensive metrics
        metrics_path = self.output_dir / 'probe_metrics_comprehensive.json'
        with open(metrics_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Create aggregated summary across debates
        self._create_probe_summary(all_results)
        
        # Create comparison plots
        self._create_probe_comparison_plots(all_results)
        
        print(f"Saved comprehensive probe metrics: {metrics_path}")
        return all_results

    def _load_manual_labels(self, debate_id: str) -> Optional[Dict]:
        """Load manual labels from debate_viz export if available"""
        # Check for manual labels in a 'manual_labels' subdirectory
        manual_dir = self.output_dir.parent / 'manual_labels'
        if not manual_dir.exists():
            return None
        
        # Try to find matching file
        for label_file in manual_dir.glob(f"*{debate_id}*.json"):
            try:
                with open(label_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load {label_file}: {e}")
                continue

        return None

    def _load_llm_labels(self, debate_id: str) -> Optional[Dict]:
        """Load LLM-generated labels if available"""
        # Check for LLM labels in a 'llm_labels' subdirectory
        llm_dir = self.output_dir.parent / 'llm_labels'
        if not llm_dir.exists():
            return None

        # Try to find matching file
        for label_file in llm_dir.glob(f"*{debate_id}*labeled*.json"):
            try:
                with open(label_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load {label_file}: {e}")
                continue
        
        return None

    def _create_probe_summary(self, all_results: Dict):
        """Aggregate probe metrics across debates with comprehensive statistics"""
        
        summary = {}
        
        for mode, debates in all_results.items():
            variant_data = {}
            
            # Collect metrics by variant
            for debate_id, metrics in debates.items():
                for variant_name, variant_metrics in metrics.items():
                    if variant_name not in variant_data:
                        variant_data[variant_name] = {
                            'aurocs': [], 'auprcs': [], 'f1s': [], 'accuracies': [],
                            'precisions': [], 'recalls': [], 'specificities': [],
                            'fprs': [], 'tprs': [],
                            'separations': [], 'n_samples': []
                        }
                    
                    variant_data[variant_name]['aurocs'].append(variant_metrics['auroc'])
                    variant_data[variant_name]['auprcs'].append(variant_metrics['auprc'])
                    variant_data[variant_name]['f1s'].append(variant_metrics['f1'])
                    variant_data[variant_name]['accuracies'].append(variant_metrics['accuracy'])
                    variant_data[variant_name]['precisions'].append(variant_metrics['precision'])
                    variant_data[variant_name]['recalls'].append(variant_metrics['recall'])
                    variant_data[variant_name]['specificities'].append(variant_metrics['specificity'])
                    variant_data[variant_name]['fprs'].append(variant_metrics['fpr'])
                    variant_data[variant_name]['tprs'].append(variant_metrics['tpr'])
                    variant_data[variant_name]['separations'].append(variant_metrics['score_separation'])
                    variant_data[variant_name]['n_samples'].append(variant_metrics['n_samples'])
            
            # Aggregate statistics
            summary[mode] = {}
            for variant_name, data in variant_data.items():
                summary[mode][variant_name] = {
                    # AUROC
                    'mean_auroc': float(np.mean(data['aurocs'])),
                    'std_auroc': float(np.std(data['aurocs'])),
                    'median_auroc': float(np.median(data['aurocs'])),
                    'min_auroc': float(np.min(data['aurocs'])),
                    'max_auroc': float(np.max(data['aurocs'])),
                    
                    # AUPRC
                    'mean_auprc': float(np.mean(data['auprcs'])),
                    'std_auprc': float(np.std(data['auprcs'])),
                    'median_auprc': float(np.median(data['auprcs'])),
                    
                    # F1
                    'mean_f1': float(np.mean(data['f1s'])),
                    'std_f1': float(np.std(data['f1s'])),
                    'median_f1': float(np.median(data['f1s'])),
                    
                    # Accuracy
                    'mean_accuracy': float(np.mean(data['accuracies'])),
                    'std_accuracy': float(np.std(data['accuracies'])),
                    'median_accuracy': float(np.median(data['accuracies'])),
                    
                    # Precision
                    'mean_precision': float(np.mean(data['precisions'])),
                    'std_precision': float(np.std(data['precisions'])),
                    
                    # Recall
                    'mean_recall': float(np.mean(data['recalls'])),
                    'std_recall': float(np.std(data['recalls'])),
                    
                    # Specificity
                    'mean_specificity': float(np.mean(data['specificities'])),
                    'std_specificity': float(np.std(data['specificities'])),
                    
                    # FPR/TPR
                    'mean_fpr': float(np.mean(data['fprs'])),
                    'mean_tpr': float(np.mean(data['tprs'])),
                    
                    # Separation
                    'mean_separation': float(np.mean(data['separations'])),
                    'std_separation': float(np.std(data['separations'])),
                    
                    # Counts
                    'total_samples': int(np.sum(data['n_samples'])),
                    'n_debates': len(data['aurocs'])
                }
        
        # Save comprehensive JSON
        summary_path = self.output_dir / 'probe_metrics_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create detailed text summary
        text_path = self.output_dir / 'probe_metrics_summary.txt'
        with open(text_path, 'w') as f:
            f.write("COMPREHENSIVE PROBE PERFORMANCE SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            for mode, variants in summary.items():
                f.write(f"Experiment Mode: {mode}\n")
                f.write("-"*80 + "\n\n")
                
                for var_name, metrics in sorted(variants.items()):
                    f.write(f"{var_name}:\n")
                    f.write(f"  N: {metrics['n_debates']} debates, {metrics['total_samples']} samples\n\n")
                    
                    f.write(f"  AUROC: {metrics['mean_auroc']:.3f} ± {metrics['std_auroc']:.3f}\n")
                    f.write(f"    Range: [{metrics['min_auroc']:.3f}, {metrics['max_auroc']:.3f}]\n")
                    f.write(f"    Median: {metrics['median_auroc']:.3f}\n\n")
                    
                    f.write(f"  AUPRC: {metrics['mean_auprc']:.3f} ± {metrics['std_auprc']:.3f}\n")
                    f.write(f"    Median: {metrics['median_auprc']:.3f}\n\n")
                    
                    f.write(f"  Accuracy: {metrics['mean_accuracy']:.3f} ± {metrics['std_accuracy']:.3f}\n")
                    f.write(f"  F1 Score: {metrics['mean_f1']:.3f} ± {metrics['std_f1']:.3f}\n")
                    f.write(f"  Precision: {metrics['mean_precision']:.3f} ± {metrics['std_precision']:.3f}\n")
                    f.write(f"  Recall (TPR): {metrics['mean_recall']:.3f} ± {metrics['std_recall']:.3f}\n")
                    f.write(f"  Specificity (TNR): {metrics['mean_specificity']:.3f} ± {metrics['std_specificity']:.3f}\n")
                    f.write(f"  FPR: {metrics['mean_fpr']:.3f}\n\n")
                    
                    f.write(f"  Score Separation: {metrics['mean_separation']:.3f} ± {metrics['std_separation']:.3f}\n")
                    f.write("\n" + "-"*40 + "\n\n")
                
                f.write("\n")
        
        print(f"Saved probe summary: {summary_path}")
        print(f"Saved text summary: {text_path}")

    def _create_probe_comparison_plots(self, all_results: Dict):
        """Create comprehensive comparison plots for probe performance"""
        
        # Load summary for plotting
        summary_path = self.output_dir / 'probe_metrics_summary.json'
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        # Create comprehensive figure
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('Comprehensive Probe Performance Comparison', 
                     fontsize=16, fontweight='bold', y=0.995)
        
        for mode_idx, (mode, variants) in enumerate(summary.items()):
            # Sort variants for consistent ordering
            sorted_variants = sorted(variants.items())
            names = [v[0].replace('_', '\n') for v in sorted_variants]
            
            # Prepare data arrays
            aurocs = [v[1]['mean_auroc'] for v in sorted_variants]
            auroc_stds = [v[1]['std_auroc'] for v in sorted_variants]
            auprcs = [v[1]['mean_auprc'] for v in sorted_variants]
            auprc_stds = [v[1]['std_auprc'] for v in sorted_variants]
            f1s = [v[1]['mean_f1'] for v in sorted_variants]
            f1_stds = [v[1]['std_f1'] for v in sorted_variants]
            accs = [v[1]['mean_accuracy'] for v in sorted_variants]
            acc_stds = [v[1]['std_accuracy'] for v in sorted_variants]
            fprs = [v[1]['mean_fpr'] for v in sorted_variants]
            tprs = [v[1]['mean_tpr'] for v in sorted_variants]
            
            x = np.arange(len(names))
            width = 0.35 if len(summary) > 1 else 0.6
            offset = width * mode_idx if len(summary) > 1 else 0
            
            # AUROC
            axes[0, 0].bar(x + offset, aurocs, width, yerr=auroc_stds, 
                           capsize=5, label=mode, alpha=0.7)
            axes[0, 0].set_ylabel('AUROC', fontweight='bold')
            axes[0, 0].set_title('Area Under ROC Curve')
            axes[0, 0].set_ylim(0, 1.05)
            axes[0, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Random')
            axes[0, 0].grid(alpha=0.3, axis='y')
            
            # AUPRC
            axes[0, 1].bar(x + offset, auprcs, width, yerr=auprc_stds,
                           capsize=5, label=mode, alpha=0.7)
            axes[0, 1].set_ylabel('AUPRC', fontweight='bold')
            axes[0, 1].set_title('Area Under Precision-Recall Curve')
            axes[0, 1].set_ylim(0, 1.05)
            axes[0, 1].grid(alpha=0.3, axis='y')
            
            # F1 Score
            axes[1, 0].bar(x + offset, f1s, width, yerr=f1_stds,
                           capsize=5, label=mode, alpha=0.7)
            axes[1, 0].set_ylabel('F1 Score', fontweight='bold')
            axes[1, 0].set_title('F1 Score at Optimal Threshold')
            axes[1, 0].set_ylim(0, 1.05)
            axes[1, 0].grid(alpha=0.3, axis='y')
            
            # Accuracy
            axes[1, 1].bar(x + offset, accs, width, yerr=acc_stds,
                           capsize=5, label=mode, alpha=0.7)
            axes[1, 1].set_ylabel('Accuracy', fontweight='bold')
            axes[1, 1].set_title('Accuracy at Optimal Threshold')
            axes[1, 1].set_ylim(0, 1.05)
            axes[1, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Random')
            axes[1, 1].grid(alpha=0.3, axis='y')
            
            # TPR
            axes[2, 0].bar(x + offset, tprs, width, label=mode, alpha=0.7)
            axes[2, 0].set_ylabel('True Positive Rate', fontweight='bold')
            axes[2, 0].set_title('TPR (Recall) at Optimal Threshold')
            axes[2, 0].set_ylim(0, 1.05)
            axes[2, 0].grid(alpha=0.3, axis='y')
            
            # FPR
            axes[2, 1].bar(x + offset, fprs, width, label=mode, alpha=0.7)
            axes[2, 1].set_ylabel('False Positive Rate', fontweight='bold')
            axes[2, 1].set_title('FPR at Optimal Threshold')
            axes[2, 1].set_ylim(0, 1.05)
            axes[2, 1].grid(alpha=0.3, axis='y')
        
        # Set x-axis labels for all plots
        for ax in axes.flat:
            ax.set_xticks(x + width/2 if len(summary) > 1 else x)
            ax.set_xticklabels(names, fontsize=8, rotation=15, ha='right')
            ax.legend(loc='best', fontsize=9)
        
        plt.tight_layout()
        save_path = self.output_dir / 'probe_comparison_comprehensive.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved comprehensive probe comparison plot: {save_path}")
    
    def _create_roc_pr_curves(self, y_true, y_scores, experiment_mode):
        """Create ROC and PR curves for probe performance"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auroc = roc_auc_score(y_true, y_scores)
        
        ax1.plot(fpr, tpr, color='#3498db', linewidth=2, 
                label=f'ROC Curve (AUROC = {auroc:.3f})')
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
        ax1.fill_between(fpr, tpr, alpha=0.2, color='#3498db')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate', fontsize=12)
        ax1.set_ylabel('True Positive Rate', fontsize=12)
        ax1.set_title('Receiver Operating Characteristic', fontsize=14, fontweight='bold')
        ax1.legend(loc="lower right")
        ax1.grid(alpha=0.3)
        
        # PR Curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        auprc = average_precision_score(y_true, y_scores)
        
        ax2.plot(recall, precision, color='#e74c3c', linewidth=2,
                label=f'PR Curve (AUPRC = {auprc:.3f})')
        ax2.fill_between(recall, precision, alpha=0.2, color='#e74c3c')
        
        # Add baseline (random classifier)
        baseline = np.sum(y_true) / len(y_true)
        ax2.axhline(y=baseline, color='k', linestyle='--', alpha=0.3, linewidth=1,
                   label=f'Random (AP = {baseline:.3f})')
        
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall', fontsize=12)
        ax2.set_ylabel('Precision', fontsize=12)
        ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax2.legend(loc="lower left")
        ax2.grid(alpha=0.3)
        
        plt.suptitle(f'Probe Performance Curves - {experiment_mode}', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / f'roc_pr_curves_{experiment_mode}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved ROC/PR curves: {save_path}")
    
    def calculate_statistical_significance(self):
        """Calculate statistical significance between different modes"""
        
        results_summary = {}
        
        for experiment_mode, eval_modes in self.results_by_mode.items():
            mode_results = {}
            
            # Extract correctness for each mode
            for eval_mode, results_list in eval_modes.items():
                if results_list:
                    correctness = [r['correct'] for r in results_list if r['correct'] is not None]
                    mode_results[eval_mode] = correctness
            
            # Define comparisons
            comparisons = [
                ('judge_only', 'baseline', 'Do probes improve judge performance?'),
                ('debater_only', 'baseline', 'Do probes help debaters?'),
                ('full_access', 'baseline', 'Does full probe access improve debates?'),
                ('judge_only', 'probe_only_mean', 'Judge vs Probe classifier (mean)'),
                ('full_access', 'probe_only_mean', 'Full access vs Probe classifier (mean)'),
            ]
            
            comparison_results = []
            
            for mode1, mode2, description in comparisons:
                if mode1 in mode_results and mode2 in mode_results:
                    data1 = mode_results[mode1]
                    data2 = mode_results[mode2]
                    
                    if len(data1) > 0 and len(data2) > 0:
                        # Calculate win rates
                        rate1 = np.mean(data1)
                        rate2 = np.mean(data2)
                        
                        # McNemar's test for paired data
                        try:
                            # Two-proportion z-test
                            count1 = sum(data1)
                            count2 = sum(data2)
                            n1 = len(data1)
                            n2 = len(data2)
                            
                            p_pooled = (count1 + count2) / (n1 + n2)
                            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
                            
                            if se > 0:
                                z = (rate1 - rate2) / se
                                p_value = 2 * (1 - stats.norm.cdf(abs(z)))
                                test_name = "Two-proportion z-test"
                            else:
                                p_value = 1.0
                                test_name = "Two-proportion (no variance)"
                            
                            comparison_results.append({
                                'comparison': description,
                                'mode1': mode1,
                                'mode2': mode2,
                                'rate1': float(rate1),
                                'rate2': float(rate2),
                                'difference': float(rate1 - rate2),
                                'p_value': float(p_value),
                                'significant': bool(p_value < 0.05),
                                'test': test_name,
                                'n1': int(n1),
                                'n2': int(n2)
                            })
                            
                        except Exception as e:
                            print(f"Warning: Statistical test failed for {description}: {e}")
            
            results_summary[experiment_mode] = comparison_results
        
        # Save results
        summary_path = self.output_dir / 'statistical_significance.json'
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Create readable summary
        readable_path = self.output_dir / 'statistical_significance.txt'
        with open(readable_path, 'w') as f:
            f.write("STATISTICAL SIGNIFICANCE ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            for experiment_mode, comparisons in results_summary.items():
                f.write(f"Experiment Mode: {experiment_mode}\n")
                f.write("-"*80 + "\n\n")
                
                if comparisons:
                    for comp in comparisons:
                        f.write(f"{comp['comparison']}\n")
                        f.write(f"  {comp['mode1']}: {comp['rate1']:.3f} (n={comp['n1']})\n")
                        f.write(f"  {comp['mode2']}: {comp['rate2']:.3f} (n={comp['n2']})\n")
                        f.write(f"  Difference: {comp['difference']:+.3f}\n")
                        f.write(f"  Test: {comp['test']}\n")
                        f.write(f"  p-value: {comp['p_value']:.4f} {'***' if comp['p_value'] < 0.001 else '**' if comp['p_value'] < 0.01 else '*' if comp['p_value'] < 0.05 else 'ns'}\n")
                        f.write(f"  Significant: {'YES' if comp['significant'] else 'NO'}\n\n")
                else:
                    f.write("No comparisons available for this mode.\n\n")
                
                f.write("\n")
        
        print(f"Saved statistical significance analysis: {summary_path}")
        print(f"Saved readable summary: {readable_path}")
    
    def create_html_visualizations(self, max_debates: int = 10):
        """Create HTML chat visualizations for debates"""
        
        viz_dir = self.output_dir / 'html_visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # Select diverse debates to visualize
        debates_to_viz = self.transcripts[:max_debates]
        
        for i, transcript in enumerate(debates_to_viz):
            debate_id = transcript['debate_id']
            output_file = viz_dir / f"{debate_id}.html"
            
            try:
                create_debate_chat_visualization(transcript, output_file)
                print(f"Created visualization: {output_file}")
            except Exception as e:
                print(f"Warning: Could not create visualization for {debate_id}: {e}")
        
        # Create index page
        self._create_index_page(viz_dir, debates_to_viz)
    
    def _create_index_page(self, viz_dir: Path, debates: List[Dict]):
        """Create index page linking to all visualizations"""
        
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Debate Visualizations Index</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .debate-list {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .debate-item {
            padding: 15px;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .debate-item:last-child {
            border-bottom: none;
        }
        .debate-info {
            flex-grow: 1;
        }
        .debate-id {
            font-weight: bold;
            color: #2c3e50;
        }
        .debate-meta {
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }
        .view-button {
            padding: 8px 16px;
            background-color: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            transition: background-color 0.3s;
        }
        .view-button:hover {
            background-color: #2980b9;
        }
    </style>
</head>
<body>
    <h1>Debate Visualizations</h1>
    <div class="debate-list">
"""
        
        for debate in debates:
            debate_id = debate['debate_id']
            debate_type = debate.get('debate_type', 'unknown')
            experiment_mode = debate.get('experiment_mode', 'unknown')
            
            html += f"""
        <div class="debate-item">
            <div class="debate-info">
                <div class="debate-id">{debate_id}</div>
                <div class="debate-meta">
                    Type: {debate_type} | Mode: {experiment_mode}
                </div>
            </div>
            <a href="{debate_id}.html" class="view-button">View Debate</a>
        </div>
"""
        
        html += """
    </div>
</body>
</html>
"""
        
        index_path = viz_dir / 'index.html'
        with open(index_path, 'w') as f:
            f.write(html)
        
        print(f"Created index page: {index_path}")
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        
        print("\n" + "="*80)
        print("DEBATE RESULTS ANALYSIS")
        print("="*80)
        
        print("\n1. Creating combined win rate plot...")
        self.create_combined_win_rate_bar_plot()
        
        print("\n2. Calculating debate accuracy metrics...")
        self.calculate_debate_accuracy_metrics()
        
        print("\n3. Calculating comprehensive probe metrics...")
        self.calculate_probe_metrics()
        
        print("\n4. Calculating statistical significance...")
        self.calculate_statistical_significance()
        
        print("\n5. Creating HTML visualizations...")
        self.create_html_visualizations()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze debate results with visualizations and statistical tests'
    )
    
    parser.add_argument(
        '--transcripts_dir',
        type=str,
        required=True,
        help='Directory containing debate transcript JSON files'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save analysis results'
    )
    
    parser.add_argument(
        '--max_visualizations',
        type=int,
        default=10,
        help='Maximum number of debates to create HTML visualizations for'
    )
    
    args = parser.parse_args()
    
    transcripts_dir = Path(args.transcripts_dir)
    output_dir = Path(args.output_dir)
    
    if not transcripts_dir.exists():
        print(f"Error: Transcripts directory not found: {transcripts_dir}")
        return 1
    
    analyzer = DebateResultsAnalyzer(transcripts_dir, output_dir)
    analyzer.run_full_analysis()
    
    return 0


if __name__ == '__main__':
    exit(main())