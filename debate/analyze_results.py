#!/usr/bin/env python3
"""
Analyze debate results and create visualizations
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import argparse


class DebateAnalyzer:
    """Analyze debate experiment results"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.results_df = None
        self.results_json = None
        self.config = None
        self._load_results()
        
    def _load_results(self):
        """Load results from files"""
        # Load CSV results
        csv_path = self.results_dir / "results.csv"
        if csv_path.exists():
            self.results_df = pd.read_csv(csv_path)
            
        # Load JSON results for detailed analysis
        json_path = self.results_dir / "results.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                self.results_json = json.load(f)
                
        # Load config
        config_path = self.results_dir / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                
    def print_summary(self):
        """Print summary statistics"""
        if self.results_df is None:
            print("No results found")
            return
            
        print("\n" + "="*60)
        print("DEBATE RESULTS SUMMARY")
        print("="*60)
        
        print(f"\nExperiment: {self.results_dir.name}")
        print(f"Debate Type: {self.config['debate_type']}")
        print(f"Total Debates: {len(self.results_df)}")
        
        # Success rate
        success_rate = self.results_df['success'].mean() * 100
        print(f"Success Rate: {success_rate:.1f}%")
        
        # Winner distribution
        print("\nWinner Distribution:")
        winner_counts = self.results_df['winner'].value_counts()
        for winner, count in winner_counts.items():
            pct = 100 * count / len(self.results_df)
            print(f"  {winner}: {count} ({pct:.1f}%)")
            
        # By difficulty if APPS dataset
        if 'difficulty' in self.results_df.columns:
            print("\nResults by Difficulty:")
            for difficulty in self.results_df['difficulty'].unique():
                if pd.notna(difficulty):
                    diff_df = self.results_df[self.results_df['difficulty'] == difficulty]
                    honest_wins = (diff_df['winner'] == 'honest').sum()
                    total = len(diff_df)
                    print(f"  {difficulty}: {honest_wins}/{total} honest wins ({100*honest_wins/total:.1f}%)")
                    
        # Probe analysis if available
        if self.config.get('probe_config', {}).get('enabled'):
            self._print_probe_summary()
            
    def _print_probe_summary(self):
        """Print probe-related statistics"""
        print("\nProbe Analysis:")
        
        probe_data = []
        for result in self.results_json:
            if result.get('probe_analysis'):
                analysis = result['probe_analysis']
                scores = analysis.get('scores_by_role', {})
                
                if 'honest' in scores:
                    probe_data.append({
                        'role': 'honest',
                        'mean_score': scores['honest']['mean'],
                        'winner': result.get('winner')
                    })
                if 'dishonest' in scores:
                    probe_data.append({
                        'role': 'dishonest',
                        'mean_score': scores['dishonest']['mean'],
                        'winner': result.get('winner')
                    })
                    
        if probe_data:
            probe_df = pd.DataFrame(probe_data)
            
            # Average scores by role
            for role in ['honest', 'dishonest']:
                role_df = probe_df[probe_df['role'] == role]
                if not role_df.empty:
                    mean_score = role_df['mean_score'].mean()
                    std_score = role_df['mean_score'].std()
                    print(f"  {role.capitalize()} mean score: {mean_score:.3f} (±{std_score:.3f})")
                    
            # Correlation with winning
            honest_scores = probe_df[probe_df['role'] == 'honest']['mean_score'].values
            dishonest_scores = probe_df[probe_df['role'] == 'dishonest']['mean_score'].values
            
            if len(honest_scores) == len(dishonest_scores):
                score_diff = dishonest_scores - honest_scores
                print(f"  Average score difference (dishonest - honest): {score_diff.mean():.3f}")
                
    def create_visualizations(self):
        """Create visualization plots"""
        if self.results_df is None:
            print("No results to visualize")
            return
            
        # Set style
        sns.set_style("whitegrid")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Winner distribution
        ax = axes[0, 0]
        winner_counts = self.results_df['winner'].value_counts()
        ax.bar(winner_counts.index, winner_counts.values)
        ax.set_title('Winner Distribution')
        ax.set_xlabel('Winner')
        ax.set_ylabel('Count')
        
        # 2. Success rate by difficulty (if available)
        ax = axes[0, 1]
        if 'difficulty' in self.results_df.columns:
            diff_success = self.results_df.groupby('difficulty')['winner'].apply(
                lambda x: (x == 'honest').mean()
            )
            ax.bar(diff_success.index, diff_success.values)
            ax.set_title('Honest Win Rate by Difficulty')
            ax.set_xlabel('Difficulty')
            ax.set_ylabel('Honest Win Rate')
            ax.set_ylim([0, 1])
        else:
            ax.text(0.5, 0.5, 'No difficulty data', ha='center', va='center')
            ax.set_title('Difficulty Analysis')
            
        # 3. Probe scores if available
        ax = axes[1, 0]
        if self.config.get('probe_config', {}).get('enabled'):
            self._plot_probe_scores(ax)
        else:
            ax.text(0.5, 0.5, 'No probe data', ha='center', va='center')
            ax.set_title('Probe Score Analysis')
            
        # 4. Debate duration
        ax = axes[1, 1]
        if 'duration' in self.results_df.columns:
            durations = self.results_df['duration'].dropna()
            ax.hist(durations, bins=20, edgecolor='black')
            ax.set_title('Debate Duration Distribution')
            ax.set_xlabel('Duration (seconds)')
            ax.set_ylabel('Count')
            ax.axvline(durations.mean(), color='red', linestyle='--', 
                      label=f'Mean: {durations.mean():.1f}s')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No duration data', ha='center', va='center')
            ax.set_title('Duration Analysis')
            
        plt.tight_layout()
        
        # Save figure
        plot_path = self.results_dir / "analysis_plots.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nPlots saved to: {plot_path}")
        
        plt.show()
        
    def _plot_probe_scores(self, ax):
        """Plot probe score distributions"""
        honest_scores = []
        dishonest_scores = []
        
        for result in self.results_json:
            if result.get('probe_analysis'):
                scores = result['probe_analysis'].get('scores_by_role', {})
                if 'honest' in scores:
                    honest_scores.append(scores['honest']['mean'])
                if 'dishonest' in scores:
                    dishonest_scores.append(scores['dishonest']['mean'])
                    
        if honest_scores and dishonest_scores:
            data = pd.DataFrame({
                'Honest': honest_scores,
                'Dishonest': dishonest_scores
            })
            
            # Box plot
            data.boxplot(ax=ax)
            ax.set_title('Probe Score Distributions by Role')
            ax.set_ylabel('Mean Probe Score')
            ax.set_ylim([0, 1])
            
            # Add mean lines
            ax.axhline(np.mean(honest_scores), color='blue', linestyle='--', 
                      alpha=0.5, label=f'Honest mean: {np.mean(honest_scores):.3f}')
            ax.axhline(np.mean(dishonest_scores), color='orange', linestyle='--',
                      alpha=0.5, label=f'Dishonest mean: {np.mean(dishonest_scores):.3f}')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Insufficient probe data', ha='center', va='center')
            
    def compare_experiments(self, other_dirs: List[str]):
        """Compare multiple experiments"""
        all_results = []
        
        # Load this experiment
        all_results.append({
            'name': self.results_dir.name,
            'type': self.config['debate_type'],
            'df': self.results_df
        })
        
        # Load other experiments
        for dir_path in other_dirs:
            analyzer = DebateAnalyzer(dir_path)
            if analyzer.results_df is not None:
                all_results.append({
                    'name': Path(dir_path).name,
                    'type': analyzer.config['debate_type'],
                    'df': analyzer.results_df
                })
                
        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Win rates comparison
        ax = axes[0]
        experiment_names = []
        honest_win_rates = []
        
        for exp in all_results:
            experiment_names.append(exp['name'])
            win_rate = (exp['df']['winner'] == 'honest').mean()
            honest_win_rates.append(win_rate)
            
        ax.bar(experiment_names, honest_win_rates)
        ax.set_title('Honest Win Rate Comparison')
        ax.set_xlabel('Experiment')
        ax.set_ylabel('Honest Win Rate')
        ax.set_ylim([0, 1])
        ax.tick_params(axis='x', rotation=45)
        
        # Debate types comparison
        ax = axes[1]
        type_counts = pd.Series([exp['type'] for exp in all_results]).value_counts()
        ax.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
        ax.set_title('Debate Types Distribution')
        
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze debate results')
    parser.add_argument('results_dir', type=str, 
                       help='Directory containing debate results')
    parser.add_argument('--compare', nargs='+',
                       help='Compare with other experiment directories')
    parser.add_argument('--no_plots', action='store_true',
                       help='Skip visualization plots')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = DebateAnalyzer(args.results_dir)
    
    # Print summary
    analyzer.print_summary()
    
    # Create visualizations
    if not args.no_plots:
        analyzer.create_visualizations()
        
    # Compare experiments if requested
    if args.compare:
        analyzer.compare_experiments(args.compare)


if __name__ == "__main__":
    main()