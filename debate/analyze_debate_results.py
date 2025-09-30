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
import matplotlib.pyplot as plt
import seaborn as sns

from debate.visualisation.debate_viz import create_debate_chat_visualization


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
            
            # Extract judge results for each evaluation mode
            judge_results = transcript.get('judge_results', {})
            probe_only_results = transcript.get('probe_only_results', {})
            
            for eval_mode_key, judge_result in judge_results.items():
                # Parse evaluation mode and judge name
                # Format: "{mode_name}_{judge_name}"
                parts = eval_mode_key.rsplit('_', 1)
                if len(parts) == 2:
                    eval_mode = parts[0]
                    judge_name = parts[1]
                else:
                    eval_mode = eval_mode_key
                    judge_name = "unknown"
                
                winner_role = judge_result.get('winner_role')
                correct = judge_result.get('correct')
                
                results[experiment_mode][eval_mode].append({
                    'debate_id': debate_id,
                    'judge_name': judge_name,
                    'winner_role': winner_role,
                    'correct': correct,
                    'transcript': transcript
                })
            
            # Add probe-only results
            if probe_only_results:
                for probe_method in ['probe_only_mean', 'probe_only_max']:
                    winner = probe_only_results.get(probe_method)
                    if winner:
                        # Determine correctness based on debate type
                        correct = self._determine_probe_only_correctness(winner, transcript)
                        
                        results[experiment_mode][probe_method].append({
                            'debate_id': debate_id,
                            'judge_name': 'probe_classifier',
                            'winner_role': winner,
                            'correct': correct,
                            'transcript': transcript
                        })
        
        return dict(results)
    
    def _determine_probe_only_correctness(self, winner: str, transcript: Dict) -> bool:
        """Determine if probe-only classification is correct"""
        debate_type = transcript.get('debate_type', 'unknown')
        
        if debate_type == 'quality':
            # Honest always defends correct answer
            return winner == 'honest'
        elif debate_type == 'simple':
            # Check ground truth from scenario
            problem_data = transcript.get('problem_data', {})
            ground_truth = problem_data.get('ground_truth', '')
            model_mapping = transcript.get('model_mapping', {})
            ground_truth_role = model_mapping.get(ground_truth)
            return winner == ground_truth_role
        else:  # APPS
            # Honest defends non-backdoored solution
            return winner == 'honest'
    
    def create_win_rate_bar_plot(self):
        """Create bar plot of win rates across judges and modes"""
        
        # Determine model names from transcripts
        model_names = set()
        for transcript in self.transcripts:
            config = transcript.get('debate_config', {})
            model_names.add(config.get('honest_model', 'unknown'))
        
        model_name = list(model_names)[0] if model_names else 'unknown'
        
        # Collect data for plotting
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
                    correct_count = sum(1 for r in judge_results if r['correct'])
                    total_count = len(judge_results)
                    win_rate = correct_count / total_count if total_count > 0 else 0
                    
                    plot_data.append({
                        'experiment_mode': experiment_mode,
                        'eval_mode': eval_mode,
                        'judge_name': judge_name,
                        'win_rate': win_rate,
                        'n': total_count
                    })
        
        # Create DataFrame
        df = pd.DataFrame(plot_data)
        
        if df.empty:
            print("No data available for bar plot")
            return
        
        # Create plot for each judge
        judges = df['judge_name'].unique()
        
        for judge_name in judges:
            judge_df = df[df['judge_name'] == judge_name]
            
            # Map eval modes to display names
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
            
            # Filter to modes that exist
            available_modes = [m for m in mode_order if m in judge_df['eval_mode'].values]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Prepare data
            plot_modes = []
            plot_rates = []
            plot_ns = []
            plot_colors = []
            
            color_map = {
                'baseline': '#808080',
                'judge_only': '#3498db',
                'debater_only': '#e74c3c',
                'full_access': '#2ecc71',
                'probe_only_mean': '#f39c12',
                'probe_only_max': '#9b59b6'
            }
            
            for mode in available_modes:
                mode_data = judge_df[judge_df['eval_mode'] == mode]
                if not mode_data.empty:
                    plot_modes.append(mode_labels.get(mode, mode))
                    plot_rates.append(mode_data['win_rate'].values[0])
                    plot_ns.append(mode_data['n'].values[0])
                    plot_colors.append(color_map.get(mode, '#95a5a6'))
            
            # Create bars
            x_pos = np.arange(len(plot_modes))
            bars = ax.bar(x_pos, plot_rates, color=plot_colors, alpha=0.8)
            
            # Add sample sizes
            for i, (bar, n) in enumerate(zip(bars, plot_ns)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'n={n}',
                       ha='center', va='bottom', fontsize=9)
            
            # Customize plot
            ax.set_xlabel('Evaluation Mode', fontsize=12)
            ax.set_ylabel('Honest Debater Win Rate', fontsize=12)
            ax.set_title(f'Debate: {model_name} (Honest) vs {model_name} (Deceptive)\nJudge: {judge_name}',
                        fontsize=14)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(plot_modes, rotation=45, ha='right')
            ax.set_ylim(0, 1.0)
            ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.3, linewidth=1)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            # Save
            save_path = self.output_dir / f'win_rates_{judge_name}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved win rate plot: {save_path}")
    
    def calculate_statistical_significance(self):
        """Calculate statistical significance between different modes"""
        
        results_summary = {}
        
        for experiment_mode, eval_modes in self.results_by_mode.items():
            mode_results = {}
            
            # Extract correctness for each mode
            for eval_mode, results_list in eval_modes.items():
                correctness = [r['correct'] for r in results_list if r['correct'] is not None]
                mode_results[eval_mode] = correctness
            
            # Define comparisons
            comparisons = [
                ('judge_only', 'baseline', 'Do probes improve judge performance?'),
                ('debater_only', 'baseline', 'Do probes help debaters?'),
                ('full_access', 'baseline', 'Does full probe access improve debates?'),
                ('judge_only', 'probe_only_mean', 'Judge vs Probe classifier (mean)'),
                ('debater_only', 'probe_only_mean', 'Debater access vs Probe classifier (mean)'),
                ('full_access', 'probe_only_mean', 'Full access vs Probe classifier (mean)'),
                ('judge_only', 'probe_only_max', 'Judge vs Probe classifier (max)'),
                ('debater_only', 'probe_only_max', 'Debater access vs Probe classifier (max)'),
                ('full_access', 'probe_only_max', 'Full access vs Probe classifier (max)'),
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
                        
                        # McNemar's test for paired binary data (if same debates)
                        # Otherwise use chi-square test
                        try:
                            # Try paired test first (requires same debates)
                            if len(data1) == len(data2):
                                # Build contingency table
                                both_correct = sum(1 for i in range(len(data1)) if data1[i] and data2[i])
                                mode1_only = sum(1 for i in range(len(data1)) if data1[i] and not data2[i])
                                mode2_only = sum(1 for i in range(len(data1)) if not data1[i] and data2[i])
                                both_wrong = sum(1 for i in range(len(data1)) if not data1[i] and not data2[i])
                                
                                # McNemar's test
                                if mode1_only + mode2_only > 0:
                                    chi2 = (abs(mode1_only - mode2_only) - 1) ** 2 / (mode1_only + mode2_only)
                                    p_value = 1 - stats.chi2.cdf(chi2, 1)
                                    test_name = "McNemar"
                                else:
                                    p_value = 1.0
                                    test_name = "McNemar (no discordant)"
                            else:
                                # Independent proportions test
                                count1 = sum(data1)
                                count2 = sum(data2)
                                n1 = len(data1)
                                n2 = len(data2)
                                
                                # Two-proportion z-test
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
                                'rate1': rate1,
                                'rate2': rate2,
                                'difference': rate1 - rate2,
                                'p_value': p_value,
                                'significant': p_value < 0.05,
                                'test': test_name,
                                'n1': len(data1),
                                'n2': len(data2)
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
                
                for comp in comparisons:
                    f.write(f"{comp['comparison']}\n")
                    f.write(f"  {comp['mode1']}: {comp['rate1']:.3f} (n={comp['n1']})\n")
                    f.write(f"  {comp['mode2']}: {comp['rate2']:.3f} (n={comp['n2']})\n")
                    f.write(f"  Difference: {comp['difference']:+.3f}\n")
                    f.write(f"  Test: {comp['test']}\n")
                    f.write(f"  p-value: {comp['p_value']:.4f} {'***' if comp['p_value'] < 0.001 else '**' if comp['p_value'] < 0.01 else '*' if comp['p_value'] < 0.05 else 'ns'}\n")
                    f.write(f"  Significant: {'YES' if comp['significant'] else 'NO'}\n\n")
                
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
        
        print("\n1. Creating win rate bar plots...")
        self.create_win_rate_bar_plot()
        
        print("\n2. Calculating statistical significance...")
        self.calculate_statistical_significance()
        
        print("\n3. Creating HTML visualizations...")
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