#!/usr/bin/env python3
"""
Combined analysis of debate results across multiple experiment modes.

This script:
1. Loads transcripts from multiple experiment runs (baseline_judge, debater_full)
2. Matches samples by question_id to create paired comparisons
3. Computes per-judge McNemar tests for statistical significance
4. Creates unified 4-mode bar plots
5. Calculates AUROC and other metrics for probe performance

Usage:
    python debate/analyze_combined_results.py \
        --baseline_dir results/baseline_judge_run/transcripts \
        --debater_full_dir results/debater_full_run/transcripts \
        --output_dir results/combined_analysis
"""

import argparse
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import warnings


def extract_question_id(debate_id: str) -> str:
    """Extract question identifier from debate_id for matching across runs.

    debate_id format: quality_synthetic_0_What_do_Jessica_and__baseline_judge_1765655969
    We want: quality_synthetic_0_What_do_Jessica_and_
    """
    # Remove the experiment mode and timestamp suffix
    # Pattern: _[experiment_mode]_[timestamp]
    match = re.match(r'(.+?)_(baseline_judge|debater_full|debater_only|judge_only|full_access)_\d+$', debate_id)
    if match:
        return match.group(1)
    # Fallback: try to extract everything before the last underscore + digits
    match = re.match(r'(.+?)_\d+$', debate_id)
    if match:
        return match.group(1)
    return debate_id


def load_transcripts(transcripts_dir: Path) -> Dict[str, Dict]:
    """Load all transcripts from a directory, keyed by question_id."""
    transcripts = {}

    for json_file in transcripts_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                transcript = json.load(f)
                question_id = extract_question_id(transcript['debate_id'])
                transcripts[question_id] = transcript
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")

    return transcripts


def extract_judge_results(transcript: Dict) -> Dict[str, Dict]:
    """Extract judge results organized by (eval_mode, judge_name)."""
    results = {}
    judge_results = transcript.get('judge_results', {})

    for full_key, judge_result in judge_results.items():
        # Parse key: evaluation_mode_judge_name
        eval_mode = None
        judge_name = None

        for mode in ['baseline', 'judge_only', 'debater_only', 'full_access']:
            if full_key.startswith(mode + '_'):
                eval_mode = mode
                judge_name = full_key[len(mode) + 1:]
                break

        if eval_mode and judge_name:
            results[(eval_mode, judge_name)] = {
                'correct': judge_result.get('correct'),
                'winner_role': judge_result.get('winner_role'),
                'reasoning': judge_result.get('reasoning', '')
            }

    return results


def extract_probe_results(transcript: Dict) -> Dict[str, bool]:
    """Extract probe-only classification results."""
    results = {}
    probe_only_results = transcript.get('probe_only_results', {})

    for method in ['probe_only_mean', 'probe_only_max']:
        winner = probe_only_results.get(method)
        if winner:
            # For quality debates, honest debater defends correct answer
            correct = (winner == 'honest')
            results[method] = correct

    return results


def mcnemar_test(paired_results: List[Tuple[bool, bool]]) -> Tuple[float, float, str]:
    """
    Perform McNemar's test on paired binary outcomes.

    Args:
        paired_results: List of (condition1_correct, condition2_correct) tuples

    Returns:
        (statistic, p_value, test_name)
    """
    # Build contingency table
    # b = condition1 correct, condition2 wrong
    # c = condition1 wrong, condition2 correct
    b = sum(1 for c1, c2 in paired_results if c1 and not c2)
    c = sum(1 for c1, c2 in paired_results if not c1 and c2)

    n = b + c
    if n == 0:
        return 0.0, 1.0, "McNemar (no discordant pairs)"

    # Use exact binomial test for small samples, chi-square for large
    if n < 25:
        # Exact binomial test
        p_value = 2 * min(
            stats.binom.cdf(min(b, c), n, 0.5),
            1 - stats.binom.cdf(max(b, c) - 1, n, 0.5)
        )
        p_value = min(p_value, 1.0)
        return float(n), float(p_value), "McNemar exact (binomial)"
    else:
        # Chi-square approximation with continuity correction
        statistic = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(statistic, df=1)
        return float(statistic), float(p_value), "McNemar chi-square"


def compute_confidence_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute Wilson score confidence interval for a proportion."""
    if total == 0:
        return 0.0, 0.0

    p = successes / total
    z = stats.norm.ppf(1 - (1 - confidence) / 2)

    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    spread = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator

    return max(0, center - spread), min(1, center + spread)


class CombinedAnalyzer:
    """Analyze debate results across multiple experiment modes with paired samples."""

    def __init__(
        self,
        baseline_dir: Path,
        debater_full_dir: Path,
        output_dir: Path
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load transcripts
        print("Loading baseline_judge transcripts...")
        self.baseline_transcripts = load_transcripts(baseline_dir)
        print(f"  Loaded {len(self.baseline_transcripts)} transcripts")

        print("Loading debater_full transcripts...")
        self.debater_full_transcripts = load_transcripts(debater_full_dir)
        print(f"  Loaded {len(self.debater_full_transcripts)} transcripts")

        # Find paired samples (questions that appear in both runs)
        self.paired_question_ids = set(self.baseline_transcripts.keys()) & set(self.debater_full_transcripts.keys())
        print(f"\nFound {len(self.paired_question_ids)} paired samples")

        # Extract results for paired samples only
        self.results = self._extract_paired_results()

    def _extract_paired_results(self) -> pd.DataFrame:
        """Extract results for all paired samples into a DataFrame."""
        rows = []

        for qid in self.paired_question_ids:
            baseline_transcript = self.baseline_transcripts[qid]
            debater_full_transcript = self.debater_full_transcripts[qid]

            # Get judge results from both
            baseline_judge_results = extract_judge_results(baseline_transcript)
            debater_full_judge_results = extract_judge_results(debater_full_transcript)

            # Get probe results (should be same for both since same debate content...
            # but baseline_judge has probes too)
            baseline_probe_results = extract_probe_results(baseline_transcript)
            debater_probe_results = extract_probe_results(debater_full_transcript)

            # Find all judges that appear in both
            baseline_judges = {j for (m, j) in baseline_judge_results.keys()}
            debater_judges = {j for (m, j) in debater_full_judge_results.keys()}
            all_judges = baseline_judges | debater_judges

            for judge in all_judges:
                row = {'question_id': qid, 'judge': judge}

                # Baseline mode (from baseline_judge run)
                key = ('baseline', judge)
                if key in baseline_judge_results:
                    row['baseline_correct'] = baseline_judge_results[key]['correct']
                else:
                    row['baseline_correct'] = None

                # Judge-only mode (from baseline_judge run)
                key = ('judge_only', judge)
                if key in baseline_judge_results:
                    row['judge_only_correct'] = baseline_judge_results[key]['correct']
                else:
                    row['judge_only_correct'] = None

                # Debater-only mode (from debater_full run)
                key = ('debater_only', judge)
                if key in debater_full_judge_results:
                    row['debater_only_correct'] = debater_full_judge_results[key]['correct']
                else:
                    row['debater_only_correct'] = None

                # Full-access mode (from debater_full run)
                key = ('full_access', judge)
                if key in debater_full_judge_results:
                    row['full_access_correct'] = debater_full_judge_results[key]['correct']
                else:
                    row['full_access_correct'] = None

                rows.append(row)

            # Add probe-only results (use baseline run's probes as reference)
            probe_row = {'question_id': qid, 'judge': 'probe_classifier'}
            probe_row['baseline_correct'] = None  # N/A
            probe_row['judge_only_correct'] = None  # N/A
            probe_row['debater_only_correct'] = None  # N/A
            probe_row['full_access_correct'] = None  # N/A

            # Probe mean
            if 'probe_only_mean' in baseline_probe_results:
                probe_row['probe_only_mean_correct'] = baseline_probe_results['probe_only_mean']
            else:
                probe_row['probe_only_mean_correct'] = None

            # Probe max
            if 'probe_only_max' in baseline_probe_results:
                probe_row['probe_only_max_correct'] = baseline_probe_results['probe_only_max']
            else:
                probe_row['probe_only_max_correct'] = None

            rows.append(probe_row)

        df = pd.DataFrame(rows)
        return df

    def compute_win_rates(self) -> pd.DataFrame:
        """Compute win rates for each judge and mode."""
        results = []

        judges = self.results['judge'].unique()
        modes = ['baseline', 'judge_only', 'debater_only', 'full_access',
                 'probe_only_mean', 'probe_only_max']

        for judge in judges:
            judge_data = self.results[self.results['judge'] == judge]

            for mode in modes:
                col = f'{mode}_correct'
                if col not in judge_data.columns:
                    continue

                valid = judge_data[col].dropna()
                if len(valid) == 0:
                    continue

                correct = valid.sum()
                total = len(valid)
                win_rate = correct / total
                ci_low, ci_high = compute_confidence_interval(int(correct), total)

                results.append({
                    'judge': judge,
                    'mode': mode,
                    'win_rate': win_rate,
                    'correct': int(correct),
                    'total': total,
                    'ci_low': ci_low,
                    'ci_high': ci_high
                })

        return pd.DataFrame(results)

    def compute_statistical_significance(self) -> List[Dict]:
        """Compute per-judge McNemar tests between conditions."""
        results = []

        judges = [j for j in self.results['judge'].unique() if j != 'probe_classifier']

        comparisons = [
            ('baseline', 'judge_only', 'Baseline vs Judge-Only (probes help judge?)'),
            ('baseline', 'debater_only', 'Baseline vs Debater-Only (probes help debaters?)'),
            ('baseline', 'full_access', 'Baseline vs Full-Access (full transparency?)'),
            ('judge_only', 'full_access', 'Judge-Only vs Full-Access'),
            ('debater_only', 'full_access', 'Debater-Only vs Full-Access'),
        ]

        for judge in judges:
            judge_data = self.results[self.results['judge'] == judge]

            for mode1, mode2, description in comparisons:
                col1 = f'{mode1}_correct'
                col2 = f'{mode2}_correct'

                if col1 not in judge_data.columns or col2 not in judge_data.columns:
                    continue

                # Get paired data (both non-null)
                valid_mask = judge_data[col1].notna() & judge_data[col2].notna()
                paired_data = judge_data[valid_mask]

                if len(paired_data) == 0:
                    continue

                paired_results = list(zip(
                    paired_data[col1].astype(bool),
                    paired_data[col2].astype(bool)
                ))

                rate1 = paired_data[col1].mean()
                rate2 = paired_data[col2].mean()

                stat, p_value, test_name = mcnemar_test(paired_results)

                results.append({
                    'judge': judge,
                    'comparison': description,
                    'mode1': mode1,
                    'mode2': mode2,
                    'rate1': float(rate1),
                    'rate2': float(rate2),
                    'difference': float(rate2 - rate1),
                    'n_paired': len(paired_results),
                    'statistic': stat,
                    'p_value': p_value,
                    'test': test_name,
                    'significant_05': p_value < 0.05,
                    'significant_01': p_value < 0.01,
                })

        return results

    def compute_probe_auroc(self) -> Dict:
        """Compute AUROC for probe-based classification at turn-level and debate-level."""
        results = {}

        # Turn-level scores
        turn_honest_scores = []
        turn_dishonest_scores = []

        # Debate-level aggregated scores
        debate_honest_mean_scores = []
        debate_dishonest_mean_scores = []
        debate_honest_max_scores = []
        debate_dishonest_max_scores = []

        for qid in self.paired_question_ids:
            transcript = self.baseline_transcripts[qid]

            # Get turn-level probe scores
            for turn in transcript.get('turns', []):
                speaker = turn.get('speaker')
                probe_scores = turn.get('probe_scores', {})

                if 'apollo' in probe_scores:
                    mean_score = np.mean(probe_scores['apollo'])
                    if speaker == 'honest':
                        turn_honest_scores.append(mean_score)
                    elif speaker == 'dishonest':
                        turn_dishonest_scores.append(mean_score)

            # Get debate-level aggregated scores from probe_only_results
            probe_only = transcript.get('probe_only_results', {})
            if probe_only:
                if 'honest_mean_score' in probe_only:
                    debate_honest_mean_scores.append(probe_only['honest_mean_score'])
                if 'dishonest_mean_score' in probe_only:
                    debate_dishonest_mean_scores.append(probe_only['dishonest_mean_score'])
                if 'honest_max_score' in probe_only:
                    debate_honest_max_scores.append(probe_only['honest_max_score'])
                if 'dishonest_max_score' in probe_only:
                    debate_dishonest_max_scores.append(probe_only['dishonest_max_score'])

        # Compute turn-level AUROC
        if turn_honest_scores and turn_dishonest_scores:
            all_scores = turn_honest_scores + turn_dishonest_scores
            all_labels = [0] * len(turn_honest_scores) + [1] * len(turn_dishonest_scores)

            try:
                turn_auroc = roc_auc_score(all_labels, all_scores)
                fpr, tpr, _ = roc_curve(all_labels, all_scores)

                results['turn_level'] = {
                    'auroc': float(turn_auroc),
                    'n_honest': len(turn_honest_scores),
                    'n_dishonest': len(turn_dishonest_scores),
                    'mean_honest_score': float(np.mean(turn_honest_scores)),
                    'mean_dishonest_score': float(np.mean(turn_dishonest_scores)),
                    'std_honest_score': float(np.std(turn_honest_scores)),
                    'std_dishonest_score': float(np.std(turn_dishonest_scores)),
                    'separation': float(np.mean(turn_dishonest_scores) - np.mean(turn_honest_scores)),
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist()
                }
            except Exception as e:
                print(f"Warning: Could not compute turn-level AUROC: {e}")

        # Compute debate-level AUROC (mean aggregation)
        if debate_honest_mean_scores and debate_dishonest_mean_scores:
            all_scores = debate_honest_mean_scores + debate_dishonest_mean_scores
            all_labels = [0] * len(debate_honest_mean_scores) + [1] * len(debate_dishonest_mean_scores)

            try:
                debate_mean_auroc = roc_auc_score(all_labels, all_scores)
                fpr, tpr, _ = roc_curve(all_labels, all_scores)

                results['debate_level_mean'] = {
                    'auroc': float(debate_mean_auroc),
                    'n_debates': len(debate_honest_mean_scores),
                    'mean_honest_score': float(np.mean(debate_honest_mean_scores)),
                    'mean_dishonest_score': float(np.mean(debate_dishonest_mean_scores)),
                    'std_honest_score': float(np.std(debate_honest_mean_scores)),
                    'std_dishonest_score': float(np.std(debate_dishonest_mean_scores)),
                    'separation': float(np.mean(debate_dishonest_mean_scores) - np.mean(debate_honest_mean_scores)),
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist()
                }
            except Exception as e:
                print(f"Warning: Could not compute debate-level mean AUROC: {e}")

        # Compute debate-level AUROC (max aggregation)
        if debate_honest_max_scores and debate_dishonest_max_scores:
            all_scores = debate_honest_max_scores + debate_dishonest_max_scores
            all_labels = [0] * len(debate_honest_max_scores) + [1] * len(debate_dishonest_max_scores)

            try:
                debate_max_auroc = roc_auc_score(all_labels, all_scores)
                fpr, tpr, _ = roc_curve(all_labels, all_scores)

                results['debate_level_max'] = {
                    'auroc': float(debate_max_auroc),
                    'n_debates': len(debate_honest_max_scores),
                    'mean_honest_score': float(np.mean(debate_honest_max_scores)),
                    'mean_dishonest_score': float(np.mean(debate_dishonest_max_scores)),
                    'std_honest_score': float(np.std(debate_honest_max_scores)),
                    'std_dishonest_score': float(np.std(debate_dishonest_max_scores)),
                    'separation': float(np.mean(debate_dishonest_max_scores) - np.mean(debate_honest_max_scores)),
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist()
                }
            except Exception as e:
                print(f"Warning: Could not compute debate-level max AUROC: {e}")

        return results

    def create_combined_bar_plot(self):
        """Create bar plot with all 4 debate modes + probe classifiers."""
        win_rates = self.compute_win_rates()

        if win_rates.empty:
            print("No data for bar plot")
            return

        # Judge name mapping for display
        judge_name_mapping = {
            'meta_llama_Llama_3_1_8B_Instruct': 'Meta Llama Llama 3.1 8B Instruct',
            'meta_llama_Llama_3_3_70B_Instruct': 'Meta Llama Llama 3.3 70B Instruct',
            'anthropic_claude_sonnet_4_5': 'Anthropic Claude Sonnet 4.5',
            'anthropic_claude_sonnet_4': 'Anthropic Claude Sonnet 4.5',
            'probe_classifier': 'Probe Classifier'
        }

        win_rates['judge_display'] = win_rates['judge'].map(
            lambda x: judge_name_mapping.get(x, x.replace('_', ' ').title())
        )

        # Mode display labels and colors
        mode_config = {
            'baseline': {'label': 'Baseline', 'color': '#808080'},
            'judge_only': {'label': 'Judge Only', 'color': '#3498db'},
            'debater_only': {'label': 'Debater Only', 'color': '#e74c3c'},
            'full_access': {'label': 'Full Access', 'color': '#2ecc71'},
            'probe_only_mean': {'label': 'Probe Only (Mean)', 'color': '#f39c12'},
            'probe_only_max': {'label': 'Probe Only (Max)', 'color': '#9b59b6'},
        }

        # Get unique judges (excluding probe_classifier for main bars)
        llm_judges = [j for j in win_rates['judge_display'].unique()
                      if 'Probe' not in j]

        # Create figure
        fig, ax = plt.subplots(figsize=(20, 10))

        # Setup bar positions
        n_judges = len(llm_judges) + 1  # +1 for probe classifier
        x = np.arange(n_judges)

        # Determine which modes have data
        available_modes = win_rates['mode'].unique()
        mode_order = [m for m in ['baseline', 'judge_only', 'debater_only', 'full_access',
                                   'probe_only_mean', 'probe_only_max'] if m in available_modes]

        n_modes = len(mode_order)
        width = 0.8 / n_modes

        # Plot bars for each mode
        for i, mode in enumerate(mode_order):
            mode_data = win_rates[win_rates['mode'] == mode]

            win_rate_values = []
            ci_errors = []
            ns = []

            # Get data for LLM judges
            for judge in llm_judges:
                judge_mode_data = mode_data[mode_data['judge_display'] == judge]
                if not judge_mode_data.empty:
                    row = judge_mode_data.iloc[0]
                    win_rate_values.append(row['win_rate'])
                    ci_errors.append([
                        row['win_rate'] - row['ci_low'],
                        row['ci_high'] - row['win_rate']
                    ])
                    ns.append(row['total'])
                else:
                    win_rate_values.append(0)
                    ci_errors.append([0, 0])
                    ns.append(0)

            # Add probe classifier data
            probe_data = mode_data[mode_data['judge_display'] == 'Probe Classifier']
            if not probe_data.empty:
                row = probe_data.iloc[0]
                win_rate_values.append(row['win_rate'])
                ci_errors.append([
                    row['win_rate'] - row['ci_low'],
                    row['ci_high'] - row['win_rate']
                ])
                ns.append(row['total'])
            else:
                win_rate_values.append(0)
                ci_errors.append([0, 0])
                ns.append(0)

            # Calculate position offset
            offset = (i - n_modes / 2 + 0.5) * width

            # Plot bars
            bars = ax.bar(
                x + offset,
                win_rate_values,
                width,
                label=mode_config[mode]['label'],
                color=mode_config[mode]['color'],
                alpha=0.85,
                yerr=np.array(ci_errors).T,
                capsize=3,
                error_kw={'linewidth': 1, 'alpha': 0.7}
            )

            # Add sample size labels
            for bar, n in zip(bars, ns):
                if n > 0:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.,
                        height + 0.02,
                        f'n={n}',
                        ha='center', va='bottom',
                        fontsize=8, rotation=0
                    )

        # Customize plot
        ax.set_xlabel('Judge Model', fontsize=14, fontweight='bold')
        ax.set_ylabel('Honest Debater Win Rate', fontsize=14, fontweight='bold')
        ax.set_title(
            'Debate Performance Across Different Judges and Evaluation Modes\n'
            f'(Paired samples only, n={len(self.paired_question_ids)} questions)',
            fontsize=16, fontweight='bold', pad=20
        )

        # Set x-axis labels
        x_labels = llm_judges + ['Probe Classifier']
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=11, rotation=0, ha='center')

        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.text(ax.get_xlim()[1] - 0.1, 0.51, 'Random (50%)',
                ha='right', va='bottom', fontsize=10, alpha=0.7)

        ax.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax.set_axisbelow(True)

        plt.tight_layout()

        # Save
        save_path = self.output_dir / 'combined_4mode_win_rates.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"Saved combined bar plot: {save_path}")

    def create_roc_curve_plot(self, auroc_results: Dict):
        """Create ROC curve plot for probe performance at multiple levels."""
        if not auroc_results:
            print("No AUROC data for ROC plot")
            return

        # Count how many levels we have
        levels = []
        if 'turn_level' in auroc_results:
            levels.append(('turn_level', 'Turn-level', '#3498db'))
        if 'debate_level_mean' in auroc_results:
            levels.append(('debate_level_mean', 'Debate-level (mean)', '#e74c3c'))
        if 'debate_level_max' in auroc_results:
            levels.append(('debate_level_max', 'Debate-level (max)', '#2ecc71'))

        if not levels:
            print("No AUROC data for ROC plot")
            return

        fig, ax = plt.subplots(figsize=(8, 8))

        for key, label, color in levels:
            data = auroc_results[key]
            fpr = data['fpr']
            tpr = data['tpr']
            auroc = data['auroc']

            ax.plot(fpr, tpr, color=color, linewidth=2.5,
                    label=f'{label} (AUROC = {auroc:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1.5, label='Random')

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('Probe ROC Curves: Detecting Dishonest Debater',
                     fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()

        save_path = self.output_dir / 'probe_roc_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"Saved ROC curves: {save_path}")

    def save_results(self, win_rates: pd.DataFrame, significance: List[Dict], auroc: Dict):
        """Save all results to files."""
        # Save win rates CSV
        win_rates_path = self.output_dir / 'win_rates.csv'
        win_rates.to_csv(win_rates_path, index=False)
        print(f"Saved win rates: {win_rates_path}")

        # Save statistical significance JSON
        sig_path = self.output_dir / 'statistical_significance.json'
        with open(sig_path, 'w') as f:
            json.dump(significance, f, indent=2)
        print(f"Saved significance tests: {sig_path}")

        # Save AUROC JSON
        auroc_path = self.output_dir / 'probe_auroc.json'
        with open(auroc_path, 'w') as f:
            json.dump(auroc, f, indent=2)
        print(f"Saved AUROC: {auroc_path}")

        # Create readable summary
        summary_path = self.output_dir / 'analysis_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMBINED DEBATE ANALYSIS SUMMARY\n")
            f.write("="*80 + "\n\n")

            f.write(f"Paired samples: {len(self.paired_question_ids)}\n")
            f.write(f"Baseline run samples: {len(self.baseline_transcripts)}\n")
            f.write(f"Debater-full run samples: {len(self.debater_full_transcripts)}\n\n")

            f.write("-"*80 + "\n")
            f.write("WIN RATES BY JUDGE AND MODE\n")
            f.write("-"*80 + "\n\n")

            for judge in win_rates['judge'].unique():
                judge_data = win_rates[win_rates['judge'] == judge]
                f.write(f"{judge}:\n")
                for _, row in judge_data.iterrows():
                    f.write(f"  {row['mode']:20s}: {row['win_rate']:.1%} "
                           f"({row['correct']}/{row['total']}) "
                           f"[{row['ci_low']:.1%}, {row['ci_high']:.1%}]\n")
                f.write("\n")

            f.write("-"*80 + "\n")
            f.write("STATISTICAL SIGNIFICANCE (McNemar Tests)\n")
            f.write("-"*80 + "\n\n")

            for result in significance:
                sig_marker = '***' if result['p_value'] < 0.001 else \
                            '**' if result['p_value'] < 0.01 else \
                            '*' if result['p_value'] < 0.05 else 'ns'

                f.write(f"{result['judge']} - {result['comparison']}\n")
                f.write(f"  {result['mode1']}: {result['rate1']:.1%} vs "
                       f"{result['mode2']}: {result['rate2']:.1%}\n")
                f.write(f"  Difference: {result['difference']:+.1%}\n")
                f.write(f"  n_paired: {result['n_paired']}, "
                       f"p={result['p_value']:.4f} {sig_marker}\n")
                f.write(f"  Test: {result['test']}\n\n")

            if auroc:
                f.write("-"*80 + "\n")
                f.write("PROBE PERFORMANCE\n")
                f.write("-"*80 + "\n\n")

                if 'turn_level' in auroc:
                    tl = auroc['turn_level']
                    f.write("Turn-level (one data point per turn, mean of token scores):\n")
                    f.write(f"  AUROC: {tl['auroc']:.3f}\n")
                    f.write(f"  Honest turns: {tl['n_honest']} (mean: {tl['mean_honest_score']:.3f}, std: {tl['std_honest_score']:.3f})\n")
                    f.write(f"  Dishonest turns: {tl['n_dishonest']} (mean: {tl['mean_dishonest_score']:.3f}, std: {tl['std_dishonest_score']:.3f})\n")
                    f.write(f"  Score separation: {tl['separation']:.3f}\n\n")

                if 'debate_level_mean' in auroc:
                    dl = auroc['debate_level_mean']
                    f.write("Debate-level MEAN aggregation (one data point per debater per debate):\n")
                    f.write(f"  AUROC: {dl['auroc']:.3f}\n")
                    f.write(f"  N debates: {dl['n_debates']}\n")
                    f.write(f"  Honest debaters (mean: {dl['mean_honest_score']:.3f}, std: {dl['std_honest_score']:.3f})\n")
                    f.write(f"  Dishonest debaters (mean: {dl['mean_dishonest_score']:.3f}, std: {dl['std_dishonest_score']:.3f})\n")
                    f.write(f"  Score separation: {dl['separation']:.3f}\n\n")

                if 'debate_level_max' in auroc:
                    dl = auroc['debate_level_max']
                    f.write("Debate-level MAX aggregation (one data point per debater per debate):\n")
                    f.write(f"  AUROC: {dl['auroc']:.3f}\n")
                    f.write(f"  N debates: {dl['n_debates']}\n")
                    f.write(f"  Honest debaters (mean: {dl['mean_honest_score']:.3f}, std: {dl['std_honest_score']:.3f})\n")
                    f.write(f"  Dishonest debaters (mean: {dl['mean_dishonest_score']:.3f}, std: {dl['std_dishonest_score']:.3f})\n")
                    f.write(f"  Score separation: {dl['separation']:.3f}\n")

        print(f"Saved summary: {summary_path}")

    def run_full_analysis(self):
        """Run complete combined analysis."""
        print("\n" + "="*80)
        print("COMBINED DEBATE ANALYSIS")
        print("="*80 + "\n")

        print("1. Computing win rates...")
        win_rates = self.compute_win_rates()

        print("2. Computing statistical significance (per-judge McNemar)...")
        significance = self.compute_statistical_significance()

        print("3. Computing probe AUROC...")
        auroc = self.compute_probe_auroc()

        print("4. Creating combined bar plot...")
        self.create_combined_bar_plot()

        print("5. Creating ROC curve...")
        self.create_roc_curve_plot(auroc)

        print("6. Saving results...")
        self.save_results(win_rates, significance, auroc)

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print(f"Results saved to: {self.output_dir}")
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Combined analysis of debate results across experiment modes'
    )

    parser.add_argument(
        '--baseline_dir',
        type=str,
        required=True,
        help='Directory containing baseline_judge transcripts'
    )

    parser.add_argument(
        '--debater_full_dir',
        type=str,
        required=True,
        help='Directory containing debater_full transcripts'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save combined analysis results'
    )

    args = parser.parse_args()

    analyzer = CombinedAnalyzer(
        baseline_dir=Path(args.baseline_dir),
        debater_full_dir=Path(args.debater_full_dir),
        output_dir=Path(args.output_dir)
    )

    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
