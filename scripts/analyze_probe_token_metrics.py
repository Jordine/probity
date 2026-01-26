#!/usr/bin/env python3
"""
Analyze probe performance with comprehensive token-level metrics.

Run this after probe_eval_deception_datasets.py to get:
- Sample-level metrics (AUROC, accuracy, etc.)
- Token-level metrics (AUPRC, Recall@K%, FPR-calibrated thresholds)
- Comprehensive rankings across all probes

Usage:
    python scripts/analyze_probe_token_metrics.py \
        --results_dir ./results/my_eval \
        --output_dir ./analysis \
        --baseline_name apollo  # optional: compare against a baseline probe
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except ImportError:
    print("Warning: sklearn not available. Install with: pip install scikit-learn")
    roc_auc_score = None
    average_precision_score = None


def load_span_scores(results_dir: Path, probe_type: str, layer: int) -> Optional[dict]:
    """Load in_span and out_span scores from span_score_distributions.json."""
    path = results_dir / "aggregated" / probe_type / f"layer_{layer}" / "span_score_distributions.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_sample_metrics(results_dir: Path, probe_type: str, layer: int) -> Optional[dict]:
    """Load sample-level metrics from metrics.json."""
    path = results_dir / "aggregated" / probe_type / f"layer_{layer}" / "metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def compute_token_metrics(in_span_scores: List[float], out_span_scores: List[float]) -> dict:
    """Compute comprehensive token-level metrics."""
    in_span = np.array(in_span_scores)
    out_span = np.array(out_span_scores)

    # Create labels: 1 for in_span, 0 for out_span
    all_scores = np.concatenate([in_span, out_span])
    all_labels = np.concatenate([np.ones(len(in_span)), np.zeros(len(out_span))])

    # Basic stats
    mean_in = float(np.mean(in_span))
    mean_out = float(np.mean(out_span))
    std_in = float(np.std(in_span))
    std_out = float(np.std(out_span))

    # Cohen's d
    pooled_std = np.sqrt((std_in**2 + std_out**2) / 2)
    cohens_d = (mean_in - mean_out) / pooled_std if pooled_std > 0 else 0

    # Token-level AUROC
    try:
        token_auroc = float(roc_auc_score(all_labels, all_scores))
    except:
        token_auroc = 0.5

    # Token-level AUPRC (Average Precision)
    try:
        token_auprc = float(average_precision_score(all_labels, all_scores))
    except:
        token_auprc = len(in_span) / len(all_scores)

    # Recall@K% and Precision@K%
    n_total = len(all_scores)
    n_positive = len(in_span)
    sorted_indices = np.argsort(all_scores)[::-1]  # descending
    sorted_labels = all_labels[sorted_indices]

    recall_at_k = {}
    precision_at_k = {}
    for k in [5, 10, 15, 20, 25, 30]:
        top_k_count = int(n_total * k / 100)
        if top_k_count == 0:
            continue
        top_k_labels = sorted_labels[:top_k_count]
        true_positives = np.sum(top_k_labels)

        recall_at_k[k] = float(true_positives / n_positive) if n_positive > 0 else 0
        precision_at_k[k] = float(true_positives / top_k_count)

    # Token-level FPR calibration
    # Find threshold such that X% of out_span tokens exceed it
    fpr_thresholds = {}
    for fpr in [1, 5, 10]:
        threshold = float(np.percentile(out_span, 100 - fpr))
        tpr = float(np.mean(in_span > threshold))
        actual_fpr = float(np.mean(out_span > threshold))
        fpr_thresholds[fpr] = {
            "threshold": threshold,
            "tpr": tpr,
            "actual_fpr": actual_fpr
        }

    return {
        "n_in_span": len(in_span),
        "n_out_span": len(out_span),
        "mean_in_span": mean_in,
        "mean_out_span": mean_out,
        "std_in_span": std_in,
        "std_out_span": std_out,
        "cohens_d": float(cohens_d),
        "token_auroc": token_auroc,
        "token_auprc": token_auprc,
        "baseline_auprc": float(n_positive / n_total),
        "recall_at_k": recall_at_k,
        "precision_at_k": precision_at_k,
        "fpr_calibrated": fpr_thresholds
    }


def discover_probes(results_dir: Path) -> List[Tuple[str, int]]:
    """Discover all probe types and layers in results directory."""
    probes = []
    aggregated_dir = results_dir / "aggregated"

    if not aggregated_dir.exists():
        return probes

    for probe_type_dir in aggregated_dir.iterdir():
        if not probe_type_dir.is_dir():
            continue
        probe_type = probe_type_dir.name

        for layer_dir in probe_type_dir.iterdir():
            if not layer_dir.is_dir():
                continue
            if not layer_dir.name.startswith("layer_"):
                continue
            try:
                layer = int(layer_dir.name.replace("layer_", ""))
                probes.append((probe_type, layer))
            except ValueError:
                continue

    return sorted(probes, key=lambda x: (x[0], x[1]))


def analyze_results(results_dir: Path, baseline_name: Optional[str] = None) -> dict:
    """Analyze all probes in results directory."""
    probes = discover_probes(results_dir)

    if not probes:
        print(f"No probes found in {results_dir}")
        return {}

    print(f"Found {len(probes)} probe configurations")

    results = {}
    baseline_metrics = None

    for probe_type, layer in probes:
        key = f"{probe_type}_L{layer}"

        # Load span scores
        span_data = load_span_scores(results_dir, probe_type, layer)
        if span_data is None:
            print(f"  {key}: No span score data")
            continue

        in_span = span_data.get("in_span_scores", [])
        out_span = span_data.get("out_span_scores", [])

        if len(in_span) == 0 or len(out_span) == 0:
            print(f"  {key}: Empty scores")
            continue

        # Compute token metrics
        token_metrics = compute_token_metrics(in_span, out_span)

        # Load sample metrics
        sample_metrics = load_sample_metrics(results_dir, probe_type, layer)
        sample_auroc = sample_metrics.get("auroc") if sample_metrics else None

        results[key] = {
            "probe_type": probe_type,
            "layer": layer,
            "sample_auroc": sample_auroc,
            **token_metrics
        }

        # Track baseline
        if baseline_name and probe_type == baseline_name:
            baseline_metrics = results[key]

        sample_str = f"{sample_auroc:.3f}" if sample_auroc else "N/A"
        print(f"  {key}: Sample AUROC={sample_str:>7}, "
              f"Token AUPRC={token_metrics['token_auprc']:.3f}, "
              f"R@10%={token_metrics['recall_at_k'].get(10, 0):.3f}")

    return results, baseline_metrics


def generate_rankings(results: dict, baseline: Optional[dict] = None) -> dict:
    """Generate various rankings of probes."""
    rows = list(results.values())

    # Filter out probes without sample AUROC
    rows_with_sample = [r for r in rows if r.get("sample_auroc") is not None]

    rankings = {}

    # By sample AUROC
    rankings["by_sample_auroc"] = sorted(
        rows_with_sample,
        key=lambda x: x["sample_auroc"],
        reverse=True
    )[:20]

    # By token AUPRC
    rankings["by_token_auprc"] = sorted(
        rows,
        key=lambda x: x["token_auprc"],
        reverse=True
    )[:20]

    # By Recall@10%
    rankings["by_recall_10"] = sorted(
        rows,
        key=lambda x: x["recall_at_k"].get(10, 0),
        reverse=True
    )[:20]

    # By TPR@5%FPR
    rankings["by_tpr_5fpr"] = sorted(
        rows,
        key=lambda x: x["fpr_calibrated"].get(5, {}).get("tpr", 0),
        reverse=True
    )[:20]

    # Best balanced (sample AUROC > 0.75 and good token metrics)
    balanced = [r for r in rows_with_sample if r["sample_auroc"] > 0.75]
    rankings["balanced"] = sorted(
        balanced,
        key=lambda x: x["token_auprc"],
        reverse=True
    )[:20]

    return rankings


def format_markdown_report(results: dict, rankings: dict, baseline: Optional[dict] = None) -> str:
    """Generate a markdown report."""
    lines = ["# Probe Token-Level Analysis Report\n"]

    # Baseline section
    if baseline:
        lines.append("## Baseline Metrics\n")
        lines.append(f"**{baseline['probe_type']} L{baseline['layer']}**\n")
        lines.append(f"- Sample AUROC: {baseline.get('sample_auroc', 'N/A')}")
        lines.append(f"- Token AUROC: {baseline['token_auroc']:.3f}")
        lines.append(f"- Token AUPRC: {baseline['token_auprc']:.3f}")
        lines.append(f"- Recall@10%: {baseline['recall_at_k'].get(10, 0):.3f}")
        lines.append(f"- TPR@5%FPR: {baseline['fpr_calibrated'].get(5, {}).get('tpr', 0):.3f}")
        lines.append(f"- Cohen's d: {baseline['cohens_d']:.3f}\n")

    # Rankings
    def format_row(r, baseline=None):
        sample = f"{r['sample_auroc']:.3f}" if r.get('sample_auroc') else "N/A"
        auprc = r['token_auprc']
        r10 = r['recall_at_k'].get(10, 0)
        tpr5 = r['fpr_calibrated'].get(5, {}).get('tpr', 0)

        vs_auprc = ""
        vs_r10 = ""
        if baseline:
            vs_auprc = f" ({(auprc - baseline['token_auprc']) / baseline['token_auprc'] * 100:+.0f}%)"
            vs_r10 = f" ({(r10 - baseline['recall_at_k'].get(10, 0)) / baseline['recall_at_k'].get(10, 1) * 100:+.0f}%)"

        return f"| {r['probe_type']} | {r['layer']} | {sample} | {auprc:.3f}{vs_auprc} | {r10:.3f}{vs_r10} | {tpr5:.3f} |"

    # Top by Token AUPRC
    lines.append("## Top 10 by Token AUPRC\n")
    lines.append("| Probe | Layer | Sample AUROC | Token AUPRC | R@10% | TPR@5%FPR |")
    lines.append("|-------|-------|--------------|-------------|-------|-----------|")
    for r in rankings["by_token_auprc"][:10]:
        lines.append(format_row(r, baseline))
    lines.append("")

    # Top by Recall@10%
    lines.append("## Top 10 by Recall@10%\n")
    lines.append("| Probe | Layer | Sample AUROC | Token AUPRC | R@10% | TPR@5%FPR |")
    lines.append("|-------|-------|--------------|-------------|-------|-----------|")
    for r in rankings["by_recall_10"][:10]:
        lines.append(format_row(r, baseline))
    lines.append("")

    # Best balanced
    lines.append("## Best Balanced (Sample AUROC > 0.75)\n")
    lines.append("| Probe | Layer | Sample AUROC | Token AUPRC | R@10% | TPR@5%FPR |")
    lines.append("|-------|-------|--------------|-------------|-------|-----------|")
    for r in rankings["balanced"][:10]:
        lines.append(format_row(r, baseline))
    lines.append("")

    # Top by Sample AUROC
    lines.append("## Top 10 by Sample AUROC\n")
    lines.append("| Probe | Layer | Sample AUROC | Token AUPRC | R@10% | TPR@5%FPR |")
    lines.append("|-------|-------|--------------|-------------|-------|-----------|")
    for r in rankings["by_sample_auroc"][:10]:
        lines.append(format_row(r, baseline))
    lines.append("")

    # Full table
    lines.append("## Full Results (sorted by Token AUPRC)\n")
    lines.append("```")
    lines.append(f"{'Probe':<18} {'Layer':>5} {'Sample':>8} {'Token':>8} {'Token':>8} "
                 f"{'R@5%':>6} {'R@10%':>6} {'R@15%':>6} {'R@20%':>6} {'Cohen':>7} {'TPR@5%':>7}")
    lines.append(f"{'':<18} {'':>5} {'AUROC':>8} {'AUROC':>8} {'AUPRC':>8} "
                 f"{'':>6} {'':>6} {'':>6} {'':>6} {'d':>7} {'FPR':>7}")
    lines.append("-" * 110)

    all_rows = sorted(results.values(), key=lambda x: x['token_auprc'], reverse=True)
    for r in all_rows:
        sample = f"{r['sample_auroc']:.3f}" if r.get('sample_auroc') else "N/A"
        lines.append(
            f"{r['probe_type']:<18} {r['layer']:>5} {sample:>8} {r['token_auroc']:>8.3f} {r['token_auprc']:>8.3f} "
            f"{r['recall_at_k'].get(5, 0):>6.3f} {r['recall_at_k'].get(10, 0):>6.3f} "
            f"{r['recall_at_k'].get(15, 0):>6.3f} {r['recall_at_k'].get(20, 0):>6.3f} "
            f"{r['cohens_d']:>7.3f} {r['fpr_calibrated'].get(5, {}).get('tpr', 0):>7.3f}"
        )
    lines.append("```")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze probe performance with token-level metrics"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to evaluation results directory (containing aggregated/)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for analysis (default: results_dir)"
    )
    parser.add_argument(
        "--baseline_name",
        type=str,
        default=None,
        help="Probe type to use as baseline for comparison (e.g., 'apollo')"
    )
    parser.add_argument(
        "--baseline_layer",
        type=int,
        default=22,
        help="Layer of baseline probe (default: 22)"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1

    print(f"Analyzing probes in: {results_dir}")

    # Analyze
    results, baseline = analyze_results(results_dir, args.baseline_name)

    if not results:
        print("No results to analyze")
        return 1

    # Generate rankings
    rankings = generate_rankings(results, baseline)

    # Save JSON
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "token_level_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to: {json_path}")

    rankings_path = output_dir / "probe_rankings.json"
    with open(rankings_path, "w") as f:
        # Convert to serializable format
        rankings_serializable = {
            k: [{kk: vv for kk, vv in r.items()} for r in v]
            for k, v in rankings.items()
        }
        json.dump(rankings_serializable, f, indent=2)
    print(f"Saved rankings to: {rankings_path}")

    # Generate markdown report
    report = format_markdown_report(results, rankings, baseline)
    report_path = output_dir / "TOKEN_LEVEL_ANALYSIS.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved report to: {report_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if baseline:
        print(f"\nBaseline ({baseline['probe_type']} L{baseline['layer']}):")
        print(f"  Token AUPRC: {baseline['token_auprc']:.3f}")
        print(f"  Recall@10%: {baseline['recall_at_k'].get(10, 0):.3f}")

    print("\nTop 5 by Token AUPRC:")
    for r in rankings["by_token_auprc"][:5]:
        vs = ""
        if baseline:
            vs = f" ({(r['token_auprc'] - baseline['token_auprc']) / baseline['token_auprc'] * 100:+.0f}%)"
        print(f"  {r['probe_type']} L{r['layer']}: {r['token_auprc']:.3f}{vs}")

    print("\nTop 5 Balanced (Sample AUROC > 0.75):")
    for r in rankings["balanced"][:5]:
        print(f"  {r['probe_type']} L{r['layer']}: Sample={r['sample_auroc']:.3f}, "
              f"TokenAUPRC={r['token_auprc']:.3f}, R@10%={r['recall_at_k'].get(10, 0):.3f}")

    return 0


if __name__ == "__main__":
    exit(main())
