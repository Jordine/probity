"""
Compute token-level metrics for all probes.

Metrics computed:
1. Token-level AUROC (in_span vs out_span)
2. Token-level AUPRC (Average Precision)
3. Recall@K% for K = 5, 10, 15, 20, 25
4. Precision@K%
5. Cohen's d effect size
6. Token-level FPR-calibrated thresholds (1%, 5% FPR)
"""

import json
import os
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

# Base path
BASE_PATH = Path(r"C:\Users\Admin\Downloads\probity_dec26\probity\big_probe_sweep_eval_full\aggregated")

# Probe types and layers
PROBE_TYPES = ["attention", "logistic", "meandiff", "sklearn_logistic"]
LAYERS = list(range(22, 58, 2))  # 22, 24, 26, ..., 56

def load_span_scores(probe_type: str, layer: int) -> dict:
    """Load in_span and out_span scores from JSON."""
    path = BASE_PATH / probe_type / f"layer_{layer}" / "span_score_distributions.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def compute_metrics(in_span_scores: list, out_span_scores: list) -> dict:
    """Compute all token-level metrics."""
    in_span = np.array(in_span_scores)
    out_span = np.array(out_span_scores)

    # Create labels: 1 for in_span, 0 for out_span
    all_scores = np.concatenate([in_span, out_span])
    all_labels = np.concatenate([np.ones(len(in_span)), np.zeros(len(out_span))])

    # Basic stats
    mean_in = np.mean(in_span)
    mean_out = np.mean(out_span)
    std_in = np.std(in_span)
    std_out = np.std(out_span)

    # Cohen's d
    pooled_std = np.sqrt((std_in**2 + std_out**2) / 2)
    cohens_d = (mean_in - mean_out) / pooled_std if pooled_std > 0 else 0

    # Token-level AUROC
    try:
        token_auroc = roc_auc_score(all_labels, all_scores)
    except:
        token_auroc = 0.5

    # Token-level AUPRC (Average Precision)
    try:
        token_auprc = average_precision_score(all_labels, all_scores)
    except:
        token_auprc = len(in_span) / len(all_scores)  # baseline = proportion positive

    # Recall@K% and Precision@K%
    # "Of all tokens, take top K% by score. What fraction of in_span tokens are captured?"
    n_total = len(all_scores)
    n_positive = len(in_span)
    sorted_indices = np.argsort(all_scores)[::-1]  # descending
    sorted_labels = all_labels[sorted_indices]

    recall_at_k = {}
    precision_at_k = {}
    for k in [5, 10, 15, 20, 25]:
        top_k_count = int(n_total * k / 100)
        top_k_labels = sorted_labels[:top_k_count]
        true_positives = np.sum(top_k_labels)

        recall_at_k[k] = true_positives / n_positive if n_positive > 0 else 0
        precision_at_k[k] = true_positives / top_k_count if top_k_count > 0 else 0

    # Token-level FPR calibration: find threshold such that X% of out_span exceed it
    # FPR = P(score > threshold | negative) = X%
    # So threshold = (100-X)th percentile of out_span scores
    fpr_thresholds = {}
    for fpr in [1, 5, 10]:
        threshold = np.percentile(out_span, 100 - fpr)
        # Verify: what fraction of out_span exceeds this threshold?
        actual_fpr = np.mean(out_span > threshold) * 100
        # TPR at this threshold
        tpr = np.mean(in_span > threshold)
        fpr_thresholds[fpr] = {
            "threshold": float(threshold),
            "actual_fpr_pct": float(actual_fpr),
            "tpr": float(tpr)
        }

    return {
        "n_in_span": len(in_span),
        "n_out_span": len(out_span),
        "mean_in_span": float(mean_in),
        "mean_out_span": float(mean_out),
        "std_in_span": float(std_in),
        "std_out_span": float(std_out),
        "cohens_d": float(cohens_d),
        "token_auroc": float(token_auroc),
        "token_auprc": float(token_auprc),
        "baseline_auprc": float(n_positive / n_total),  # random baseline
        "recall_at_k": {str(k): float(v) for k, v in recall_at_k.items()},
        "precision_at_k": {str(k): float(v) for k, v in precision_at_k.items()},
        "fpr_calibrated_thresholds": fpr_thresholds
    }

def main():
    results = {}

    for probe_type in PROBE_TYPES:
        print(f"\nProcessing {probe_type}...")
        results[probe_type] = {}

        for layer in LAYERS:
            data = load_span_scores(probe_type, layer)
            if data is None:
                print(f"  Layer {layer}: No data found")
                continue

            in_span = data.get("in_span_scores", [])
            out_span = data.get("out_span_scores", [])

            if len(in_span) == 0 or len(out_span) == 0:
                print(f"  Layer {layer}: Empty scores")
                continue

            metrics = compute_metrics(in_span, out_span)
            results[probe_type][layer] = metrics

            print(f"  Layer {layer}: AUROC={metrics['token_auroc']:.3f}, "
                  f"AUPRC={metrics['token_auprc']:.3f}, "
                  f"R@10%={metrics['recall_at_k']['10']:.3f}, "
                  f"Cohen's d={metrics['cohens_d']:.3f}")

    # Save results
    output_path = Path(r"C:\Users\Admin\Downloads\probity_dec26\token_level_metrics.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Also load sample-level AUROC from metrics.json and create combined table
    print("\n" + "="*100)
    print("COMBINED RESULTS TABLE (sorted by Token AUPRC)")
    print("="*100)

    all_rows = []
    for probe_type in PROBE_TYPES:
        for layer in LAYERS:
            if layer not in results.get(probe_type, {}):
                continue

            token_metrics = results[probe_type][layer]

            # Load sample-level AUROC
            sample_auroc_path = BASE_PATH / probe_type / f"layer_{layer}" / "metrics.json"
            sample_auroc = None
            if sample_auroc_path.exists():
                with open(sample_auroc_path) as f:
                    sample_metrics = json.load(f)
                    sample_auroc = sample_metrics.get("auroc", None)

            all_rows.append({
                "probe": probe_type,
                "layer": layer,
                "sample_auroc": sample_auroc,
                "token_auroc": token_metrics["token_auroc"],
                "token_auprc": token_metrics["token_auprc"],
                "recall_5": token_metrics["recall_at_k"]["5"],
                "recall_10": token_metrics["recall_at_k"]["10"],
                "recall_15": token_metrics["recall_at_k"]["15"],
                "recall_20": token_metrics["recall_at_k"]["20"],
                "recall_25": token_metrics["recall_at_k"]["25"],
                "cohens_d": token_metrics["cohens_d"],
                "fpr_1_tpr": token_metrics["fpr_calibrated_thresholds"][1]["tpr"],
                "fpr_5_tpr": token_metrics["fpr_calibrated_thresholds"][5]["tpr"],
            })

    # Sort by token_auprc descending
    all_rows.sort(key=lambda x: x["token_auprc"], reverse=True)

    # Print header
    print(f"{'Probe':<18} {'Layer':>5} {'Sample':>7} {'Token':>7} {'Token':>7} "
          f"{'R@5%':>6} {'R@10%':>6} {'R@15%':>6} {'R@20%':>6} {'R@25%':>6} "
          f"{'Cohen':>6} {'TPR@':>6} {'TPR@':>6}")
    print(f"{'':18} {'':>5} {'AUROC':>7} {'AUROC':>7} {'AUPRC':>7} "
          f"{'':>6} {'':>6} {'':>6} {'':>6} {'':>6} "
          f"{'d':>6} {'1%FPR':>6} {'5%FPR':>6}")
    print("-"*120)

    for row in all_rows:
        sample_auroc_str = f"{row['sample_auroc']:.3f}" if row['sample_auroc'] else "N/A"
        print(f"{row['probe']:<18} {row['layer']:>5} {sample_auroc_str:>7} "
              f"{row['token_auroc']:.3f}  {row['token_auprc']:.3f}  "
              f"{row['recall_5']:.3f}  {row['recall_10']:.3f}  {row['recall_15']:.3f}  "
              f"{row['recall_20']:.3f}  {row['recall_25']:.3f}  "
              f"{row['cohens_d']:.3f}  {row['fpr_1_tpr']:.3f}  {row['fpr_5_tpr']:.3f}")

    # Print top 10 by each metric
    print("\n" + "="*100)
    print("TOP 10 BY SAMPLE AUROC (> 0.8)")
    print("="*100)
    high_sample = [r for r in all_rows if r['sample_auroc'] and r['sample_auroc'] > 0.8]
    high_sample.sort(key=lambda x: x['sample_auroc'], reverse=True)
    for row in high_sample[:10]:
        print(f"{row['probe']:<18} L{row['layer']}: Sample={row['sample_auroc']:.3f}, "
              f"TokenAUPRC={row['token_auprc']:.3f}, R@10%={row['recall_10']:.3f}")

    print("\n" + "="*100)
    print("TOP 10 BY TOKEN AUPRC")
    print("="*100)
    by_auprc = sorted(all_rows, key=lambda x: x['token_auprc'], reverse=True)
    for row in by_auprc[:10]:
        sample_str = f"{row['sample_auroc']:.3f}" if row['sample_auroc'] else "N/A"
        print(f"{row['probe']:<18} L{row['layer']}: TokenAUPRC={row['token_auprc']:.3f}, "
              f"Sample={sample_str}, R@10%={row['recall_10']:.3f}")

    print("\n" + "="*100)
    print("TOP 10 BY RECALL@10%")
    print("="*100)
    by_recall = sorted(all_rows, key=lambda x: x['recall_10'], reverse=True)
    for row in by_recall[:10]:
        sample_str = f"{row['sample_auroc']:.3f}" if row['sample_auroc'] else "N/A"
        print(f"{row['probe']:<18} L{row['layer']}: R@10%={row['recall_10']:.3f}, "
              f"Sample={sample_str}, TokenAUPRC={row['token_auprc']:.3f}")

    print("\n" + "="*100)
    print("BEST BALANCED (Sample AUROC > 0.8 AND Token AUPRC > 0.22)")
    print("="*100)
    balanced = [r for r in all_rows if r['sample_auroc'] and r['sample_auroc'] > 0.8 and r['token_auprc'] > 0.22]
    balanced.sort(key=lambda x: x['token_auprc'], reverse=True)
    for row in balanced[:10]:
        print(f"{row['probe']:<18} L{row['layer']}: Sample={row['sample_auroc']:.3f}, "
              f"TokenAUPRC={row['token_auprc']:.3f}, R@10%={row['recall_10']:.3f}, "
              f"TPR@5%FPR={row['fpr_5_tpr']:.3f}")

if __name__ == "__main__":
    main()
