import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from typing import List, Tuple, Optional

def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC score."""
    if len(np.unique(labels)) < 2:
        return 0.5  # Can't compute AUROC with single class
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return 0.5

def find_optimal_threshold_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Find threshold that maximizes AUROC (Youden's J statistic)."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    # Youden's J statistic: maximize TPR - FPR
    optimal_idx = np.argmax(tpr - fpr)
    return float(thresholds[optimal_idx])

def find_threshold_for_fpr(scores: np.ndarray, labels: np.ndarray, 
                          target_fpr: float = 0.01) -> float:
    """Find threshold that achieves target false positive rate."""
    
    # Special case: all negative labels (common for calibration datasets)
    if np.all(labels == 0):
        # For all-negative data, FPR = fraction classified as positive
        # So we want the threshold where (100-target_fpr*100)% of scores are below it
        percentile = (1 - target_fpr) * 100
        threshold = np.percentile(scores, percentile)
        return float(threshold)
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    # Remove inf values if present
    valid_mask = np.isfinite(thresholds)
    fpr = fpr[valid_mask]
    tpr = tpr[valid_mask]
    thresholds = thresholds[valid_mask]
    
    if len(thresholds) == 0:
        # Fallback to percentile method
        percentile = (1 - target_fpr) * 100
        return float(np.percentile(scores, percentile))
    
    # Find threshold closest to target FPR
    valid_indices = np.where(fpr <= target_fpr)[0]
    if len(valid_indices) == 0:
        return float(thresholds[0])
    
    optimal_idx = valid_indices[-1]
    return float(thresholds[optimal_idx])

def calculate_threshold_metrics(scores: np.ndarray, labels: np.ndarray, 
                               threshold: float) -> dict:
    """Calculate metrics at a specific threshold."""
    predictions = (scores >= threshold).astype(int)
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    tn = np.sum((predictions == 0) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0
    
    return {
        'threshold': threshold,
        'tpr': tpr,
        'fpr': fpr,
        'precision': precision,
        'accuracy': accuracy
    }