"""
Utilities for loading labeled span data and computing span-level metrics.

Labeled files contain character-level spans marking deceptive content.
This module provides:
1. Loading spans from *_labeled.json files
2. Converting character spans to token indices
3. Computing span-level evaluation metrics
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np

from probity.datasets.position_finder import Position, PositionFinder


@dataclass
class LabeledSpan:
    """A labeled span of deceptive content."""
    text: str
    char_start: int
    char_end: int
    confidence: str  # low, medium, high
    reasoning: Optional[str] = None
    sample_id: Optional[str] = None


def load_labeled_spans(filepath: str) -> Dict[str, List[LabeledSpan]]:
    """Load spans from *_labeled.json files.

    Args:
        filepath: Path to labeled JSON file (e.g., ai_liar_labeled.json)

    Returns:
        Dict mapping sample_id -> list of LabeledSpan objects
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    spans_by_sample = {}

    for record in data:
        if 'spans' not in record or not record['spans']:
            continue

        # Use original_id or generate one from index
        sample_id = record.get('original_id', 'unknown')
        if sample_id == 'unknown':
            # Try to get sample_index from original_data
            orig_data = record.get('original_data', {})
            if 'metadata' in orig_data:
                sample_id = f"sample_{orig_data['metadata'].get('sample_index', 0)}"
            elif 'sample_index' in orig_data:
                sample_id = f"sample_{orig_data['sample_index']}"

        labeled_spans = []
        for span in record['spans']:
            labeled_spans.append(LabeledSpan(
                text=span.get('text', ''),
                char_start=span.get('char_start', 0),
                char_end=span.get('char_end', 0),
                confidence=span.get('confidence', 'medium'),
                reasoning=span.get('reasoning'),
                sample_id=sample_id
            ))

        if labeled_spans:
            spans_by_sample[sample_id] = labeled_spans

    return spans_by_sample


def get_response_text(record: Dict) -> Optional[str]:
    """Extract the model response text from a labeled record.

    Args:
        record: A record from the labeled JSON file

    Returns:
        The assistant/model response text, or None if not found
    """
    orig_data = record.get('original_data', {})

    # ai_liar format: conversations.main.messages
    if 'conversations' in orig_data:
        conv = orig_data['conversations'].get('main', {})
        messages = conv.get('messages', [])
        for msg in reversed(messages):
            if msg.get('role') == 'assistant':
                return msg.get('content')

    # Other formats: messages at top level
    if 'messages' in orig_data:
        messages = orig_data['messages']
        for msg in reversed(messages):
            if msg.get('role') == 'assistant':
                return msg.get('content')

    return None


def convert_spans_to_token_indices(
    spans: List[LabeledSpan],
    text: str,
    tokenizer,
    add_special_tokens: bool = True
) -> List[Tuple[int, int]]:
    """Convert character spans to (start_token, end_token) tuples.

    Args:
        spans: List of LabeledSpan objects
        text: The full text that was tokenized
        tokenizer: HuggingFace tokenizer
        add_special_tokens: Whether special tokens were added

    Returns:
        List of (start_token_idx, end_token_idx) tuples
    """
    token_spans = []

    for span in spans:
        try:
            pos = Position(start=span.char_start, end=span.char_end)
            token_indices = PositionFinder.convert_char_span_to_token_positions(
                pos, text, tokenizer, add_special_tokens=add_special_tokens
            )
            if token_indices:
                token_spans.append((min(token_indices), max(token_indices)))
        except (ValueError, IndexError) as e:
            # Skip spans that don't align with tokens
            print(f"Warning: Could not convert span '{span.text[:30]}...': {e}")
            continue

    return token_spans


def compute_span_metrics(
    token_scores: List[float],
    labeled_token_spans: List[Tuple[int, int]],
    threshold: float = 0.5
) -> Dict[str, float]:
    """Compute span-level precision, recall, and other metrics.

    Args:
        token_scores: Per-token probe scores (normalized 0-1)
        labeled_token_spans: List of (start, end) token index tuples for labeled spans
        threshold: Score threshold for flagging tokens as deceptive

    Returns:
        Dict with span-level metrics
    """
    n_tokens = len(token_scores)
    scores_array = np.array(token_scores)

    # Create binary mask for labeled span tokens
    in_span_mask = np.zeros(n_tokens, dtype=bool)
    for start, end in labeled_token_spans:
        # Clamp to valid range
        start = max(0, start)
        end = min(n_tokens - 1, end)
        in_span_mask[start:end+1] = True

    # Create binary mask for flagged tokens (above threshold)
    flagged_mask = scores_array > threshold

    # Compute overlap metrics
    true_positives = np.sum(flagged_mask & in_span_mask)
    false_positives = np.sum(flagged_mask & ~in_span_mask)
    false_negatives = np.sum(~flagged_mask & in_span_mask)

    # Span precision: of flagged tokens, how many are in labeled spans?
    span_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0

    # Span recall: of labeled span tokens, how many are flagged?
    span_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    # F1
    span_f1 = 2 * span_precision * span_recall / (span_precision + span_recall) if (span_precision + span_recall) > 0 else 0.0

    # IoU (intersection over union)
    union = np.sum(flagged_mask | in_span_mask)
    iou = true_positives / union if union > 0 else 0.0

    # Max score inside vs outside spans
    in_span_scores = scores_array[in_span_mask] if np.any(in_span_mask) else np.array([0.0])
    outside_span_scores = scores_array[~in_span_mask] if np.any(~in_span_mask) else np.array([0.0])

    max_in_span = float(np.max(in_span_scores))
    max_outside_span = float(np.max(outside_span_scores))
    mean_in_span = float(np.mean(in_span_scores))
    mean_outside_span = float(np.mean(outside_span_scores))

    return {
        'span_precision': float(span_precision),
        'span_recall': float(span_recall),
        'span_f1': float(span_f1),
        'iou': float(iou),
        'max_in_span': max_in_span,
        'max_outside_span': max_outside_span,
        'mean_in_span': mean_in_span,
        'mean_outside_span': mean_outside_span,
        'n_labeled_tokens': int(np.sum(in_span_mask)),
        'n_flagged_tokens': int(np.sum(flagged_mask)),
        'n_total_tokens': n_tokens,
        'true_positives': int(true_positives),
        'false_positives': int(false_positives),
        'false_negatives': int(false_negatives)
    }


def aggregate_span_metrics(sample_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate span metrics across multiple samples.

    Args:
        sample_metrics: List of per-sample span metric dicts

    Returns:
        Aggregated metrics dict
    """
    if not sample_metrics:
        return {
            'span_precision': 0.0,
            'span_recall': 0.0,
            'span_f1': 0.0,
            'mean_iou': 0.0,
            'max_in_span': 0.0,
            'max_outside_span': 0.0,
            'mean_in_span': 0.0,
            'mean_outside_span': 0.0,
            'n_samples_with_spans': 0,
            'n_total_spans': 0
        }

    # Micro-average: total TP / (TP + FP), etc.
    total_tp = sum(m['true_positives'] for m in sample_metrics)
    total_fp = sum(m['false_positives'] for m in sample_metrics)
    total_fn = sum(m['false_negatives'] for m in sample_metrics)

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    return {
        'span_precision': float(micro_precision),
        'span_recall': float(micro_recall),
        'span_f1': float(micro_f1),
        'mean_iou': float(np.mean([m['iou'] for m in sample_metrics])),
        'max_in_span': float(np.mean([m['max_in_span'] for m in sample_metrics])),
        'max_outside_span': float(np.mean([m['max_outside_span'] for m in sample_metrics])),
        'mean_in_span': float(np.mean([m['mean_in_span'] for m in sample_metrics])),
        'mean_outside_span': float(np.mean([m['mean_outside_span'] for m in sample_metrics])),
        'n_samples_with_spans': len(sample_metrics),
        'n_total_labeled_tokens': sum(m['n_labeled_tokens'] for m in sample_metrics)
    }
