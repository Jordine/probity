"""Utilities for loading labeled span data and computing span-level metrics."""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np
from probity.datasets.position_finder import Position, PositionFinder

@dataclass
class LabeledSpan:
    text: str
    char_start: int
    char_end: int
    confidence: str
    reasoning: Optional[str] = None
    sample_id: Optional[str] = None

def load_labeled_spans(filepath: str) -> Dict[str, List[LabeledSpan]]:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    spans_by_sample = {}
    for record in data:
        if "spans" not in record or not record["spans"]:
            continue
        sample_id = record.get("original_id", "unknown")
        if sample_id == "unknown":
            orig_data = record.get("original_data", {})
            if "metadata" in orig_data:
                sample_id = f"sample_{orig_data['metadata'].get('sample_index', 0)}"
            elif "sample_index" in orig_data:
                sample_id = f"sample_{orig_data['sample_index']}"
        labeled_spans = []
        for span in record["spans"]:
            labeled_spans.append(LabeledSpan(
                text=span.get("text", ""),
                char_start=span.get("char_start", 0),
                char_end=span.get("char_end", 0),
                confidence=span.get("confidence", "medium"),
                reasoning=span.get("reasoning"),
                sample_id=sample_id
            ))
        if labeled_spans:
            spans_by_sample[sample_id] = labeled_spans
    return spans_by_sample

def get_response_text(record: Dict) -> Optional[str]:
    orig_data = record.get("original_data", {})
    if "conversations" in orig_data:
        conv = orig_data["conversations"].get("main", {})
        messages = conv.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                return msg.get("content")
    if "messages" in orig_data:
        messages = orig_data["messages"]
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                return msg.get("content")
    return None

def find_span_by_text_match(span_text: str, tokens, tokenizer=None) -> Optional[Tuple[int, int]]:
    """Find span by fuzzy matching decoded token text.

    tokens can be List[int] (token IDs) or List[str] (already decoded tokens).
    """
    if not span_text or not tokens:
        return None
    full_decoded = ""
    char_to_token = []
    for i, tok in enumerate(tokens):
        # Handle both token IDs and token strings
        if isinstance(tok, str):
            decoded = tok.replace("▁", " ").replace("Ġ", " ")  # Clean tokenizer artifacts
        elif tokenizer is not None:
            decoded = tokenizer.decode([tok], skip_special_tokens=True)
        else:
            decoded = str(tok)
        for _ in decoded:
            char_to_token.append(i)
        full_decoded += decoded
    if not full_decoded or not char_to_token:
        return None
    span_normalized = " ".join(span_text.split())
    match_pos = full_decoded.find(span_normalized)
    if match_pos != -1:
        end_pos = match_pos + len(span_normalized) - 1
        if match_pos < len(char_to_token) and end_pos < len(char_to_token):
            return (char_to_token[match_pos], char_to_token[end_pos])
    full_normalized = " ".join(full_decoded.split())
    match_pos = full_normalized.find(span_normalized)
    if match_pos != -1:
        ratio = len(full_decoded) / max(len(full_normalized), 1)
        approx_start = int(match_pos * ratio)
        approx_end = int((match_pos + len(span_normalized)) * ratio)
        approx_start = max(0, min(approx_start, len(char_to_token) - 1))
        approx_end = max(0, min(approx_end, len(char_to_token) - 1))
        return (char_to_token[approx_start], char_to_token[approx_end])
    span_words = span_normalized.split()
    if len(span_words) >= 3:
        prefix = " ".join(span_words[:3])
        match_pos = full_decoded.find(prefix)
        if match_pos == -1:
            match_pos = full_normalized.find(prefix)
            if match_pos != -1:
                match_pos = int(match_pos * len(full_decoded) / max(len(full_normalized), 1))
        if match_pos != -1 and match_pos < len(char_to_token):
            est_end = min(match_pos + len(span_text), len(char_to_token) - 1)
            return (char_to_token[match_pos], char_to_token[est_end])
    return None

def convert_spans_to_token_indices(spans: List[LabeledSpan], text: str, tokenizer, add_special_tokens: bool = True, tokens: Optional[List[int]] = None) -> List[Tuple[int, int]]:
    """Convert character spans to token indices with fuzzy fallback."""
    token_spans = []
    if tokens is None:
        tokens = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    for span in spans:
        try:
            pos = Position(start=span.char_start, end=span.char_end)
            token_indices = PositionFinder.convert_char_span_to_token_positions(pos, text, tokenizer, add_special_tokens=add_special_tokens)
            if token_indices:
                token_spans.append((min(token_indices), max(token_indices)))
                continue
        except (ValueError, IndexError):
            pass
        if span.text:
            result = find_span_by_text_match(span.text, tokens, tokenizer)
            if result:
                token_spans.append(result)
                continue
        print(f"Warning: Could not convert span: No alignment found")
    return token_spans

def compute_span_metrics(token_scores: List[float], labeled_token_spans: List[Tuple[int, int]], threshold: float = 0.5) -> Dict[str, float]:
    n_tokens = len(token_scores)
    scores_array = np.array(token_scores)
    in_span_mask = np.zeros(n_tokens, dtype=bool)
    for start, end in labeled_token_spans:
        start = max(0, start)
        end = min(n_tokens - 1, end)
        in_span_mask[start:end+1] = True
    flagged_mask = scores_array > threshold
    true_positives = np.sum(flagged_mask & in_span_mask)
    false_positives = np.sum(flagged_mask & ~in_span_mask)
    false_negatives = np.sum(~flagged_mask & in_span_mask)
    span_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    span_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    span_f1 = 2 * span_precision * span_recall / (span_precision + span_recall) if (span_precision + span_recall) > 0 else 0.0
    union = np.sum(flagged_mask | in_span_mask)
    iou = true_positives / union if union > 0 else 0.0
    in_span_scores = scores_array[in_span_mask] if np.any(in_span_mask) else np.array([0.0])
    outside_span_scores = scores_array[~in_span_mask] if np.any(~in_span_mask) else np.array([0.0])
    return {"span_precision": float(span_precision), "span_recall": float(span_recall), "span_f1": float(span_f1), "iou": float(iou), "max_in_span": float(np.max(in_span_scores)), "max_outside_span": float(np.max(outside_span_scores)), "mean_in_span": float(np.mean(in_span_scores)), "mean_outside_span": float(np.mean(outside_span_scores)), "n_labeled_tokens": int(np.sum(in_span_mask)), "n_flagged_tokens": int(np.sum(flagged_mask)), "n_total_tokens": n_tokens, "true_positives": int(true_positives), "false_positives": int(false_positives), "false_negatives": int(false_negatives)}

def aggregate_span_metrics(sample_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not sample_metrics:
        return {"span_precision": 0.0, "span_recall": 0.0, "span_f1": 0.0, "mean_iou": 0.0, "max_in_span": 0.0, "max_outside_span": 0.0, "mean_in_span": 0.0, "mean_outside_span": 0.0, "n_samples_with_spans": 0, "n_total_spans": 0}
    total_tp = sum(m["true_positives"] for m in sample_metrics)
    total_fp = sum(m["false_positives"] for m in sample_metrics)
    total_fn = sum(m["false_negatives"] for m in sample_metrics)
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    return {"span_precision": float(micro_precision), "span_recall": float(micro_recall), "span_f1": float(micro_f1), "mean_iou": float(np.mean([m["iou"] for m in sample_metrics])), "max_in_span": float(np.mean([m["max_in_span"] for m in sample_metrics])), "max_outside_span": float(np.mean([m["max_outside_span"] for m in sample_metrics])), "mean_in_span": float(np.mean([m["mean_in_span"] for m in sample_metrics])), "mean_outside_span": float(np.mean([m["mean_outside_span"] for m in sample_metrics])), "n_samples_with_spans": len(sample_metrics), "n_total_labeled_tokens": sum(m["n_labeled_tokens"] for m in sample_metrics)}
