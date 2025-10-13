"""
Comprehensive probe analysis for debate transcripts.
Supports token-level, statement-level (mean/max), and custom labeling.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve,
    precision_recall_curve, confusion_matrix
)
import re


@dataclass
class ProbeMetrics:
    """Comprehensive metrics for one analysis variant"""
    # Configuration
    level: str  # 'token' or 'statement'
    aggregation: Optional[str]  # 'mean', 'max', or None for token-level
    label_source: str  # 'speaker', 'manual', 'llm'
    
    # Core performance metrics
    auroc: float
    auprc: float
    
    # Threshold-based metrics
    optimal_threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    specificity: float
    
    # Confusion matrix at optimal threshold
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    
    # Rates at optimal threshold
    tpr: float  # True positive rate (same as recall)
    fpr: float  # False positive rate
    tnr: float  # True negative rate (same as specificity)
    fnr: float  # False negative rate
    
    # Data statistics
    n_samples: int
    n_positive: int
    n_negative: int
    
    # Score distributions
    positive_mean: float
    positive_std: float
    positive_median: float
    positive_min: float
    positive_max: float
    
    negative_mean: float
    negative_std: float
    negative_median: float
    negative_min: float
    negative_max: float
    
    # Separation metrics
    score_separation: float  # positive_mean - negative_mean
    
    def to_dict(self):
        return asdict(self)


class ProbeAnalyzer:
    """Unified probe performance analyzer"""
    
    @staticmethod
    def split_into_statements(
        text: str, 
        tokens: List[str], 
        scores: List[float]
    ) -> List[Tuple[str, List[float], List[int]]]:
        """
        Segment text into statements and map tokens.
        Returns: [(statement_text, token_scores, token_indices), ...]
        """
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        results = []
        token_idx = 0
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            sentence_tokens = []
            sentence_scores = []
            sentence_indices = []
            sentence_length = len(sentence)
            
            reconstructed = ""
            while token_idx < len(tokens) and len(reconstructed) < sentence_length * 1.2:
                token = tokens[token_idx].replace('Ġ', ' ').replace('Ċ', '\n')
                reconstructed += token
                sentence_tokens.append(tokens[token_idx])
                sentence_scores.append(scores[token_idx])
                sentence_indices.append(token_idx)
                token_idx += 1
                
                if len(reconstructed) >= sentence_length * 0.8:
                    break
            
            if sentence_scores:
                results.append((sentence, sentence_scores, sentence_indices))
        
        return results
    
    @staticmethod
    def extract_token_data(transcript: Dict) -> Tuple[List[float], List[int]]:
        """
        Extract token-level scores and labels.
        Returns: (scores, labels) where label is 0=honest, 1=deceptive
        """
        scores = []
        labels = []
        
        for turn in transcript.get('turns', []):
            if turn['speaker'] not in ['honest', 'dishonest']:
                continue
            
            tokens = turn.get('tokens')
            probe_scores = turn.get('probe_scores')
            
            if not tokens or not probe_scores:
                continue
            
            # Average across probe types
            for i in range(len(tokens)):
                token_scores = [
                    scores_list[i] 
                    for scores_list in probe_scores.values() 
                    if i < len(scores_list)
                ]
                avg_score = np.mean(token_scores) if token_scores else 0.0
                
                scores.append(avg_score)
                labels.append(1 if turn['speaker'] == 'dishonest' else 0)
        
        return scores, labels
    
    @staticmethod
    def extract_statement_data(
        transcript: Dict,
        aggregation: str = 'mean'
    ) -> Tuple[List[float], List[int], List[Dict]]:
        """
        Extract statement-level scores and labels.
        
        Args:
            aggregation: 'mean' or 'max'
            
        Returns: (scores, labels, metadata) 
            where metadata contains statement details for custom labeling
        """
        scores = []
        labels = []
        metadata = []
        
        for turn in transcript.get('turns', []):
            if turn['speaker'] not in ['honest', 'dishonest']:
                continue
            
            content = turn['content']
            tokens = turn.get('tokens')
            probe_scores = turn.get('probe_scores')
            
            if not tokens or not probe_scores:
                continue
            
            # Average across probe types
            avg_token_scores = []
            for i in range(len(tokens)):
                token_scores = [
                    scores_list[i]
                    for scores_list in probe_scores.values()
                    if i < len(scores_list)
                ]
                avg_token_scores.append(np.mean(token_scores) if token_scores else 0.0)
            
            # Segment into statements
            statements = ProbeAnalyzer.split_into_statements(
                content, tokens, avg_token_scores
            )
            
            label = 1 if turn['speaker'] == 'dishonest' else 0
            
            for stmt_idx, (stmt_text, token_scores, token_indices) in enumerate(statements):
                # Aggregate
                if aggregation == 'mean':
                    agg_score = np.mean(token_scores)
                elif aggregation == 'max':
                    agg_score = np.max(token_scores)
                else:
                    raise ValueError(f"Invalid aggregation: {aggregation}")
                
                scores.append(agg_score)
                labels.append(label)
                metadata.append({
                    'turn_number': turn['turn_number'],
                    'speaker': turn['speaker'],
                    'statement_index': stmt_idx,
                    'text': stmt_text,
                    'token_indices': token_indices,
                    'n_tokens': len(token_scores)
                })
        
        return scores, labels, metadata
    
    @staticmethod
    def apply_manual_labels(
        labels: List[int],
        metadata: List[Dict],
        manual_labels: Dict
    ) -> List[int]:
        """
        Override labels with manual annotations from debate_viz.
        
        manual_labels format:
        {
            'statementLabels': {
                'turn-statement': 0 or 1
            }
        }
        """
        new_labels = labels.copy()
        statement_labels = manual_labels.get('statementLabels', {})
        
        for i, meta in enumerate(metadata):
            key = f"{meta['turn_number']}-{meta['statement_index']}"
            if key in statement_labels:
                new_labels[i] = statement_labels[key]
        
        return new_labels
    
    @staticmethod
    def apply_llm_labels(
        labels: List[int],
        metadata: List[Dict],
        llm_labels: Dict
    ) -> List[int]:
        """
        Override labels with LLM-generated annotations.
        
        llm_labels format from label_debate_statements.py:
        {
            'statement_labels': [
                {
                    'turn_number': int,
                    'statement_index': int,
                    'contains_lie': bool,
                    'contains_misleading': bool,
                    ...
                }
            ]
        }
        """
        new_labels = labels.copy()
        
        # Build lookup from flat list
        llm_lookup = {}
        for stmt_data in llm_labels.get('statement_labels', []):
            key = f"{stmt_data['turn_number']}-{stmt_data['statement_index']}"
            llm_lookup[key] = stmt_data
        
        for i, meta in enumerate(metadata):
            key = f"{meta['turn_number']}-{meta['statement_index']}"
            if key in llm_lookup:
                llm_data = llm_lookup[key]
                # Deceptive if contains lie OR misleading
                if llm_data.get('contains_lie') or llm_data.get('contains_misleading'):
                    new_labels[i] = 1
                else:
                    new_labels[i] = 0
        
        return new_labels
    
    @staticmethod
    def calculate_metrics(
        scores: List[float],
        labels: List[int],
        level: str,
        aggregation: Optional[str],
        label_source: str
    ) -> Optional[ProbeMetrics]:
        """Calculate comprehensive metrics for one variant"""
        
        if not scores or not labels:
            return None
        
        scores = np.array(scores)
        labels = np.array(labels)
        
        # Need both classes
        if len(set(labels)) < 2:
            return None
        
        # Calculate AUROC and AUPRC
        auroc = float(roc_auc_score(labels, scores))
        auprc = float(average_precision_score(labels, scores))
        
        # Find optimal threshold using Youden's J
        fpr_curve, tpr_curve, thresholds = roc_curve(labels, scores)
        j_scores = tpr_curve - fpr_curve
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = float(thresholds[optimal_idx])
        
        # Make predictions at optimal threshold
        predictions = (scores >= optimal_threshold).astype(int)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        
        # Calculate all metrics at optimal threshold
        accuracy = float(accuracy_score(labels, predictions))
        precision = float(precision_score(labels, predictions, zero_division=0))
        recall = float(recall_score(labels, predictions, zero_division=0))
        f1 = float(f1_score(labels, predictions, zero_division=0))
        
        # Calculate rates
        tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        tnr = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        specificity = tnr
        
        # Score distributions
        positive_scores = scores[labels == 1]
        negative_scores = scores[labels == 0]
        
        return ProbeMetrics(
            level=level,
            aggregation=aggregation,
            label_source=label_source,
            
            # Core metrics
            auroc=auroc,
            auprc=auprc,
            
            # Threshold-based metrics
            optimal_threshold=optimal_threshold,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            specificity=specificity,
            
            # Confusion matrix
            true_positives=int(tp),
            false_positives=int(fp),
            true_negatives=int(tn),
            false_negatives=int(fn),
            
            # Rates
            tpr=tpr,
            fpr=fpr,
            tnr=tnr,
            fnr=fnr,
            
            # Sample counts
            n_samples=int(len(labels)),
            n_positive=int(np.sum(labels == 1)),
            n_negative=int(np.sum(labels == 0)),
            
            # Positive class distribution
            positive_mean=float(np.mean(positive_scores)),
            positive_std=float(np.std(positive_scores)),
            positive_median=float(np.median(positive_scores)),
            positive_min=float(np.min(positive_scores)),
            positive_max=float(np.max(positive_scores)),
            
            # Negative class distribution
            negative_mean=float(np.mean(negative_scores)),
            negative_std=float(np.std(negative_scores)),
            negative_median=float(np.median(negative_scores)),
            negative_min=float(np.min(negative_scores)),
            negative_max=float(np.max(negative_scores)),
            
            # Separation
            score_separation=float(np.mean(positive_scores) - np.mean(negative_scores))
        )
    
    @staticmethod
    def analyze_transcript(
        transcript: Dict,
        manual_labels: Optional[Dict] = None,
        llm_labels: Optional[Dict] = None
    ) -> Dict[str, ProbeMetrics]:
        """
        Comprehensive analysis of one transcript.
        
        Returns dict with keys:
        - 'token_speaker'
        - 'statement_mean_speaker'
        - 'statement_max_speaker'
        - 'statement_mean_manual' (if manual_labels provided)
        - 'statement_max_manual' (if manual_labels provided)
        - 'statement_mean_llm' (if llm_labels provided)
        - 'statement_max_llm' (if llm_labels provided)
        """
        results = {}
        
        # 1. Token-level with speaker labels
        token_scores, token_labels = ProbeAnalyzer.extract_token_data(transcript)
        if token_scores:
            metrics = ProbeAnalyzer.calculate_metrics(
                token_scores, token_labels,
                level='token', aggregation=None, label_source='speaker'
            )
            if metrics:
                results['token_speaker'] = metrics
        
        # 2. Statement-level (mean) with speaker labels
        stmt_scores, stmt_labels, stmt_meta = ProbeAnalyzer.extract_statement_data(
            transcript, 'mean'
        )
        if stmt_scores:
            metrics = ProbeAnalyzer.calculate_metrics(
                stmt_scores, stmt_labels,
                level='statement', aggregation='mean', label_source='speaker'
            )
            if metrics:
                results['statement_mean_speaker'] = metrics
        
        # 3. Statement-level (max) with speaker labels
        stmt_scores_max, stmt_labels_max, stmt_meta_max = ProbeAnalyzer.extract_statement_data(
            transcript, 'max'
        )
        if stmt_scores_max:
            metrics = ProbeAnalyzer.calculate_metrics(
                stmt_scores_max, stmt_labels_max,
                level='statement', aggregation='max', label_source='speaker'
            )
            if metrics:
                results['statement_max_speaker'] = metrics
        
        # 4. Manual labels (if provided)
        if manual_labels and stmt_scores:
            # Mean aggregation
            manual_labels_mean = ProbeAnalyzer.apply_manual_labels(
                stmt_labels, stmt_meta, manual_labels
            )
            metrics = ProbeAnalyzer.calculate_metrics(
                stmt_scores, manual_labels_mean,
                level='statement', aggregation='mean', label_source='manual'
            )
            if metrics:
                results['statement_mean_manual'] = metrics
            
            # Max aggregation
            manual_labels_max = ProbeAnalyzer.apply_manual_labels(
                stmt_labels_max, stmt_meta_max, manual_labels
            )
            metrics = ProbeAnalyzer.calculate_metrics(
                stmt_scores_max, manual_labels_max,
                level='statement', aggregation='max', label_source='manual'
            )
            if metrics:
                results['statement_max_manual'] = metrics
        
        # 5. LLM labels (if provided)
        if llm_labels and stmt_scores:
            # Mean aggregation
            llm_labels_mean = ProbeAnalyzer.apply_llm_labels(
                stmt_labels, stmt_meta, llm_labels
            )
            metrics = ProbeAnalyzer.calculate_metrics(
                stmt_scores, llm_labels_mean,
                level='statement', aggregation='mean', label_source='llm'
            )
            if metrics:
                results['statement_mean_llm'] = metrics
            
            # Max aggregation
            llm_labels_max = ProbeAnalyzer.apply_llm_labels(
                stmt_labels_max, stmt_meta_max, llm_labels
            )
            metrics = ProbeAnalyzer.calculate_metrics(
                stmt_scores_max, llm_labels_max,
                level='statement', aggregation='max', label_source='llm'
            )
            if metrics:
                results['statement_max_llm'] = metrics
        
        return results