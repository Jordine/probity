import argparse
import json
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
from jinja2 import Template

from probity.evaluation.batch_evaluator import OptimizedBatchProbeEvaluator
from probity.utils.dataset_loading import apply_chat_template_unified, detect_model_type
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def parse_dataset_spec(spec: str) -> Tuple[str, int]:
    """Parse dataset specification like 'path.jsonl:10' or just 'path.jsonl'"""
    if ':' in spec:
        path, count = spec.rsplit(':', 1)
        return path, int(count)
    return spec, None

def load_deception_samples(jsonl_path: str, n_samples: int = None) -> List[Dict]:
    """Load deception detection samples from JSONL."""
    samples = []
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f):
            if n_samples and len(samples) >= n_samples:
                break
            try:
                data = json.loads(line)
                # Filter for honest/deceptive labels only
                if data.get('label') in ['honest', 'deceptive']:
                    # Add source dataset info
                    data['source_dataset'] = Path(jsonl_path).stem
                    samples.append(data)
            except json.JSONDecodeError:
                continue
    return samples

def aggregate_metrics(all_results: List[Dict]) -> Dict:
    """Aggregate metrics across multiple datasets."""
    aggregated = {}
    
    for (layer, probe_type), results_list in all_results.items():
        all_labels = []
        all_scores = []
        all_predictions = []
        
        for result in results_list:
            all_labels.extend(result['labels'])
            all_scores.extend(result['mean_scores'])
            all_predictions.extend(result['predictions'])
        
        # Calculate aggregated metrics
        if all_labels:
            metrics = {
                'accuracy': float(accuracy_score(all_labels, all_predictions)),
                'precision': float(precision_score(all_labels, all_predictions, zero_division=0)),
                'recall': float(recall_score(all_labels, all_predictions, zero_division=0)),
                'f1': float(f1_score(all_labels, all_predictions, zero_division=0)),
                'auroc': float(roc_auc_score(all_labels, all_scores)) if len(set(all_labels)) > 1 else 0.5,
                'total_samples': len(all_labels),
                'positive_samples': sum(all_labels),
                'negative_samples': len(all_labels) - sum(all_labels)
            }
            
            aggregated[(layer, probe_type)] = {
                'metrics': metrics,
                'per_dataset': results_list
            }
    
    return aggregated
    

def extract_final_assistant_tokens(formatted_text: str, model_type: str) -> Tuple[int, int]:
    """Extract character positions of final assistant message."""
    if model_type == 'qwen':
        assistant_marker = "<|im_start|>assistant"
        end_marker = "<|im_end|>"
        
        last_assistant = formatted_text.rfind(assistant_marker)
        if last_assistant == -1:
            return None, None
            
        content_start = last_assistant + len(assistant_marker)
        if content_start < len(formatted_text) and formatted_text[content_start] == '\n':
            content_start += 1
            
        end_pos = formatted_text.find(end_marker, content_start)
        content_end = end_pos if end_pos != -1 else len(formatted_text)
        
    else:  # Llama format
        assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"
        eot_marker = "<|eot_id|>"
        
        last_assistant = formatted_text.rfind(assistant_marker)
        if last_assistant == -1:
            return None, None
            
        content_start = last_assistant + len(assistant_marker)
        while content_start < len(formatted_text) and formatted_text[content_start] in ['\n', ' ', '\t']:
            content_start += 1
            
        eot_pos = formatted_text.find(eot_marker, content_start)
        content_end = eot_pos if eot_pos != -1 else len(formatted_text)
    
    return content_start, content_end

def load_probe_from_checkpoint(probe_path: Path, device: str) -> Optional[object]:
    """Load probe from checkpoint file (matching debate inference approach)."""
    try:
        checkpoint = torch.load(
            probe_path,
            map_location=device,
            weights_only=False,  # Important: same as debate inference
        )
        
        # Import probe classes
        from probity.probes import (
            LogisticProbe, LinearProbe, PCAProbe, 
            MeanDifferenceProbe, KMeansProbe, 
            MLPProbe, AttentionProbe
        )
        from probity.probes.directional import DirectionalProbe
        
        probe_cls_map = {
            "LogisticProbe": LogisticProbe,
            "LinearProbe": LinearProbe,
            "PCAProbe": PCAProbe,
            "MeanDifferenceProbe": MeanDifferenceProbe,
            "KMeansProbe": KMeansProbe,
            "MLPProbe": MLPProbe,
            "AttentionProbe": AttentionProbe,
        }
        
        cls_name = checkpoint.get("probe_type")
        probe_cls = probe_cls_map.get(cls_name)
        
        if probe_cls is None:
            print(f"[WARN] Unknown probe class '{cls_name}' in {probe_path}")
            return None

        # Store the optimal thresholds before reconstruction
        config = checkpoint["config"]
        saved_thresholds = getattr(config, 'optimal_thresholds', {})
            
        
        # Reconstruct probe
        probe = probe_cls(checkpoint["config"])


        # Restore the optimal thresholds after reconstruction
        if saved_thresholds and hasattr(probe, 'config'):
            probe.config.optimal_thresholds = saved_thresholds
            print(f"  Loaded probe with thresholds: {list(saved_thresholds.keys())}")
        
        probe.load_state_dict(checkpoint["state_dict"])

        if isinstance(probe, DirectionalProbe):
            probe.has_fit = True
        
        probe = probe.to(device)
        probe.eval()
        
        return probe
        
    except Exception as e:
        print(f"Error loading probe from {probe_path}: {e}")
        return None

def evaluate_on_assistant_tokens(evaluator: OptimizedBatchProbeEvaluator, 
                                probe_configs: Dict, samples: List[Dict],
                                tokenizer_name: str) -> Dict:
    """Evaluate probes only on final assistant message tokens."""
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_type = detect_model_type(tokenizer_name)
    
    # Prepare texts and labels
    formatted_texts = []
    labels = []
    assistant_ranges = []
    
    for sample in tqdm(samples, desc="Formatting conversations"):
        # Apply chat template
        formatted = apply_chat_template_unified(
            tokenizer,
            sample['messages'],
            model_type=model_type,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Extract final assistant message position
        start, end = extract_final_assistant_tokens(formatted, model_type)
        if start is None:
            continue
            
        formatted_texts.append(formatted)
        labels.append(1 if sample['label'] == 'deceptive' else 0)
        assistant_ranges.append((start, end))
    
    print(f"Processing {len(formatted_texts)} valid samples")
    
    # Get activations for all texts
    print("Getting activations...")
    layers = list(set(layer for layer, _ in probe_configs.keys()))
    activation_data = evaluator.get_batch_activations(formatted_texts, layers, batch_size=4, disk_cache_dir="./cache/val_acts")
    activations = activation_data['activations']
    tokens_by_text = activation_data['tokens_by_text']
    
    # Evaluate each probe
    results = {}
    for (layer, probe_type), probe in tqdm(probe_configs.items(), desc="Evaluating probes"):
        layer_activations = activations[layer]
        
        # Collect scores only for assistant tokens
        all_scores = []
        all_labels = []
        token_details = []
        
        for i, (start_char, end_char) in enumerate(assistant_ranges):
            # Map character positions to token positions
            tokens = tokens_by_text[i]
            
            # Better token mapping using character offsets
            text = formatted_texts[i]
            char_count = 0
            start_token = None
            end_token = None
            
            for j, token in enumerate(tokens):
                token_str = token.replace('Ġ', ' ').replace('Ċ', '\n')
                if start_token is None and char_count >= start_char:
                    start_token = j
                if char_count >= end_char:
                    end_token = j
                    break
                char_count += len(token_str)
            
            if start_token is None:
                start_token = 0
            if end_token is None:
                end_token = len(tokens)
            
            # Get activations for assistant tokens only
            assistant_activations = layer_activations[i, start_token:end_token, :]
            
            # Apply probe
            with torch.no_grad():
                assistant_activations = assistant_activations.to(probe.dtype)
                token_scores = probe(assistant_activations)
                
                # Apply sigmoid for logistic probes
                if probe.__class__.__name__ == 'LogisticProbe':
                    token_scores = torch.sigmoid(token_scores)
                    
                token_scores = token_scores.cpu().squeeze().tolist()
                
                if isinstance(token_scores, float):
                    token_scores = [token_scores]
            
            # Calculate mean score for classification
            mean_score = np.mean(token_scores) if token_scores else 0.5
            all_scores.append(mean_score)
            all_labels.append(labels[i])
            
            # Store for visualization
            assistant_tokens = tokens[start_token:end_token]
            assistant_text = text[start_char:end_char]
            
            token_details.append({
                'text': assistant_text,
                'label': labels[i],
                'tokens': assistant_tokens,
                'token_scores': token_scores,
                'mean_score': mean_score,
                'min_score': float(np.min(token_scores)) if token_scores else 0.5,
                'max_score': float(np.max(token_scores)) if token_scores else 0.5
            })
        
        # Determine all thresholds to evaluate
        thresholds_to_evaluate = {
            'default_0.5': 0.5
        }
        
        # Check for probe-specific thresholds
        if hasattr(probe, 'config') and hasattr(probe.config, 'optimal_thresholds'):
            opt_thresholds = probe.config.optimal_thresholds
            
            # Handle directional probes
            if 'score_min' in opt_thresholds and 'score_max' in opt_thresholds:
                # Directional probe - normalize if needed
                score_min = opt_thresholds['score_min']
                score_max = opt_thresholds['score_max']
                
                if 'train_auroc' in opt_thresholds and score_max > score_min:
                    # Normalize the threshold
                    raw_threshold = opt_thresholds['train_auroc']
                    normalized_threshold = (raw_threshold - score_min) / (score_max - score_min)
                    thresholds_to_evaluate['train_auroc'] = normalized_threshold
                    
                    # Also normalize scores for directional probes
                    all_scores = [(s - score_min) / (score_max - score_min) if score_max > score_min else 0.5 
                                  for s in all_scores]
            else:
                # Probabilistic probes
                if 'train_auroc' in opt_thresholds:
                    thresholds_to_evaluate['train_auroc'] = opt_thresholds['train_auroc']
                
                if 'fpr_1pct' in opt_thresholds:
                    threshold_data = opt_thresholds['fpr_1pct']
                    if isinstance(threshold_data, dict):
                        thresholds_to_evaluate['fpr_1pct'] = threshold_data.get('threshold', 0.5)
                    else:
                        thresholds_to_evaluate['fpr_1pct'] = threshold_data
                
                if 'fpr_5pct' in opt_thresholds:
                    threshold_data = opt_thresholds['fpr_5pct']
                    if isinstance(threshold_data, dict):
                        thresholds_to_evaluate['fpr_5pct'] = threshold_data.get('threshold', 0.5)
                    else:
                        thresholds_to_evaluate['fpr_5pct'] = threshold_data
        
        # Calculate metrics for each threshold
        metrics_by_threshold = {}
        for threshold_name, threshold_value in thresholds_to_evaluate.items():
            predictions = (np.array(all_scores) > threshold_value).astype(int)
            
            metrics = {
                'threshold_value': float(threshold_value),
                'accuracy': float(accuracy_score(all_labels, predictions)),
                'precision': float(precision_score(all_labels, predictions, zero_division=0)),
                'recall': float(recall_score(all_labels, predictions, zero_division=0)),
                'f1': float(f1_score(all_labels, predictions, zero_division=0)),
                'auroc': float(roc_auc_score(all_labels, all_scores)) if len(set(all_labels)) > 1 else 0.5
            }
            
            # Calculate confusion matrix elements
            tp = np.sum((predictions == 1) & (np.array(all_labels) == 1))
            fp = np.sum((predictions == 1) & (np.array(all_labels) == 0))
            tn = np.sum((predictions == 0) & (np.array(all_labels) == 0))
            fn = np.sum((predictions == 0) & (np.array(all_labels) == 1))
            
            metrics['confusion_matrix'] = {
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn)
            }
            
            metrics['fpr'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            metrics['tpr'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            
            metrics_by_threshold[threshold_name] = metrics
        
        # Determine best threshold
        best_threshold_name = 'fpr_1pct' if 'fpr_1pct' in thresholds_to_evaluate else \
                             'train_auroc' if 'train_auroc' in thresholds_to_evaluate else \
                             'default_0.5'
        
        # Print which threshold is being used
        if best_threshold_name != 'default_0.5':
            print(f"  {probe_type} L{layer}: Using {best_threshold_name} threshold = {thresholds_to_evaluate[best_threshold_name]:.4f}")
        
        results[(layer, probe_type)] = {
            'metrics': metrics_by_threshold[best_threshold_name],  # Primary metrics
            'metrics_all_thresholds': metrics_by_threshold,
            'best_threshold_name': best_threshold_name,
            'thresholds_evaluated': thresholds_to_evaluate,
            'token_details': token_details,
            'mean_scores': all_scores,
            'predictions': (np.array(all_scores) > thresholds_to_evaluate[best_threshold_name]).astype(int).tolist(),
            'labels': all_labels
        }
    
    return results

def generate_enhanced_visualization(token_details: List[Dict], output_path: Path):
    """Generate enhanced HTML visualization inspired by debate viz."""
    
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Deception Detection - Token Analysis</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .legend {
            display: flex;
            justify-content: center;
            gap: 30px;
            padding: 20px;
            background: #f8f9fa;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .legend-color {
            width: 24px;
            height: 24px;
            border-radius: 4px;
        }
        
        .samples {
            padding: 20px;
        }
        
        .sample {
            margin-bottom: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .sample-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #dee2e6;
        }
        
        .label-badge {
            padding: 6px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 14px;
        }
        
        .label-badge.honest {
            background: #d4edda;
            color: #155724;
        }
        
        .label-badge.deceptive {
            background: #f8d7da;
            color: #721c24;
        }
        
        .token-container {
            line-height: 2;
            margin: 15px 0;
        }
        
        .token {
            display: inline-block;
            padding: 3px 6px;
            margin: 1px;
            border-radius: 3px;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 14px;
            cursor: help;
            transition: transform 0.2s;
            white-space: pre-wrap;
        }
        
        .token:hover {
            transform: scale(1.1);
            z-index: 10;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        
        .stat-card {
            background: white;
            padding: 12px;
            border-radius: 6px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 20px;
            font-weight: 700;
            color: #667eea;
        }
        
        .stat-label {
            font-size: 12px;
            color: #6c757d;
            text-transform: uppercase;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Deception Detection Analysis</h1>
            <p>Token-level probe scores for assistant responses</p>
        </div>
        
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background-color: rgba(40, 167, 69, 0.7);"></div>
                <span>Honest (Low Score)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: rgba(128, 128, 128, 0.2);"></div>
                <span>Neutral</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: rgba(220, 53, 69, 0.7);"></div>
                <span>Deceptive (High Score)</span>
            </div>
        </div>
        
        <div class="samples">
            {% for sample in samples %}
            <div class="sample">
                <div class="sample-header">
                    <div>
                        <strong>Sample {{ loop.index }}</strong>
                        <span style="margin-left: 10px; color: #6c757d;">
                            {{ sample.token_count }} tokens
                        </span>
                    </div>
                    <div class="label-badge {{ 'honest' if sample.label == 0 else 'deceptive' }}">
                        {{ 'HONEST' if sample.label == 0 else 'DECEPTIVE' }}
                    </div>
                </div>
                
                <div class="token-container">
                    {% for token, score, color in sample.tokens_with_colors %}
                    <span class="token" 
                          style="background-color: {{ color }}"
                          title="Score: {{ '%.3f'|format(score) }}">{{ token }}</span>
                    {% endfor %}
                </div>
                
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-value">{{ '%.3f'|format(sample.mean_score) }}</div>
                        <div class="stat-label">Mean Score</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{{ '%.3f'|format(sample.min_score) }}</div>
                        <div class="stat-label">Min Score</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{{ '%.3f'|format(sample.max_score) }}</div>
                        <div class="stat-label">Max Score</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{{ 'MATCH' if sample.prediction == sample.label else 'MISS' }}</div>
                        <div class="stat-label">Prediction</div>
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
</body>
</html>
"""
    
    def normalize_scores(scores):
        """Normalize scores to [0, 1]"""
        if not scores:
            return scores
        min_val, max_val = min(scores), max(scores)
        if max_val == min_val:
            return [0.5] * len(scores)
        return [(s - min_val) / (max_val - min_val) for s in scores]
    
    def score_to_color(score):
        """Convert normalized score to color"""
        if score >= 0.5:
            intensity = (score - 0.5) * 2
            return f"rgba(220, 53, 69, {intensity * 0.7:.3f})"
        else:
            intensity = (0.5 - score) * 2
            return f"rgba(40, 167, 69, {intensity * 0.7:.3f})"
    
    def clean_tokens(tokens):
        """Clean tokenizer artifacts"""
        return [token.replace('Ġ', ' ').replace('Ċ', '\n') for token in tokens]
    
    # Process samples
    samples_data = []
    for detail in token_details[:20]:  # Limit to 20 for performance
        normalized = normalize_scores(detail['token_scores'])
        clean = clean_tokens(detail['tokens'])
        
        tokens_with_colors = [
            (token, score, score_to_color(norm_score))
            for token, score, norm_score in zip(clean, detail['token_scores'], normalized)
        ]
        
        samples_data.append({
            'tokens_with_colors': tokens_with_colors,
            'label': detail['label'],
            'mean_score': detail['mean_score'],
            'min_score': detail['min_score'],
            'max_score': detail['max_score'],
            'token_count': len(clean),
            'prediction': 1 if detail['mean_score'] > 0.5 else 0
        })
    
    # Render template
    template = Template(html_template)
    html_content = template.render(samples=samples_data)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate probes on deception detection datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single dataset
  python probe_eval_deception.py --model_name llama --dataset_path data.jsonl --n_samples 10
  
  # Multiple datasets with different sample counts
  python probe_eval_deception.py --model_name llama \\
    --datasets data1.jsonl:10 data2.jsonl:20 data3.jsonl:15
  
  # Mix of full and limited datasets
  python probe_eval_deception.py --model_name llama \\
    --datasets data1.jsonl data2.jsonl:50
        """
    )
    
    # Dataset specification - either single or multiple
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument('--dataset_path', type=str, help='Single dataset path')
    dataset_group.add_argument('--datasets', nargs='+', 
                              help='Multiple datasets with optional sample counts (format: path.jsonl or path.jsonl:N)')
    
    parser.add_argument('--model_name', type=str, required=True, help='Model used for probe training')
    parser.add_argument('--probe_dir', type=str, required=True, help='Directory with trained probes')
    parser.add_argument('--results_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--n_samples', type=int, help='Default sample count for datasets without explicit count')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Prepare dataset specifications
    dataset_specs = []
    if args.dataset_path:
        # Single dataset
        dataset_specs.append((args.dataset_path, args.n_samples))
    else:
        # Multiple datasets
        for spec in args.datasets:
            path, count = parse_dataset_spec(spec)
            if count is None:
                count = args.n_samples  # Use default if not specified
            dataset_specs.append((path, count))
    
    # Load all samples
    print("Loading datasets...")
    all_samples = []
    dataset_summary = []
    
    for dataset_path, n_samples in dataset_specs:
        dataset_name = Path(dataset_path).stem
        print(f"  Loading {dataset_name}" + (f" (max {n_samples} samples)" if n_samples else ""))
        
        samples = load_deception_samples(dataset_path, n_samples)
        all_samples.extend(samples)
        
        dataset_summary.append({
            'name': dataset_name,
            'path': dataset_path,
            'requested': n_samples or 'all',
            'loaded': len(samples),
            'honest': sum(1 for s in samples if s['label'] == 'honest'),
            'deceptive': sum(1 for s in samples if s['label'] == 'deceptive')
        })
        
        print(f"    Loaded {len(samples)} valid samples")
    
    print(f"\nTotal samples loaded: {len(all_samples)}")
    print(f"  Honest: {sum(1 for s in all_samples if s['label'] == 'honest')}")
    print(f"  Deceptive: {sum(1 for s in all_samples if s['label'] == 'deceptive')}")
    
    if not all_samples:
        print("No valid samples found!")
        return
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save dataset summary
    with open(results_dir / 'dataset_summary.json', 'w') as f:
        json.dump(dataset_summary, f, indent=2)
    
    # Load probes
    probe_configs = {}
    probe_dir = Path(args.probe_dir)
    
    print("\nLoading probes...")
    for probe_type_dir in probe_dir.iterdir():
        if not probe_type_dir.is_dir() or probe_type_dir.name.startswith('.'):
            continue
            
        probe_type = probe_type_dir.name
        probe_files = sorted(probe_type_dir.glob("layer_*_probe.pt"))
        
        for probe_file in probe_files:
            layer = int(probe_file.stem.split('_')[1])
            probe = load_probe_from_checkpoint(probe_file, args.device)
            
            if probe is not None:
                probe_configs[(layer, probe_type)] = probe
                print(f"  Loaded {probe_type} layer {layer}")
    
    print(f"Loaded {len(probe_configs)} probes")
    
    if not probe_configs:
        print("No probes loaded!")
        return
    
    # Create evaluator
    evaluator = OptimizedBatchProbeEvaluator(args.model_name, args.device)
    
# Evaluate on all samples
    print("\nEvaluating probes on assistant tokens...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    results = evaluate_on_assistant_tokens(evaluator, probe_configs, all_samples, args.model_name)
    
    # Group results by dataset for per-dataset metrics
    results_by_dataset = {}
    for (layer, probe_type), result in results.items():
        # Get the best threshold being used
        best_threshold_name = result.get('best_threshold_name', 'default_0.5')
        
        # Group by source dataset
        for i in range(len(result['labels'])):
            source = all_samples[i]['source_dataset']
            key = (layer, probe_type, source)
            
            if key not in results_by_dataset:
                results_by_dataset[key] = {
                    'labels': [],
                    'mean_scores': [],
                    'token_details': []
                }
            
            results_by_dataset[key]['labels'].append(result['labels'][i])
            results_by_dataset[key]['mean_scores'].append(result['mean_scores'][i])
            if i < len(result['token_details']):
                results_by_dataset[key]['token_details'].append(result['token_details'][i])
    
    # Calculate and save per-dataset metrics for all thresholds
    print("\n" + "="*60)
    print("PER-DATASET RESULTS")
    print("="*60)
    
    for (layer, probe_type, dataset_name), data in results_by_dataset.items():
        if not data['labels']:
            continue
            
        # Get threshold values from the original result
        original_result = results[(layer, probe_type)]
        thresholds_evaluated = original_result.get('thresholds_evaluated', {'default_0.5': 0.5})
        
        # Calculate metrics for each threshold
        per_dataset_metrics = {}
        for thresh_name, thresh_value in thresholds_evaluated.items():
            predictions = (np.array(data['mean_scores']) > thresh_value).astype(int)
            
            per_dataset_metrics[thresh_name] = {
                'threshold_value': float(thresh_value),
                'accuracy': float(accuracy_score(data['labels'], predictions)),
                'auroc': float(roc_auc_score(data['labels'], data['mean_scores'])) 
                        if len(set(data['labels'])) > 1 else 0.5,
                'n_samples': len(data['labels'])
            }
        
        # Print best threshold results
        best_thresh = original_result.get('best_threshold_name', 'default_0.5')
        best_metrics = per_dataset_metrics.get(best_thresh, per_dataset_metrics.get('default_0.5'))
        print(f"{probe_type} L{layer} @ {dataset_name}: "
              f"Acc={best_metrics['accuracy']:.3f}, AUROC={best_metrics['auroc']:.3f} "
              f"(using {best_thresh})")
        
        # Save per-dataset results
        dataset_dir = results_dir / dataset_name / probe_type / f"layer_{layer}"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all threshold metrics
        with open(dataset_dir / 'metrics_all_thresholds.json', 'w') as f:
            json.dump({
                'best_threshold_name': best_thresh,
                'all_thresholds': per_dataset_metrics
            }, f, indent=2)
        
        # Save backward-compatible single metrics file
        with open(dataset_dir / 'metrics.json', 'w') as f:
            json.dump(best_metrics, f, indent=2)
    
    # Save aggregated results (across all datasets)
    print("\n" + "="*60)
    print("AGGREGATED RESULTS (ALL DATASETS)")
    print("="*60)
    
    for (layer, probe_type), result in results.items():
        # Primary metrics (using best threshold)
        metrics = result['metrics']
        best_threshold_name = result.get('best_threshold_name', 'default_0.5')
        
        print(f"\n{probe_type} Layer {layer}:")
        print(f"  Best threshold: {best_threshold_name}")
        print(f"  Performance: Acc={metrics['accuracy']:.3f}, AUROC={metrics['auroc']:.3f}, "
              f"F1={metrics['f1']:.3f}, FPR={metrics.get('fpr', 0):.3f}, N={len(result['labels'])}")
        
        # Save aggregated results
        agg_dir = results_dir / 'aggregated' / probe_type / f"layer_{layer}"
        agg_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all threshold metrics
        all_thresholds_path = agg_dir / 'metrics_all_thresholds.json'
        with open(all_thresholds_path, 'w') as f:
            json.dump({
                'primary_metrics': metrics,
                'best_threshold_name': best_threshold_name,
                'all_thresholds': result.get('metrics_all_thresholds', {}),
                'thresholds_evaluated': result.get('thresholds_evaluated', {})
            }, f, indent=2)
        
        # Save backward-compatible single metrics file
        with open(agg_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Generate visualization
        generate_enhanced_visualization(
            result['token_details'],
            agg_dir / 'token_visualization.html'
        )
        
        # Print threshold comparison if available
        if result.get('metrics_all_thresholds'):
            print("\n  Threshold Comparison:")
            for thresh_name, thresh_metrics in result['metrics_all_thresholds'].items():
                marker = ">>> " if thresh_name == best_threshold_name else "    "
                print(f"  {marker}{thresh_name}: "
                      f"Acc={thresh_metrics['accuracy']:.3f}, "
                      f"FPR={thresh_metrics.get('fpr', 0):.3f}, "
                      f"F1={thresh_metrics['f1']:.3f}, "
                      f"Threshold={thresh_metrics['threshold_value']:.4f}")
    
    # Final summary
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {results_dir}")
    print(f"  • Per-dataset results: {results_dir}/<dataset_name>/<probe_type>/")
    print(f"  • Aggregated results: {results_dir}/aggregated/<probe_type>/")
    print(f"  • For each result:")
    print(f"    - metrics.json: Best threshold metrics (backward compatible)")
    print(f"    - metrics_all_thresholds.json: Results for all thresholds")
    print(f"    - token_visualization.html: Token-level score visualization")
if __name__ == "__main__":
    main()