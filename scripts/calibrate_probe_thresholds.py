import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from probity.utils.threshold_optimization import find_threshold_for_fpr, calculate_threshold_metrics
from probity.utils.dataset_loading import apply_chat_template_unified, detect_model_type
from probity.evaluation.batch_evaluator import OptimizedBatchProbeEvaluator

def load_alpaca_samples(path: str, n_samples: int = 100) -> List[Dict]:
    """Load Alpaca samples from JSONL file (all labeled as honest)."""
    samples = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if len(samples) >= n_samples:
                break
            try:
                data = json.loads(line)
                # Format as conversation (same as deception eval expects)
                # Check if it's already in the right format
                if 'messages' in data and 'label' in data:
                    samples.append(data)
                else:
                    # Convert to expected format if needed
                    messages = data.get('messages', [
                        {"role": "user", "content": data.get('instruction', '')},
                        {"role": "assistant", "content": data.get('output', '')}
                    ])
                    samples.append({
                        'messages': messages,
                        'label': 'honest'  # All Alpaca samples are honest
                    })
            except json.JSONDecodeError:
                continue
    return samples

def calibrate_thresholds(probe_path: str, reference_dataset: str, 
                        model_name: str, target_fprs: List[float] = [0.01, 0.05],
                        n_samples: int = 100, device: str = 'cuda'):
    """Calibrate probe thresholds on reference dataset."""
    
    # Load probe
    checkpoint = torch.load(probe_path, map_location=device, weights_only=False)
    
    # Reconstruct probe from checkpoint
    from probity.probes import (
        LogisticProbe, LinearProbe, PCAProbe, 
        MeanDifferenceProbe, KMeansProbe, AttentionProbe, MLPProbe
    )
    
    probe_cls_map = {
        "LogisticProbe": LogisticProbe,
        "LinearProbe": LinearProbe,
        "PCAProbe": PCAProbe,
        "MeanDifferenceProbe": MeanDifferenceProbe,
        "KMeansProbe": KMeansProbe,
        "MLPProbe": MLPProbe,
        "AttentionProbe": AttentionProbe
    }
    
    cls_name = checkpoint.get("probe_type")
    probe_cls = probe_cls_map.get(cls_name)
    
    if probe_cls is None:
        print(f"Unknown probe class: {cls_name}")
        return None
    
    probe = probe_cls(checkpoint["config"])
    probe.load_state_dict(checkpoint["state_dict"])
    probe = probe.to(device)
    probe.eval()
    
    # Mark directional probes as fitted
    from probity.probes.directional import DirectionalProbe
    if isinstance(probe, DirectionalProbe):
        probe.has_fit = True
    
    # Load reference samples
    if 'alpaca' in reference_dataset.lower():
        samples = load_alpaca_samples(reference_dataset, n_samples)
    else:
        raise ValueError(f"Unknown reference dataset: {reference_dataset}")
    
    # Create evaluator
    evaluator = OptimizedBatchProbeEvaluator(model_name, device)
    
    # Get probe predictions
    texts = []
    labels = []
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_type = detect_model_type(model_name)
    
    for sample in samples:
        formatted = apply_chat_template_unified(
            tokenizer, sample['messages'], 
            model_type=model_type,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(formatted)
        labels.append(0)  # All honest = 0
    
    # Get activations and scores
    layer = checkpoint["config"].hook_layer
    activation_data = evaluator.get_batch_activations(texts, [layer], batch_size=8)
    activations = activation_data['activations'][layer]
    
    # Apply probe
    with torch.no_grad():
        all_scores = []
        for i in range(len(texts)):
            text_activations = activations[i].mean(dim=0).unsqueeze(0)  # Mean pool over tokens
            text_activations = text_activations.to(probe.dtype)
            scores = probe(text_activations)
            
            if probe.__class__.__name__ == 'LogisticProbe':
                scores = torch.sigmoid(scores)
            
            mean_score = scores.mean().item()
            all_scores.append(mean_score)
    
    # Calculate thresholds for different FPRs
    thresholds = {}
    for target_fpr in target_fprs:
        threshold = find_threshold_for_fpr(
            np.array(all_scores),
            np.array(labels),
            target_fpr
        )
        
        metrics = calculate_threshold_metrics(
            np.array(all_scores),
            np.array(labels),
            threshold
        )
        
        key = f'fpr_{int(target_fpr*100)}pct'
        thresholds[key] = {
            'threshold': threshold,
            'target_fpr': target_fpr,
            'actual_fpr': metrics['fpr'],
            'n_samples': n_samples,
            'reference_dataset': reference_dataset
        }
        
        print(f"Threshold for {target_fpr*100}% FPR: {threshold:.4f}")
        print(f"  Actual FPR: {metrics['fpr']*100:.2f}%")
    
    # Update probe config and save
    if not hasattr(checkpoint["config"], 'optimal_thresholds'):
        checkpoint["config"].optimal_thresholds = {}
    
    checkpoint["config"].optimal_thresholds.update(thresholds)
    
    # Save updated probe
    torch.save(checkpoint, probe_path)
    
    # Also save thresholds separately as JSON
    threshold_path = Path(probe_path).with_suffix('.thresholds.json')
    with open(threshold_path, 'w') as f:
        json.dump(checkpoint["config"].optimal_thresholds, f, indent=2)
    
    print(f"Saved thresholds to {threshold_path}")
    
    return thresholds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe_dir', required=True, help='Directory with trained probes')
    parser.add_argument('--reference_dataset', required=True, help='Path to reference dataset')
    parser.add_argument('--model_name', required=True, help='Model name')
    parser.add_argument('--target_fprs', nargs='+', type=float, default=[0.01, 0.05])
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Process all probes in directory
    probe_dir = Path(args.probe_dir)
    for probe_file in probe_dir.glob('**/*_probe.pt'):
        print(f"\nCalibrating {probe_file}")
        calibrate_thresholds(
            str(probe_file),
            args.reference_dataset,
            args.model_name,
            args.target_fprs,
            args.n_samples,
            args.device
        )