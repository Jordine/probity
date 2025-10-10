import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
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
                if 'messages' in data and 'label' in data:
                    samples.append(data)
                else:
                    messages = data.get('messages', [
                        {"role": "user", "content": data.get('instruction', '')},
                        {"role": "assistant", "content": data.get('output', '')}
                    ])
                    samples.append({
                        'messages': messages,
                        'label': 'honest'
                    })
            except json.JSONDecodeError:
                continue
    return samples


def load_probe_from_checkpoint(probe_path: Path, device: str):
    """Load probe from checkpoint (same as evaluation script)."""
    checkpoint = torch.load(probe_path, map_location=device, weights_only=False)
    
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
        return None, None
    
    probe = probe_cls(checkpoint["config"])
    probe.load_state_dict(checkpoint["state_dict"])
    probe = probe.to(device)
    probe.eval()
    
    if isinstance(probe, DirectionalProbe):
        probe.has_fit = True
    
    layer = checkpoint["config"].hook_layer
    return probe, layer


def calibrate_all_probes(probe_dir: str, reference_dataset: str, 
                         model_name: str, target_fprs: List[float],
                         n_samples: int, device: str):
    """Calibrate all probes efficiently."""
    
    # 1. Load samples ONCE
    print(f"Loading {n_samples} reference samples...")
    if 'alpaca' in reference_dataset.lower():
        samples = load_alpaca_samples(reference_dataset, n_samples)

    elif 'wildchat' in reference_dataset.lower():
        samples = load_alpaca_samples(reference_dataset, n_samples)
        print('using wildchat')
    else:
        raise ValueError(f"Unknown reference dataset: {reference_dataset}")
    
    # 2. Format texts ONCE
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_type = detect_model_type(model_name)
    
    texts = []
    labels = []
    for sample in samples:
        formatted = apply_chat_template_unified(
            tokenizer, sample['messages'], 
            model_type=model_type,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(formatted)
        labels.append(0)  # All honest
    
    # 3. Load all probes and group by layer
    print("Loading all probes...")
    probes_by_layer = {}  # {layer: [(probe, probe_path, probe_type), ...]}
    
    for probe_file in Path(probe_dir).glob('**/*_probe.pt'):
        probe, layer = load_probe_from_checkpoint(probe_file, device)
        if probe is None:
            continue
        
        if layer not in probes_by_layer:
            probes_by_layer[layer] = []
        
        probe_type = probe.__class__.__name__
        probes_by_layer[layer].append((probe, probe_file, probe_type))
        print(f"  Loaded {probe_type} for layer {layer}")
    
    # 4. Create evaluator ONCE
    print("Loading model...")
    evaluator = OptimizedBatchProbeEvaluator(model_name, device)
    
    # 5. Collect activations for all needed layers ONCE
    layers = sorted(probes_by_layer.keys())
    print(f"Collecting activations for layers {layers}...")
    activation_data = evaluator.get_batch_activations(texts, layers, batch_size=8)
    activations_by_layer = activation_data['activations']
    
    # 6. Apply each probe to its layer's activations
    results = {}
    
    for layer in tqdm(layers, desc="Calibrating layers"):
        layer_activations = activations_by_layer[layer]
        
        for probe, probe_file, probe_type in probes_by_layer[layer]:
            print(f"\nCalibrating {probe_type} layer {layer}")
            
            # Get scores for this probe
            all_scores = []
            with torch.no_grad():
                for i in range(len(texts)):
                    text_activations = layer_activations[i].mean(dim=0).unsqueeze(0)
                    text_activations = text_activations.to(probe.dtype)
                    scores = probe(text_activations)
                    
                    if probe_type == 'LogisticProbe':
                        scores = torch.sigmoid(scores)
                    
                    all_scores.append(scores.mean().item())
            
            # Calculate thresholds
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
                
                print(f"  {key}: threshold={threshold:.4f}, actual_fpr={metrics['fpr']*100:.2f}%")
            
            # Update probe checkpoint
            checkpoint = torch.load(probe_file, map_location=device, weights_only=False)
            if not hasattr(checkpoint["config"], 'optimal_thresholds'):
                checkpoint["config"].optimal_thresholds = {}
            checkpoint["config"].optimal_thresholds.update(thresholds)
            
            # Save updated probe
            torch.save(checkpoint, probe_file)
            
            # Save thresholds separately
            threshold_path = probe_file.with_suffix('.thresholds.json')
            with open(threshold_path, 'w') as f:
                json.dump(checkpoint["config"].optimal_thresholds, f, indent=2)
            
            results[(layer, probe_type)] = thresholds
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe_dir', required=True)
    parser.add_argument('--reference_dataset', required=True)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--target_fprs', nargs='+', type=float, default=[0.01, 0.05])
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    results = calibrate_all_probes(
        args.probe_dir,
        args.reference_dataset,
        args.model_name,
        args.target_fprs,
        args.n_samples,
        args.device
    )
    
    print("\n✅ Calibration complete!")
    print(f"Calibrated {len(results)} probes")