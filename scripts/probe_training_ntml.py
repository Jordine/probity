import argparse
import os
import torch
import json
from pathlib import Path
from typing import Dict, List, Optional
from transformer_lens import HookedTransformer
from tqdm import tqdm
import hashlib

from probity.collection.activation_store import ActivationStore

from probity.training.configs import (
    get_probe_config,
    get_probe_class,
    get_trainer_config,
    get_trainer_class
)
from probity.utils.caching import get_dataset_hash, smart_cache_activations
from probity.utils.dataset_loading import load_contrastive_ntml_dataset, get_model_dtype


def train_all_probes_for_layer(layer: int, activation_store: ActivationStore, 
                              probe_types: List[str], args, 
                              model_name: str, hidden_size: int, 
                              device: str, dtype: torch.dtype) -> Dict[str, Dict]:
    """Train all probe types for a single layer efficiently"""
    
    hook_point = f"blocks.{layer}.hook_resid_pre"
    layer_results = {}
    
    for probe_type in probe_types:
        print(f"Training {probe_type} probe on layer {layer}")
        
        # Get configurations
        probe_config = get_probe_config(
            probe_type, hidden_size, model_name, 
            hook_point, layer, dtype
        )
        probe_cls = get_probe_class(probe_type)
        trainer_config = get_trainer_config(probe_type, device, args.batch_size)
        trainer_cls = get_trainer_class(probe_type)
        
        # Initialize probe and trainer
        probe = probe_cls(probe_config).to(device)
        trainer = trainer_cls(trainer_config)
        
        # Prepare data once per layer (shared across probe types)
        train_loader, val_loader = trainer.prepare_supervised_data(
            activation_store, "LIE_SPAN"  # Same position key as original
        )
        
        # Train
        history = trainer.train(probe, train_loader, val_loader)
        
        # Save probe immediately (CHANGED: .pt instead of .json)
        save_dir = Path(args.probe_save_dir) / probe_type
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"layer_{layer}_probe.pt"
        probe.save(str(save_path))  # CHANGED: .save() instead of .save_json()
        
        layer_results[probe_type] = {
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1] if 'val_loss' in history else None,
            'save_path': str(save_path)
        }
        
        print(f"Saved {probe_type} probe for layer {layer} to {save_path}")
        
        # Clear probe from memory
        del probe
        torch.cuda.empty_cache()
    
    return layer_results


def find_dataset_file(dataset_name: str) -> Optional[Path]:
    """Find a contrastive dataset file by name."""
    # Look in the contrastive datasets directory
    contrastive_dir = Path("./data/NTML-datasets/contrastive")
    
    if not contrastive_dir.exists():
        return None
    
    # Try exact match first
    exact_path = contrastive_dir / f"{dataset_name}.json"
    if exact_path.exists():
        return exact_path
    
    # Try pattern matching
    patterns = [
        f"{dataset_name}*.json",
        f"*{dataset_name}*.json"
    ]
    
    for pattern in patterns:
        matches = list(contrastive_dir.glob(pattern))
        if matches:
            return matches[0]
    
    return None


def list_available_datasets() -> List[str]:
    """List all available contrastive NTML datasets."""
    contrastive_dir = Path("./data/NTML-datasets/contrastive")
    
    if not contrastive_dir.exists():
        return []
    
    json_files = list(contrastive_dir.glob("*.json"))
    return sorted([f.stem for f in json_files])


def parse_args():
    parser = argparse.ArgumentParser(description='Train contrastive NTML probes efficiently')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--train_dataset_dir', type=str, help='Path to contrastive JSON file')
    parser.add_argument('--dataset_name', type=str, help='Name of dataset (will auto-find .json file)')
    parser.add_argument('--probe_types', nargs='+', 
                       choices=['logistic', 'linear', 'pca', 'meandiff', 'kmeans', 'mlp', 'attention'],
                       default=['logistic', 'pca', 'meandiff'])
    parser.add_argument('--hyperparams', type=json.loads, default=None,
                       help='JSON dict of hyperparameters')
    parser.add_argument('--sweep_config', type=str, 
                       help='Path to hyperparameter sweep config file')
    parser.add_argument('--use_sklearn_logistic', action='store_true',
                       help='Use sklearn logistic instead of PyTorch version')
    parser.add_argument('--layers', nargs='+', default=['all'])
    parser.add_argument('--probe_save_dir', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default='./cache/contrastive')
    parser.add_argument('--max_length', type=int, default=512, required=True,
                   help='Maximum token length for sequences')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--activation_batch_size', type=int, default=16, 
                       help='Batch size for activation collection (separate from training)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--force_recache', action='store_true', 
                       help='Force recollection of activations even if cache exists')
    parser.add_argument('--list_datasets', action='store_true',
                       help='List available datasets and exit')
    parser.add_argument('--dishonest_mode', type=str, choices=['all', 'diff'], 
                       default='all', required=True,
                       help='Which statements from dishonest sample: all or diff only')
    parser.add_argument('--honest_mode', type=str, choices=['none', 'diff', 'all'], 
                       default='none', required=True,
                       help='Which statements from honest sample: none, diff only, or all')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Handle --list_datasets
    if args.list_datasets:
        datasets = list_available_datasets()
        if datasets:
            print("📋 Available Contrastive NTML Datasets:")
            for dataset in datasets:
                print(f"   • {dataset}")
            print(f"\nUsage: python contrastive_probe_training.py --dataset_name <name> --model_name <model>")
        else:
            print("❌ No contrastive NTML datasets found.")
            print("   Run: python generate_contrastive_ntml_datasets.py")
        return 0
    
    # Determine dataset path
    if args.train_dataset_dir:
        dataset_path = Path(args.train_dataset_dir)
    elif args.dataset_name:
        dataset_path = find_dataset_file(args.dataset_name)
        if not dataset_path:
            print(f"❌ Dataset '{args.dataset_name}' not found.")
            available = list_available_datasets()
            if available:
                print("Available datasets:")
                for dataset in available[:5]:
                    print(f"   • {dataset}")
            return 1
    else:
        print("❌ Must specify either --train_dataset_dir or --dataset_name")
        return 1
    
    if not dataset_path.exists():
        print(f"❌ Dataset file not found: {dataset_path}")
        return 1
    
    print("🚀 Contrastive NTML Probe Training")
    print(f"📄 Dataset: {dataset_path.name}")
    print(f"🤖 Model: {args.model_name}")
   
    
    # Load dataset using the new loader
    print(f"Loading contrastive dataset from {dataset_path}")
    dataset = load_contrastive_ntml_dataset(str(dataset_path), 
                                            args.model_name, 
                                            max_length=args.max_length, 
                                            dishonest_mode=args.dishonest_mode,
                                            honest_mode=args.honest_mode)
    print(f"Dataset size: {len(dataset.examples)}")
    
    # Load model once
    print(f"Loading model {args.model_name}")
    model_dtype = get_model_dtype(args.model_name)
    print(f"🤖 Model dtype: {model_dtype}")
    
    try:
        model = HookedTransformer.from_pretrained_no_processing(
            args.model_name, 
            device="cuda",  
            n_devices=2,
            dtype=model_dtype
        )

        
    except Exception as e:
        print(f"Error with from_pretrained_no_processing: {e}")
        print("Attempting alternative loading method...")
        from transformers import AutoModelForCausalLM
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=model_dtype,
            device_map=args.device
        )
        model = HookedTransformer.from_pretrained(
            args.model_name,
            hf_model=hf_model,
            device=args.device,
            dtype=model_dtype,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
        )
    
    hidden_size = model.cfg.d_model
    
    # Determine layers
    if 'all' in args.layers:
        layers = list(range(model.cfg.n_layers))
    else:
        layers = [int(l) for l in args.layers]
    
    print(f"Training on layers: {layers}")
    print(f"Model dtype: {model_dtype}")
    
    # Collect activations for all layers at once using smart caching
    print("Collecting/loading activations...")
    activation_stores = smart_cache_activations(
        model, dataset, layers, args.cache_dir, 
        args.activation_batch_size, args.device, model_dtype, 
        args.force_recache
    )
    
    # Free model memory after collecting activations
    del model
    torch.cuda.empty_cache()
    
    # Train probes efficiently - one layer at a time, all probe types per layer
    results = {}
    
    for layer in tqdm(layers, desc="Training layers"):
        hook_point = f"blocks.{layer}.hook_resid_pre"
        activation_store = activation_stores[hook_point]
        
        # Train all probe types for this layer
        layer_results = train_all_probes_for_layer(
            layer, activation_store, args.probe_types, args,
            args.model_name, hidden_size, args.device, model_dtype
        )
        
        results[layer] = layer_results
        
        # Optional: Clear activation store to save memory if processing many layers
        if len(layers) > 16:
            del activation_stores[hook_point]
            torch.cuda.empty_cache()
    
    # Save training summary
    summary_path = Path(args.probe_save_dir) / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining complete. Summary saved to {summary_path}")
    
    # Print summary statistics
    print("\nTraining Summary:")
    for layer, layer_results in results.items():
        print(f"\nLayer {layer}:")
        for probe_type, probe_results in layer_results.items():
            final_loss = probe_results['final_train_loss']
            print(f"  {probe_type}: Final loss = {final_loss:.6f}")


if __name__ == "__main__":
    main()