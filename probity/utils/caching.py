import os
import torch
import json
from pathlib import Path
from typing import Dict, List
from transformer_lens import HookedTransformer
from tqdm import tqdm
import hashlib

from probity.collection.activation_store import ActivationStore



def get_dataset_hash(dataset) -> str:
    """Generate a hash for the dataset to enable smart caching"""
    # Create a hash based on dataset content
    content_str = ""
    for ex in dataset.examples[:10]:  # Sample first 10 examples for efficiency
        content_str += f"{ex.text[:100]}_{ex.label}_"  # First 100 chars + label
    
    content_str += f"_size_{len(dataset.examples)}"
    return hashlib.md5(content_str.encode()).hexdigest()[:16]


def smart_cache_activations(model: HookedTransformer, dataset, layers: List[int], 
                           cache_dir: str, batch_size: int, device: str, 
                           dtype: torch.dtype, force_recache: bool = False) -> Dict[str, ActivationStore]:
    """Smart caching that checks dataset compatibility and model compatibility"""
    
    # Create cache path with dataset and model info
    model_name_clean = model.cfg.model_name.replace('/', '_').replace('-', '_')
    dataset_hash = get_dataset_hash(dataset)
    
    cache_base = Path(cache_dir) / f"{model_name_clean}_{dataset_hash}"
    cache_metadata_path = cache_base / "cache_metadata.json"
    
    # Check if cache exists and is valid
    if not force_recache and cache_metadata_path.exists():
        try:
            with open(cache_metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Verify cache is compatible
            if (metadata.get("model_name") == model.cfg.model_name and
                metadata.get("dataset_size") == len(dataset.examples) and
                metadata.get("dtype") == str(dtype) and
                set(metadata.get("layers", [])) >= set(layers)):
                
                print(f"Loading compatible cached activations from {cache_base}")
                
                # Load cached activation stores
                stores = {}
                for layer in layers:
                    store_path = cache_base / f"layer_{layer}.pt"
                    if store_path.exists():
                        stores[f"blocks.{layer}.hook_resid_pre"] = torch.load(store_path, map_location=device)
                        # Ensure correct dtype
                        store = stores[f"blocks.{layer}.hook_resid_pre"]
                        if store.raw_activations.dtype != dtype:
                            store.raw_activations = store.raw_activations.to(dtype)
                
                if len(stores) == len(layers):
                    print(f"Successfully loaded {len(stores)} cached activation stores")
                    return stores
                else:
                    print(f"Cache incomplete: found {len(stores)}/{len(layers)} layers")
        
        except Exception as e:
            print(f"Failed to load cache: {e}. Recollecting...")
    
    # Collect activations
    print(f"Collecting activations for {len(layers)} layers...")
    
    hook_points = [f"blocks.{layer}.hook_resid_pre" for layer in layers]
    all_activations = {hook: [] for hook in hook_points}
    
    # Process in batches
    model.eval()
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(dataset.examples), batch_size), 
                               desc="Collecting activations"):
            batch_end = min(batch_start + batch_size, len(dataset.examples))
            batch_indices = list(range(batch_start, batch_end))
            
            # Get batch tensors
            batch = dataset.get_batch_tensors(batch_indices)
            input_ids = batch["input_ids"].to(device)
            
            # Run model with caching for all layers at once
            _, cache = model.run_with_cache(
                input_ids,
                names_filter=hook_points,
                return_cache_object=True,
                stop_at_layer=max(layers) + 1
            )
            
            # Store activations for each hook point
            for hook in hook_points:
                all_activations[hook].append(cache[hook].cpu())
    
    # Create ActivationStore objects
    activation_stores = {}
    cache_base.mkdir(parents=True, exist_ok=True)
    
    for hook, activations in all_activations.items():
        layer = int(hook.split(".")[1])
        
        # Stack all activations
        raw_activations = torch.cat(activations, dim=0)
        
        store = ActivationStore(
            raw_activations=raw_activations,
            hook_point=hook,
            example_indices=torch.arange(len(dataset.examples)),
            sequence_lengths=torch.tensor(dataset.get_token_lengths()),
            hidden_size=raw_activations.shape[-1],
            dataset=dataset,
            labels=torch.tensor([ex.label for ex in dataset.examples]),
            label_texts=[ex.label_text for ex in dataset.examples],
        )
        
        activation_stores[hook] = store
        
        # Save individual layer cache
        layer_cache_path = cache_base / f"layer_{layer}.pt"
        # torch.save(store, layer_cache_path)
        store.save(str(layer_cache_path))
    
    # Save cache metadata
    metadata = {
        "model_name": model.cfg.model_name,
        "dataset_size": len(dataset.examples),
        "dtype": str(dtype),
        "layers": layers,
        "cache_version": "1.0"
    }
    
    with open(cache_metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved activations cache to {cache_base}")
    
    return activation_stores

