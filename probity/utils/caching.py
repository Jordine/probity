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
    """Generate a hash for the dataset to enable smart caching.

    Uses ALL examples to avoid cache collisions when datasets differ
    only in later examples.
    """
    hasher = hashlib.md5()
    for ex in dataset.examples:
        # Hash text and label for each example
        hasher.update(f"{ex.text}_{ex.label}_".encode())
    hasher.update(f"_size_{len(dataset.examples)}".encode())
    return hasher.hexdigest()[:16]


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
                        # ActivationStore.save() creates a directory, use ActivationStore.load()
                        stores[f"blocks.{layer}.hook_resid_pre"] = ActivationStore.load(str(store_path))
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

    # CRITICAL FIX: Collect adjusted positions for each position type
    # These positions account for left padding and are the correct indices into raw_activations
    all_adjusted_positions = {}  # position_type -> list of positions (one per example)

    # Process in batches
    model.eval()
    with torch.no_grad():
        for batch_start in tqdm(range(0, len(dataset.examples), batch_size),
                               desc="Collecting activations"):
            batch_end = min(batch_start + batch_size, len(dataset.examples))
            batch_indices = list(range(batch_start, batch_end))

            # Get batch tensors - this returns ADJUSTED positions that account for padding
            batch = dataset.get_batch_tensors(batch_indices)
            input_ids = batch["input_ids"].to(device)

            # CRITICAL: Collect adjusted positions from batch["positions"]
            if "positions" in batch:
                for pos_type, pos_tensor in batch["positions"].items():
                    if pos_type not in all_adjusted_positions:
                        all_adjusted_positions[pos_type] = []
                    # Convert tensor to list of positions (one per example in batch)
                    for i in range(len(batch_indices)):
                        pos_val = pos_tensor[i]
                        if pos_val.dim() == 0:  # Single position (scalar)
                            all_adjusted_positions[pos_type].append(int(pos_val.item()))
                        else:  # Multiple positions (list)
                            all_adjusted_positions[pos_type].append(pos_val.tolist())

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
    
    # FIX: Find maximum sequence length across all batches for each hook point
    max_seq_lens = {}
    for hook in hook_points:
        max_seq_len = 0
        for batch_activations in all_activations[hook]:
            max_seq_len = max(max_seq_len, batch_activations.shape[1])
        max_seq_lens[hook] = max_seq_len
        print(f"Maximum sequence length for {hook}: {max_seq_len}")
    
    # FIX: Pad all batches to the same length for each hook point
    for hook in hook_points:
        max_seq_len = max_seq_lens[hook]
        padded_batches = []
        for batch_activations in all_activations[hook]:
            batch_size_actual, seq_len, hidden_size = batch_activations.shape
            if seq_len < max_seq_len:
                # Pad with zeros - maintain dtype
                padding = torch.zeros(
                    batch_size_actual, max_seq_len - seq_len, hidden_size, 
                    dtype=batch_activations.dtype, device=batch_activations.device
                )
                padded_batch = torch.cat([batch_activations, padding], dim=1)
            else:
                padded_batch = batch_activations
            padded_batches.append(padded_batch)
        all_activations[hook] = padded_batches
    
    # Create ActivationStore objects
    activation_stores = {}
    cache_base.mkdir(parents=True, exist_ok=True)
    
    for hook, activations in all_activations.items():
        layer = int(hook.split(".")[1])
        
        # Stack all activations (now they should have consistent shapes)
        raw_activations = torch.cat(activations, dim=0)
        print(f"Final activations for {hook}: {raw_activations.shape}, dtype: {raw_activations.dtype}")
        
        store = ActivationStore(
            raw_activations=raw_activations,
            hook_point=hook,
            example_indices=torch.arange(len(dataset.examples)),
            sequence_lengths=torch.tensor(dataset.get_token_lengths()),
            hidden_size=raw_activations.shape[-1],
            dataset=dataset,
            labels=torch.tensor([ex.label for ex in dataset.examples]),
            label_texts=[ex.label_text for ex in dataset.examples],
            adjusted_positions=all_adjusted_positions if all_adjusted_positions else None,  # CRITICAL FIX
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

