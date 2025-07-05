#!/usr/bin/env python3
"""
NTML Cache Management Utility

Manages activation caches for NTML datasets. Provides functions to clean dataset-specific 
caches without confirmation prompts.

Usage:
    python cache_management.py --clean-dataset data/dataset.jsonl --model meta-llama/Llama-3.1-8B-Instruct
    python cache_management.py --list-caches
    python cache_management.py --clean-all-caches
"""

import os
import sys
import shutil
import json
import hashlib
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Add probity to path for imports
probity_root = Path(__file__).parent.parent
sys.path.insert(0, str(probity_root))

from ntml_efficient_scripts.config import NTMLBinaryTrainingConfig
from ntml_efficient_scripts.data_loading import NTMLBinaryDataset

logger = logging.getLogger(__name__)


@dataclass
class CacheInfo:
    """Information about a cache directory."""
    path: Path
    dataset_name: str
    model_name: str
    cache_type: str  # "single_layer" or "multi_layer"
    size_bytes: int
    size_str: str
    layers: List[int]  # For multi-layer caches
    hash_key: str


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def get_directory_size(path: Path) -> int:
    """Calculate total size of directory in bytes."""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, FileNotFoundError):
                    pass
    except (OSError, FileNotFoundError):
        pass
    return total_size


def find_cache_directories(base_cache_dir: str = "./cache") -> List[CacheInfo]:
    """Find all NTML cache directories and return info about them."""
    
    base_path = Path(base_cache_dir)
    if not base_path.exists():
        return []
    
    caches = []
    
    # Pattern 1: Single layer caches in ntml_binary subdirectory
    # Structure: cache/ntml_binary/{model}_{dataset}/ntml_{hash}/
    ntml_binary_dir = base_path / "ntml_binary"
    if ntml_binary_dir.exists():
        for model_dataset_dir in ntml_binary_dir.iterdir():
            if model_dataset_dir.is_dir():
                # Look for ntml_{hash} subdirectories
                for hash_dir in model_dataset_dir.iterdir():
                    if hash_dir.is_dir() and hash_dir.name.startswith("ntml_"):
                        cache_info = _analyze_single_layer_cache(hash_dir, model_dataset_dir.name)
                        if cache_info:
                            caches.append(cache_info)
    
    # Pattern 2: Multi-layer caches directly in cache directory
    # Structure: cache/{hash_key}/layer_{N}.pt + cache_metadata.json
    for cache_dir in base_path.iterdir():
        if cache_dir.is_dir() and cache_dir.name != "ntml_binary":
            cache_info = _analyze_multi_layer_cache(cache_dir)
            if cache_info:
                caches.append(cache_info)
    
    return caches


def _analyze_single_layer_cache(cache_dir: Path, model_dataset_name: str) -> Optional[CacheInfo]:
    """Analyze a single-layer cache directory."""
    
    # Check for required files
    activations_file = cache_dir / "activations.pt"
    metadata_file = cache_dir / "metadata.json"
    
    if not (activations_file.exists() and metadata_file.exists()):
        return None
    
    try:
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        model_name = metadata.get("model_name", "unknown")
        dataset_hash = metadata.get("dataset_hash", cache_dir.name)
        
        # Extract dataset name from model_dataset_name if possible
        # Format is usually {model}_{dataset}
        dataset_name = "unknown"
        if "_" in model_dataset_name:
            parts = model_dataset_name.split("_")
            # Try to find dataset part (usually at the end)
            dataset_name = "_".join(parts[1:])  # Everything after first underscore
        
        size_bytes = get_directory_size(cache_dir)
        
        return CacheInfo(
            path=cache_dir,
            dataset_name=dataset_name,
            model_name=model_name,
            cache_type="single_layer",
            size_bytes=size_bytes,
            size_str=format_size(size_bytes),
            layers=[metadata.get("hook_layer", -1)],
            hash_key=dataset_hash
        )
        
    except Exception as e:
        logger.warning(f"Error analyzing cache {cache_dir}: {e}")
        return None


def _analyze_multi_layer_cache(cache_dir: Path) -> Optional[CacheInfo]:
    """Analyze a multi-layer cache directory."""
    
    metadata_file = cache_dir / "cache_metadata.json"
    if not metadata_file.exists():
        return None
    
    try:
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        model_name = metadata.get("model_name", "unknown")
        dataset_path = metadata.get("dataset_path", "unknown")
        layers = metadata.get("layers", [])
        
        # Extract dataset name from path
        dataset_name = Path(dataset_path).stem if dataset_path != "unknown" else "unknown"
        
        # Verify layer files exist
        layer_files = list(cache_dir.glob("layer_*.pt"))
        if not layer_files:
            return None
        
        size_bytes = get_directory_size(cache_dir)
        
        return CacheInfo(
            path=cache_dir,
            dataset_name=dataset_name,
            model_name=model_name,
            cache_type="multi_layer",
            size_bytes=size_bytes,
            size_str=format_size(size_bytes),
            layers=layers,
            hash_key=cache_dir.name
        )
        
    except Exception as e:
        logger.warning(f"Error analyzing cache {cache_dir}: {e}")
        return None


def generate_dataset_hash(dataset_path: str, model_name: str, max_length: int = 512) -> str:
    """Generate the same hash that would be used for a dataset (for matching existing caches)."""
    
    # This mirrors the logic in NTMLActivationCache.get_dataset_hash()
    try:
        # Create a minimal config to load dataset
        config = NTMLBinaryTrainingConfig(
            dataset_path=dataset_path,
            model_name=model_name,
            max_length=max_length
        )
        
        dataset = NTMLBinaryDataset(config)
        dataset.load_and_process()
        
        # Create hash based on dataset characteristics (matching activation_utils.py)
        hash_components = [
            dataset_path,
            model_name,
            "blocks.15.hook_resid_pre",  # Default hook point
            max_length,
            len(dataset.examples),
            str(dataset.label_distribution),
        ]
        
        # Sample first few examples for content-based hash
        if dataset.examples:
            for ex in dataset.examples[:3]:
                hash_components.extend([
                    ex.conversation_id,
                    str(ex.tokens[:10]),  # First 10 tokens
                    str(ex.labels[:10]),  # First 10 labels
                ])
        
        content_str = "|".join(str(comp) for comp in hash_components)
        return hashlib.md5(content_str.encode()).hexdigest()[:16]
        
    except Exception as e:
        logger.warning(f"Could not generate dataset hash: {e}")
        # Fallback to simple hash
        content_str = f"{dataset_path}|{model_name}|{max_length}"
        return hashlib.md5(content_str.encode()).hexdigest()[:16]


def find_dataset_caches(dataset_path: str, model_name: str, base_cache_dir: str = "./cache") -> List[CacheInfo]:
    """Find all caches for a specific dataset and model combination."""
    
    all_caches = find_cache_directories(base_cache_dir)
    dataset_name = Path(dataset_path).stem
    
    matching_caches = []
    
    for cache in all_caches:
        # Match by dataset name and model name
        if (cache.dataset_name == dataset_name and 
            cache.model_name == model_name):
            matching_caches.append(cache)
    
    return matching_caches


def clean_dataset_cache(dataset_path: str, model_name: str, base_cache_dir: str = "./cache") -> Tuple[int, int]:
    """
    Clean all caches for a specific dataset and model.
    
    Returns:
        (num_deleted, total_size_freed)
    """
    
    matching_caches = find_dataset_caches(dataset_path, model_name, base_cache_dir)
    
    if not matching_caches:
        print(f"No caches found for dataset: {Path(dataset_path).name} + model: {model_name}")
        return 0, 0
    
    total_size_freed = 0
    num_deleted = 0
    
    for cache in matching_caches:
        print(f"Deleting {cache.cache_type} cache: {cache.path}")
        print(f"  Size: {cache.size_str}")
        print(f"  Layers: {cache.layers}")
        
        try:
            shutil.rmtree(cache.path)
            total_size_freed += cache.size_bytes
            num_deleted += 1
            print(f"  ✅ Deleted")
        except Exception as e:
            print(f"  ❌ Error deleting: {e}")
    
    return num_deleted, total_size_freed


def clean_all_caches(base_cache_dir: str = "./cache") -> Tuple[int, int]:
    """
    Clean ALL NTML caches.
    
    Returns:
        (num_deleted, total_size_freed)
    """
    
    all_caches = find_cache_directories(base_cache_dir)
    
    if not all_caches:
        print("No NTML caches found")
        return 0, 0
    
    total_size_freed = 0
    num_deleted = 0
    
    print(f"Deleting {len(all_caches)} cache directories...")
    
    for cache in all_caches:
        print(f"Deleting: {cache.path} ({cache.size_str})")
        
        try:
            shutil.rmtree(cache.path)
            total_size_freed += cache.size_bytes
            num_deleted += 1
        except Exception as e:
            print(f"  ❌ Error deleting {cache.path}: {e}")
    
    return num_deleted, total_size_freed


def list_all_caches(base_cache_dir: str = "./cache") -> None:
    """List all NTML caches with details."""
    
    all_caches = find_cache_directories(base_cache_dir)
    
    if not all_caches:
        print("No NTML caches found")
        return
    
    # Sort by size (largest first)
    all_caches.sort(key=lambda x: x.size_bytes, reverse=True)
    
    total_size = sum(cache.size_bytes for cache in all_caches)
    
    print(f"Found {len(all_caches)} NTML cache directories")
    print(f"Total size: {format_size(total_size)}")
    print()
    
    # Group by dataset
    by_dataset = {}
    for cache in all_caches:
        key = f"{cache.dataset_name} + {cache.model_name}"
        if key not in by_dataset:
            by_dataset[key] = []
        by_dataset[key].append(cache)
    
    for dataset_model, caches in by_dataset.items():
        dataset_size = sum(c.size_bytes for c in caches)
        print(f"📊 {dataset_model} ({format_size(dataset_size)})")
        
        for cache in caches:
            layers_str = f"layers {cache.layers}" if len(cache.layers) > 1 else f"layer {cache.layers[0]}"
            print(f"  • {cache.cache_type} ({layers_str}) - {cache.size_str}")
            print(f"    {cache.path}")
        print()


def setup_logging(verbose: bool = False):
    """Setup logging."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s'
    )


def main():
    """Main CLI interface."""
    
    parser = argparse.ArgumentParser(
        description="Manage NTML activation caches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List all caches
    python cache_management.py --list-caches
    
    # Clean caches for specific dataset
    python cache_management.py --clean-dataset data/2T1L_500samples.jsonl \\
        --model meta-llama/Llama-3.1-8B-Instruct
    
    # Clean all caches (nuclear option)
    python cache_management.py --clean-all-caches
    
    # Show cache usage
    python cache_management.py --cache-usage
        """
    )
    
    parser.add_argument(
        "--list-caches",
        action="store_true",
        help="List all existing caches"
    )
    
    parser.add_argument(
        "--clean-dataset",
        type=str,
        help="Clean caches for specific dataset path"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name (required with --clean-dataset)"
    )
    
    parser.add_argument(
        "--clean-all-caches",
        action="store_true",
        help="Delete ALL NTML caches (nuclear option)"
    )
    
    parser.add_argument(
        "--cache-usage",
        action="store_true",
        help="Show cache usage summary"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache",
        help="Base cache directory (default: ./cache)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # Handle commands
    if args.list_caches or args.cache_usage:
        list_all_caches(args.cache_dir)
    
    elif args.clean_dataset:
        if not Path(args.clean_dataset).exists():
            print(f"❌ Dataset file not found: {args.clean_dataset}")
            return 1
        
        print(f"Cleaning caches for dataset: {Path(args.clean_dataset).name}")
        print(f"Model: {args.model}")
        print()
        
        num_deleted, size_freed = clean_dataset_cache(
            args.clean_dataset, 
            args.model, 
            args.cache_dir
        )
        
        if num_deleted > 0:
            print(f"\n✅ Deleted {num_deleted} cache directories")
            print(f"✅ Freed {format_size(size_freed)} of disk space")
        else:
            print("No matching caches found to delete")
    
    elif args.clean_all_caches:
        print("🗑️  Deleting ALL NTML caches...")
        print()
        
        num_deleted, size_freed = clean_all_caches(args.cache_dir)
        
        if num_deleted > 0:
            print(f"\n✅ Deleted {num_deleted} cache directories")
            print(f"✅ Freed {format_size(size_freed)} of disk space")
        else:
            print("No caches found to delete")
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())