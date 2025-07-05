#!/usr/bin/env python3
"""
NTML Binary Token Training CLI

Efficient training of binary classification probes on NTML conversational datasets.
Trains probes to classify individual assistant tokens as truthful or deceptive.

Usage:
    python ntml_efficient_scripts/train_ntml_binary.py --train_dataset_dir data/NTML-datasets/2T1L_500samples.jsonl --model_name meta-llama/Llama-3.1-8B-Instruct --layers all --probe_save_dir ./trained_probes
    python ntml_efficient_scripts/train_ntml_binary.py --train_dataset_dir data/NTML-datasets/5T1L_100samples.jsonl --layers 10 15 20 --epochs 10
    python ntml_efficient_scripts/train_ntml_binary.py --config configs/fast_debug.json
"""

import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Optional

# Add probity to path for imports
probity_root = Path(__file__).parent.parent
sys.path.insert(0, str(probity_root))

from ntml_efficient_scripts.config import NTMLBinaryTrainingConfig, FAST_DEBUG_CONFIG, PRODUCTION_CONFIG, LARGE_MODEL_CONFIG
from ntml_efficient_scripts.activation_utils import prepare_ntml_training_data
from ntml_efficient_scripts.training import NTMLBinaryTrainer


def setup_logging(verbose: bool = True, log_file: Optional[str] = None):
    """Setup logging configuration."""
    
    log_level = logging.INFO if verbose else logging.WARNING
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers,
        force=True  # Override any existing configuration
    )


def parse_args():
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(
        description="Train binary classification probes on NTML conversational datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training on single layer
    python train_ntml_binary.py --train_dataset_dir data/NTML-datasets/2T1L_500samples.jsonl
    
    # Train on all layers (like original probe_training.py)
    python train_ntml_binary.py --train_dataset_dir data/NTML-datasets/2T1L_500samples.jsonl \\
        --model_name meta-llama/Llama-3.1-8B-Instruct --layers all --probe_save_dir ./trained_probes
    
    # Train on specific layers
    python train_ntml_binary.py --train_dataset_dir data/NTML-datasets/5T1L_100samples.jsonl \\
        --layers 10 15 20 25 --preset production
    
    # Fast debug run
    python train_ntml_binary.py --train_dataset_dir data/NTML-datasets/2T1L_10samples.jsonl \\
        --preset fast_debug --layers 15
    
    # Production training with custom settings
    python train_ntml_binary.py --train_dataset_dir data/NTML-datasets/10T10L_1000samples.jsonl \\
        --preset production --epochs 30 --batch_size 64 --layers all
        
    # Load config from file
    python train_ntml_binary.py --config my_config.json
        """
    )
    
    # Dataset and model arguments
    parser.add_argument(
        "--train_dataset_dir", 
        type=str, 
        help="Path to NTML JSONL dataset file"
    )
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        help="Path to NTML JSONL dataset file (alias for --train_dataset_dir)"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name for activation extraction (default: Llama-3.1-8B-Instruct)"
    )
    parser.add_argument(
        "--layers", 
        nargs='+', 
        default=['15'],
        help="Layers to train probes on. Use 'all' for all layers or specify layer numbers (default: 15)"
    )
    parser.add_argument(
        "--hook_layer", 
        type=int, 
        help="Single layer to extract activations from (deprecated, use --layers)"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=512,
        help="Maximum sequence length (default: 512)"
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="Training batch size (default: 32)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=20,
        help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--train_ratio", 
        type=float, 
        default=0.8,
        help="Training/validation split ratio (default: 0.8)"
    )
    
    # Probe method arguments
    parser.add_argument(
        "--probe_method",
        type=str,
        choices=["sklearn", "pytorch"],
        default="pytorch",
        help="Probe training method (default: pytorch)"
    )
    
    # Sklearn-specific arguments
    parser.add_argument(
        "--sklearn_C",
        type=float,
        default=1.0,
        help="Sklearn regularization parameter C (default: 1.0)"
    )
    parser.add_argument(
        "--sklearn_C_sweep",
        action="store_true",
        help="Perform regularization sweep to find best C"
    )
    parser.add_argument(
        "--sklearn_C_values",
        nargs='+',
        type=float,
        help="Custom C values for sweep (e.g., --sklearn_C_values 0.1 1.0 10.0)"
    )
    parser.add_argument(
        "--sklearn_solver",
        type=str,
        choices=["liblinear", "newton-cg", "lbfgs", "sag", "saga"],
        default="liblinear",
        help="Sklearn solver (default: liblinear)"
    )
    
    # PyTorch-specific arguments
    parser.add_argument(
        "--pytorch_bias",
        action="store_true",
        default=True,
        help="Use bias term in PyTorch probe (default: True)"
    )
    parser.add_argument(
        "--pytorch_no_bias",
        action="store_true",
        help="Disable bias term in PyTorch probe"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-3,
        help="L2 regularization for PyTorch training (default: 1e-3)"
    )
    
    # Update presets
    parser.add_argument(
        "--preset", 
        type=str, 
        choices=[
            "fast_debug", "production", "large_model", 
            "sklearn_fast", "sklearn_sweep", "pytorch_interpretability"
        ],
        help="Use predefined configuration preset"
    )
    parser.add_argument(
        "--config", 
        type=str,
        help="Load configuration from JSON file"
    )
    
    # Output arguments
    parser.add_argument(
        "--probe_save_dir", 
        type=str, 
        default="./trained_probes/ntml_binary",
        help="Output directory for trained probes (default: ./trained_probes/ntml_binary)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        help="Output directory for trained probes (alias for --probe_save_dir)"
    )
    parser.add_argument(
        "--probe_name", 
        type=str,
        help="Name for the trained probe (default: auto-generated)"
    )
    parser.add_argument(
        "--cache_dir", 
        type=str,
        help="Cache directory for activations (default: auto-generated)"
    )
    
    # System arguments
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use (cuda/cpu/auto, default: auto)"
    )
    parser.add_argument(
        "--dtype", 
        type=str, 
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Model dtype (default: bfloat16)"
    )
    parser.add_argument(
        "--activation_batch_size", 
        type=int, 
        default=16,
        help="Batch size for activation collection (default: 16)"
    )
    
    # Utility arguments
    parser.add_argument(
        "--force_recache", 
        action="store_true",
        help="Force recollection of activations"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--dry_run", 
        action="store_true",
        help="Setup and validate configuration without training"
    )
    parser.add_argument(
        "--list_datasets", 
        action="store_true",
        help="List available NTML datasets and exit"
    )
    
    return parser.parse_args()


def list_available_datasets():
    """List all available NTML datasets."""
    
    data_dir = probity_root / "data" / "NTML-datasets"
    
    if not data_dir.exists():
        print("❌ NTML datasets directory not found")
        print(f"   Expected: {data_dir}")
        print("   Run: python generate_ntml_datasets.py")
        return
    
    jsonl_files = list(data_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        print("❌ No NTML datasets found")
        print(f"   Directory: {data_dir}")
        print("   Run: python generate_ntml_datasets.py")
        return
    
    print("📋 Available NTML Datasets:")
    print(f"   Directory: {data_dir}")
    print()
    
    for jsonl_file in sorted(jsonl_files):
        try:
            # Quick peek at first line to get basic info
            with open(jsonl_file, 'r') as f:
                first_line = f.readline()
                if first_line.strip():
                    data = json.loads(first_line)
                    ratio = data.get("labels", {}).get("ratio", "unknown")
                    print(f"   • {jsonl_file.name} ({ratio})")
                else:
                    print(f"   • {jsonl_file.name} (empty)")
        except Exception as e:
            print(f"   • {jsonl_file.name} (error reading: {e})")
    
    print()
    print("Usage:")
    print(f"   python train_ntml_binary.py --dataset_path {data_dir}/2T1L_500samples.jsonl")


def determine_layers(layers_arg: List[str], model_name: str) -> List[int]:
    """Determine which layers to train on."""
    
    if 'all' in layers_arg:
        # Load model to get number of layers
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            if hasattr(config, 'num_hidden_layers'):
                num_layers = config.num_hidden_layers
            elif hasattr(config, 'n_layer'):
                num_layers = config.n_layer
            elif hasattr(config, 'num_layers'):
                num_layers = config.num_layers
            else:
                print("⚠️  Could not determine number of layers, using default range 0-31")
                num_layers = 32
            
            return list(range(num_layers))
        except Exception as e:
            print(f"⚠️  Error determining layers from model config: {e}")
            print("   Using default range 0-31")
            return list(range(32))
    else:
        # Parse specific layer numbers
        try:
            return [int(layer) for layer in layers_arg]
        except ValueError as e:
            raise ValueError(f"Invalid layer specification: {layers_arg}. Use 'all' or layer numbers.")


def create_config_from_args(args, layer: int) -> NTMLBinaryTrainingConfig:
    """Create training configuration from command line arguments."""
    
    # Start with base configuration
    config_dict = {}
    
    # Apply preset if specified
    if args.preset == "fast_debug":
        config_dict.update(FAST_DEBUG_CONFIG)
    elif args.preset == "production":
        config_dict.update(PRODUCTION_CONFIG)
    elif args.preset == "large_model":
        config_dict.update(LARGE_MODEL_CONFIG)

    # Apply new presets
    if args.preset == "sklearn_fast":
        config_dict.update(SKLEARN_FAST_CONFIG)
    elif args.preset == "sklearn_sweep":
        config_dict.update(SKLEARN_SWEEP_CONFIG)
    elif args.preset == "pytorch_interpretability":
        config_dict.update(PYTORCH_INTERPRETABILITY_CONFIG)
    
    # Override with command line arguments
    dataset_path = args.train_dataset_dir or args.dataset_path
    if dataset_path:
        config_dict["dataset_path"] = dataset_path
    if args.model_name:
        config_dict["model_name"] = args.model_name
    
    # Set layer-specific configuration
    config_dict["hook_layer"] = layer
    config_dict["hook_point"] = f"blocks.{layer}.hook_resid_pre"
    
    if args.max_length:
        config_dict["max_length"] = args.max_length
    if args.batch_size:
        config_dict["batch_size"] = args.batch_size
    if args.epochs:
        config_dict["num_epochs"] = args.epochs
    if args.learning_rate:
        config_dict["learning_rate"] = args.learning_rate
    if args.train_ratio:
        config_dict["train_ratio"] = args.train_ratio
    
    # Handle output directory
    output_dir = args.probe_save_dir or args.output_dir
    if output_dir:
        config_dict["output_dir"] = output_dir
    
    if args.probe_name:
        config_dict["probe_name"] = args.probe_name
    if args.cache_dir:
        config_dict["cache_dir"] = args.cache_dir
    if args.dtype:
        config_dict["dtype"] = args.dtype
    if args.activation_batch_size:
        config_dict["activation_batch_size"] = args.activation_batch_size
    if args.force_recache:
        config_dict["force_recache"] = args.force_recache
    if args.verbose:
        config_dict["verbose"] = args.verbose
    
    # Add new probe method overrides
    if args.probe_method:
        config_dict["probe_method"] = args.probe_method
    if args.sklearn_C:
        config_dict["sklearn_C"] = args.sklearn_C
    if args.sklearn_C_sweep:
        config_dict["sklearn_C_sweep"] = args.sklearn_C_sweep
    if args.sklearn_C_values:
        config_dict["sklearn_C_values"] = args.sklearn_C_values
    if args.sklearn_solver:
        config_dict["sklearn_solver"] = args.sklearn_solver
    if args.pytorch_no_bias:
        config_dict["pytorch_bias"] = False
    elif args.pytorch_bias is not None:
        config_dict["pytorch_bias"] = args.pytorch_bias
    if args.weight_decay:
        config_dict["weight_decay"] = args.weight_decay
    
    # Handle device
    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    config_dict["device"] = device
    
    return NTMLBinaryTrainingConfig(**config_dict)


def load_config_from_file(config_path: str) -> NTMLBinaryTrainingConfig:
    """Load configuration from JSON file."""
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    return NTMLBinaryTrainingConfig.from_dict(config_dict)


def validate_config(config: NTMLBinaryTrainingConfig) -> bool:
    """Validate configuration and check prerequisites."""
    
    # Check dataset exists
    if not Path(config.dataset_path).exists():
        print(f"❌ Dataset not found: {config.dataset_path}")
        return False
    
    # Check device compatibility
    if config.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            print("❌ CUDA requested but not available")
            return False
    
    # Check output directory is writable
    output_dir = Path(config.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"❌ Cannot create output directory {output_dir}: {e}")
        return False
    
    return True


def print_config_summary(config: NTMLBinaryTrainingConfig):
    """Print a summary of the training configuration."""
    
    print("🎯 NTML Binary Token Training Configuration")
    print("=" * 60)
    print(f"📄 Dataset: {Path(config.dataset_path).name}")
    print(f"🤖 Model: {config.model_name}")
    print(f"📍 Hook Point: {config.hook_point}")
    print(f"💾 Device: {config.device} ({config.dtype})")
    print(f"🏋️ Training: {config.num_epochs} epochs, batch size {config.batch_size}")
    print(f"📚 Learning Rate: {config.learning_rate}")
    print(f"📊 Train/Val Split: {config.train_ratio:.1%}/{1-config.train_ratio:.1%}")
    print(f"💾 Output: {config.output_dir}")
    print(f"💰 Cache: {config.cache_dir}")
    print("=" * 60)


def main():
    """Main training function."""
    
    args = parse_args()
    
    # Handle special commands
    if args.list_datasets:
        list_available_datasets()
        return 0
    
    # Determine dataset path
    dataset_path = args.train_dataset_dir or args.dataset_path
    if not dataset_path:
        if args.config:
            pass  # Will be handled by config loading
        else:
            print("❌ Must specify either --train_dataset_dir, --dataset_path, or --config")
            return 1



    # Determine layers to train on
    try:
        if args.config:
            # Single layer from config file
            config = load_config_from_file(args.config)
            layers = [config.hook_layer]
        else:
            # Multiple layers from command line
            layers = determine_layers(args.layers, args.model_name)
            print(f"🎯 Training on {len(layers)} layers: {layers}")
    except Exception as e:
        print(f"❌ Error determining layers: {e}")
        return 1
    
    # Setup base logging (will be updated per layer)
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)
    
    all_results = {}
    successful_layers = []
    failed_layers = []
    
    start_time = time.time()
    
    # NEW EFFICIENT APPROACH: Collect activations for all layers at once
    if len(layers) > 1:
        print(f"\n{'='*60}")
        print(f"🚀 Efficient Multi-Layer Training ({len(layers)} layers)")
        print(f"{'='*60}")
        
        # Create a base config for activation collection
        if args.config:
            base_config = load_config_from_file(args.config)
        else:
            base_config = create_config_from_args(args, layers[0])  # Use first layer for base config
        
        try:
            # Import the new efficient functions
            from ntml_efficient_scripts.activation_utils import collect_all_layers_activations, extract_layer_training_data
            
            # Phase 1: Collect activations for all layers at once
            print(f"\n📊 Phase 1: Collecting activations for all {len(layers)} layers...")
            cached_activations = collect_all_layers_activations(base_config, layers)
            print(f"✅ Cached activations for {len(cached_activations)} layers")
            
            # Phase 2: Train each layer using cached activations
            print(f"\n🏋️ Phase 2: Training probes on cached activations...")
            
            for layer_idx, layer in enumerate(layers):
                print(f"\n{'='*60}")
                print(f"🏋️ Training Layer {layer} ({layer_idx + 1}/{len(layers)}) - Using Cached Activations")
                print(f"{'='*60}")
                
                try:
                    # Load configuration for this layer
                    if args.config:
                        config = load_config_from_file(args.config)
                    else:
                        config = create_config_from_args(args, layer)
                    
                    # Update probe name to include layer
                    base_name = config.probe_name or f"ntml_binary_{Path(args.dataset_path or args.train_dataset_dir).stem}"
                    config.probe_name = f"{base_name}_layer_{layer}"
                    
                    # Validate configuration
                    if not validate_config(config):
                        print(f"❌ Configuration validation failed for layer {layer}")
                        failed_layers.append(layer)
                        continue
                    
                    # Setup layer-specific logging
                    output_paths = config.get_output_paths()
                    setup_logging(verbose=config.verbose, log_file=str(output_paths["log"]))
                    
                    # Print configuration summary
                    print_config_summary(config)
                    
                    if args.dry_run:
                        print(f"✅ Dry run completed for layer {layer} - configuration is valid")
                        continue
                    
                    # Extract training data for this layer from cache
                    logger.info(f"Extracting cached training data for layer {layer}...")
                    assistant_activations, assistant_labels, metadata = extract_layer_training_data(cached_activations, layer)
                    
                    # Train the probe
                    logger.info(f"Training binary token classifier for layer {layer}...")
                    trainer = NTMLBinaryTrainer(config)
                    training_results = trainer.train(assistant_activations, assistant_labels)
                    
                    # Save the trained model
                    logger.info(f"Saving trained probe for layer {layer}...")
                    trainer.save_model(str(output_paths["probe"]), {
                        **metadata,
                        **training_results,
                    })

                    # Save all C sweep models if sklearn C sweep was performed
                    if (config.probe_method == "sklearn" and 
                        getattr(config, 'sklearn_C_sweep', False)):
                        
                        logger.info(f"Saving all C sweep models for layer {layer}...")
                        
                        # Create sweep models base path (avoid overwriting main probe)
                        sweep_base_path = output_paths["probe"].with_name(
                            f"{output_paths['probe'].stem}_sweep{output_paths['probe'].suffix}"
                        )
                        
                        saved_sweep_models = trainer.save_all_sweep_models(
                            str(sweep_base_path), 
                            {**metadata, **training_results}
                        )
                        
                        if saved_sweep_models:
                            logger.info(f"✅ Saved {len(saved_sweep_models)} C sweep models:")
                            for C, paths in saved_sweep_models.items():
                                val_score = None
                                if 'C_sweep_results' in training_results:
                                    C_values = training_results['C_sweep_results']['C_values']
                                    val_scores = training_results['C_sweep_results']['val_scores']
                                    if C in C_values:
                                        idx = C_values.index(C)
                                        val_score = val_scores[idx]
                                
                                score_info = f" (val_acc: {val_score:.4f})" if val_score else ""
                                logger.info(f"   C={C}: {Path(paths['model_path']).name}{score_info}")
                        else:
                            logger.warning("No sweep models were saved")
                    
                    
                    # Save configuration and metrics
                    with open(output_paths["config"], 'w') as f:
                        json.dump(config.to_dict(), f, indent=2)
                    
                    with open(output_paths["metrics"], 'w') as f:
                        json.dump(training_results["final_metrics"], f, indent=2)
                    
                    # Store results
                    all_results[layer] = {
                        "config": config.to_dict(),
                        "training_results": training_results,
                        "paths": {str(k): str(v) for k, v in output_paths.items()},
                    }
                    
                    successful_layers.append(layer)
                    
                    # Print layer summary
                    print(f"\n✅ Layer {layer} completed successfully!")
                    print(f"📊 Final validation F1: {training_results['final_metrics']['f1']:.4f}")
                    print(f"📊 Final validation AUROC: {training_results['final_metrics']['auroc']:.4f}")
                    print(f"💾 Probe saved: {output_paths['probe']}")
                    
                except KeyboardInterrupt:
                    print(f"\n❌ Training interrupted by user at layer {layer}")
                    failed_layers.append(layer)
                    break
                except Exception as e:
                    logger.error(f"Training failed for layer {layer}: {e}", exc_info=args.verbose)
                    print(f"❌ Layer {layer} failed: {e}")
                    failed_layers.append(layer)
                    continue
                    
        except Exception as e:
            print(f"❌ Efficient multi-layer training failed: {e}")
            print("Falling back to single-layer approach...")
            # Fall back to original approach
            layers_to_process = [l for l in layers if l not in successful_layers]
        else:
            layers_to_process = []  # All layers processed successfully
    else:
        # Single layer - use original approach
        layers_to_process = layers
    
    # FALLBACK: Original single-layer approach for remaining layers
    if layers_to_process:
        print(f"\n{'='*60}")
        print(f"🔄 Single-Layer Training ({len(layers_to_process)} layers)")
        print(f"{'='*60}")
        
        for layer_idx, layer in enumerate(layers_to_process):
            # ... existing single-layer training code ...
            print(f"\n{'='*60}")
            print(f"🏋️ Training Layer {layer} ({layer_idx + 1}/{len(layers)})")
            print(f"{'='*60}")
            
            try:
                # Load configuration for this layer
                if args.config:
                    config = load_config_from_file(args.config)
                else:
                    config = create_config_from_args(args, layer)
                
                # Update probe name to include layer
                base_name = config.probe_name or f"ntml_binary_{Path(dataset_path).stem}"
                config.probe_name = f"{base_name}_layer_{layer}"
    
                # Validate configuration
                if not validate_config(config):
                    print(f"❌ Configuration validation failed for layer {layer}")
                    failed_layers.append(layer)
                    continue
    
                # Setup layer-specific logging
                output_paths = config.get_output_paths()
                setup_logging(verbose=config.verbose, log_file=str(output_paths["log"]))
                
    
                
                # Print configuration summary
                print_config_summary(config)
                
                if args.dry_run:
                    print(f"✅ Dry run completed for layer {layer} - configuration is valid")
                    continue
                
                # Prepare training data (will use cached activations if available)
                logger.info(f"Preparing NTML training data for layer {layer}...")
                assistant_activations, assistant_labels, metadata = prepare_ntml_training_data(config)
                
                # Train the probe
                logger.info(f"Training binary token classifier for layer {layer}...")
                trainer = NTMLBinaryTrainer(config)
                training_results = trainer.train(assistant_activations, assistant_labels)
                
                # Save the trained model
                logger.info(f"Saving trained probe for layer {layer}...")
                trainer.save_model(str(output_paths["probe"]), {
                    **metadata,
                    **training_results,
                })
                
                # Save configuration and metrics
                with open(output_paths["config"], 'w') as f:
                    json.dump(config.to_dict(), f, indent=2)
                
                with open(output_paths["metrics"], 'w') as f:
                    json.dump(training_results["final_metrics"], f, indent=2)
                
                # Store results
                all_results[layer] = {
                    "config": config.to_dict(),
                    "training_results": training_results,
                    "paths": {str(k): str(v) for k, v in output_paths.items()},
                }
                
                successful_layers.append(layer)
                
                # Print layer summary
                print(f"\n✅ Layer {layer} completed successfully!")
                print(f"📊 Final validation F1: {training_results['final_metrics']['f1']:.4f}")
                print(f"📊 Final validation AUROC: {training_results['final_metrics']['auroc']:.4f}")
                print(f"💾 Probe saved: {output_paths['probe']}")
                
            except KeyboardInterrupt:
                print(f"\n❌ Training interrupted by user at layer {layer}")
                failed_layers.append(layer)
                break
            except Exception as e:
                logger.error(f"Training failed for layer {layer}: {e}", exc_info=args.verbose)
                print(f"❌ Layer {layer} failed: {e}")
                failed_layers.append(layer)
                continue
    
    total_time = time.time() - start_time
    
    # Print final summary
    print(f"\n{'='*60}")
    print("🏁 NTML Binary Training Summary")
    print(f"{'='*60}")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    print(f"✅ Successful layers: {len(successful_layers)}/{len(layers)}")
    
    if successful_layers:
        print(f"📊 Successful: {successful_layers}")
        
        # Show best performing layer
        best_layer = None
        best_f1 = 0.0
        for layer in successful_layers:
            f1 = all_results[layer]["training_results"]["final_metrics"]["f1"]
            if f1 > best_f1:
                best_f1 = f1
                best_layer = layer
        
        if best_layer is not None:
            print(f"🏆 Best layer: {best_layer} (F1: {best_f1:.4f})")
    
    if failed_layers:
        print(f"❌ Failed layers: {failed_layers}")
    
    # Save summary results
    if successful_layers and not args.dry_run:
        summary_path = Path(args.probe_save_dir or args.output_dir or "./trained_probes/ntml_binary") / "training_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary_data = {
            "dataset_path": dataset_path,
            "model_name": args.model_name,
            "layers_attempted": layers,
            "successful_layers": successful_layers,
            "failed_layers": failed_layers,
            "total_time": total_time,
            "results": all_results,
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"📄 Summary saved: {summary_path}")
    
    print(f"{'='*60}")
    
    # Return exit code
    if failed_layers and not successful_layers:
        return 1  # All failed
    elif failed_layers:
        return 2  # Some failed
    else:
        return 0  # All succeeded


if __name__ == "__main__":
    exit(main())