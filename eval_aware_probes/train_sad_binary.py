#!/usr/bin/env python3
"""
SAD Binary Token Training CLI

Efficient training of binary classification probes on SAD deployment vs evaluation datasets.
Trains probes to classify tokens as deployment (0) or evaluation (1) context.

Usage:
    python sad_probe_training/train_sad_binary.py --deploy_data path/to/deploy.jsonl --eval_data path/to/eval.jsonl --model_name Qwen/Qwen3-32B --layers all
    python sad_probe_training/train_sad_binary.py --deploy_data deploy.zip --eval_data eval.zip --layers 10 15 20 --epochs 10
    python sad_probe_training/train_sad_binary.py --config configs/sad_config.json
"""

import sys
import argparse
import logging
import json
import time
from pathlib import Path
from typing import List, Dict, Optional

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent))

from config import SADBinaryTrainingConfig
from data_loading import prepare_sad_training_data

# Import existing NTML infrastructure
ntml_path = current_dir.parent / "ntml_efficient_scripts"
sys.path.insert(0, str(ntml_path))

from activation_utils import collect_all_layers_activations, extract_layer_training_data
from training import NTMLBinaryTrainer


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
        force=True
    )


def parse_args():
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(
        description="Train binary classification probes on SAD deployment vs evaluation datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training on single layer
    python train_sad_binary.py --deploy_data ./sad/sad/stages/private_data_gen.zip --eval_data ./sad/sad/stages/private_data_gen.zip
    
    # Train on all layers (comprehensive analysis)
    python train_sad_binary.py --deploy_data deploy.zip --eval_data eval.zip \\
        --model_name Qwen/Qwen3-32B --layers all --probe_save_dir ./trained_probes
    
    # Train on specific layers with Qwen3
    python train_sad_binary.py --deploy_data deploy.zip --eval_data eval.zip \\
        --layers 10 15 20 25 --model_name Qwen/Qwen3-32B --enable_thinking
    
    # Fast debug run
    python train_sad_binary.py --deploy_data deploy.zip --eval_data eval.zip \\
        --preset fast_debug --layers 15
    
    # Production training with custom settings
    python train_sad_binary.py --deploy_data deploy.zip --eval_data eval.zip \\
        --preset production --epochs 30 --batch_size 64 --layers all
        
    # Load config from file
    python train_sad_binary.py --config my_config.json
        """
    )
    
    # Dataset and model arguments
    parser.add_argument(
        "--deploy_data", 
        type=str, 
        help="Path to deployment dataset (JSONL or ZIP file)"
    )
    parser.add_argument(
        "--eval_data", 
        type=str, 
        help="Path to evaluation dataset (JSONL or ZIP file)"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="Qwen/Qwen3-32B",
        help="Model name for activation extraction (default: Qwen/Qwen3-32B)"
    )
    parser.add_argument(
        "--layers", 
        nargs='+', 
        default=['15'],
        help="Layers to train probes on. Use 'all' for all layers or specify layer numbers (default: 15)"
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
    
    # PyTorch-specific arguments
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-3,
        help="L2 regularization for PyTorch training (default: 1e-3)"
    )
    
    # Qwen3-specific arguments
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Enable thinking mode for Qwen3 models"
    )
    
    # Preset configurations
    parser.add_argument(
        "--preset", 
        type=str, 
        choices=["fast_debug", "production", "large_model", "sklearn_fast", "sklearn_sweep"],
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
        default="./trained_probes/sad_binary",
        help="Output directory for trained probes (default: ./trained_probes/sad_binary)"
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
    
    return parser.parse_args()


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


def create_config_from_args(args, layer: int) -> SADBinaryTrainingConfig:
    """Create training configuration from command line arguments."""
    
    # Start with base configuration
    config_dict = {}
    
    # Apply preset configurations (simplified for SAD)
    if args.preset == "fast_debug":
        config_dict.update({
            "batch_size": 8,
            "num_epochs": 2,
            "activation_batch_size": 4,
            "log_every": 5,
            "max_length": 256,
        })
    elif args.preset == "production":
        config_dict.update({
            "batch_size": 32,
            "num_epochs": 20,
            "activation_batch_size": 16,
            "log_every": 50,
            "max_length": 512,
            "save_checkpoints": True,
        })
    elif args.preset == "large_model":
        config_dict.update({
            "batch_size": 16,
            "activation_batch_size": 8,
            "dtype": "bfloat16",
            "gradient_clip_norm": 0.5,
        })
    elif args.preset == "sklearn_fast":
        config_dict.update({
            "probe_method": "sklearn",
            "sklearn_C": 1.0,
            "sklearn_C_sweep": False,
            "handle_class_imbalance": True,
            "dtype": "float32",
        })
    elif args.preset == "sklearn_sweep":
        config_dict.update({
            "probe_method": "sklearn", 
            "sklearn_C_sweep": True,
            "handle_class_imbalance": True,
            "dtype": "float32",
        })
    
    # Override with command line arguments
    if args.deploy_data:
        config_dict["deploy_data_path"] = args.deploy_data
    if args.eval_data:
        config_dict["eval_data_path"] = args.eval_data
    if args.model_name:
        config_dict["model_name"] = args.model_name
    
    # Set layer-specific configuration
    config_dict["hook_layer"] = layer
    config_dict["hook_point"] = f"blocks.{layer}.hook_resid_pre"
    
    # Override other arguments
    for arg_name, config_name in [
        ("max_length", "max_length"),
        ("batch_size", "batch_size"),
        ("epochs", "num_epochs"),
        ("learning_rate", "learning_rate"),
        ("train_ratio", "train_ratio"),
        ("probe_save_dir", "output_dir"),
        ("probe_name", "probe_name"),
        ("cache_dir", "cache_dir"),
        ("dtype", "dtype"),
        ("activation_batch_size", "activation_batch_size"),
        ("force_recache", "force_recache"),
        ("verbose", "verbose"),
        ("probe_method", "probe_method"),
        ("sklearn_C", "sklearn_C"),
        ("sklearn_C_sweep", "sklearn_C_sweep"),
        ("weight_decay", "weight_decay"),
        ("enable_thinking", "enable_thinking"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            config_dict[config_name] = value
    
    # Handle device
    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    config_dict["device"] = device
    
    return SADBinaryTrainingConfig(**config_dict)


def load_config_from_file(config_path: str) -> SADBinaryTrainingConfig:
    """Load configuration from JSON file."""
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    return SADBinaryTrainingConfig.from_dict(config_dict)


def validate_config(config: SADBinaryTrainingConfig) -> bool:
    """Validate configuration and check prerequisites."""
    
    # Check datasets exist
    for path_name, path_value in [("deploy_data_path", config.deploy_data_path), ("eval_data_path", config.eval_data_path)]:
        if not Path(path_value).exists():
            print(f"❌ {path_name} not found: {path_value}")
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


def print_config_summary(config: SADBinaryTrainingConfig):
    """Print a summary of the training configuration."""
    
    print("🎯 SAD Binary Token Training Configuration")
    print("=" * 60)
    print(f"📄 Deploy Data: {Path(config.deploy_data_path).name}")
    print(f"📄 Eval Data: {Path(config.eval_data_path).name}")
    print(f"🤖 Model: {config.model_name}")
    print(f"📍 Hook Point: {config.hook_point}")
    print(f"💾 Device: {config.device} ({config.dtype})")
    print(f"🏋️ Training: {config.num_epochs} epochs, batch size {config.batch_size}")
    print(f"📚 Learning Rate: {config.learning_rate}")
    print(f"📊 Train/Val Split: {config.train_ratio:.1%}/{1-config.train_ratio:.1%}")
    print(f"⚙️  Probe Method: {config.probe_method}")
    if config.enable_thinking:
        print(f"🧠 Thinking Mode: Enabled")
    print(f"💾 Output: {config.output_dir}")
    print(f"💰 Cache: {config.cache_dir}")
    print("=" * 60)


def main():
    """Main training function."""
    
    args = parse_args()
    
    # Determine dataset paths
    if not args.config:
        if not args.deploy_data or not args.eval_data:
            print("❌ Must specify both --deploy_data and --eval_data, or use --config")
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
    
    # Setup base logging
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)
    
    all_results = {}
    successful_layers = []
    failed_layers = []
    
    start_time = time.time()
    
    # Use efficient multi-layer training if more than one layer
    if len(layers) > 1:
        print(f"\n{'='*60}")
        print(f"🚀 Efficient Multi-Layer Training ({len(layers)} layers)")
        print(f"{'='*60}")
        
        # Create a base config for activation collection
        if args.config:
            base_config = load_config_from_file(args.config)
        else:
            base_config = create_config_from_args(args, layers[0])
        
        try:
            # Phase 1: Collect activations for all layers at once
            print(f"\n📊 Phase 1: Collecting activations for all {len(layers)} layers...")
            
            # Use the SAD data loading infrastructure
            from data_loading import SADBinaryDataset
            dataset = SADBinaryDataset(base_config)
            dataset.load_and_process()
            
            # Collect activations using existing infrastructure
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
                    base_name = config.probe_name or f"sad_binary_{Path(config.deploy_data_path).stem}"
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
    
    # Handle any remaining layers with single-layer approach
    for layer_idx, layer in enumerate(layers_to_process):
        print(f"\n{'='*60}")
        print(f"🏋️ Training Layer {layer} ({layer_idx + 1}/{len(layers_to_process)}) - Single Layer")
        print(f"{'='*60}")
        
        try:
            # Load configuration for this layer
            if args.config:
                config = load_config_from_file(args.config)
            else:
                config = create_config_from_args(args, layer)
            
            # Update probe name to include layer
            base_name = config.probe_name or f"sad_binary_{Path(config.deploy_data_path).stem}"
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
            
            # Prepare training data
            logger.info(f"Preparing SAD training data for layer {layer}...")
            assistant_activations, assistant_labels, metadata = prepare_sad_training_data(config)
            
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
    print("🏁 SAD Binary Training Summary")
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
        summary_path = Path(args.probe_save_dir or "./trained_probes/sad_binary") / "training_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary_data = {
            "deploy_data_path": args.deploy_data,
            "eval_data_path": args.eval_data,
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