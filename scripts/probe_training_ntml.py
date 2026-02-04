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
from probity.training.parallel import (
    train_probes_parallel,
    train_all_layers_parallel,
    _ensure_spawn_start_method,
)
from probity.utils.caching import get_dataset_hash, smart_cache_activations
from probity.utils.dataset_loading import load_contrastive_ntml_dataset, get_model_dtype


def train_single_probe_with_hyperparams(
    layer: int, activation_store: ActivationStore,
    probe_type: str, hyperparams: Optional[Dict],
    args, model_name: str, hidden_size: int,
    device: str, dtype: torch.dtype,
    name_suffix: str = ""
) -> Dict:
    """Train a single probe with specific hyperparameters."""
    hook_point = f"blocks.{layer}.hook_resid_pre"

    # Build mode string for logging
    loss_mode = args.loss_mode
    mode_str = f" [{loss_mode}]" if loss_mode != "sample_mean" else ""

    hp_str = f" {hyperparams}" if hyperparams else ""
    print(f"Training {probe_type}{name_suffix} probe on layer {layer}{mode_str}{hp_str}")

    # Get configurations with hyperparams
    probe_config = get_probe_config(
        probe_type, hidden_size, model_name,
        hook_point, layer, dtype,
        hyperparams=hyperparams
    )
    probe_cls = get_probe_class(probe_type)
    trainer_config = get_trainer_config(probe_type, device, args.batch_size)
    trainer_cls = get_trainer_class(probe_type)

    # Apply loss mode configuration
    if hasattr(trainer_config, 'loss_mode'):
        trainer_config.loss_mode = args.loss_mode
        trainer_config.joint_alpha = args.joint_alpha
        trainer_config.anneal_warmup = args.anneal_warmup
        trainer_config.sparsity_penalty = args.sparsity_penalty
        # Localization loss hyperparameters
        trainer_config.margin = args.margin
        trainer_config.temperature = args.temperature
        trainer_config.num_pairs = args.num_pairs

    # Apply epoch/patience settings
    trainer_config.num_epochs = args.num_epochs
    trainer_config.patience = args.patience

    # Initialize probe and trainer
    probe = probe_cls(probe_config).to(device)
    trainer = trainer_cls(trainer_config)

    # Prepare data - use spans for token-level loss modes
    token_level_modes = ['token_all', 'token_spans_only', 'joint', 'joint_span_max', 'annealed', 'span_max', 'span_mean',
                         'margin', 'ranking', 'contrastive_intra', 'soft_recall', 'topk_overlap']
    needs_spans = args.loss_mode in token_level_modes

    if needs_spans and hasattr(trainer, 'prepare_supervised_data_with_spans'):
        train_loader, val_loader, _, _ = trainer.prepare_supervised_data_with_spans(
            activation_store, "LIE_SPAN"
        )
    else:
        train_loader, val_loader = trainer.prepare_supervised_data(
            activation_store, "LIE_SPAN"
        )

    # Train
    history = trainer.train(probe, train_loader, val_loader)

    # Determine save name
    probe_name = f"{probe_type}{name_suffix}"
    save_dir = Path(args.probe_save_dir) / probe_name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"layer_{layer}_probe.pt"
    probe.save(str(save_path))

    result = {
        'final_train_loss': history['train_loss'][-1],
        'final_val_loss': history['val_loss'][-1] if 'val_loss' in history and history['val_loss'] else None,
        'save_path': str(save_path),
        'loss_mode': args.loss_mode,
        'hyperparams': hyperparams,
    }

    # Add AUROC if available
    if hasattr(probe, 'config') and hasattr(probe.config, 'optimal_thresholds'):
        thresholds = probe.config.optimal_thresholds
        if 'train_auroc_score' in thresholds:
            result['train_auroc'] = thresholds['train_auroc_score']

    # Track omega for span_max and similar loss modes
    if 'omega' in history and history['omega']:
        result['final_omega'] = history['omega'][-1]
        result['anneal_warmup'] = args.anneal_warmup

    print(f"Saved {probe_name} probe for layer {layer} to {save_path}")

    del probe
    torch.cuda.empty_cache()

    return result


def train_all_probes_for_layer(layer: int, activation_store: ActivationStore,
                              probe_types: List[str], args,
                              model_name: str, hidden_size: int,
                              device: str, dtype: torch.dtype,
                              sweep_config: Optional[Dict] = None) -> Dict[str, Dict]:
    """Train all probe types for a single layer efficiently.

    If sweep_config is provided, trains multiple hyperparameter variants per probe type.
    sweep_config format:
    {
        "attention": [
            {"n_heads": 1, "temperature": 1.0, "name_suffix": "_h1"},
            {"n_heads": 4, "temperature": 1.0, "name_suffix": "_h4"}
        ],
        "sklearn_logistic": [
            {"C": 0.1, "name_suffix": "_C0.1"},
            {"C": 1.0, "name_suffix": "_C1.0"}
        ]
    }
    """
    hook_point = f"blocks.{layer}.hook_resid_pre"
    layer_results = {}

    for probe_type in probe_types:
        # Check if we have sweep configs for this probe type
        if sweep_config and probe_type in sweep_config:
            # Train multiple variants
            for hp_config in sweep_config[probe_type]:
                name_suffix = hp_config.pop("name_suffix", "")
                result = train_single_probe_with_hyperparams(
                    layer, activation_store, probe_type, hp_config,
                    args, model_name, hidden_size, device, dtype,
                    name_suffix=name_suffix
                )
                # Put name_suffix back for logging
                hp_config["name_suffix"] = name_suffix
                result_key = f"{probe_type}{name_suffix}"
                layer_results[result_key] = result
        else:
            # Train single probe with args.hyperparams (or None)
            result = train_single_probe_with_hyperparams(
                layer, activation_store, probe_type, args.hyperparams,
                args, model_name, hidden_size, device, dtype
            )
            layer_results[probe_type] = result

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
    parser.add_argument('--probe_types', nargs='+', required=True,
                       choices=['logistic', 'linear', 'pca', 'meandiff', 'kmeans', 'mlp', 'attention', 'sklearn_logistic'],
                       help='Probe types to train (REQUIRED - no default to prevent mistakes)')
    parser.add_argument('--hyperparams', type=json.loads, default=None,
                       help='JSON dict of hyperparameters')
    parser.add_argument('--sweep_config', type=str,
                       help='Path to hyperparameter sweep config file')
    parser.add_argument('--layers', nargs='+', default=['all'])
    parser.add_argument('--probe_save_dir', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, default='./cache/contrastive')
    parser.add_argument('--max_length', type=int, default=512, required=True,
                   help='Maximum token length for sequences')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience - epochs without improvement before stopping (default: 5)')
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
    parser.add_argument('--last_k_tokens', type=int, default=0,
                       help='Only use last K tokens of each span (Apollo style). 0=all tokens, 5=last 5')

    # ========== SPAN EXTRACTION MODE ==========
    # Controls how deceptive spans are identified in the text
    parser.add_argument('--use_llm_spans', type=str, default='auto',
                       choices=['auto', 'always', 'never'],
                       help='''Span extraction mode:
  - auto (default): Use LLM-tagged spans if present in dataset, else sentence parsing
  - always: Require LLM-tagged spans (error if not present)
  - never: Always use sentence-level parsing, ignore any LLM spans

LLM-tagged spans provide fine-grained phrase-level deception labels
(e.g., "red Honda Civic") from running scripts/run_tagging.py.
Sentence parsing splits by .!? and uses full sentences.''')

    # Loss mode options (see LOSS_DESIGN.md)
    parser.add_argument('--loss_mode', type=str, default='sample_mean',
                       choices=['sample_mean', 'sample_max', 'token_all', 'token_spans_only',
                                'span_mean', 'span_max', 'joint', 'joint_span_max', 'annealed',
                                'margin', 'ranking', 'contrastive_intra', 'soft_recall', 'topk_overlap'],
                       help='''Loss function mode:
  - sample_mean (default): BCE on mean of all token scores
  - sample_max: BCE on max of all token scores
  - token_all: Per-token BCE on ALL tokens (best for localization)
  - token_spans_only: Per-token BCE only on tokens in labeled spans
  - span_max: BCE on max score per span (good for localization)
  - joint: α * sample_loss + (1-α) * token_loss
  - joint_span_max: α * sample_mean + (1-α) * span_max (detection + localization)
  - annealed: Curriculum from sample_mean to token_all over epochs
  - margin: Explicit margin between in-span and out-span mean scores (LOCALIZATION)
  - ranking: Pairwise ranking loss for lie vs truth tokens (LOCALIZATION)
  - contrastive_intra: Within-sample contrastive (mean_lie > mean_truth) (LOCALIZATION)
  - soft_recall: Differentiable approximation of R@Oracle (LOCALIZATION)
  - topk_overlap: Adaptive - top k predictions should match k actual lie tokens (LOCALIZATION)''')
    parser.add_argument('--joint_alpha', type=float, default=0.5,
                       help='Weight for sample loss in joint mode (1.0 = sample only, 0.0 = token only)')
    parser.add_argument('--anneal_warmup', type=float, default=0.3,
                       help='Fraction of epochs for warmup in annealed mode (default: 0.3)')
    parser.add_argument('--margin', type=float, default=1.0,
                       help='Margin for margin/ranking losses (default: 1.0)')
    parser.add_argument('--temperature', type=float, default=0.5,
                       help='Temperature for soft_recall loss (default: 0.5)')
    parser.add_argument('--num_pairs', type=int, default=32,
                       help='Number of pairs to sample for ranking loss (default: 32)')

    parser.add_argument('--sparsity_penalty', type=float, default=0.0,
                       help='Penalty for high average activation (default: 0.0, off)')

    # ========== PARALLEL TRAINING ==========
    parser.add_argument('--parallel_probes', action='store_true',
                       help='''Enable parallel probe training across GPUs.
After activation collection completes for a layer, trains all probe
configs (e.g., h1_t0.25, h2_t0.25) in parallel using torch.multiprocessing.
Requires multiple GPUs for best performance (8xH200 recommended).''')
    parser.add_argument('--cross_layer_parallel', action='store_true',
                       help='''Enable cross-layer parallel training (more aggressive).
Trains ALL probes across ALL layers simultaneously after activation
collection. Example: 10 layers x 6 configs = 60 parallel tasks.
Best for clusters with many GPUs (8+). Implies --parallel_probes.''')
    parser.add_argument('--num_gpus', type=int, default=None,
                       help='Number of GPUs to use for parallel training (default: all available)')
    parser.add_argument('--max_parallel_workers', type=int, default=None,
                       help='Max parallel workers (default: num_gpus * 2, capped at 16/32)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Set spawn start method early if parallel training requested
    if args.parallel_probes or args.cross_layer_parallel:
        _ensure_spawn_start_method()

    # Cross-layer parallel implies parallel_probes
    if args.cross_layer_parallel:
        args.parallel_probes = True

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

    # Load sweep config if provided
    sweep_config = None
    if args.sweep_config:
        with open(args.sweep_config) as f:
            raw_config = json.load(f)
        # Handle both formats: {"probe_configs": {...}} or direct {...}
        sweep_config = raw_config.get("probe_configs", raw_config)
        # Remove non-probe keys if present
        sweep_config = {k: v for k, v in sweep_config.items()
                        if isinstance(v, list) and k not in ["layers", "training", "training_modes"]}
        print(f"📋 Sweep config loaded: {list(sweep_config.keys())}")
        # Count total probes to train
        total_variants = sum(len(v) for v in sweep_config.values())
        print(f"   Training {total_variants} probe variants per layer")

    # Load dataset using the new loader
    print(f"Loading contrastive dataset from {dataset_path}")
    dataset = load_contrastive_ntml_dataset(str(dataset_path),
                                            args.model_name,
                                            max_length=args.max_length,
                                            dishonest_mode=args.dishonest_mode,
                                            honest_mode=args.honest_mode,
                                            last_k_tokens=args.last_k_tokens,
                                            use_llm_spans=args.use_llm_spans)
    print(f"Dataset size: {len(dataset.examples)}")
    
    # Load model once
    print(f"Loading model {args.model_name}")
    model_dtype = get_model_dtype(args.model_name)
    print(f"🤖 Model dtype: {model_dtype}")
    
    try:
        model = HookedTransformer.from_pretrained_no_processing(
            args.model_name,
            device="cuda",
            n_devices=torch.cuda.device_count() or 1,
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
    
    # Train probes efficiently
    results = {}

    # Choose training strategy based on parallelization mode
    if args.cross_layer_parallel:
        # Most aggressive: train ALL probes across ALL layers in parallel
        print(f"\n{'='*60}")
        print("CROSS-LAYER PARALLEL TRAINING ENABLED")
        print(f"  Layers: {len(layers)}")
        print(f"  GPUs: {args.num_gpus or torch.cuda.device_count()}")
        print(f"  Max workers: {args.max_parallel_workers or 'auto'}")
        print(f"{'='*60}\n")

        results = train_all_layers_parallel(
            layers=layers,
            activation_stores=activation_stores,
            probe_types=args.probe_types,
            args=args,
            model_name=args.model_name,
            hidden_size=hidden_size,
            dtype=model_dtype,
            sweep_config=sweep_config,
            num_gpus=args.num_gpus,
            max_workers=args.max_parallel_workers,
        )

    elif args.parallel_probes:
        # Per-layer parallel: train probes in parallel within each layer
        print(f"\n{'='*60}")
        print("PER-LAYER PARALLEL TRAINING ENABLED")
        print(f"  GPUs: {args.num_gpus or torch.cuda.device_count()}")
        print(f"  Max workers: {args.max_parallel_workers or 'auto'}")
        print(f"{'='*60}\n")

        for layer in tqdm(layers, desc="Training layers"):
            hook_point = f"blocks.{layer}.hook_resid_pre"
            activation_store = activation_stores[hook_point]

            layer_results = train_probes_parallel(
                layer=layer,
                activation_store=activation_store,
                probe_types=args.probe_types,
                args=args,
                model_name=args.model_name,
                hidden_size=hidden_size,
                dtype=model_dtype,
                sweep_config=sweep_config,
                num_gpus=args.num_gpus,
                max_workers=args.max_parallel_workers,
            )
            results[layer] = layer_results

            # Clear activation store to save memory if processing many layers
            if len(layers) > 16:
                del activation_stores[hook_point]
                torch.cuda.empty_cache()

    else:
        # Sequential: train one probe at a time (original behavior)
        for layer in tqdm(layers, desc="Training layers"):
            hook_point = f"blocks.{layer}.hook_resid_pre"
            activation_store = activation_stores[hook_point]

            layer_results = train_all_probes_for_layer(
                layer, activation_store, args.probe_types, args,
                args.model_name, hidden_size, args.device, model_dtype,
                sweep_config=sweep_config
            )
            results[layer] = layer_results

            # Clear activation store to save memory if processing many layers
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
            if 'error' in probe_results:
                print(f"  {probe_type}: FAILED - {probe_results['error'][:100]}...")
            else:
                final_loss = probe_results.get('final_train_loss', float('nan'))
                print(f"  {probe_type}: Final loss = {final_loss:.6f}")


if __name__ == "__main__":
    main()