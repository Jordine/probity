# probity/training/parallel.py
"""Parallel probe training utilities.

This module provides utilities for training multiple probes in parallel
across GPUs using torch.multiprocessing with spawn start method.

Key considerations:
- CUDA tensors cannot be pickled, so activations must be on CPU before forking
- Uses 'spawn' start method for CUDA compatibility
- Each worker gets its own GPU assignment
- Uses shared memory tensors to minimize memory overhead

Example usage:
    # In your training script:
    from probity.training.parallel import train_probes_parallel, _ensure_spawn_start_method

    # Call once at start of main()
    _ensure_spawn_start_method()

    # Then for each layer:
    results = train_probes_parallel(
        layer=5,
        activation_store=activation_stores[f"blocks.5.hook_resid_pre"],
        probe_types=["logistic", "sklearn_logistic"],
        args=args,
        model_name="meta-llama/Llama-3.1-70B",
        hidden_size=8192,
        dtype=torch.bfloat16,
        sweep_config=sweep_config,  # Optional
        num_gpus=8,  # Optional, defaults to all available
    )
"""

import os
import sys
import torch
import torch.multiprocessing as mp
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import traceback


def _ensure_spawn_start_method():
    """Ensure spawn start method is set (call once from main process).

    This must be called before any CUDA operations or multiprocessing.
    Safe to call multiple times - subsequent calls are no-ops.
    """
    try:
        current = mp.get_start_method(allow_none=True)
        if current is None:
            mp.set_start_method('spawn')
        elif current != 'spawn':
            # Already set to something else, try to force
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                print(f"Warning: Could not set spawn start method (current: {current})")
    except RuntimeError:
        # Already set - that's fine
        pass


@dataclass
class ProbeTrainTask:
    """A single probe training task to be executed by a worker."""
    layer: int
    probe_type: str
    hyperparams: Optional[Dict]
    name_suffix: str
    task_id: int  # For tracking


@dataclass
class ProbeTrainResult:
    """Result from a probe training task."""
    task_id: int
    layer: int
    probe_type: str
    name_suffix: str
    success: bool
    result: Optional[Dict]  # Training results if success
    error: Optional[str]  # Error message if failed


def _worker_train_probe(
    task: ProbeTrainTask,
    # CPU tensors for data (will be moved to GPU in worker)
    raw_activations_cpu: torch.Tensor,
    labels_cpu: torch.Tensor,
    activation_store_metadata: Dict,
    # Training config (all picklable)
    args_dict: Dict,
    model_name: str,
    hidden_size: int,
    dtype_str: str,
    gpu_id: int,
    result_queue: mp.Queue,
) -> None:
    """Worker function that trains a single probe.

    This runs in a separate process with its own CUDA context.
    """
    try:
        # Import here to avoid issues with multiprocessing
        from probity.collection.activation_store import ActivationStore
        from probity.datasets.tokenized import TokenizedProbingDataset
        from probity.training.configs import (
            get_probe_config, get_probe_class,
            get_trainer_config, get_trainer_class
        )

        # Set device for this worker
        device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(gpu_id)

        # Convert dtype string back to torch dtype
        dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float32

        # Reconstruct minimal activation store from CPU tensors
        # Move to GPU for training
        raw_activations = raw_activations_cpu.to(device)
        labels = labels_cpu.to(device)

        # Load the dataset from the saved path
        dataset = TokenizedProbingDataset.load(activation_store_metadata['dataset_path'])

        # Recreate activation store
        activation_store = ActivationStore(
            raw_activations=raw_activations,
            hook_point=activation_store_metadata['hook_point'],
            labels=labels,
            label_texts=activation_store_metadata['label_texts'],
            example_indices=torch.tensor(activation_store_metadata['example_indices']),
            sequence_lengths=torch.tensor(activation_store_metadata['sequence_lengths']),
            hidden_size=activation_store_metadata['hidden_size'],
            dataset=dataset,
        )

        # Recreate args namespace from dict
        from argparse import Namespace
        args = Namespace(**args_dict)

        # Now train the probe
        layer = task.layer
        probe_type = task.probe_type
        hyperparams = task.hyperparams
        name_suffix = task.name_suffix

        hook_point = f"blocks.{layer}.hook_resid_pre"

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

        # Apply epoch/patience settings
        trainer_config.num_epochs = args.num_epochs
        trainer_config.patience = args.patience

        # Initialize probe and trainer
        probe = probe_cls(probe_config).to(device)
        trainer = trainer_cls(trainer_config)

        # Prepare data
        token_level_modes = ['token_all', 'token_spans_only', 'joint', 'annealed', 'span_max', 'span_mean']
        needs_spans = use_max_aggr or args.loss_mode in token_level_modes

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

        # Save probe
        probe_name = f"{probe_type}{name_suffix}"
        save_dir = Path(args.probe_save_dir) / probe_name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"layer_{layer}_probe.pt"
        probe.save(str(save_path))

        result = {
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1] if 'val_loss' in history and history['val_loss'] else None,
            'save_path': str(save_path),
            'max_aggregation': use_max_aggr,
            'hyperparams': hyperparams,
        }

        if hasattr(probe, 'config') and hasattr(probe.config, 'optimal_thresholds'):
            thresholds = probe.config.optimal_thresholds
            if 'train_auroc_score' in thresholds:
                result['train_auroc'] = thresholds['train_auroc_score']

        if use_max_aggr and 'omega' in history and history['omega']:
            result['final_omega'] = history['omega'][-1]
            result['anneal_warmup'] = args.anneal_warmup

        # Clean up GPU memory
        del probe
        torch.cuda.empty_cache()

        # Send success result
        result_queue.put(ProbeTrainResult(
            task_id=task.task_id,
            layer=task.layer,
            probe_type=task.probe_type,
            name_suffix=task.name_suffix,
            success=True,
            result=result,
            error=None,
        ))

    except Exception as e:
        # Send error result
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        result_queue.put(ProbeTrainResult(
            task_id=task.task_id,
            layer=task.layer,
            probe_type=task.probe_type,
            name_suffix=task.name_suffix,
            success=False,
            result=None,
            error=error_msg,
        ))


def _prepare_activation_store_for_workers(
    activation_store,
    cache_dir: str,
    layer: int,
    use_shared_memory: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Prepare activation store data for passing to workers.

    Returns CPU tensors (optionally in shared memory) and metadata that can be pickled.

    Args:
        activation_store: The activation store to prepare
        cache_dir: Directory for temporary files
        layer: Layer number (for temp file naming)
        use_shared_memory: If True, use shared memory tensors to avoid copies

    Returns:
        Tuple of (raw_activations_cpu, labels_cpu, metadata)
    """
    # Save dataset to a temporary location that workers can load
    dataset_path = Path(cache_dir) / f"parallel_worker_dataset_L{layer}"
    dataset_path.mkdir(parents=True, exist_ok=True)
    activation_store.dataset.save(str(dataset_path))

    # Move tensors to CPU for pickling
    raw_activations_cpu = activation_store.raw_activations.cpu()
    labels_cpu = activation_store.labels.cpu()

    # Use shared memory to avoid copies when forking
    if use_shared_memory:
        raw_activations_cpu = raw_activations_cpu.share_memory_()
        labels_cpu = labels_cpu.share_memory_()

    # Extract metadata
    metadata = {
        'hook_point': activation_store.hook_point,
        'label_texts': activation_store.label_texts,
        'example_indices': activation_store.example_indices.tolist(),
        'sequence_lengths': activation_store.sequence_lengths.tolist(),
        'hidden_size': activation_store.hidden_size,
        'dataset_path': str(dataset_path),
    }

    return raw_activations_cpu, labels_cpu, metadata


def train_probes_parallel(
    layer: int,
    activation_store,
    probe_types: List[str],
    args,
    model_name: str,
    hidden_size: int,
    dtype: torch.dtype,
    sweep_config: Optional[Dict] = None,
    num_gpus: Optional[int] = None,
    max_workers: Optional[int] = None,
) -> Dict[str, Dict]:
    """Train all probe types for a layer in parallel across GPUs.

    Args:
        layer: Layer number
        activation_store: ActivationStore with collected activations
        probe_types: List of probe types to train
        args: Training arguments namespace
        model_name: Model name string
        hidden_size: Model hidden size
        dtype: Model dtype
        sweep_config: Optional hyperparameter sweep configuration
        num_gpus: Number of GPUs to use (default: all available)
        max_workers: Max parallel workers (default: num_gpus * 2)

    Returns:
        Dictionary mapping probe names to their training results

    Note:
        This uses torch.multiprocessing with spawn start method for CUDA
        compatibility. Activations are moved to CPU and optionally shared
        via shared memory to reduce memory overhead.
    """
    import time
    start_time = time.time()

    # Detect available GPUs
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available for parallel training")

    if max_workers is None:
        max_workers = min(num_gpus * 2, 16)  # Cap at 16 workers

    print(f"Parallel probe training: {num_gpus} GPUs, up to {max_workers} workers")

    # Build list of all training tasks
    tasks: List[ProbeTrainTask] = []
    task_id = 0

    for probe_type in probe_types:
        if sweep_config and probe_type in sweep_config:
            for hp_config in sweep_config[probe_type]:
                hp_copy = hp_config.copy()
                name_suffix = hp_copy.pop("name_suffix", "")
                tasks.append(ProbeTrainTask(
                    layer=layer,
                    probe_type=probe_type,
                    hyperparams=hp_copy,
                    name_suffix=name_suffix,
                    task_id=task_id,
                ))
                task_id += 1
        else:
            tasks.append(ProbeTrainTask(
                layer=layer,
                probe_type=probe_type,
                hyperparams=getattr(args, 'hyperparams', None),
                name_suffix="",
                task_id=task_id,
            ))
            task_id += 1

    if not tasks:
        return {}

    print(f"Training {len(tasks)} probes for layer {layer} in parallel")

    # Prepare activation store for workers (CPU tensors + metadata)
    raw_activations_cpu, labels_cpu, metadata = _prepare_activation_store_for_workers(
        activation_store, args.cache_dir, layer
    )

    # Convert args to dict for pickling
    args_dict = vars(args).copy()

    # Convert dtype to string for pickling
    dtype_str = "bfloat16" if dtype == torch.bfloat16 else "float32"

    # Create result queue
    result_queue = mp.Queue()

    # Track active processes
    active_processes: List[mp.Process] = []
    task_idx = 0
    results: Dict[str, Dict] = {}
    completed = 0

    # Process tasks with worker pool
    while completed < len(tasks):
        # Start new workers up to max_workers
        while len(active_processes) < max_workers and task_idx < len(tasks):
            task = tasks[task_idx]
            gpu_id = task_idx % num_gpus

            p = mp.Process(
                target=_worker_train_probe,
                args=(
                    task,
                    raw_activations_cpu,
                    labels_cpu,
                    metadata,
                    args_dict,
                    model_name,
                    hidden_size,
                    dtype_str,
                    gpu_id,
                    result_queue,
                ),
            )
            p.start()
            active_processes.append(p)
            task_idx += 1
            print(f"  Started task {task.task_id}: {task.probe_type}{task.name_suffix} on GPU {gpu_id}")

        # Wait for a result
        try:
            result = result_queue.get(timeout=600)  # 10 minute timeout per probe
            completed += 1

            # Process result
            result_key = f"{result.probe_type}{result.name_suffix}"
            if result.success:
                results[result_key] = result.result
                print(f"  Completed {result_key}: loss={result.result['final_train_loss']:.6f}")
            else:
                print(f"  FAILED {result_key}: {result.error[:200]}...")
                results[result_key] = {'error': result.error}

            # Clean up finished processes
            active_processes = [p for p in active_processes if p.is_alive()]

        except Exception as e:
            print(f"  Error waiting for result: {e}")
            # Check for dead processes
            for p in active_processes:
                if not p.is_alive() and p.exitcode != 0:
                    print(f"  Worker died with exit code {p.exitcode}")
            active_processes = [p for p in active_processes if p.is_alive()]

    # Clean up any remaining processes
    for p in active_processes:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    # Clean up temporary dataset
    import shutil
    temp_dataset_path = Path(args.cache_dir) / f"parallel_worker_dataset_L{layer}"
    if temp_dataset_path.exists():
        shutil.rmtree(temp_dataset_path, ignore_errors=True)

    elapsed = time.time() - start_time
    successful = sum(1 for r in results.values() if 'error' not in r)
    print(f"Layer {layer} parallel training complete: {successful}/{len(tasks)} succeeded in {elapsed:.1f}s")

    return results


def train_all_layers_parallel(
    layers: List[int],
    activation_stores: Dict[str, Any],  # hook_point -> ActivationStore
    probe_types: List[str],
    args,
    model_name: str,
    hidden_size: int,
    dtype: torch.dtype,
    sweep_config: Optional[Dict] = None,
    num_gpus: Optional[int] = None,
    max_workers: Optional[int] = None,
) -> Dict[int, Dict[str, Dict]]:
    """Train ALL probes across ALL layers in parallel.

    This is more efficient than per-layer parallelization because it can
    maximize GPU utilization when you have many layers and probe configs.

    Example: 10 layers x 6 probe configs = 60 parallel tasks distributed
    across 8 GPUs with round-robin scheduling.

    Args:
        layers: List of layer numbers to train
        activation_stores: Dict mapping hook_point to ActivationStore
        probe_types: List of probe types to train
        args: Training arguments namespace
        model_name: Model name string
        hidden_size: Model hidden size
        dtype: Model dtype
        sweep_config: Optional hyperparameter sweep configuration
        num_gpus: Number of GPUs to use (default: all available)
        max_workers: Max parallel workers (default: num_gpus * 2)

    Returns:
        Nested dict: {layer: {probe_name: result_dict}}
    """
    import time
    import shutil
    start_time = time.time()

    # Detect available GPUs
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available for parallel training")

    if max_workers is None:
        max_workers = min(num_gpus * 2, 32)  # Higher cap for cross-layer

    # Build ALL tasks across all layers
    all_tasks: List[Tuple[ProbeTrainTask, int]] = []  # (task, layer)
    task_id = 0

    for layer in layers:
        for probe_type in probe_types:
            if sweep_config and probe_type in sweep_config:
                for hp_config in sweep_config[probe_type]:
                    hp_copy = hp_config.copy()
                    name_suffix = hp_copy.pop("name_suffix", "")
                    all_tasks.append((ProbeTrainTask(
                        layer=layer,
                        probe_type=probe_type,
                        hyperparams=hp_copy,
                        name_suffix=name_suffix,
                        task_id=task_id,
                    ), layer))
                    task_id += 1
            else:
                all_tasks.append((ProbeTrainTask(
                    layer=layer,
                    probe_type=probe_type,
                    hyperparams=getattr(args, 'hyperparams', None),
                    name_suffix="",
                    task_id=task_id,
                ), layer))
                task_id += 1

    total_tasks = len(all_tasks)
    print(f"\n{'='*60}")
    print(f"CROSS-LAYER PARALLEL TRAINING")
    print(f"  Layers: {len(layers)} ({min(layers)}-{max(layers)})")
    print(f"  Probe configs: {total_tasks // len(layers)} per layer")
    print(f"  Total tasks: {total_tasks}")
    print(f"  GPUs: {num_gpus}, Max workers: {max_workers}")
    print(f"{'='*60}\n")

    # Prepare ALL activation stores for workers upfront
    # This converts them to CPU shared memory tensors
    prepared_stores: Dict[int, Tuple[torch.Tensor, torch.Tensor, Dict]] = {}
    print("Preparing activation stores for parallel workers...")
    for layer in layers:
        hook_point = f"blocks.{layer}.hook_resid_pre"
        activation_store = activation_stores[hook_point]
        prepared_stores[layer] = _prepare_activation_store_for_workers(
            activation_store, args.cache_dir, layer
        )
    print(f"Prepared {len(prepared_stores)} activation stores")

    # Convert args to dict for pickling
    args_dict = vars(args).copy()
    dtype_str = "bfloat16" if dtype == torch.bfloat16 else "float32"

    # Create result queue
    result_queue = mp.Queue()

    # Track active processes
    active_processes: List[mp.Process] = []
    task_idx = 0
    results: Dict[int, Dict[str, Dict]] = {layer: {} for layer in layers}
    completed = 0

    # Process all tasks with worker pool
    while completed < total_tasks:
        # Start new workers up to max_workers
        while len(active_processes) < max_workers and task_idx < total_tasks:
            task, layer = all_tasks[task_idx]
            gpu_id = task_idx % num_gpus

            # Get the prepared activation data for this layer
            raw_activations_cpu, labels_cpu, metadata = prepared_stores[layer]

            p = mp.Process(
                target=_worker_train_probe,
                args=(
                    task,
                    raw_activations_cpu,
                    labels_cpu,
                    metadata,
                    args_dict,
                    model_name,
                    hidden_size,
                    dtype_str,
                    gpu_id,
                    result_queue,
                ),
            )
            p.start()
            active_processes.append(p)
            task_idx += 1
            print(f"  [{task_idx}/{total_tasks}] L{layer} {task.probe_type}{task.name_suffix} -> GPU {gpu_id}")

        # Wait for results
        try:
            result = result_queue.get(timeout=600)  # 10 minute timeout
            completed += 1

            # Process result
            result_key = f"{result.probe_type}{result.name_suffix}"
            if result.success:
                results[result.layer][result_key] = result.result
                loss = result.result['final_train_loss']
                print(f"  [{completed}/{total_tasks}] Done L{result.layer} {result_key}: loss={loss:.6f}")
            else:
                print(f"  [{completed}/{total_tasks}] FAILED L{result.layer} {result_key}: {result.error[:100]}...")
                results[result.layer][result_key] = {'error': result.error}

            # Clean up finished processes
            active_processes = [p for p in active_processes if p.is_alive()]

        except Exception as e:
            print(f"  Error waiting for result: {e}")
            for p in active_processes:
                if not p.is_alive() and p.exitcode != 0:
                    print(f"  Worker died with exit code {p.exitcode}")
            active_processes = [p for p in active_processes if p.is_alive()]

    # Clean up processes
    for p in active_processes:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    # Clean up temporary datasets
    for layer in layers:
        temp_path = Path(args.cache_dir) / f"parallel_worker_dataset_L{layer}"
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)

    # Summary
    elapsed = time.time() - start_time
    total_successful = sum(
        sum(1 for r in layer_results.values() if 'error' not in r)
        for layer_results in results.values()
    )
    print(f"\n{'='*60}")
    print(f"PARALLEL TRAINING COMPLETE")
    print(f"  {total_successful}/{total_tasks} probes trained successfully")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Avg per probe: {elapsed/total_tasks:.1f}s")
    print(f"{'='*60}\n")

    return results


