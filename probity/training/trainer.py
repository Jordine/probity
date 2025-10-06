from abc import ABC
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from typing import Optional, Dict, List, Tuple, Literal
from tqdm.auto import tqdm
import math

from probity.collection.activation_store import ActivationStore
from probity.probes import (
    BaseProbe,
    DirectionalProbe,
    MultiClassLogisticProbe,
    LogisticProbe,
    LinearProbe,
)


@dataclass
class BaseTrainerConfig:
    """Enhanced base configuration shared by all trainers."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: Optional[str] = None
    batch_size: int = 32
    learning_rate: float = 1e-3
    end_learning_rate: float = 1e-5  # For LR scheduling
    weight_decay: float = 0.01
    num_epochs: int = 10
    show_progress: bool = True
    optimizer_type: Literal["Adam", "SGD", "AdamW"] = "Adam"
    handle_class_imbalance: bool = True
    standardize_activations: bool = False  # Option to standardize *during* training


class BaseProbeTrainer(ABC):
    """Enhanced abstract base class for all probe trainers. Handles standardization during training."""
    def __init__(self, config: BaseTrainerConfig):
        self.config = config
        self.feature_mean: Optional[torch.Tensor] = None
        self.feature_std: Optional[torch.Tensor] = None

    def _get_lr_scheduler(
        self, optimizer: optim.Optimizer, start_lr: float, end_lr: float, num_steps: int
    ) -> optim.lr_scheduler.LRScheduler:
        """Create exponential learning rate scheduler."""
        if start_lr <= 0 or end_lr <= 0:
            return optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

        if num_steps <= 0:
            return optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

        if start_lr == end_lr:
            gamma = 1.0
        else:
            ratio = end_lr / start_lr
            if ratio <= 0:
                return optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
            gamma = math.exp(math.log(ratio) / num_steps)

        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    def _calculate_pos_weights(self, y: torch.Tensor) -> torch.Tensor:
        """Calculate positive weights for handling class imbalance."""
        if y.dim() == 1:
            y = y.unsqueeze(1)

        num_pos = y.sum(dim=0)
        num_neg = len(y) - num_pos
        weights = num_neg / (num_pos + 1e-8)
        return weights

    def _calculate_class_weights(
        self, y: torch.Tensor, num_classes: int
    ) -> Optional[torch.Tensor]:
        """Calculate class weights for multi-class CrossEntropyLoss."""
        if y.dim() == 2 and y.shape[1] == 1:
            y = y.squeeze(1)
        if y.dim() != 1:
            return None

        if y.dtype != torch.long:
            return None

        counts = torch.bincount(y, minlength=num_classes)
        total_samples = counts.sum()
        if total_samples == 0:
            return None

        weights = total_samples / (num_classes * (counts + 1e-8))
        return weights

    def _create_optimizer(self, model: torch.nn.Module) -> optim.Optimizer:
        """Create optimizer based on config."""
        optimizer_class = getattr(optim, self.config.optimizer_type, None)
        if optimizer_class is None:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")

        params_to_optimize = filter(lambda p: p.requires_grad, model.parameters())

        if self.config.optimizer_type in ["Adam", "AdamW"]:
            optimizer = optimizer_class(
                params_to_optimize,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer_type == "SGD":
            optimizer = optimizer_class(
                params_to_optimize,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.config.optimizer_type}")

        return optimizer

    def prepare_data(
        self,
        activation_store: ActivationStore,
        position_key: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare data for training, optionally applying standardization."""
        X_expected, y_expected = activation_store.get_probe_data(position_key)
        X_train = X_expected

        print(f"DEBUG [Training Data]:")
        print(f"  X shape: {X_expected.shape}")
        print(f"  y shape: {y_expected.shape}")
        print(f"  Number of examples in dataset: {len(activation_store.dataset.examples)}")
        print(f"  Ratio: {X_expected.shape[0] / len(activation_store.dataset.examples):.2f} activations per example")
        
        # Check a specific example
        if len(activation_store.dataset.examples) > 0:
            ex = activation_store.dataset.examples[0]
            if hasattr(ex, 'token_positions') and ex.token_positions:
                pos = ex.token_positions.positions.get(position_key)
                print(f"  First example position type: {type(pos)}")
                if isinstance(pos, list):
                    print(f"  First example has {len(pos)} tokens in statement")
            

        if self.config.standardize_activations:
            if self.feature_mean is None or self.feature_std is None:
                self.feature_mean = X_expected.mean(dim=0, keepdim=True)
                self.feature_std = X_expected.std(dim=0, keepdim=True) + 1e-8

                if self.config.device:
                    target_device = torch.device(self.config.device)
                    self.feature_mean = self.feature_mean.to(target_device)
                    self.feature_std = self.feature_std.to(target_device)

            if self.feature_mean is not None and self.feature_std is not None:
                if hasattr(self.feature_mean, "device"):
                    X_orig_dev = X_expected.to(self.feature_mean.device)
                    X_train = (X_orig_dev - self.feature_mean) / self.feature_std
                else:
                    X_train = (X_expected - self.feature_mean) / self.feature_std

        return X_train, y_expected, X_expected


@dataclass
class SupervisedTrainerConfig(BaseTrainerConfig):
    """Enhanced config for supervised training methods."""
    train_ratio: float = 0.8
    patience: int = 5
    min_delta: float = 1e-4


class SupervisedProbeTrainer(BaseProbeTrainer):
    """Enhanced trainer for supervised probes with progress tracking and LR scheduling."""
    def __init__(self, config: SupervisedTrainerConfig):
        super().__init__(config)
        if not isinstance(config, SupervisedTrainerConfig):
            raise TypeError("SupervisedProbeTrainer requires a SupervisedTrainerConfig")
        self.config: SupervisedTrainerConfig = config

    def prepare_supervised_data(self, activation_store: ActivationStore, position_key: str) -> Tuple[DataLoader, DataLoader]:
        """Prepare train/val splits with DataLoader creation."""
        X_train_all, y_all, X_orig_all = self.prepare_data(activation_store, position_key)
        
        n_total = len(X_orig_all)
        
        # Check if we have contrastive pairs that need to stay together
        has_contrastive_pairs = any(
            hasattr(ex, 'group_id') and ex.group_id and "contrastive" in str(ex.group_id)
            for ex in activation_store.dataset.examples[:min(100, len(activation_store.dataset.examples))]
        )
        
        if has_contrastive_pairs:
            # Group-aware split for contrastive pairs
            print("Using group-aware train/val split for contrastive pairs")
            
            # Build mapping of indices to group IDs
            # Note: After converting to tokens, we may have multiple indices per original example
            # We need to track which activation indices belong to which group
            
            group_to_indices = {}
            for i, dataset_ex_idx in enumerate(activation_store.example_indices):
                dataset_ex = activation_store.dataset.examples[dataset_ex_idx]
                group_id = getattr(dataset_ex, 'group_id', f'default_{dataset_ex_idx}')
                if group_id not in group_to_indices:
                    group_to_indices[group_id] = []
                group_to_indices[group_id].append(i)
            
            # Shuffle groups
            import random
            group_ids = list(group_to_indices.keys())
            random.shuffle(group_ids)
            
            # Split groups
            n_train_groups = int(len(group_ids) * self.config.train_ratio)
            train_groups = group_ids[:n_train_groups]
            
            # Get indices based on groups
            train_indices = []
            val_indices = []
            for group_id in group_ids:
                indices = group_to_indices[group_id]
                if group_id in train_groups:
                    train_indices.extend(indices)
                else:
                    val_indices.extend(indices)
            
            train_indices = torch.tensor(train_indices)
            val_indices = torch.tensor(val_indices)
            
        else:
            # Standard random split
            n_train = int(n_total * self.config.train_ratio)
            if n_train == n_total:
                n_train = max(0, n_total - 1)
            
            indices = torch.randperm(n_total)
            train_indices = indices[:n_train]
            val_indices = indices[n_train:]

        X_train_split, X_val_split = X_train_all[train_indices], X_train_all[val_indices]
        X_orig_train_split, X_orig_val_split = X_orig_all[train_indices], X_orig_all[val_indices]
        y_train_split, y_val_split = y_all[train_indices], y_all[val_indices]

        y_train_split = y_train_split if y_train_split.dim() > 1 else y_train_split.unsqueeze(1)
        y_val_split = y_val_split if y_val_split.dim() > 1 else y_val_split.unsqueeze(1)

        train_dataset = TensorDataset(X_train_split, y_train_split, X_orig_train_split)
        val_dataset = TensorDataset(X_val_split, y_val_split, X_orig_val_split)

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)

        return train_loader, val_loader

    def train_epoch(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        epoch: int,
        num_epochs: int,
        is_multi_class: bool = False,
    ) -> float:
        """Run one epoch of training with progress tracking."""
        model.train()
        total_loss = 0

        batch_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            disable=not self.config.show_progress,
            leave=False,
        )

        for _, (batch_x_train, batch_y, _) in enumerate(batch_pbar):
            optimizer.zero_grad()
            batch_x_train = batch_x_train.to(self.config.device)
            batch_y = batch_y.to(self.config.device)

            model_dtype = next(model.parameters()).dtype
            if is_multi_class:
                if batch_y.dim() == 2 and batch_y.shape[1] == 1:
                    batch_y = batch_y.squeeze(1)
                batch_y = batch_y.long()
            else:
                batch_y = batch_y.to(dtype=model_dtype)

            outputs = model(batch_x_train)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

        return total_loss / len(train_loader)

    def train(
        self,
        model: BaseProbe,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:
        """Train model, potentially unscale direction if standardization was used."""
        if not isinstance(model, BaseProbe):
            raise TypeError("SupervisedProbeTrainer expects model to be an instance of BaseProbe")

        target_device = torch.device(self.config.device)
        model.to(target_device)

        is_multi_class = isinstance(model, MultiClassLogisticProbe)
        optimizer = self._create_optimizer(model)
        scheduler = self._get_lr_scheduler(
            optimizer,
            self.config.learning_rate,
            self.config.end_learning_rate,
            self.config.num_epochs,
        )

        loss_fn: nn.Module
        all_train_y: Optional[torch.Tensor] = None
        if self.config.handle_class_imbalance:
            all_train_y = torch.cat([y for _, y, _ in train_loader])

        model_dtype = next(model.parameters()).dtype

        if isinstance(model, MultiClassLogisticProbe):
            weights_arg: Optional[torch.Tensor] = None
            if self.config.handle_class_imbalance and all_train_y is not None:
                num_classes = model.config.output_size
                class_weights = self._calculate_class_weights(all_train_y, num_classes)
                if class_weights is not None:
                    weights_arg = class_weights.to(target_device).to(dtype=model_dtype)
            loss_fn = model.get_loss_fn(class_weights=weights_arg)

        elif isinstance(model, LogisticProbe):
            weights_arg: Optional[torch.Tensor] = None
            if self.config.handle_class_imbalance and all_train_y is not None:
                pos_weight = self._calculate_pos_weights(all_train_y)
                if pos_weight is not None:
                    weights_arg = pos_weight.to(target_device).to(dtype=model_dtype)
            loss_fn = model.get_loss_fn(pos_weight=weights_arg)

        elif isinstance(model, LinearProbe):
            loss_fn = model.get_loss_fn()
            loss_fn = loss_fn.to(dtype=model_dtype)
            if self.config.handle_class_imbalance:
                pass
        else:
            weights_arg: Optional[torch.Tensor] = None
            if self.config.handle_class_imbalance and all_train_y is not None:
                pos_weight = self._calculate_pos_weights(all_train_y)
                if pos_weight is not None:
                    weights_arg = pos_weight.to(target_device).to(dtype=model_dtype)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=weights_arg).to(dtype=model_dtype)

        loss_fn = loss_fn.to(device=target_device)

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": [],
        }

        epoch_pbar = tqdm(
            range(self.config.num_epochs),
            desc="Training",
            disable=not self.config.show_progress,
        )

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in epoch_pbar:
            train_loss = self.train_epoch(
                model,
                train_loader,
                optimizer,
                loss_fn,
                epoch,
                self.config.num_epochs,
                is_multi_class=is_multi_class,
            )
            history["train_loss"].append(train_loss)

            if val_loader is not None:
                val_loss = self.validate(model, val_loader, loss_fn, is_multi_class=is_multi_class)
                history["val_loss"].append(val_loss)

                if val_loss < best_val_loss - self.config.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.config.patience:
                    break

                epoch_pbar.set_postfix(
                    {
                        "Train Loss": f"{train_loss:.6f}",
                        "Val Loss": f"{val_loss:.6f}",
                        "LR": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                )
            else:
                epoch_pbar.set_postfix(
                    {
                        "Train Loss": f"{train_loss:.6f}",
                        "LR": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                )

            history["learning_rate"].append(scheduler.get_last_lr()[0])
            scheduler.step()



        # Calculate optimal threshold on training data
        if hasattr(model, 'config') and hasattr(model.config, 'optimal_thresholds'):
            print("Calculating optimal thresholds...")

            # Collect predictions on full training set
            model.eval()
            all_scores = []
            all_labels = []
            
            with torch.no_grad():
                for batch_x, batch_y, _ in train_loader:
                    batch_x = batch_x.to(self.config.device)
                    batch_y = batch_y.to(self.config.device)
                    
                    outputs = model(batch_x)
                    
                    # Apply sigmoid for logistic probes
                    if model.__class__.__name__ == 'LogisticProbe':
                        outputs = torch.sigmoid(outputs)
                    
                    all_scores.extend(outputs.cpu().squeeze().tolist())
                    all_labels.extend(batch_y.cpu().squeeze().tolist())
            
            # Calculate optimal threshold
            from probity.utils.threshold_optimization import find_optimal_threshold_auroc
            import numpy as np
            
            optimal_threshold = find_optimal_threshold_auroc(
                np.array(all_scores), 
                np.array(all_labels)
            )
            
            model.config.optimal_thresholds['train_auroc'] = optimal_threshold
            print(f"Optimal threshold (train AUROC): {optimal_threshold:.4f}")


        if self.config.standardize_activations and self.feature_std is not None:
            with torch.no_grad():
                learned_direction = model._get_raw_direction_representation()
                std_dev = self.feature_std.squeeze().to(learned_direction.device)

                if (
                    learned_direction.dim() == 2
                    and learned_direction.shape[0] == 1
                    and std_dev.dim() == 1
                ):
                    unscaled_direction = learned_direction / std_dev.unsqueeze(0)
                elif learned_direction.shape == std_dev.shape:
                    unscaled_direction = learned_direction / std_dev
                elif (
                    learned_direction.dim() == 1
                    and std_dev.dim() == 1
                    and learned_direction.shape[0] == std_dev.shape[0]
                ):
                    unscaled_direction = learned_direction / std_dev
                else:
                    unscaled_direction = learned_direction

                model._set_raw_direction_representation(unscaled_direction)

        return history

    def validate(
        self,
        model: torch.nn.Module,
        val_loader: DataLoader,
        loss_fn: torch.nn.Module,
        is_multi_class: bool = False,
    ) -> float:
        """Run validation with progress tracking."""
        model.eval()
        total_loss = 0
        model_dtype = next(model.parameters()).dtype

        with torch.no_grad():
            for _, (_, batch_y, batch_x_orig) in enumerate(val_loader):
                batch_x_orig = batch_x_orig.to(self.config.device)
                batch_y = batch_y.to(self.config.device)

                if is_multi_class:
                    if batch_y.dim() == 2 and batch_y.shape[1] == 1:
                        batch_y = batch_y.squeeze(1)
                    batch_y = batch_y.long()
                else:
                    batch_y = batch_y.to(dtype=model_dtype)

                outputs = model(batch_x_orig)
                loss = loss_fn(outputs, batch_y)
                total_loss += loss.item()

        return total_loss / len(val_loader)


@dataclass
class DirectionalTrainerConfig(BaseTrainerConfig):
    """Configuration for training direction-finding probes."""
    pass


class DirectionalProbeTrainer(BaseProbeTrainer):
    """Trainer for probes that find directions through direct computation."""
    def __init__(self, config: DirectionalTrainerConfig):
        super().__init__(config)
        if not isinstance(config, DirectionalTrainerConfig):
            raise TypeError("DirectionalProbeTrainer requires a DirectionalTrainerConfig")
        self.config: DirectionalTrainerConfig = config

    def prepare_supervised_data(
        self,
        activation_store: ActivationStore,
        position_key: str,
    ) -> Tuple[DataLoader, DataLoader]:
        """Prepare data for directional probe computation."""
        X_train_all, y_all, X_orig_all = self.prepare_data(activation_store, position_key)

        y_all = y_all if y_all.dim() > 1 else y_all.unsqueeze(1)
        dataset = TensorDataset(X_train_all, y_all, X_orig_all)

        batch_size_fit = len(dataset)
        if batch_size_fit == 0:
            return DataLoader([]), DataLoader([])

        all_data_loader = DataLoader(dataset, batch_size=batch_size_fit, shuffle=False)
        return all_data_loader, all_data_loader

    def train(
        self,
        model: DirectionalProbe,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:
        """Train method for directional probes."""
        if not isinstance(model, DirectionalProbe):
            raise TypeError("DirectionalProbeTrainer expects model to be an instance of DirectionalProbe")

        target_device = torch.device(self.config.device)
        model.to(target_device)

        try:
            x_train_tensor, y_train_tensor, x_orig_tensor = next(iter(train_loader))
        except StopIteration:
            return {"train_loss": [], "val_loss": []}

        x_train_tensor = x_train_tensor.to(target_device)
        y_train_tensor = y_train_tensor.to(target_device)
        x_orig_tensor = x_orig_tensor.to(target_device)

        initial_direction = model.fit(x_train_tensor, y_train_tensor)
        final_direction = initial_direction

        if self.config.standardize_activations and self.feature_std is not None:
            std_dev = self.feature_std.squeeze().to(initial_direction.device)
            
            if initial_direction.shape == std_dev.shape:
                final_direction = initial_direction / std_dev
            elif (
                initial_direction.dim() == 2
                and initial_direction.shape[0] == 1
                and std_dev.dim() == 1
            ):
                final_direction = initial_direction / std_dev.unsqueeze(0)

        model._set_raw_direction_representation(final_direction)
        model.has_fit = True  

        # Calculate optimal threshold on training data
        if hasattr(model, 'config') and hasattr(model.config, 'optimal_thresholds'):
            print("Calculating optimal threshold for directional probe...")
            
            model.eval()
            all_scores = []
            all_labels = []
            
            with torch.no_grad():
                # Get predictions on the full training set
                outputs = model(x_orig_tensor)
                
                # No sigmoid needed for directional probes
                all_scores = outputs.cpu().squeeze().tolist()
                all_labels = y_train_tensor.cpu().squeeze().tolist()
            
            # Normalize scores to [0, 1] range for threshold calculation
            import numpy as np
            scores_array = np.array(all_scores)
            min_score = scores_array.min()
            max_score = scores_array.max()
            
            if max_score > min_score:
                normalized_scores = (scores_array - min_score) / (max_score - min_score)
                
                from probity.utils.threshold_optimization import find_optimal_threshold_auroc
                optimal_threshold_normalized = find_optimal_threshold_auroc(
                    normalized_scores, 
                    np.array(all_labels)
                )
                
                # Convert back to original scale
                optimal_threshold = min_score + optimal_threshold_normalized * (max_score - min_score)
                
                model.config.optimal_thresholds['train_auroc'] = optimal_threshold
                model.config.optimal_thresholds['score_min'] = min_score
                model.config.optimal_thresholds['score_max'] = max_score
                model.config.optimal_thresholds['normalized_threshold'] = optimal_threshold_normalized
                
                print(f"Optimal threshold (train AUROC): {optimal_threshold:.4f}")
                print(f"  Score range: [{min_score:.4f}, {max_score:.4f}]")
                print(f"  Normalized threshold: {optimal_threshold_normalized:.4f}")
            else:
                print("Warning: All scores are identical, cannot calculate optimal threshold")
                model.config.optimal_thresholds['train_auroc'] = 0.0

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

        with torch.no_grad():
            preds = model(x_orig_tensor)
            model_dtype = model.dtype
            y_target = y_train_tensor.to(dtype=model_dtype)
            
            if preds.dim() == 1 and y_target.dim() == 2 and y_target.shape[1] == 1:
                y_target = y_target.squeeze(1)
            elif preds.shape != y_target.shape:
                try:
                    y_target = y_target.view_as(preds)
                except RuntimeError:
                    history["train_loss"].append(float("nan"))
                    history["val_loss"].append(float("nan"))
                    return history

            loss_item = float("nan")
            try:
                if isinstance(model, MultiClassLogisticProbe):
                    loss_fn = nn.CrossEntropyLoss().to(device=target_device, dtype=model_dtype)
                    y_target = y_target.long().squeeze()
                else:
                    loss_fn = nn.BCEWithLogitsLoss().to(device=target_device, dtype=model_dtype)
                
                loss = loss_fn(preds, y_target)
                loss_item = loss.item()
            except Exception as e:
                pass

        history["train_loss"].append(loss_item)
        history["val_loss"].append(loss_item)

        return history