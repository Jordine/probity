from abc import ABC
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    SklearnLogisticProbe,
)
from probity.training.losses import (
    compute_probe_bce_loss,
    compute_max_aggregation_loss,
    compute_max_aggregation_loss_vectorized,
    precompute_span_masks,
    compute_annealed_loss,
    compute_sparsity_loss,
    LossMode,
    ProbeLoss,
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
    # Advanced training options (ported from hallucination probes)
    use_max_aggregation: bool = False  # DEPRECATED: kept for backward compat with old saved configs. Use loss_mode="span_max" instead
    anneal_warmup: float = 0.3  # Fraction of training for annealing transition
    pos_weight_override: float = 0.0  # Manual pos_weight (0 = auto-calculate)
    neg_weight_override: float = 0.0  # Manual neg_weight (0 = equal to pos)
    sparsity_penalty: float = 0.0  # Penalize high average activation (0 = off)
    # New unified loss mode system (see LOSS_DESIGN.md)
    loss_mode: str = "sample_mean"  # sample_mean, sample_max, token_all, joint, annealed, span_max, etc.
    joint_alpha: float = 0.5  # Weight for sample loss in joint mode (1.0 = sample only)
    # Localization loss hyperparameters
    margin: float = 1.0  # Margin for margin/ranking losses
    temperature: float = 0.5  # Temperature for soft_recall loss
    num_pairs: int = 32  # Number of pairs for ranking loss


class SupervisedProbeTrainer(BaseProbeTrainer):
    """Enhanced trainer for supervised probes with progress tracking and LR scheduling."""
    def __init__(self, config: SupervisedTrainerConfig):
        super().__init__(config)
        if not isinstance(config, SupervisedTrainerConfig):
            raise TypeError("SupervisedProbeTrainer requires a SupervisedTrainerConfig")
        self.config: SupervisedTrainerConfig = config
        # Storage for span-aware training
        self._train_spans: Optional[List[List[Tuple[int, int]]]] = None
        self._val_spans: Optional[List[List[Tuple[int, int]]]] = None
        # Precomputed GPU masks for vectorized training
        self._train_pos_masks: Optional[torch.Tensor] = None
        self._train_neg_masks: Optional[torch.Tensor] = None
        self._train_token_labels: Optional[torch.Tensor] = None

    def prepare_supervised_data_with_spans(
        self, activation_store: ActivationStore, position_key: str
    ) -> Tuple[DataLoader, DataLoader, List[List[Tuple[int, int]]], List[List[Tuple[int, int]]]]:
        """Prepare train/val data with span boundaries for max aggregation.

        Returns:
            Tuple of (train_loader, val_loader, train_spans, val_spans)
        """
        # Get full sequence activations with spans
        full_acts, labels, spans, span_labels = activation_store.get_probe_data_with_spans(position_key)

        n_total = len(full_acts)
        n_train = int(n_total * self.config.train_ratio)

        indices = torch.randperm(n_total)
        train_indices = indices[:n_train].tolist()
        val_indices = indices[n_train:].tolist()

        # Split activations and labels
        X_train = full_acts[train_indices]
        X_val = full_acts[val_indices]
        y_train = labels[train_indices]
        y_val = labels[val_indices]

        # Split spans
        train_spans = [spans[i] for i in train_indices]
        val_spans = [spans[i] for i in val_indices]

        # Create DataLoaders
        train_dataset = TensorDataset(X_train, y_train.unsqueeze(1), X_train)  # X_orig same as X for now
        val_dataset = TensorDataset(X_val, y_val.unsqueeze(1), X_val)

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=False)  # Don't shuffle to keep span alignment
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)

        # Store spans for use in train_epoch
        self._train_spans = train_spans
        self._val_spans = val_spans

        # Precompute GPU masks for vectorized training
        seq_len = X_train.shape[1] if X_train.dim() == 3 else 1
        self._train_pos_masks, self._train_neg_masks = precompute_span_masks(
            train_spans, y_train, seq_len, device=self.config.device, dtype=torch.float32
        )
        # Also precompute token_labels for BCE loss (positive sample tokens in spans)
        self._train_token_labels = self._train_pos_masks.clone()

        # Precompute validation masks for validate_with_probe_loss
        val_seq_len = X_val.shape[1] if X_val.dim() == 3 else 1
        self._val_pos_masks, self._val_neg_masks = precompute_span_masks(
            val_spans, y_val, val_seq_len, device=self.config.device, dtype=torch.float32
        )
        self._val_token_labels = self._val_pos_masks.clone()

        return train_loader, val_loader, train_spans, val_spans

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

    def train_epoch_with_max_aggr(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        num_epochs: int,
    ) -> Tuple[float, float]:
        """Run one epoch with max aggregation loss + annealing (GPU-optimized).

        Returns:
            Tuple of (total_loss, omega) where omega is the annealing coefficient
        """
        model.train()
        total_loss = 0.0

        # Check for precomputed masks (GPU-optimized path)
        use_vectorized = (
            self._train_pos_masks is not None and
            self._train_neg_masks is not None and
            self._train_token_labels is not None
        )

        if not use_vectorized and self._train_spans is None:
            raise ValueError("Must call prepare_supervised_data_with_spans() before training with max aggregation")

        batch_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs} [max_aggr]",
            disable=not self.config.show_progress,
            leave=False,
        )

        cumulative_idx = 0  # Track position in dataset
        omega = 0.0  # Initialize omega in case loader is empty

        for batch_x, batch_y, _ in batch_pbar:
            optimizer.zero_grad()
            batch_x = batch_x.to(self.config.device)
            batch_y = batch_y.to(self.config.device).float()

            batch_size = batch_x.shape[0]
            start_idx = cumulative_idx
            end_idx = min(start_idx + batch_size, len(self._train_spans) if self._train_spans else start_idx + batch_size)

            # Handle different input shapes
            if batch_x.dim() == 3:
                # (batch, seq_len, hidden) -> need to process per-position
                B, S, H = batch_x.shape
                batch_x_flat = batch_x.view(B * S, H)
                logits_flat = model(batch_x_flat)  # (B*S, 1)
                logits = logits_flat.view(B, S)  # (B, S)
            else:
                # (batch, hidden) - standard case, shouldn't happen with max aggr
                logits = model(batch_x).squeeze(-1)

            if use_vectorized:
                # GPU-optimized path: use precomputed masks
                batch_token_labels = self._train_token_labels[start_idx:end_idx].to(self.config.device)
                batch_pos_masks = self._train_pos_masks[start_idx:end_idx].to(self.config.device)
                batch_neg_masks = self._train_neg_masks[start_idx:end_idx].to(self.config.device)

                # BCE loss with precomputed token labels
                bce_loss = F.binary_cross_entropy_with_logits(logits, batch_token_labels)

                # Vectorized max aggregation loss
                max_aggr_loss = compute_max_aggregation_loss_vectorized(
                    logits, batch_pos_masks, batch_neg_masks
                )
            else:
                # Fallback to Python loops (slower, for debugging)
                batch_spans = self._train_spans[start_idx:end_idx]

                # Create token-level labels from spans
                token_labels = torch.zeros_like(logits)
                for i, spans in enumerate(batch_spans):
                    if i < len(batch_y) and batch_y[i].item() > 0.5:
                        for span in spans:
                            if isinstance(span, (tuple, list)) and len(span) >= 2:
                                start, end = span[0], span[1]
                                if 0 <= start <= end < logits.shape[1]:
                                    token_labels[i, start:end+1] = 1.0

                bce_loss = F.binary_cross_entropy_with_logits(logits, token_labels)

                # Build pos/neg span lists
                pos_spans = []
                neg_spans = []
                for i, spans in enumerate(batch_spans):
                    if i < len(batch_y):
                        label = batch_y[i].item()
                        if label > 0.5:
                            pos_spans.append(spans)
                            neg_spans.append([])
                        else:
                            pos_spans.append([])
                            neg_spans.append(spans)
                    else:
                        pos_spans.append([])
                        neg_spans.append([])

                max_aggr_loss = compute_max_aggregation_loss(logits, pos_spans, neg_spans)

            cumulative_idx = end_idx

            # Compute annealed loss
            combined_loss, omega = compute_annealed_loss(
                bce_loss, max_aggr_loss,
                epoch, num_epochs,
                self.config.anneal_warmup
            )

            # Add sparsity penalty if configured
            if self.config.sparsity_penalty > 0:
                sparsity = compute_sparsity_loss(logits)
                combined_loss = combined_loss + self.config.sparsity_penalty * sparsity

            combined_loss.backward()
            optimizer.step()

            total_loss += combined_loss.item()
            batch_pbar.set_postfix({
                "Loss": f"{combined_loss.item():.4f}",
                "omega": f"{omega:.2f}"
            })

        return total_loss / max(len(train_loader), 1), omega

    def train_epoch_with_probe_loss(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        probe_loss: ProbeLoss,
        epoch: int,
        num_epochs: int,
    ) -> Tuple[float, float]:
        """Run one epoch with configurable ProbeLoss (GPU-optimized).

        This is the new unified training method that supports all loss modes
        defined in LOSS_DESIGN.md.

        Returns:
            Tuple of (total_loss, alpha) where alpha is the annealing coefficient
        """
        model.train()
        total_loss = 0.0

        # Update epoch for annealed mode
        probe_loss.set_epoch(epoch, num_epochs)

        # Check for precomputed masks
        has_token_labels = self._train_token_labels is not None
        has_span_masks = self._train_pos_masks is not None

        batch_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs} [{probe_loss.mode.value}]",
            disable=not self.config.show_progress,
            leave=False,
        )

        cumulative_idx = 0
        alpha = probe_loss._get_annealed_alpha() if probe_loss.mode == LossMode.ANNEALED else 1.0

        for batch_x, batch_y, _ in batch_pbar:
            optimizer.zero_grad()
            batch_x = batch_x.to(self.config.device)
            batch_y = batch_y.to(self.config.device).float().squeeze()

            batch_size = batch_x.shape[0]
            start_idx = cumulative_idx
            end_idx = start_idx + batch_size

            # Handle different input shapes
            if batch_x.dim() == 3:
                # (batch, seq_len, hidden) -> need to process per-position
                B, S, H = batch_x.shape
                batch_x_flat = batch_x.view(B * S, H)
                logits_flat = model(batch_x_flat)  # (B*S, 1)
                logits = logits_flat.view(B, S)  # (B, S)
            else:
                # (batch, hidden) - shouldn't happen for token-level modes
                logits = model(batch_x).squeeze(-1)

            # Prepare tensors for loss
            attention_mask = torch.ones_like(logits)  # TODO: use actual mask if available

            token_labels = None
            span_masks = None

            if has_token_labels and self._train_token_labels is not None:
                token_labels = self._train_token_labels[start_idx:end_idx].to(self.config.device)

            if has_span_masks and self._train_pos_masks is not None:
                span_masks = self._train_pos_masks[start_idx:end_idx].to(self.config.device)

            cumulative_idx = end_idx

            # Compute loss using ProbeLoss
            loss = probe_loss(
                token_scores=logits,
                sample_labels=batch_y,
                token_labels=token_labels,
                attention_mask=attention_mask,
                span_masks=span_masks,
            )

            # Add sparsity penalty if configured
            if self.config.sparsity_penalty > 0:
                sparsity = compute_sparsity_loss(logits)
                loss = loss + self.config.sparsity_penalty * sparsity

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update alpha for annealed mode
            if probe_loss.mode == LossMode.ANNEALED:
                alpha = probe_loss._get_annealed_alpha()

            batch_pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "alpha": f"{alpha:.2f}" if probe_loss.mode in [LossMode.ANNEALED, LossMode.JOINT] else "N/A"
            })

        return total_loss / max(len(train_loader), 1), alpha

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
            "omega": [],  # Track annealing coefficient when using max aggregation
            "alpha": [],  # Track annealing coefficient for ProbeLoss modes
        }

        # Determine which training method to use
        use_probe_loss = self.config.loss_mode != "sample_mean" and self._train_spans is not None
        use_legacy_max_aggr = self.config.use_max_aggregation and self._train_spans is not None and not use_probe_loss

        # Create ProbeLoss if using new loss modes
        probe_loss_fn: Optional[ProbeLoss] = None
        if use_probe_loss:
            try:
                loss_mode = LossMode(self.config.loss_mode)
            except ValueError:
                print(f"Warning: Unknown loss_mode '{self.config.loss_mode}', falling back to sample_mean")
                loss_mode = LossMode.SAMPLE_MEAN
                use_probe_loss = False

            if use_probe_loss:
                probe_loss_fn = ProbeLoss(
                    mode=loss_mode,
                    joint_alpha=self.config.joint_alpha,
                    anneal_warmup=self.config.anneal_warmup,
                    margin=getattr(self.config, 'margin', 1.0),
                    temperature=getattr(self.config, 'temperature', 0.5),
                    num_pairs=getattr(self.config, 'num_pairs', 32),
                )

        mode_str = ""
        if use_probe_loss and probe_loss_fn:
            mode_str = f" [{probe_loss_fn.mode.value}]"
        elif use_legacy_max_aggr:
            mode_str = " [max_aggr]"

        epoch_pbar = tqdm(
            range(self.config.num_epochs),
            desc=f"Training{mode_str}",
            disable=not self.config.show_progress,
        )

        best_val_loss = float("inf")
        patience_counter = 0
        warmup_complete = False  # Track when annealing warmup is done

        for epoch in epoch_pbar:
            # Choose training method based on config
            if use_probe_loss and probe_loss_fn is not None:
                train_loss, alpha = self.train_epoch_with_probe_loss(
                    model,
                    train_loader,
                    optimizer,
                    probe_loss_fn,
                    epoch,
                    self.config.num_epochs,
                )
                history["alpha"].append(alpha)
            elif use_legacy_max_aggr:
                train_loss, omega = self.train_epoch_with_max_aggr(
                    model,
                    train_loader,
                    optimizer,
                    epoch,
                    self.config.num_epochs,
                )
                history["omega"].append(omega)
            else:
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
                # Choose validation method based on training mode
                if use_probe_loss and probe_loss_fn is not None:
                    val_loss = self.validate_with_probe_loss(model, val_loader, probe_loss_fn)
                elif use_legacy_max_aggr:
                    val_loss = self.validate_with_max_aggr(model, val_loader)
                else:
                    val_loss = self.validate(model, val_loader, loss_fn, is_multi_class=is_multi_class)
                history["val_loss"].append(val_loss)

                # Handle early stopping with warmup for annealing modes
                # (prevents comparing BCE-phase losses with token-phase losses)
                uses_annealing = (
                    (use_legacy_max_aggr and history["omega"]) or
                    (use_probe_loss and probe_loss_fn and probe_loss_fn.mode == LossMode.ANNEALED and history["alpha"])
                )

                if uses_annealing:
                    # Get current annealing coefficient
                    if use_legacy_max_aggr and history["omega"]:
                        current_coef = history["omega"][-1]
                        coef_threshold = 1.0  # omega goes 0->1
                    else:
                        current_coef = history["alpha"][-1]
                        coef_threshold = 0.0  # alpha goes 1->0 (fully token-level when 0)

                    # Check if warmup is complete
                    is_warmup_complete = (
                        (use_legacy_max_aggr and current_coef >= coef_threshold) or
                        (use_probe_loss and current_coef <= coef_threshold)
                    )

                    if is_warmup_complete:
                        if not warmup_complete:
                            warmup_complete = True
                            best_val_loss = val_loss
                            patience_counter = 0
                        elif val_loss < best_val_loss - self.config.min_delta:
                            best_val_loss = val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                else:
                    # No annealing - normal early stopping
                    if val_loss < best_val_loss - self.config.min_delta:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                if patience_counter >= self.config.patience:
                    break

                postfix = {
                    "Train Loss": f"{train_loss:.6f}",
                    "Val Loss": f"{val_loss:.6f}",
                    "LR": f"{scheduler.get_last_lr()[0]:.2e}",
                }
                if use_legacy_max_aggr and history["omega"]:
                    postfix["omega"] = f"{history['omega'][-1]:.2f}"
                if use_probe_loss and history["alpha"]:
                    postfix["alpha"] = f"{history['alpha'][-1]:.2f}"
                epoch_pbar.set_postfix(postfix)
            else:
                postfix = {
                    "Train Loss": f"{train_loss:.6f}",
                    "LR": f"{scheduler.get_last_lr()[0]:.2e}",
                }
                if use_legacy_max_aggr and history["omega"]:
                    postfix["omega"] = f"{history['omega'][-1]:.2f}"
                if use_probe_loss and history["alpha"]:
                    postfix["alpha"] = f"{history['alpha'][-1]:.2f}"
                epoch_pbar.set_postfix(postfix)

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
                cumulative_idx = 0  # Track position in spans list
                for batch_x, batch_y, _ in train_loader:
                    batch_x = batch_x.to(self.config.device)
                    batch_y = batch_y.to(self.config.device)

                    outputs = model(batch_x)

                    # Apply sigmoid for logistic probes
                    if model.__class__.__name__ == 'LogisticProbe':
                        outputs = torch.sigmoid(outputs)

                    # Check if outputs have a sequence dimension (not already sample-level)
                    # Supervised probes output [B, 1], so we should NOT try span aggregation
                    if outputs.dim() == 3:
                        outputs = outputs.squeeze(-1)

                    outputs_have_seq_dim = outputs.dim() == 2 and outputs.shape[1] > 1

                    # Aggregate using span boundaries only if outputs have sequence dimension
                    if self._train_spans is not None and self.config.use_max_aggregation and outputs_have_seq_dim:
                        # Get spans for this batch using cumulative index
                        batch_size = batch_x.shape[0]
                        start_idx = cumulative_idx
                        end_idx = min(start_idx + batch_size, len(self._train_spans))
                        batch_spans = self._train_spans[start_idx:end_idx]
                        cumulative_idx = end_idx

                        # Aggregate within spans for each sample
                        for i, spans in enumerate(batch_spans):
                            if i < outputs.shape[0] and spans:
                                # Get max score within all spans for this sample
                                span_maxes = []
                                for start, end in spans:
                                    if start < outputs.shape[1] and end <= outputs.shape[1]:
                                        span_scores = outputs[i, start:end]
                                        if len(span_scores) > 0:
                                            span_maxes.append(span_scores.max().item())
                                if span_maxes:
                                    all_scores.append(max(span_maxes))
                                    all_labels.append(batch_y[i].item())
                    else:
                        # Fallback: use sample-level scores directly
                        # For probes outputting [B, 1], just squeeze and use as sample scores
                        if outputs.dim() == 2 and outputs.shape[1] == 1:
                            sample_scores = outputs.squeeze(-1)  # [B, 1] -> [B]
                        elif outputs.dim() == 2:
                            sample_scores = outputs.max(dim=1).values
                        elif outputs.dim() == 1:
                            sample_scores = outputs
                        else:
                            sample_scores = outputs.squeeze()
                        all_scores.extend(sample_scores.cpu().view(-1).tolist())
                        all_labels.extend(batch_y.cpu().view(-1).tolist())

            # Calculate optimal threshold and AUROC
            from probity.utils.threshold_optimization import find_optimal_threshold_auroc, compute_auroc
            import numpy as np

            scores_arr = np.array(all_scores)
            labels_arr = np.array(all_labels)

            optimal_threshold = find_optimal_threshold_auroc(scores_arr, labels_arr)
            train_auroc = compute_auroc(scores_arr, labels_arr)

            model.config.optimal_thresholds['train_auroc'] = optimal_threshold
            model.config.optimal_thresholds['train_auroc_score'] = train_auroc
            print(f"Train AUROC: {train_auroc:.4f} | Optimal threshold: {optimal_threshold:.4f}")


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

    def validate_with_max_aggr(
        self,
        model: torch.nn.Module,
        val_loader: DataLoader,
    ) -> float:
        """Run validation with max aggregation loss."""
        model.eval()
        total_loss = 0.0

        if self._val_spans is None:
            raise ValueError("Must call prepare_supervised_data_with_spans() before validating with max aggregation")

        cumulative_idx = 0  # Track position in spans list
        with torch.no_grad():
            for batch_x, batch_y, _ in val_loader:
                batch_x = batch_x.to(self.config.device)
                batch_y = batch_y.to(self.config.device).float()

                batch_size = batch_x.shape[0]

                # Get spans for this batch using cumulative index
                start_idx = cumulative_idx
                end_idx = min(start_idx + batch_size, len(self._val_spans))
                batch_spans = self._val_spans[start_idx:end_idx]
                cumulative_idx = end_idx

                # Handle different input shapes
                if batch_x.dim() == 3:
                    B, S, H = batch_x.shape
                    batch_x_flat = batch_x.view(B * S, H)
                    logits_flat = model(batch_x_flat)
                    logits = logits_flat.view(B, S)
                else:
                    logits = model(batch_x).squeeze(-1)

                # Split spans into positive/negative
                pos_spans = []
                neg_spans = []
                for i, spans in enumerate(batch_spans):
                    if i < len(batch_y):
                        label = batch_y[i].item()
                        if label > 0.5:
                            pos_spans.append(spans)
                            neg_spans.append([])
                        else:
                            pos_spans.append([])
                            neg_spans.append(spans)
                    else:
                        pos_spans.append([])
                        neg_spans.append([])

                loss = compute_max_aggregation_loss(logits, pos_spans, neg_spans)
                total_loss += loss.item()

        return total_loss / max(len(val_loader), 1)

    def validate_with_probe_loss(
        self,
        model: torch.nn.Module,
        val_loader: DataLoader,
        probe_loss: ProbeLoss,
    ) -> float:
        """Run validation with ProbeLoss for token-level loss modes (span_max, token_all, etc.)."""
        model.eval()
        total_loss = 0.0

        # Check if we have precomputed validation masks
        has_val_masks = (
            hasattr(self, '_val_pos_masks') and self._val_pos_masks is not None
        )
        has_token_labels = (
            hasattr(self, '_val_token_labels') and self._val_token_labels is not None
        )

        if not has_val_masks and self._val_spans is None:
            raise ValueError("Must call prepare_supervised_data_with_spans() before validating with probe loss")

        cumulative_idx = 0
        with torch.no_grad():
            for batch_x, batch_y, _ in val_loader:
                batch_x = batch_x.to(self.config.device)
                batch_y = batch_y.to(self.config.device).float().squeeze()

                batch_size = batch_x.shape[0]
                start_idx = cumulative_idx
                end_idx = start_idx + batch_size

                # Handle different input shapes (critical for logistic probes)
                if batch_x.dim() == 3:
                    B, S, H = batch_x.shape
                    batch_x_flat = batch_x.view(B * S, H)
                    logits_flat = model(batch_x_flat)  # (B*S, 1)
                    logits = logits_flat.view(B, S)  # (B, S)
                else:
                    logits = model(batch_x).squeeze(-1)

                # Prepare tensors for loss
                attention_mask = torch.ones_like(logits)

                token_labels = None
                span_masks = None

                if has_token_labels and self._val_token_labels is not None:
                    token_labels = self._val_token_labels[start_idx:end_idx].to(self.config.device)

                if has_val_masks and self._val_pos_masks is not None:
                    span_masks = self._val_pos_masks[start_idx:end_idx].to(self.config.device)

                cumulative_idx = end_idx

                # Compute loss using ProbeLoss
                loss = probe_loss(
                    token_scores=logits,
                    sample_labels=batch_y,
                    token_labels=token_labels,
                    attention_mask=attention_mask,
                    span_masks=span_masks,
                )

                total_loss += loss.item()

        return total_loss / max(len(val_loader), 1)


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
        if not isinstance(model, (DirectionalProbe, SklearnLogisticProbe)):
            raise TypeError("DirectionalProbeTrainer expects model to be an instance of DirectionalProbe or SklearnLogisticProbe")

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

                from probity.utils.threshold_optimization import find_optimal_threshold_auroc, compute_auroc
                optimal_threshold_normalized = find_optimal_threshold_auroc(
                    normalized_scores,
                    np.array(all_labels)
                )
                train_auroc = compute_auroc(scores_array, np.array(all_labels))

                # Convert back to original scale
                optimal_threshold = min_score + optimal_threshold_normalized * (max_score - min_score)

                model.config.optimal_thresholds['train_auroc'] = optimal_threshold
                model.config.optimal_thresholds['train_auroc_score'] = train_auroc
                model.config.optimal_thresholds['score_min'] = min_score
                model.config.optimal_thresholds['score_max'] = max_score
                model.config.optimal_thresholds['normalized_threshold'] = optimal_threshold_normalized

                print(f"Train AUROC: {train_auroc:.4f} | Optimal threshold: {optimal_threshold:.4f}")
                print(f"  Score range: [{min_score:.4f}, {max_score:.4f}]")
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