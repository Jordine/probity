"""
Training module for NTML binary token classification.

Supports both sklearn logistic regression and PyTorch interpretability-focused probes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

from config import NTMLBinaryTrainingConfig

logger = logging.getLogger(__name__)


class InterpretabilityBinaryProbe(nn.Module):
    """Simple logistic regression probe for interpretability research."""
    
    def __init__(self, input_size: int, bias: bool = True, normalize_weights: bool = True):
        super().__init__()
        self.input_size = input_size
        self.normalize_weights = normalize_weights
        
        # Simple linear layer - no dropout!
        self.linear = nn.Linear(input_size, 1, bias=bias)
        
        # Zero initialization (common in interpretability work)
        nn.init.zeros_(self.linear.weight)
        if bias:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returns logits."""
        return self.linear(x).squeeze(-1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probabilities using sigmoid."""
        logits = self.forward(x)
        return torch.sigmoid(logits)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return binary predictions."""
        probs = self.predict_proba(x)
        return (probs > threshold).long()
    
    def get_direction(self, normalized: bool = True) -> torch.Tensor:
        """Get the learned probe direction."""
        direction = self.linear.weight.data.clone().squeeze(0)
        
        if normalized and self.normalize_weights:
            direction = direction / (torch.norm(direction) + 1e-8)
            
        return direction
    
    def set_direction(self, direction: torch.Tensor) -> None:
        """Set the probe direction (useful for interventions)."""
        if direction.dim() == 1:
            direction = direction.unsqueeze(0)
        with torch.no_grad():
            self.linear.weight.copy_(direction)
    
    def get_bias(self) -> float:
        """Get the learned bias term."""
        if self.linear.bias is not None:
            return self.linear.bias.item()
        return 0.0


class SklearnProbeTrainer:
    """Sklearn-based probe trainer for fast, stable training."""
    
    def __init__(self, config: NTMLBinaryTrainingConfig):
        self.config = config
        self.pipeline = None
        self.training_history = {}
        self.used_dtype = None
        self.all_sweep_models = {} 
        
    def train(self, activations: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
        """Train sklearn logistic regression probe."""
        
        logger.info("Starting sklearn logistic regression training")
        
        # Convert to numpy with adaptive dtype handling
        X = self._convert_activations_for_sklearn(activations)
        y = labels.cpu().numpy()
        
        # Split data
        num_samples = len(X)
        num_train = int(num_samples * self.config.train_ratio)
        indices = np.random.permutation(num_samples)
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]
        
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        
        logger.info(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}")
        logger.info(f"Train label distribution: {np.bincount(y_train).tolist()}")
        logger.info(f"Val label distribution: {np.bincount(y_val).tolist()}")
        logger.info(f"Using dtype: {X.dtype}")
        
        start_time = time.time()
        
        # Regularization sweep if requested
        if getattr(self.config, 'sklearn_C_sweep', False):
            results = self._train_with_C_sweep(X_train, y_train, X_val, y_val)
        else:
            # Single C value training
            C = getattr(self.config, 'sklearn_C', 1.0)
            results = self._train_single_C(X_train, y_train, X_val, y_val, C)
        
        total_time = time.time() - start_time
        
        # Final validation metrics
        final_val_metrics = self._evaluate_pipeline(X_val, y_val)
        
        return {
            "training_history": self.training_history,
            "final_metrics": final_val_metrics,
            "training_time": total_time,
            "num_parameters": X.shape[1] + 1,  # weights + bias
            "regularization_results": results.get("C_sweep_results", {}),
            "best_C": results.get("best_C", getattr(self.config, 'sklearn_C', 1.0)),
            "used_dtype": str(self.used_dtype),
        }
    
    def _convert_activations_for_sklearn(self, activations: torch.Tensor) -> np.ndarray:
        """Convert activations to sklearn-compatible dtype with adaptive fallback."""
        
        original_dtype = activations.dtype
        logger.info(f"Converting activations from {original_dtype} for sklearn")
        
        # Strategy 1: Try to preserve memory with float16 (if coming from bfloat16/float16)
        if original_dtype in [torch.bfloat16, torch.float16]:
            try:
                if original_dtype == torch.bfloat16:
                    X = activations.half().cpu().numpy()  # bfloat16 → float16
                else:
                    X = activations.cpu().numpy()  # already float16
                
                # Test sklearn compatibility with float16
                self._test_sklearn_float16_compatibility(X[:min(100, len(X))])
                
                logger.info(f"✅ Successfully converted {original_dtype} → float16 for sklearn")
                self.used_dtype = np.float16
                return X
                
            except Exception as e:
                logger.warning(f"⚠️  float16 conversion failed: {e}")
                logger.info("Falling back to float32...")
        
        # Strategy 2: Fallback to float32 (guaranteed compatibility)
        if original_dtype == torch.bfloat16:
            X = activations.float().cpu().numpy()
            logger.info(f"✅ Converted {original_dtype} → float32 for sklearn (fallback)")
        elif original_dtype == torch.float16:
            X = activations.float().cpu().numpy()
            logger.info(f"✅ Converted {original_dtype} → float32 for sklearn (fallback)")
        else:
            X = activations.cpu().numpy()
            if X.dtype not in [np.float32, np.float64]:
                logger.warning(f"Converting from {X.dtype} to float32")
                X = X.astype(np.float32)
        
        self.used_dtype = X.dtype
        return X
    
    def _test_sklearn_float16_compatibility(self, X_sample: np.ndarray):
        """Test if sklearn pipeline works with float16."""
        if X_sample.dtype != np.float16:
            return
        
        if len(X_sample) < 2:
            return  # Need at least 2 samples for meaningful test
        
        try:
            # Test StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_sample)
            
            # Test LogisticRegression with minimal data
            if len(X_sample) >= 2:
                y_sample = np.array([0, 1] * (len(X_sample) // 2))[:len(X_sample)]
                lr = LogisticRegression(max_iter=10, solver='liblinear')
                
                # Test full pipeline
                pipeline = Pipeline([
                    ("scaler", StandardScaler()), 
                    ("classifier", lr)
                ])
                pipeline.fit(X_sample, y_sample)
                
        except (ValueError, TypeError, RuntimeError, np.core._exceptions._ArrayMemoryError) as e:
            raise RuntimeError(f"Sklearn compatibility test failed with float16: {e}")
    
    def _train_with_C_sweep(self, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        """Train with regularization sweep."""
        
        C_values = getattr(self.config, 'sklearn_C_values', np.logspace(-4, 4, 9))
        
        logger.info(f"Training with C sweep: {C_values}")
        
        train_scores = []
        val_scores = []
        best_val_score = 0.0
        best_C = C_values[0]
        best_pipeline = None
        
        for C in tqdm(C_values, desc="C regularization sweep"):
            try:
                pipeline = self._create_pipeline(C)
                pipeline.fit(X_train, y_train)
                
                train_score = pipeline.score(X_train, y_train)
                val_score = pipeline.score(X_val, y_val)
                
                train_scores.append(train_score)
                val_scores.append(val_score)

                self.all_sweep_models[C] = pipeline
                
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_C = C
                    best_pipeline = pipeline
                    
            except Exception as e:
                logger.warning(f"Training failed for C={C}: {e}")
                train_scores.append(0.0)
                val_scores.append(0.0)
        
        if best_pipeline is None:
            raise RuntimeError("All C values failed during training")
        
        self.pipeline = best_pipeline
        self.training_history["C_values"] = C_values.tolist()
        self.training_history["train_scores"] = train_scores
        self.training_history["val_scores"] = val_scores
        
        logger.info(f"✅ Best C: {best_C} (val accuracy: {best_val_score:.4f})")
        
        return {
            "best_C": best_C,
            "best_val_score": best_val_score,
            "C_sweep_results": {
                "C_values": C_values.tolist(),
                "train_scores": train_scores,
                "val_scores": val_scores,
            }
        }
    
    def _train_single_C(self, X_train, y_train, X_val, y_val, C: float) -> Dict[str, Any]:
        """Train with single C value."""
        
        logger.info(f"Training with C = {C}")
        
        self.pipeline = self._create_pipeline(C)
        self.pipeline.fit(X_train, y_train)
        
        train_score = self.pipeline.score(X_train, y_train)
        val_score = self.pipeline.score(X_val, y_val)
        
        logger.info(f"✅ Train accuracy: {train_score:.4f}, Val accuracy: {val_score:.4f}")
        
        return {
            "best_C": C,
            "best_val_score": val_score,
        }
    
    def _create_pipeline(self, C: float) -> Pipeline:
        """Create sklearn pipeline."""
        
        class_weight = "balanced" if self.config.handle_class_imbalance else None
        solver = getattr(self.config, 'sklearn_solver', 'liblinear')
        max_iter = getattr(self.config, 'sklearn_max_iter', 1000)
        
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=C,
                class_weight=class_weight,
                solver=solver,
                random_state=42,
                max_iter=max_iter
            ))
        ])
    
    def _evaluate_pipeline(self, X_val, y_val) -> Dict[str, float]:
        """Comprehensive evaluation of the pipeline."""
        
        if self.pipeline is None:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auroc": 0.0}
        
        y_pred = self.pipeline.predict(X_val)
        y_proba = self.pipeline.predict_proba(X_val)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0),
        }
        
        try:
            metrics["auroc"] = roc_auc_score(y_val, y_proba)
        except ValueError:
            metrics["auroc"] = 0.0
        
        return metrics
    
    def get_direction(self, normalized: bool = True) -> np.ndarray:
        """Get the learned probe direction."""
        
        if self.pipeline is None:
            raise ValueError("Pipeline not trained yet")
        
        classifier = self.pipeline.named_steps['classifier']
        direction = classifier.coef_[0].copy()
        
        if normalized:
            direction = direction / (np.linalg.norm(direction) + 1e-8)
        
        return direction
    
    def get_bias(self) -> float:
        """Get the learned bias term."""
        
        if self.pipeline is None:
            raise ValueError("Pipeline not trained yet")
        
        return float(self.pipeline.named_steps['classifier'].intercept_[0])
    
    def save_model(self, output_path: str, metadata: Dict[str, Any]):
        """Save the sklearn pipeline."""
        
        if self.pipeline is None:
            raise ValueError("Pipeline not trained yet")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save sklearn pipeline
        joblib.dump(self.pipeline, output_path)
        logger.info(f"✅ Saved sklearn pipeline: {output_path}")
        
        # Save metadata
        metadata_path = output_path.with_suffix(".json")
        full_metadata = {
            "model_type": "sklearn_logistic_regression",
            "training_history": self.training_history,
            "metadata": metadata,
            "config": self.config.to_dict(),
            "used_dtype": str(self.used_dtype),
        }
        
        with open(metadata_path, "w") as f:
            json.dump(full_metadata, f, indent=2)
        
        logger.info(f"✅ Saved metadata: {metadata_path}")

    def save_all_sweep_models(self, base_output_path: str, metadata: Dict[str, Any]):
        """Save all models from C sweep, not just the best one."""
        
        if not self.all_sweep_models:
            logger.warning("No sweep models to save. Was C sweep performed?")
            return {}
        
        base_path = Path(base_output_path)
        base_dir = base_path.parent
        base_name = base_path.stem
        base_suffix = base_path.suffix
        
        saved_models = {}
        
        logger.info(f"Saving {len(self.all_sweep_models)} sweep models...")
        
        for C, pipeline in self.all_sweep_models.items():
            # Create filename like: probe_name_C_1.0.pt
            model_filename = f"{base_name}_C_{C}{base_suffix}"
            model_path = base_dir / model_filename
            
            # Save the model
            joblib.dump(pipeline, model_path)
            
            # Save metadata for this specific C
            metadata_path = model_path.with_suffix(".json")
            model_metadata = {
                "model_type": "sklearn_logistic_regression",
                "C_value": float(C),
                "training_history": self.training_history,
                "metadata": metadata,
                "config": self.config.to_dict(),
                "used_dtype": str(self.used_dtype),
            }
            
            with open(metadata_path, "w") as f:
                json.dump(model_metadata, f, indent=2)
            
            saved_models[C] = {
                "model_path": str(model_path),
                "metadata_path": str(metadata_path)
            }
            
            logger.info(f"✅ Saved C={C} model: {model_path}")
        
        return saved_models

    def get_sweep_model(self, C: float):
        """Get a specific model from the C sweep."""
        if C not in self.all_sweep_models:
            available_C = list(self.all_sweep_models.keys())
            raise ValueError(f"C={C} not found. Available: {available_C}")
        return self.all_sweep_models[C]


class PyTorchProbeTrainer:
    """PyTorch-based interpretability probe trainer."""
    
    def __init__(self, config: NTMLBinaryTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Training state
        self.model = None
        self.optimizer = None
        
        # Metrics tracking
        self.training_history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "val_auroc": [],
            "learning_rate": [],
        }
    
    def _safe_tensor_to_numpy(self, tensor: torch.Tensor, target_dtype: type = None) -> np.ndarray:
        """Safely convert torch tensor to numpy, handling bfloat16 and other unsupported dtypes."""
        
        # Move to CPU first
        tensor = tensor.cpu()
        
        # Handle dtypes not supported by numpy
        if tensor.dtype == torch.bfloat16:
            logger.debug("Converting bfloat16 tensor to float32 for numpy compatibility")
            tensor = tensor.float()  # Convert to float32
        elif tensor.dtype == torch.float16:
            # float16 is supported by numpy, but float32 is more stable for metrics
            tensor = tensor.float()
        
        # Convert to numpy
        numpy_array = tensor.numpy()
        
        # Cast to target dtype if specified
        if target_dtype is not None:
            numpy_array = numpy_array.astype(target_dtype)
        
        return numpy_array
    
    def train(self, activations: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
        """Train PyTorch interpretability probe."""
        
        logger.info("Starting PyTorch interpretability probe training")
        logger.info(f"Input dtype: {activations.dtype}, device: {activations.device}")
        
        # Prepare data
        train_loader, val_loader = self._prepare_data(activations, labels)
        
        # Setup model
        self._setup_model(activations.shape[-1])
        
        # Training loop
        best_val_f1 = 0.0
        best_model_state = None
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_metrics = self._train_epoch(train_loader)
            
            # Validate
            if epoch % self.config.eval_every == 0:
                val_metrics = self._validate(val_loader)
            else:
                val_metrics = {"loss": 0.0, "accuracy": 0.0, "f1": 0.0, "auroc": 0.0}
            
            # Update history
            self.training_history["train_loss"].append(train_metrics["loss"])
            self.training_history["train_accuracy"].append(train_metrics["accuracy"])
            self.training_history["val_loss"].append(val_metrics["loss"])
            self.training_history["val_accuracy"].append(val_metrics["accuracy"])
            self.training_history["val_f1"].append(val_metrics["f1"])
            self.training_history["val_auroc"].append(val_metrics["auroc"])
            self.training_history["learning_rate"].append(train_metrics["learning_rate"])
            
            # Save best model
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_model_state = self.model.state_dict().copy()
            
            # Log metrics
            if self.config.verbose and epoch % 5 == 0:
                logger.info(f"Epoch {epoch+1}: Train Loss: {train_metrics['loss']:.4f}, "
                          f"Val F1: {val_metrics['f1']:.4f}")
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info(f"✅ Restored best model with validation F1: {best_val_f1:.4f}")
        
        total_time = time.time() - start_time
        logger.info(f"✅ Training completed in {total_time:.1f} seconds")
        
        # Final validation
        final_val_metrics = self._validate(val_loader)
        
        return {
            "training_history": self.training_history,
            "final_metrics": final_val_metrics,
            "best_val_f1": best_val_f1,
            "training_time": total_time,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
        }
    
    def _prepare_data(self, activations: torch.Tensor, labels: torch.Tensor) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data loaders."""
        
        # Move to device if needed
        if activations.device != self.device:
            activations = activations.to(self.device)
        if labels.device != self.device:
            labels = labels.to(self.device)
        
        # Split into train/val
        num_samples = len(activations)
        num_train = int(num_samples * self.config.train_ratio)
        
        indices = torch.randperm(num_samples, device=self.device)
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]
        
        train_X = activations[train_indices]
        train_y = labels[train_indices]
        val_X = activations[val_indices]
        val_y = labels[val_indices]
        
        logger.info(f"Train samples: {len(train_X)}, Val samples: {len(val_X)}")
        logger.info(f"Train label distribution: {torch.bincount(train_y).tolist()}")
        logger.info(f"Val label distribution: {torch.bincount(val_y).tolist()}")
        
        # Create datasets
        train_dataset = TensorDataset(train_X, train_y)
        val_dataset = TensorDataset(val_X, val_y)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,  # Data already on device
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        
        return train_loader, val_loader
    
    def _setup_model(self, input_size: int):
        """Initialize model and optimizer."""
        
        bias = getattr(self.config, 'pytorch_bias', True)
        normalize_weights = getattr(self.config, 'pytorch_normalize_weights', True)
        
        self.model = InterpretabilityBinaryProbe(
            input_size, 
            bias=bias, 
            normalize_weights=normalize_weights
        ).to(self.device)
        
        # Convert model to appropriate dtype
        if hasattr(self.config, 'torch_dtype'):
            self.model = self.model.to(dtype=self.config.torch_dtype)
        
        # Use only weight decay for regularization (L2)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        logger.info(f"✅ Created interpretability probe: input_size={input_size}, bias={bias}")
        logger.info(f"Model dtype: {next(self.model.parameters()).dtype}")
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        
        self.model.train()
        criterion = nn.BCEWithLogitsLoss()
        
        total_loss = 0.0
        all_preds = []
        all_labels = []
        num_batches = 0
        
        for batch_X, batch_y in train_loader:
            # Ensure data matches model dtype
            if batch_X.dtype != next(self.model.parameters()).dtype:
                batch_X = batch_X.to(dtype=next(self.model.parameters()).dtype)
            
            batch_y = batch_y.float()
            
            self.optimizer.zero_grad()
            
            logits = self.model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            
            # Gradient clipping if configured
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
            
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).long()
                # Use safe conversion to handle bfloat16
                all_preds.extend(self._safe_tensor_to_numpy(preds, np.int64))
                all_labels.extend(self._safe_tensor_to_numpy(batch_y.long(), np.int64))
        
        avg_loss = total_loss / num_batches
        accuracy = accuracy_score(all_labels, all_preds)
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "learning_rate": self.optimizer.param_groups[0]['lr']
        }
    
    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        
        self.model.eval()
        criterion = nn.BCEWithLogitsLoss()
        
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_labels = []
        num_batches = 0
        
        for batch_X, batch_y in val_loader:
            # Ensure data matches model dtype
            if batch_X.dtype != next(self.model.parameters()).dtype:
                batch_X = batch_X.to(dtype=next(self.model.parameters()).dtype)
            
            batch_y = batch_y.float()
            
            logits = self.model(batch_X)
            loss = criterion(logits, batch_y)
            
            total_loss += loss.item()
            num_batches += 1
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            
            # Use safe conversion to handle bfloat16
            all_preds.extend(self._safe_tensor_to_numpy(preds, np.int64))
            all_probs.extend(self._safe_tensor_to_numpy(probs, np.float64))
            all_labels.extend(self._safe_tensor_to_numpy(batch_y.long(), np.int64))
        
        # Calculate metrics
        avg_loss = total_loss / num_batches
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        try:
            auroc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auroc = 0.0
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auroc": auroc,
        }
    
    def get_direction(self, normalized: bool = True) -> torch.Tensor:
        """Get the learned probe direction."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.get_direction(normalized)
    
    def get_bias(self) -> float:
        """Get the learned bias term."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.get_bias()
    
    def save_model(self, output_path: str, metadata: Dict[str, Any]):
        """Save the trained model and metadata."""
        
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_data = {
            "model_state_dict": self.model.state_dict(),
            "model_config": {
                "input_size": self.model.input_size,
                "model_type": "InterpretabilityBinaryProbe",
                "bias": self.model.linear.bias is not None,
                "normalize_weights": self.model.normalize_weights,
            },
            "training_config": self.config.to_dict(),
            "metadata": metadata,
        }
        
        torch.save(model_data, output_path)
        logger.info(f"✅ Saved PyTorch probe: {output_path}")
        
        # Save training history as JSON
        history_path = output_path.with_suffix(".json")
        with open(history_path, "w") as f:
            json.dump({
                "model_type": "pytorch_interpretability_probe",
                "training_history": self.training_history,
                "metadata": metadata,
                "config": self.config.to_dict(),
            }, f, indent=2)
        
        logger.info(f"✅ Saved training history: {history_path}")


class NTMLBinaryTrainer:
    """Unified trainer that delegates to sklearn or PyTorch implementation."""
    
    def __init__(self, config: NTMLBinaryTrainingConfig):
        self.config = config
        
        # Choose implementation based on config
        probe_method = getattr(config, 'probe_method', 'pytorch')
        
        if probe_method == 'sklearn':
            self.trainer = SklearnProbeTrainer(config)
        elif probe_method == 'pytorch':
            self.trainer = PyTorchProbeTrainer(config)
        else:
            raise ValueError(f"Unknown probe_method: {probe_method}. Use 'sklearn' or 'pytorch'")
        
        logger.info(f"🎯 Using {probe_method} probe trainer")
    
    def train(self, activations: torch.Tensor, labels: torch.Tensor) -> Dict[str, Any]:
        """Delegate training to the chosen implementation."""
        return self.trainer.train(activations, labels)
    
    def save_model(self, output_path: str, metadata: Dict[str, Any]):
        """Delegate model saving to the chosen implementation."""
        return self.trainer.save_model(output_path, metadata)
        
    def save_all_sweep_models(self, base_output_path: str, metadata: Dict[str, Any]) -> Dict:
        """Save all C sweep models if using sklearn trainer."""
        if hasattr(self.trainer, 'save_all_sweep_models'):
            return self.trainer.save_all_sweep_models(base_output_path, metadata)
        else:
            logger.warning("save_all_sweep_models only available for sklearn trainer")
            return {}
    
    def get_sweep_model(self, C: float):
        """Get specific C value model if using sklearn trainer."""
        if hasattr(self.trainer, 'get_sweep_model'):
            return self.trainer.get_sweep_model(C)
        else:
            raise ValueError("get_sweep_model only available for sklearn trainer")
    
    def get_direction(self, normalized: bool = True) -> Union[torch.Tensor, np.ndarray]:
        """Get the learned probe direction."""
        return self.trainer.get_direction(normalized)
    
    def get_bias(self) -> float:
        """Get the learned bias term."""
        return self.trainer.get_bias()
    
    @classmethod
    def load_model(cls, model_path: str, device: str = "cpu") -> Tuple[Union[InterpretabilityBinaryProbe, Pipeline], Dict]:
        """Load a trained model."""
        
        model_path = Path(model_path)
        
        # Try to determine model type from metadata
        metadata_path = model_path.with_suffix(".json")
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            model_type = metadata.get("model_type", "unknown")
            
            if model_type == "sklearn_logistic_regression":
                # Load sklearn pipeline
                pipeline = joblib.load(model_path)
                logger.info(f"✅ Loaded sklearn pipeline from {model_path}")
                return pipeline, metadata
            
            elif model_type == "pytorch_interpretability_probe":
                # Load PyTorch model
                model_data = torch.load(model_path, map_location=device)
                model_config = model_data["model_config"]
                model = InterpretabilityBinaryProbe(
                    model_config["input_size"],
                    bias=model_config.get("bias", True),
                    normalize_weights=model_config.get("normalize_weights", True)
                )
                model.load_state_dict(model_data["model_state_dict"])
                model.to(device)
                model.eval()
                logger.info(f"✅ Loaded PyTorch probe from {model_path}")
                return model, model_data["metadata"]
        
        # Fallback: try to load as PyTorch first, then sklearn
        try:
            model_data = torch.load(model_path, map_location=device)
            if "model_state_dict" in model_data:
                model_config = model_data["model_config"]
                model = InterpretabilityBinaryProbe(model_config["input_size"])
                model.load_state_dict(model_data["model_state_dict"])
                model.to(device)
                model.eval()
                logger.info(f"✅ Loaded PyTorch probe (fallback) from {model_path}")
                return model, model_data.get("metadata", {})
        except Exception:
            pass
        
        try:
            pipeline = joblib.load(model_path)
            logger.info(f"✅ Loaded sklearn pipeline (fallback) from {model_path}")
            return pipeline, {}
        except Exception:
            pass
        
        raise ValueError(f"❌ Could not load model from {model_path}")