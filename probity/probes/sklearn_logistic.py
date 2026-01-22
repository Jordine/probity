import torch
import numpy as np
from typing import Optional, Literal
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .base import BaseProbe
from .config import SklearnLogisticProbeConfig

# Define the literal type for solver more precisely
SolverLiteral = Literal[
    "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
]


class SklearnLogisticProbe(BaseProbe[SklearnLogisticProbeConfig]):
    """Logistic regression probe using scikit-learn. Handles its own standardization internally."""

    # Explicitly type hint buffers and internal models
    unscaled_coef_: Optional[torch.Tensor]
    intercept_: Optional[torch.Tensor]
    scaler: Optional[StandardScaler]
    model: LogisticRegression

    def __init__(self, config: SklearnLogisticProbeConfig):
        super().__init__(config)
        # Store scaler and model internally
        self.scaler = StandardScaler() if config.standardize else None
        # Validate solver type
        solver: SolverLiteral = (
            config.solver if config.solver in SolverLiteral.__args__ else "lbfgs"
        )
        if solver != config.solver:
            print(
                f"Warning: Invalid solver '{config.solver}' specified. Using default 'lbfgs'. Valid options: {SolverLiteral.__args__}"
            )

        self.model = LogisticRegression(
            max_iter=config.max_iter,
            random_state=config.random_state,
            fit_intercept=config.bias,
            solver=solver,
            C=getattr(config, 'C', 1.0),
            class_weight=getattr(config, 'class_weight', None),
        )
        # Store the final, unscaled coefficients and intercept as tensors
        # Initialize buffers as None; they will be populated by fit() or load()
        self.register_buffer("unscaled_coef_", None, persistent=True)
        self.register_buffer("intercept_", None, persistent=True)

        # Flag for DirectionalProbeTrainer compatibility
        self.has_fit: bool = False

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Fit the probe using sklearn's LogisticRegression.
        Stores unscaled coefficients internally.
        Input x is expected to be in the original activation space for this fit method.
        """
        x_np = x.cpu().numpy().astype(np.float32)
        y_np = y.cpu().numpy()
        if y_np.ndim > 1:
            y_np = y_np.squeeze()  # Ensure y is 1D

        # Apply internal standardization if requested
        if self.scaler is not None:
            try:
                x_np_scaled = self.scaler.fit_transform(x_np)
            except ValueError as e:
                # Handle cases where scaler fails (e.g., constant features)
                print(
                    f"Warning: StandardScaler failed during fit: {e}. Proceeding without scaling."
                )
                x_np_scaled = x_np
                self.scaler = None  # Disable scaler if it failed
        else:
            x_np_scaled = x_np

        # Fit logistic regression on potentially scaled data
        try:
            self.model.fit(x_np_scaled, y_np)
        except Exception as e:
            raise RuntimeError(f"Sklearn LogisticRegression fitting failed: {e}")

        # Get coefficients (potentially scaled) and intercept
        coef_ = (
            self.model.coef_.squeeze()
            if self.model.coef_.shape[0] == 1
            else self.model.coef_
        )
        intercept_ = self.model.intercept_

        # Unscale coefficients if internal standardization was used *and successful*
        if (
            self.scaler is not None
            and hasattr(self.scaler, "scale_")
            and self.scaler.scale_ is not None
        ):
            # Ensure scaler has 'scale_' attribute before accessing
            scale_ = self.scaler.scale_
            # Add epsilon to avoid division by zero or very small numbers
            coef_unscaled = coef_ / (scale_ + 1e-8)
        else:
            coef_unscaled = coef_

        # Store unscaled coefficients and intercept as tensors (buffers)
        # Convert numpy arrays to tensors on the correct device and dtype
        coef_tensor = torch.tensor(
            coef_unscaled, dtype=self.dtype, device=self.config.device
        )
        intercept_tensor = torch.tensor(
            intercept_, dtype=self.dtype, device=self.config.device
        )

        # Update buffers using _set_raw... or directly via setattr/copy_
        self._set_raw_direction_representation(coef_tensor)
        # Update intercept buffer separately
        if self.intercept_ is None:
            self.register_buffer("intercept_", intercept_tensor.clone())
        else:
            with torch.no_grad():
                self.intercept_.copy_(intercept_tensor)

        # Return direction for DirectionalProbeTrainer compatibility
        return coef_tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict logits using the learned coefficients. Input x is in original activation space."""
        if self.unscaled_coef_ is None:
            raise RuntimeError(
                "SklearnLogisticProbe must be fitted or loaded before calling forward."
            )

        # Forward pass uses the stored unscaled coefficients and intercept
        x = x.to(dtype=self.dtype)
        # Ensure coef and intercept are tensors before use
        coef = self.unscaled_coef_
        intercept = self.intercept_

        if not isinstance(coef, torch.Tensor):
            raise RuntimeError("unscaled_coef_ is not a valid tensor.")

        # Calculate logits: (x @ coef^T) + intercept
        # Need to handle potential shape mismatch between x and coef (e.g., [B, D] vs [D] or [C, D])
        if coef.dim() == 1:
            # Binary classification case: coef is [D]
            logits = torch.matmul(x, coef)
        elif coef.dim() == 2 and coef.shape[0] == 1:
            # Binary classification case (sklearn default): coef is [1, D]
            logits = torch.matmul(x, coef.squeeze(0))
        elif coef.dim() == 2 and x.dim() >= 2:
            # Multi-class classification: coef is [C, D], x is [..., D]
            # Result should be [..., C]
            logits = torch.matmul(x, coef.t())
        else:
            raise RuntimeError(
                f"Unexpected shape combination for forward pass: x {x.shape}, coef {coef.shape}"
            )

        if intercept is not None:
            if isinstance(intercept, torch.Tensor):
                # Ensure intercept shape aligns with logits shape for broadcasting
                # Logits: [B] or [..., C]. Intercept: [1] or [C]
                try:
                    logits += intercept
                except RuntimeError as e:
                    raise RuntimeError(
                        f"Error adding intercept during forward pass. Logits shape: {logits.shape}, Intercept shape: {intercept.shape}. Original error: {e}"
                    )
            else:
                raise RuntimeError("intercept_ is not a valid tensor.")

        return logits

    def _get_raw_direction_representation(self) -> torch.Tensor:
        """Return the stored unscaled coefficients."""
        if self.unscaled_coef_ is None:
            # Return zero vector if not fitted/loaded, consistent with DirectionalProbe
            print(
                f"Warning: Accessing direction for probe {self.name} before fit/load. Returning zero vector."
            )
            input_size = getattr(self.config, "input_size", None)
            device = getattr(self.config, "device", "cpu")
            if input_size is None:
                raise ValueError("Cannot determine input_size to create zero vector.")
            return torch.zeros(input_size, dtype=self.dtype, device=device)
        # Ensure it's a tensor before returning
        if not isinstance(self.unscaled_coef_, torch.Tensor):
            raise RuntimeError(
                "_get_raw_direction_representation: unscaled_coef_ is not a Tensor"
            )
        return self.unscaled_coef_

    def _set_raw_direction_representation(self, vector: torch.Tensor) -> None:
        """Set the unscaled coefficients. Used primarily for loading."""
        # vector should already be unscaled when loading

        # Get target device and dtype
        device = getattr(self.config, "device", "cpu")
        dtype = self.dtype

        # Ensure vector is on correct device and dtype
        vector = vector.to(device=device, dtype=dtype)

        # Check shape if buffer already exists
        if hasattr(self, "unscaled_coef_") and self.unscaled_coef_ is not None:
            if self.unscaled_coef_.shape != vector.shape:
                # Attempt common reshape scenarios (e.g., [1, dim] vs [dim])
                try:
                    target_shape = self.unscaled_coef_.shape
                    print(
                        f"Warning: Reshaping loaded vector from {vector.shape} to match existing buffer shape {target_shape}"
                    )
                    vector = vector.reshape(target_shape)
                except Exception as e:
                    raise ValueError(
                        f"Shape mismatch loading vector for {self.name}. Buffer shape: {self.unscaled_coef_.shape}, Loaded vector shape: {vector.shape}. Reshape failed: {e}"
                    )

        # Update or register the buffer
        if not hasattr(self, "unscaled_coef_") or self.unscaled_coef_ is None:
            self.register_buffer("unscaled_coef_", vector.clone())
        else:
            with torch.no_grad():
                self.unscaled_coef_.copy_(vector)

    def get_direction(self, normalized: bool = True) -> torch.Tensor:
        """Get the probe direction (unscaled coefficients), applying normalization."""
        # Direction is already unscaled
        direction = self._get_raw_direction_representation().clone()

        # Handle potential multi-class case (coef shape [C, D])
        # For multi-class, we might return the matrix or require selecting a class direction
        # Current implementation normalizes each row independently if normalize_weights=True
        is_multiclass = direction.dim() == 2 and direction.shape[0] > 1

        # Normalize if requested and configured
        should_normalize = normalized and self.config.normalize_weights
        if should_normalize:
            if is_multiclass:
                # Normalize each class direction (row) independently
                norms = torch.norm(direction, dim=1, keepdim=True)
                direction = direction / (norms + 1e-8)
            else:
                # Normalize the single direction vector (potentially squeezing [1,D] to [D])
                if direction.dim() == 2 and direction.shape[0] == 1:
                    direction = direction.squeeze(0)
                norm = torch.norm(direction)
                if norm > 1e-8:
                    direction = direction / norm
                else:
                    direction = torch.zeros_like(direction)
        else:
            # If not normalizing, still squeeze the [1, D] case for consistency?
            # Let's keep it as [C, D] or [1, D] or [D] as determined by _get_raw...
            pass

        return direction
