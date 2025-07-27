"""
Configuration classes for SAD binary token training.
"""

import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class SADBinaryTrainingConfig:
    """Configuration for SAD binary token-level training (deployment vs evaluation)."""
    
    # Dataset configuration
    deploy_data_path: str  # Path to deployment data (JSONL)
    eval_data_path: str    # Path to evaluation data (JSONL)
    max_length: int = 512
    
    # Model configuration  
    model_name: str = "Qwen/Qwen3-32B"
    hook_point: str = "blocks.15.hook_resid_pre" 
    hook_layer: int = 15
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    dtype: str = "bfloat16"  # "float32", "float16", "bfloat16"
    
    # Training configuration
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 20
    weight_decay: float = 0.01
    train_ratio: float = 0.8
    handle_class_imbalance: bool = True
    
    # Optimization settings
    optimizer_type: str = "AdamW"  # "Adam", "AdamW", "SGD"
    scheduler_type: str = "cosine"  # "cosine", "linear", "constant"
    warmup_ratio: float = 0.1
    gradient_clip_norm: float = 1.0
    
    # Caching configuration
    cache_dir: Optional[str] = None
    activation_batch_size: int = 16  # For activation collection
    force_recache: bool = False
    
    # Output configuration
    output_dir: str = "./trained_probes/sad_binary"
    probe_name: Optional[str] = None
    save_checkpoints: bool = True
    checkpoint_every: int = 5  # Save every N epochs
    
    # Logging and debugging
    verbose: bool = True
    log_every: int = 10  # Log every N batches
    eval_every: int = 1   # Evaluate every N epochs
    
    # Probe method configuration
    probe_method: str = "pytorch"  # "sklearn" or "pytorch"
    
    # Sklearn-specific options
    sklearn_C: float = 1.0  # Regularization parameter
    sklearn_C_sweep: bool = False  # Whether to do regularization sweep
    sklearn_C_values: Optional[List[float]] = None  # Custom C values for sweep
    sklearn_solver: str = "liblinear"  # "liblinear", "newton-cg", "lbfgs", etc.
    sklearn_max_iter: int = 1000  # Maximum iterations
    
    # PyTorch-specific options  
    pytorch_bias: bool = True  # Whether to use bias in linear layer
    pytorch_normalize_weights: bool = True  # Whether to normalize probe directions

    enable_thinking: bool = False  # Whether to enable thinking mode for Qwen3
    model_family: str = field(default="auto")  # "llama", "qwen", or "auto"
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Validate paths
        if not Path(self.deploy_data_path).exists():
            raise FileNotFoundError(f"Deployment data not found: {self.deploy_data_path}")
        if not Path(self.eval_data_path).exists():
            raise FileNotFoundError(f"Evaluation data not found: {self.eval_data_path}")
        
        # Set default cache directory
        if self.cache_dir is None:
            deploy_name = Path(self.deploy_data_path).stem
            eval_name = Path(self.eval_data_path).stem
            model_clean = self.model_name.replace("/", "_").replace("-", "_")
            self.cache_dir = f"./cache/sad_binary/{model_clean}_{deploy_name}_{eval_name}"
        
        # Set default probe name
        if self.probe_name is None:
            deploy_name = Path(self.deploy_data_path).stem
            eval_name = Path(self.eval_data_path).stem
            self.probe_name = f"sad_binary_{deploy_name}_{eval_name}"
        
        # Validate dtype
        if self.dtype not in ["float32", "float16", "bfloat16"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        # Auto-detect model family
        if self.model_family == "auto":
            if "qwen" in self.model_name.lower():
                self.model_family = "qwen"
            elif "llama" in self.model_name.lower():
                self.model_family = "llama"
            else:
                self.model_family = "unknown"
        
        # Validate device
        if self.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            self.device = "cpu"

        # Validate probe method
        if self.probe_method not in ["sklearn", "pytorch"]:
            raise ValueError(f"Invalid probe_method: {self.probe_method}")
        
        # Set sklearn defaults
        if self.sklearn_C_values is None:
            self.sklearn_C_values = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4]
        
        # Adjust defaults based on probe method
        if self.probe_method == "sklearn":
            # Sklearn is fast, so we can afford more evaluation
            self.eval_every = 1
            # Warn about dtype compatibility
            if self.dtype in ['bfloat16', 'float16']:
                print(f"ℹ️  Info: sklearn with {self.dtype} will attempt float16, fallback to float32")
        elif self.probe_method == "pytorch":
            # Keep existing PyTorch defaults
            pass
    
    @property 
    def torch_dtype(self) -> torch.dtype:
        """Convert dtype string to torch dtype."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16, 
            "bfloat16": torch.bfloat16
        }
        return dtype_map[self.dtype]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            field.name: getattr(self, field.name) 
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SADBinaryTrainingConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def get_output_paths(self) -> Dict[str, Path]:
        """Get standardized output paths."""
        output_dir = Path(self.output_dir)
        return {
            "probe": output_dir / f"{self.probe_name}.pt",
            "config": output_dir / f"{self.probe_name}_config.json",
            "metrics": output_dir / f"{self.probe_name}_metrics.json",
            "log": output_dir / f"{self.probe_name}_training.log",
            "checkpoints": output_dir / "checkpoints",
        }