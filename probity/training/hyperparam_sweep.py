import itertools
from typing import Dict, List, Any, Union
from dataclasses import dataclass

@dataclass
class HyperparamSweep:
    """Defines hyperparameter sweep ranges."""
    param_ranges: Dict[str, Union[List[Any], Any]]
    
    def generate_configs(self):
        """Generate all combinations of hyperparameters."""
        sweep_params = {k: v for k, v in self.param_ranges.items() 
                       if isinstance(v, list)}
        fixed_params = {k: v for k, v in self.param_ranges.items() 
                       if not isinstance(v, list)}
        
        if not sweep_params:
            yield fixed_params
        else:
            keys, values = zip(*sweep_params.items())
            for combo in itertools.product(*values):
                config = fixed_params.copy()
                config.update(dict(zip(keys, combo)))
                yield config

# Example sweep configurations
SWEEP_CONFIGS = {
    'attention': {
        'n_heads': [1, 2, 4, 8],
        'temperature': [0.5, 1.0, 2.0],
        'use_position_weights': [True, False],
    },
    'mlp': {
        'hidden_dims': [[256], [256, 128], [512, 256, 128]],
        'dropout': [0.0, 0.1, 0.3],
        'activation': ['relu', 'gelu'],
    },
    'logistic': {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0],
        'solver': ['lbfgs', 'liblinear'],
    }
}