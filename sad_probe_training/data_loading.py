"""
Data loading and processing for SAD binary token training.

Converts SAD deployment vs evaluation datasets into binary token-level training data.
"""

import torch
import json
import zipfile
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
from transformers import AutoTokenizer
import logging

# Import the BinaryTokenExample from NTML (reuse the data structure)
import sys
ntml_path = Path(__file__).parent.parent / "ntml_efficient_scripts"
sys.path.insert(0, str(ntml_path))

import data_loading as ntml_data_loading
BinaryTokenExample = ntml_data_loading.BinaryTokenExample
from .config import SADBinaryTrainingConfig

logger = logging.getLogger(__name__)


class SADBinaryDataset:
    """Dataset class for SAD binary token training (deployment vs evaluation)."""
    
    def __init__(self, config: SADBinaryTrainingConfig):
        self.config = config
        self.tokenizer = None
        self.examples: List[BinaryTokenExample] = []
        self.label_distribution: Dict[str, int] = {}
        
    def load_and_process(self) -> "SADBinaryDataset":
        """Load SAD dataset and convert to binary token format."""
        logger.info(f"Loading SAD datasets:")
        logger.info(f"  Deployment data: {self.config.deploy_data_path}")
        logger.info(f"  Evaluation data: {self.config.eval_data_path}")
        
        # Load data from both sources
        deploy_data = self._load_jsonl_data(self.config.deploy_data_path, "deploy")
        eval_data = self._load_jsonl_data(self.config.eval_data_path, "eval")
        
        logger.info(f"Loaded {len(deploy_data)} deployment samples and {len(eval_data)} evaluation samples")
        
        # Initialize tokenizer
        logger.info(f"Loading tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Process both datasets
        logger.info("Converting to binary token format...")
        total_tokens = 0
        
        # Process deployment data (label = 0)
        for i, item in enumerate(deploy_data):
            binary_example = self._process_sad_item(item, label=0, item_id=f"deploy_{i}")
            if binary_example is not None:
                self.examples.append(binary_example)
                total_tokens += len(binary_example.tokens)
        
        # Process evaluation data (label = 1) 
        for i, item in enumerate(eval_data):
            binary_example = self._process_sad_item(item, label=1, item_id=f"eval_{i}")
            if binary_example is not None:
                self.examples.append(binary_example)
                total_tokens += len(binary_example.tokens)
        
        logger.info(f"Created {len(self.examples)} binary token examples")
        logger.info(f"Total tokens: {total_tokens}")
        
        # Calculate label distribution
        self._calculate_label_distribution()
        
        return self
    
    def _load_jsonl_data(self, file_path: str, data_type: str = "deploy") -> List[Dict]:
        """Load data from JSONL file, handling both regular files and ZIP files.
        
        Args:
            file_path: Path to the data file
            data_type: Either "deploy" or "eval" - used for ZIP files to determine internal file
        """
        file_path = Path(file_path)
        
        if file_path.suffix == '.zip':
            # Handle ZIP files (like the SAD dataset)
            return self._load_jsonl_from_zip(file_path, data_type)
        else:
            # Handle regular JSONL files
            with open(file_path, 'r') as f:
                return [json.loads(line) for line in f if line.strip()]
    
    def _load_jsonl_from_zip(self, zip_path: Path, data_type: str = "deploy", password: bytes = b'sadtimesforthesetimes') -> List[Dict]:
        """Load JSONL data from ZIP file with password protection.
        
        Args:
            zip_path: Path to the ZIP file
            data_type: Either "deploy" or "eval" to specify which internal file to load
            password: Password for the ZIP file
        """
        # Determine the internal file name based on data type
        if data_type == "deploy":
            internal_file = 'out/deploy_oversight.jsonl'
        elif data_type == "eval":
            internal_file = 'out/test_oversight.jsonl'
        else:
            # Fallback: try to infer from the zip path
            if 'deploy' in str(zip_path).lower():
                internal_file = 'out/deploy_oversight.jsonl'
            else:
                internal_file = 'out/test_oversight.jsonl'
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            content = zip_ref.read(internal_file, pwd=password).decode('utf-8')
            return [json.loads(line) for line in content.strip().split('\n')]
    
    def _process_sad_item(self, item: Dict, label: int, item_id: str) -> Optional[BinaryTokenExample]:
        """Convert a single SAD item to binary token format."""
        
        # Extract the body text from the SAD item
        body_text = item.get('body', '')
        if not body_text:
            logger.warning(f"Empty body text for item {item_id}")
            return None
        
        # Create chat messages format
        is_qwen = 'qwen' in self.config.model_name.lower()
        
        messages = [
            {"role": "system", "content": ""},  # Empty system prompt as in example
            {"role": "user", "content": body_text}
        ]
        
        # Apply chat template to get properly formatted text
        try:
            if is_qwen:
                full_text = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=self.config.enable_thinking
                )
            else:
                # Llama format or other models
                full_text = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False,
                    add_generation_prompt=False
                )
        except Exception as e:
            logger.warning(f"Failed to apply chat template for item {item_id}: {e}")
            # Fallback to simple concatenation
            full_text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{body_text}<|eot_id|>"
        
        # Tokenize the full text
        try:
            encoding = self.tokenizer(
                full_text,
                max_length=self.config.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False  # Chat template already adds special tokens
            )
        except Exception as e:
            logger.warning(f"Failed to tokenize item {item_id}: {e}")
            return None
        
        tokens = encoding["input_ids"][0].tolist()
        attention_mask = encoding["attention_mask"][0].tolist()
        
        # For SAD dataset, ALL tokens get the same label (deployment=0, evaluation=1)
        # Create labels and assistant_mask
        token_labels = [label] * len(tokens)
        
        # For simplicity, treat all tokens as "assistant" tokens for training
        # This means we'll train the probe on all tokens in the sequence
        assistant_mask = attention_mask.copy()  # Use attention mask as assistant mask
        
        # Get token texts for debugging
        token_texts = None
        if self.config.verbose:
            token_texts = [self.tokenizer.decode([t]) for t in tokens]
        
        return BinaryTokenExample(
            tokens=tokens,
            labels=token_labels,
            attention_mask=attention_mask,
            assistant_mask=assistant_mask,
            conversation_id=item_id,
            statement_count=1,  # Treat each item as one "statement"
            truth_count=1 if label == 0 else 0,
            lie_count=1 if label == 1 else 0,
            token_texts=token_texts,
            statement_assignments=[0] * len(tokens)  # All tokens belong to statement 0
        )
    
    def _calculate_label_distribution(self):
        """Calculate distribution of deployment/evaluation labels."""
        total_tokens = 0
        deploy_tokens = 0  # label = 0
        eval_tokens = 0    # label = 1
        
        for example in self.examples:
            for i, (is_active, label) in enumerate(zip(example.assistant_mask, example.labels)):
                if is_active:  # Only count active tokens
                    total_tokens += 1
                    if label == 0:
                        deploy_tokens += 1
                    else:
                        eval_tokens += 1
        
        self.label_distribution = {
            "total_tokens": total_tokens,
            "deploy_tokens": deploy_tokens,
            "eval_tokens": eval_tokens,
            "deploy_ratio": deploy_tokens / total_tokens if total_tokens > 0 else 0,
            "eval_ratio": eval_tokens / total_tokens if total_tokens > 0 else 0,
        }
        
        logger.info(f"Label distribution: {self.label_distribution}")
    
    def get_training_data(self, train_ratio: float = 0.8) -> Tuple[List[BinaryTokenExample], List[BinaryTokenExample]]:
        """Split dataset into training and validation sets."""
        
        # Shuffle examples
        indices = np.random.permutation(len(self.examples))
        split_idx = int(len(self.examples) * train_ratio)
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_examples = [self.examples[i] for i in train_indices]
        val_examples = [self.examples[i] for i in val_indices]
        
        logger.info(f"Training examples: {len(train_examples)}, Validation examples: {len(val_examples)}")
        
        return train_examples, val_examples
    
    def save_debug_info(self, output_dir: str):
        """Save debugging information about tokenization and labeling."""
        
        if not self.config.verbose:
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        debug_data = []
        for example in self.examples[:5]:  # Save first 5 examples
            debug_entry = {
                "conversation_id": example.conversation_id,
                "statement_count": example.statement_count,
                "truth_count": example.truth_count,
                "lie_count": example.lie_count,
                "tokens_and_labels": []
            }
            
            for i, (token, label, is_active, stmt_assign, token_text) in enumerate(zip(
                example.tokens, example.labels, example.assistant_mask, 
                example.statement_assignments, example.token_texts or [""] * len(example.tokens)
            )):
                debug_entry["tokens_and_labels"].append({
                    "index": i,
                    "token_id": token,
                    "token_text": token_text,
                    "label": label,
                    "is_active": bool(is_active),
                    "statement_assignment": stmt_assign
                })
            
            debug_data.append(debug_entry)
        
        import json
        with open(output_path / "tokenization_debug.json", "w") as f:
            json.dump(debug_data, f, indent=2)
        
        # Save label distribution
        with open(output_path / "label_distribution.json", "w") as f:
            json.dump(self.label_distribution, f, indent=2)


def prepare_sad_training_data(config: SADBinaryTrainingConfig) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Main function to prepare training data from SAD dataset.
    
    Returns:
        activations: Activations for all active tokens [num_tokens, hidden_size]
        labels: Binary labels for all active tokens [num_tokens]
        metadata: Dictionary with dataset and processing information
    """
    
    # Load and process dataset
    logger.info("Loading SAD dataset...")
    dataset = SADBinaryDataset(config)
    dataset.load_and_process()
    
    # Save debug info if requested
    if config.verbose:
        debug_dir = Path(config.output_dir) / "debug"
        dataset.save_debug_info(str(debug_dir))
    
    # Use the existing activation collection infrastructure
    # Import the activation preparation function from NTML
    from activation_utils import prepare_ntml_training_data
    
    # Create a minimal adapter to make NTML infrastructure work with SAD data
    # We'll monkey-patch the dataset into the expected format
    class SADAdapter:
        def __init__(self, sad_dataset):
            self.examples = sad_dataset.examples
            self.label_distribution = sad_dataset.label_distribution
    
    # Create adapter and use existing infrastructure
    sad_adapter = SADAdapter(dataset)
    
    # Create a modified config that matches NTML format expectations
    ntml_config = type('Config', (), {
        'dataset_path': config.deploy_data_path,  # Just for caching key
        'model_name': config.model_name,
        'hook_point': config.hook_point,
        'hook_layer': config.hook_layer,
        'device': config.device,
        'torch_dtype': config.torch_dtype,
        'max_length': config.max_length,
        'activation_batch_size': config.activation_batch_size,
        'force_recache': config.force_recache,
        'cache_dir': config.cache_dir,
        'output_dir': config.output_dir,
        'verbose': config.verbose,
        'dtype': config.dtype,
        'enable_thinking': config.enable_thinking,
        'model_family': config.model_family,
    })()




    # Monkey-patch the dataset loading to return our SAD data
    original_load = None
    try:
        import data_loading as ntml_data_loading
        original_load = ntml_data_loading.NTMLBinaryDataset
        ntml_data_loading.NTMLBinaryDataset = lambda cfg: type('Dataset', (), {
            'examples': sad_adapter.examples,
            'label_distribution': sad_adapter.label_distribution,
            'load_and_process': lambda: None,
            'save_debug_info': lambda path: None,
        })()
    
        
        # Use existing preparation function
        activations, labels, metadata = prepare_ntml_training_data(ntml_config)
        
        return activations, labels, metadata
        
    finally:
        # Restore original function
        if original_load:
            ntml_data_loading.NTMLBinaryDataset = original_load