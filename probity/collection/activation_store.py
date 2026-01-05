import os
import torch
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict
from probity.datasets.tokenized import TokenizedProbingDataset

@dataclass
class ActivationStore:
    """Stores and provides access to model activations."""
    
    # Core storage
    raw_activations: torch.Tensor  # Shape: (num_examples, seq_len, hidden_size)
    hook_point: str  # Which part of model these came from
    
    # Dataset information
    labels: torch.Tensor  # Shape: (num_examples,) numeric labels
    label_texts: List[str]  # Original text labels
    example_indices: torch.Tensor  # Shape: (num_examples,) maps to dataset indices
    sequence_lengths: torch.Tensor  # Shape: (num_examples,) actual lengths before padding
    hidden_size: int
    
    # Keep reference to original dataset for position lookups
    dataset: TokenizedProbingDataset

    def get_full_sequence_activations(self) -> torch.Tensor:
        """Get activations for complete sequences.
        
        Returns:
            Tensor of shape (num_examples, seq_len, hidden_size)
        """
        return self.raw_activations

    def get_position_activations(self, position_key: str) -> torch.Tensor:
        """Get activations for a specific position key from dataset.
        
        Args:
            position_key: Key from TokenPositions dictionary
            
        Returns:
            If single position: (num_examples, hidden_size)
            If multiple positions: (total_positions, hidden_size)
        """
        positions = []

    
        # print(f"DEBUG [Actual Training Tokens]:")
        
        # Check first 5 examples
        # for i, idx in enumerate(self.example_indices[:5]):
        #     example = self.dataset.examples[idx]
        #     if example.token_positions:
        #         pos = example.token_positions[position_key]
        #         if isinstance(pos, int):
        #             # Get the actual token at this position
        #             if hasattr(example, 'tokens') and pos < len(example.tokens):
        #                 token = example.tokens[pos]
        #                 print(f"  Example {i}: Training on token position {pos}: '{token}'")
        
        # print(f"DEBUG [Activation Extraction] Getting positions for key: {position_key}")

        for store_idx, dataset_idx in enumerate(self.example_indices):
            example = self.dataset.examples[dataset_idx]
            if example.token_positions:
                pos = example.token_positions[position_key]
                # print(f"  Example {dataset_idx}:")
                # print(f"    Position value: {pos}")
                # print(f"    Type: {type(pos)}")

                if isinstance(pos, int):
                    positions.append(self.raw_activations[store_idx, pos])
                else:  # List[int]
                    positions.extend([self.raw_activations[store_idx, p] for p in pos])
        
        return torch.stack(positions)

    def get_activations_by_fn(self, position_fn: Callable) -> torch.Tensor:
        """Get activations at positions specified by a function.
        
        Args:
            position_fn: Function that takes example and returns position(s)
            
        Returns:
            Tensor of gathered activations
        """
        # Stub for now
        raise NotImplementedError
    

    def get_probe_data_with_spans(self, position_key: str) -> Tuple[torch.Tensor, torch.Tensor, List[List[Tuple[int, int]]], List[int]]:
        """Get activations with span boundaries for max aggregation training.

        Args:
            position_key: Key from TokenPositions dictionary

        Returns:
            Tuple of:
            - full_activations: (num_examples, seq_len, hidden_size)
            - labels: (num_examples,) label per example
            - spans: List[List[Tuple[int, int]]] - span boundaries per example
            - span_labels: List[int] - label for each span (same as example label)
        """
        spans_per_example = []
        span_labels = []

        for i, idx in enumerate(self.example_indices):
            example = self.dataset.examples[idx]
            if example.token_positions and position_key in example.token_positions.positions:
                pos = example.token_positions[position_key]
                if isinstance(pos, list) and len(pos) > 0:
                    # Convert list of positions to (start, end) span
                    start = min(pos)
                    end = max(pos)
                    spans_per_example.append([(start, end)])
                    span_labels.append(int(self.labels[i].item()))
                elif isinstance(pos, int):
                    spans_per_example.append([(pos, pos)])
                    span_labels.append(int(self.labels[i].item()))
                else:
                    spans_per_example.append([])
            else:
                spans_per_example.append([])

        return self.raw_activations, self.labels, spans_per_example, span_labels

    def get_probe_data(self, position_key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get activations and labels formatted for probe training.
        
        Args:
            position_key: Key from TokenPositions dictionary
            
        Returns:
            Tuple of (activations, labels) where:
            - For single positions: (num_examples, hidden_size), (num_examples,)
            - For multiple positions: (total_positions, hidden_size), (total_positions,)
              Labels are repeated for examples with multiple positions
        """
        activations = self.get_position_activations(position_key)
        # Detailed debug
        print(f"DEBUG [get_probe_data]:")
        print(f"  Dataset has {len(self.dataset.examples)} examples")
        print(f"  Returning {activations.shape[0]} activation vectors")
        print(f"  Ratio: {activations.shape[0] / len(self.dataset.examples):.2f}x")
        
        # Check if we're getting multiple positions per example
        if activations.shape[0] == len(self.dataset.examples):
            print("  ⚠️ WARNING: Only one activation per example - likely still using first token only!")
        else:
            avg_tokens = activations.shape[0] / len(self.dataset.examples)
            print(f"  ✓ Average {avg_tokens:.1f} tokens per statement")
        
        
        # Handle label replication for multiple positions
        if len(activations) > len(self.labels):
            # Count positions per example to know how many times to repeat each label
            position_counts = [
                len(ex.token_positions[position_key]) if isinstance(ex.token_positions[position_key], list)
                else 1 
                for ex in self.dataset.examples
            ]
            labels = torch.repeat_interleave(self.labels, torch.tensor(position_counts))
        else:
            labels = self.labels
            
        return activations, labels

    def save(self, path: str) -> None:
        """Save cache to disk."""
        os.makedirs(path, exist_ok=True)
        torch.save({
            'raw_activations': self.raw_activations,
            'hook_point': self.hook_point,
            'labels': self.labels,
            'label_texts': self.label_texts,
            'example_indices': self.example_indices,
            'sequence_lengths': self.sequence_lengths,
            'hidden_size': self.hidden_size
        }, os.path.join(path, 'cache.pt'))
        
        # Save dataset separately since it contains complex objects
        self.dataset.save(os.path.join(path, 'dataset'))

    @classmethod
    def load(cls, path: str) -> 'ActivationStore':
        """Load cache from disk."""
        cache_data = torch.load(os.path.join(path, 'cache.pt'))
        dataset = TokenizedProbingDataset.load(os.path.join(path, 'dataset'))
        
        return cls(
            dataset=dataset,
            **cache_data
        )