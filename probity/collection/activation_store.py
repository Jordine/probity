import os
import torch
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Dict, Optional, Union
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

    # CRITICAL FIX: Store adjusted positions (accounts for left padding)
    # Dict mapping position_type -> List of positions (one per example)
    # These are the ACTUAL indices into raw_activations after padding adjustment
    adjusted_positions: Optional[Dict[str, List[Union[int, List[int]]]]] = None

    def get_full_sequence_activations(self) -> torch.Tensor:
        """Get activations for complete sequences.
        
        Returns:
            Tensor of shape (num_examples, seq_len, hidden_size)
        """
        return self.raw_activations

    def get_position_activations(self, position_key: str) -> torch.Tensor:
        """Get activations for a specific position key from dataset.

        CRITICAL: Uses adjusted_positions when available to account for left padding.
        The raw_activations tensor is indexed with PADDED positions, so we must use
        the adjusted positions that were computed during batching.

        Args:
            position_key: Key from TokenPositions dictionary

        Returns:
            If single position: (num_examples, hidden_size)
            If multiple positions: (total_positions, hidden_size)
        """
        activations = []
        max_pos = self.raw_activations.shape[1] - 1

        # Use adjusted positions if available (correct behavior)
        if self.adjusted_positions is not None and position_key in self.adjusted_positions:
            adjusted_pos_list = self.adjusted_positions[position_key]
            for store_idx, pos in enumerate(adjusted_pos_list):
                if isinstance(pos, int):
                    if pos <= max_pos:
                        activations.append(self.raw_activations[store_idx, pos])
                else:  # List[int]
                    valid_positions = [p for p in pos if p <= max_pos]
                    activations.extend([self.raw_activations[store_idx, p] for p in valid_positions])
        else:
            # Fallback to original behavior (DEPRECATED - may have position mismatch)
            # This path is kept for backward compatibility with old cached activations
            import warnings
            warnings.warn(
                f"Using unadjusted positions for key '{position_key}'. "
                "This may cause position mismatch with left-padded inputs. "
                "Re-cache activations to fix.",
                DeprecationWarning
            )
            for store_idx, dataset_idx in enumerate(self.example_indices):
                example = self.dataset.examples[dataset_idx]
                if example.token_positions:
                    pos = example.token_positions[position_key]
                    if isinstance(pos, int):
                        if pos <= max_pos:
                            activations.append(self.raw_activations[store_idx, pos])
                    else:  # List[int]
                        valid_positions = [p for p in pos if p <= max_pos]
                        activations.extend([self.raw_activations[store_idx, p] for p in valid_positions])

        return torch.stack(activations)

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

        CRITICAL: Uses adjusted_positions when available to account for left padding.

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

        # Use adjusted positions if available
        use_adjusted = (self.adjusted_positions is not None and
                       position_key in self.adjusted_positions)

        for i, idx in enumerate(self.example_indices):
            if use_adjusted:
                pos = self.adjusted_positions[position_key][i]
            else:
                example = self.dataset.examples[idx]
                if example.token_positions and position_key in example.token_positions.positions:
                    pos = example.token_positions[position_key]
                else:
                    pos = None

            if pos is not None:
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

    def get_token_level_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get full-sequence activations with per-token binary labels.

        For use with token-level training (load_ntml_token_level datasets).
        Each example has 'token_labels' in its attributes: a list of 0/1 per token.

        Returns:
            Tuple of:
            - activations: (num_examples, seq_len, hidden_size)
            - token_labels: (num_examples, seq_len) binary labels per token
            - attention_mask: (num_examples, seq_len) 1 for real tokens, 0 for padding
        """
        seq_len = self.raw_activations.shape[1]
        num_examples = self.raw_activations.shape[0]

        token_labels = torch.zeros(num_examples, seq_len, dtype=torch.float32)
        attention_mask = torch.zeros(num_examples, seq_len, dtype=torch.float32)

        for i, idx in enumerate(self.example_indices):
            example = self.dataset.examples[idx]
            actual_len = int(self.sequence_lengths[i].item())

            # Use stored attention mask from tokenized example if available
            if hasattr(example, 'attention_mask') and example.attention_mask is not None:
                am = example.attention_mask
                for j in range(min(len(am), seq_len)):
                    attention_mask[i, j] = float(am[j])
            else:
                # Reconstruct from sequence length (left-padding: content right-aligned)
                pad_len = seq_len - actual_len
                if pad_len >= 0:
                    attention_mask[i, pad_len:] = 1.0
                else:
                    attention_mask[i, :] = 1.0

            # Get token labels from example attributes
            if hasattr(example, 'attributes') and example.attributes and 'token_labels' in example.attributes:
                labels = example.attributes['token_labels']
                # Find where content starts (first 1 in attention mask)
                content_start = 0
                if hasattr(example, 'attention_mask') and example.attention_mask is not None:
                    for j, m in enumerate(example.attention_mask):
                        if m == 1:
                            content_start = j
                            break
                else:
                    content_start = max(0, seq_len - actual_len)

                # Place labels aligned with content tokens
                n_labels = min(len(labels), seq_len - content_start)
                for j in range(n_labels):
                    token_labels[i, content_start + j] = float(labels[j])

        return self.raw_activations, token_labels, attention_mask

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
            # CRITICAL: Put position_counts tensor on same device as labels
            labels = torch.repeat_interleave(
                self.labels,
                torch.tensor(position_counts, device=self.labels.device)
            )
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
            'hidden_size': self.hidden_size,
            'adjusted_positions': self.adjusted_positions,  # CRITICAL: Save adjusted positions
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