from typing import Optional, List, Dict, Set, Union, cast, Any
import dataclasses
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from .position_finder import Position
from dataclasses import dataclass, field
import json
import torch
from datasets import Dataset
from .base import ProbingDataset, ProbingExample
from .position_finder import PositionFinder


@dataclass
class TokenizationConfig:
    """Stores information about how the dataset was tokenized."""

    tokenizer_name: str
    tokenizer_kwargs: Dict
    vocab_size: int
    pad_token_id: Optional[int]
    eos_token_id: Optional[int]
    bos_token_id: Optional[int] = None  # Add BOS token ID
    padding_side: str = "right"  # "right" or "left" padding


@dataclass
class TokenPositions:
    """Container for token positions of interest."""

    positions: Dict[str, Union[int, List[int]]]

    def __getitem__(self, key: str) -> Union[int, List[int]]:
        return self.positions[key]

    def keys(self) -> Set[str]:
        return set(self.positions.keys())


@dataclass
class TokenizedProbingExample(ProbingExample):
    """Extends ProbingExample with tokenization information."""

    tokens: List[int] = field(default_factory=list)
    attention_mask: Optional[List[int]] = None
    token_positions: Optional[TokenPositions] = None


class TokenizedProbingDataset(ProbingDataset):
    """ProbingDataset with tokenization information and guarantees."""

    def __init__(
        self,
        examples: List[TokenizedProbingExample],
        tokenization_config: TokenizationConfig,
        position_types: Optional[Set[str]] = None,
        **kwargs,
    ):
        # Call parent constructor with the examples
        super().__init__(examples=examples, **kwargs)  # type: ignore
        self.tokenization_config = tokenization_config
        self.position_types = position_types or set()

        # Validate that all examples are properly tokenized
        self._validate_tokenization()

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        # Delegate to the underlying list of examples
        return len(self.examples)

    def _validate_tokenization(self):
        """Ensure all examples have valid tokens."""
        for example in self.examples:
            if not isinstance(example, TokenizedProbingExample):
                raise TypeError(
                    "All examples must be TokenizedProbingExample instances"
                )
            if any(t >= self.tokenization_config.vocab_size for t in example.tokens):
                raise ValueError(f"Invalid token ID found in example: {example.text}")

    def _to_hf_dataset(self) -> Dataset:
        """Override parent method to include tokenization data."""
        # Get base dictionary from parent
        data_dict = super()._to_hf_dataset().to_dict()

        # Add tokenization data
        examples = [cast(TokenizedProbingExample, ex) for ex in self.examples]
        data_dict["tokens"] = [ex.tokens for ex in examples]
        if all(ex.attention_mask is not None for ex in examples):
            data_dict["attention_mask"] = [ex.attention_mask for ex in examples]

        # Save token positions if they exist
        position_keys = set()
        for ex in examples:
            if ex.token_positions:
                position_keys.update(ex.token_positions.keys())

        # Add token position data columns
        for key in position_keys:
            data_dict[f"token_pos_{key}"] = []

        # Fill in token position data
        for ex in examples:
            for key in position_keys:
                if ex.token_positions and key in ex.token_positions.keys():
                    data_dict[f"token_pos_{key}"].append(ex.token_positions[key])
                else:
                    # Add a placeholder for missing positions
                    data_dict[f"token_pos_{key}"].append(None)

        return Dataset.from_dict(data_dict)

    @classmethod
    def from_probing_dataset(
        cls,
        dataset: ProbingDataset,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        add_special_tokens: bool = True,  # Explicitly control special tokens
        **tokenizer_kwargs,
    ) -> "TokenizedProbingDataset":
        """Create TokenizedProbingDataset from ProbingDataset."""
        tokenized_examples = []

        # Get the tokenizer's padding side
        padding_side = getattr(tokenizer, "padding_side", "right")
        is_left_padding = padding_side == "left"

        # Ensure add_special_tokens is in tokenizer_kwargs for consistency
        tokenizer_kwargs["add_special_tokens"] = add_special_tokens

        for example in dataset.examples:
            # First, tokenize without padding to get accurate token positions
            no_padding_kwargs = {
                k: v for k, v in tokenizer_kwargs.items() if k != "padding"
            }
            # Remove return_tensors if present, as we explicitly set it below
            no_padding_kwargs.pop("return_tensors", None)
            no_padding_kwargs["padding"] = False  # Explicitly set padding to False

            # Tokenize the text without padding but with special tokens if requested
            no_pad_tokens_info = tokenizer(
                example.text, return_tensors="pt", **no_padding_kwargs
            )
            no_pad_tokens = no_pad_tokens_info["input_ids"][0].tolist()

            # Now tokenize with all requested kwargs (may include padding)
            # Remove return_tensors if present, as we explicitly set it below
            full_tokenizer_kwargs = tokenizer_kwargs.copy()
            full_tokenizer_kwargs.pop("return_tensors", None)
            tokens_info = tokenizer(
                example.text, return_tensors="pt", **full_tokenizer_kwargs
            )
            padded_tokens = tokens_info["input_ids"][0].tolist()
            attention_mask = (
                tokens_info["attention_mask"][0].tolist()
                if "attention_mask" in tokens_info
                else None
            )

            # Find padding length if left padding is used
            pad_length = 0
            if is_left_padding and attention_mask:
                # Find where the content starts (first 1 in attention mask)
                for i, mask in enumerate(attention_mask):
                    if mask == 1:
                        pad_length = i
                        break

            # Convert character positions to token positions if they exist
            token_positions = None
            if example.character_positions:
                positions = {}
                for key, pos in example.character_positions.positions.items():
                    # Use the no-padding tokens for position conversion to get accurate positions
                    if isinstance(pos, Position):
                        # Check if this is a span (has both start and end) or single position
                        if pos.end is not None and pos.end > pos.start:
                            # It's a span - convert to multiple token positions
                            token_positions_list = PositionFinder.convert_char_span_to_token_positions(
                                pos,
                                example.text,
                                tokenizer,
                                add_special_tokens=add_special_tokens,
                            )
                            token_pos = token_positions_list  # Store as list
                        else:
                            # Single position - use original method
                            token_pos = PositionFinder.convert_to_token_position(
                                pos,
                                example.text,
                                tokenizer,
                                add_special_tokens=add_special_tokens,
                                padding_side=padding_side,
                            )

                    
                        if isinstance(token_pos, list):
                            # print(f"DEBUG [Token Span]: Converted {pos.start}-{pos.end} to {len(token_pos)} tokens")
                            if token_pos and no_pad_tokens:
                                sample_tokens = [tokenizer.convert_ids_to_tokens([no_pad_tokens[tp]])[0] 
                                                for tp in token_pos[:5] if tp < len(no_pad_tokens)]
                                # print(f"  First few tokens: {sample_tokens}")
                        # else:
                            # print(f"DEBUG [Single Token]: Position {pos.start} -> token {token_pos}")


                        
                        # If using left padding, the positions need to be adjusted
                        if is_left_padding and pad_length > 0:
                            if isinstance(token_pos, list):
                                # Adjust all positions in the list
                                positions[key] = [tp + pad_length for tp in token_pos]
                            else:
                                positions[key] = token_pos + pad_length
                        else:
                            positions[key] = token_pos  # Can be int or list

                    else:  # List[Position]
                        # Convert list of positions
                        token_positions = [
                            PositionFinder.convert_to_token_position(
                                p,
                                example.text,
                                tokenizer,
                                add_special_tokens=add_special_tokens,
                                padding_side=padding_side,
                            )
                            for p in pos
                        ]

                        # Adjust for left padding if needed
                        if is_left_padding and pad_length > 0:
                            positions[key] = [tp + pad_length for tp in token_positions]
                        else:
                            positions[key] = token_positions

                token_positions = TokenPositions(positions)

            # Create tokenized example with accurate token positions
            tokenized_example = TokenizedProbingExample(
                text=example.text,
                label=example.label,
                label_text=example.label_text,
                character_positions=example.character_positions,
                token_positions=token_positions,
                group_id=example.group_id,
                attributes=example.attributes,
                tokens=padded_tokens,  # Use the full tokens with padding
                attention_mask=attention_mask,
            )
            tokenized_examples.append(tokenized_example)

        # Create tokenization config with explicit record of special token handling
        tokenization_config = TokenizationConfig(
            tokenizer_name=tokenizer.name_or_path,
            tokenizer_kwargs=tokenizer_kwargs,
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            padding_side=padding_side,
        )

        return cls(
            examples=tokenized_examples,
            tokenization_config=tokenization_config,
            task_type=dataset.task_type,
            valid_layers=dataset.valid_layers,
            label_mapping=dataset.label_mapping,
            position_types=dataset.position_types,
            dataset_attributes=dataset.dataset_attributes,
        )

    def get_token_lengths(self) -> List[int]:
        """Get length of each sequence in tokens."""
        return [len(cast(TokenizedProbingExample, ex).tokens) for ex in self.examples]

    def get_max_sequence_length(self) -> int:
        """Get maximum sequence length in dataset."""
        return max(self.get_token_lengths())

    def validate_positions(self) -> bool:
        """Check if all token positions are valid given the sequence lengths."""
        for example in self.examples:
            tokenized_ex = cast(TokenizedProbingExample, example)
            if tokenized_ex.token_positions is None:
                continue
            seq_len = len(tokenized_ex.tokens)

            # Skip validation for BOS token if it exists
            bos_token_id = self.tokenization_config.bos_token_id
            bos_offset = 0
            if (
                bos_token_id is not None
                and tokenized_ex.tokens
                and tokenized_ex.tokens[0] == bos_token_id
            ):
                bos_offset = 1  # Adjust sequence length to exclude BOS token

            # Note: For un-padded sequences, we don't need to adjust for padding side
            # since positions are stored relative to the unpadded sequence
            for key, pos in tokenized_ex.token_positions.positions.items():
                if isinstance(pos, int):
                    if pos < 0 or pos >= (seq_len - bos_offset):
                        return False
                elif isinstance(pos, list):
                    if any(p < 0 or p >= (seq_len - bos_offset) for p in pos):
                        return False
        return True

    def get_batch_tensors(
        self, indices: List[int], pad: bool = True
    ) -> Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Get a batch of examples as tensors.

        Args:
            indices: Which examples to include in batch
            pad: Whether to pad sequences to max length in batch

        Returns:
            Dictionary with keys 'input_ids', 'attention_mask', and 'positions'
        """
        # Get examples and cast them to TokenizedProbingExample
        examples = [cast(TokenizedProbingExample, self.examples[i]) for i in indices]

        # Get max length for this batch
        max_len = max(len(ex.tokens) for ex in examples)

        # Prepare tensors
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        positions: Dict[str, List[Union[int, List[int]]]] = {
            pt: [] for pt in self.position_types
        }

        # Check padding direction
        is_left_padding = self.tokenization_config.padding_side == "left"

        for ex in examples:
            seq_len = len(ex.tokens)
            padding_length = max_len - seq_len

            if pad:
                if is_left_padding:
                    # Left padding (add pad tokens at the beginning)
                    pad_token = self.tokenization_config.pad_token_id or 0
                    padded_tokens = [pad_token] * padding_length + ex.tokens
                    input_ids.append(padded_tokens)

                    # Left pad the attention mask
                    if ex.attention_mask is not None:
                        padded_mask = [0] * padding_length + ex.attention_mask
                        attention_mask.append(padded_mask)
                else:
                    # Right padding (add pad tokens at the end)
                    pad_token = self.tokenization_config.pad_token_id or 0
                    padded_tokens = ex.tokens + [pad_token] * padding_length
                    input_ids.append(padded_tokens)

                    # Right pad the attention mask
                    if ex.attention_mask is not None:
                        padded_mask = ex.attention_mask + [0] * padding_length
                        attention_mask.append(padded_mask)
            else:
                input_ids.append(ex.tokens)
                if ex.attention_mask is not None:
                    attention_mask.append(ex.attention_mask)

            # Add positions for each position type
            for pt in self.position_types:
                if ex.token_positions is not None and pt in ex.token_positions.keys():
                    token_pos = ex.token_positions[pt]

                    # For left padding, we need to adjust positions when padding
                    # This is critical because the token positions from PositionFinder.convert_to_token_position
                    # are based on unpadded tokens. When using left padding, all token indices need to be
                    # shifted by the padding length to account for pad tokens inserted at the beginning.
                    #
                    # Example:
                    # Original tokens: [BOS, token1, token2, token3]
                    # Original position: 1 (points to token1)
                    # After left padding with 2 tokens: [PAD, PAD, BOS, token1, token2, token3]
                    # Adjusted position: 3 (1 + padding_length of 2)
                    if pad and is_left_padding:
                        if isinstance(token_pos, int):
                            # Shift position by padding length
                            adjusted_pos = token_pos + padding_length
                            positions[pt].append(adjusted_pos)
                        else:  # List of positions
                            # Shift all positions by padding length
                            adjusted_pos = [pos + padding_length for pos in token_pos]
                            positions[pt].append(adjusted_pos)
                    else:
                        # For right padding or no padding, positions stay the same
                        # since right padding adds tokens at the end, which doesn't affect
                        # the indices of the existing tokens
                        positions[pt].append(token_pos)
                else:
                    # Use a default position when position type doesn't exist for this example
                    positions[pt].append(0)

        # Convert to tensors
        batch: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]] = {
            "input_ids": torch.tensor(input_ids),
            "positions": {
                pt: self._flatten_positions_to_tensor(positions[pt]) for pt in self.position_types
            },
        }

        if attention_mask:
            batch["attention_mask"] = torch.tensor(attention_mask)

        return batch


    def _flatten_positions_to_tensor(self, position_list: List[Union[int, List[int]]]) -> torch.Tensor:
        """Flatten variable-length position lists into a single tensor.
        
        For positions that are lists of tokens (from character spans), we flatten them
        so each token becomes its own training example.
        """
        flattened = []
        for pos in position_list:
            if isinstance(pos, list):
                # Add each token position as separate entry
                flattened.extend(pos)
            else:
                # Single position
                flattened.append(pos)
        
        return torch.tensor(flattened, dtype=torch.long)
    
    def save(self, path: str) -> None:
        """Override save to include tokenization info."""
        super().save(path)

        # Save tokenization info
        with open(f"{path}/tokenization_config.json", "w") as f:
            json.dump(dataclasses.asdict(self.tokenization_config), f)

    @classmethod
    def load(cls, path: str) -> "TokenizedProbingDataset":
        """Override load to include tokenization info."""
        # Load base dataset
        base_dataset = ProbingDataset.load(path)

        # Load tokenization info
        with open(f"{path}/tokenization_config.json", "r") as f:
            tokenization_config = TokenizationConfig(**json.load(f))

        # Convert examples to TokenizedProbingExamples
        tokenized_examples = []
        hf_dataset = Dataset.load_from_disk(f"{path}/hf_dataset")

        # Extract position information from the dataset
        position_types = base_dataset.position_types
        position_data = {}

        # Find token position columns if they exist
        token_position_columns = [
            col for col in hf_dataset.column_names if col.startswith("token_pos_")
        ]

        if token_position_columns:
            for col in token_position_columns:
                # Extract the position key from the column name
                key = col.replace("token_pos_", "")
                position_data[key] = hf_dataset[col]

        for i, (base_ex, hf_item) in enumerate(zip(base_dataset.examples, hf_dataset)):
            # Reconstruct token positions only if base example had character positions
            token_positions = None
            if base_ex.character_positions and position_types:
                positions = {}
                # Try to get from token_position columns if they exist
                if position_data:
                    for key in position_types:
                        # Only add if key exists for this example in the loaded data
                        if (
                            key in position_data
                            and i < len(position_data[key])
                            and position_data[key][i] is not None
                        ):
                            positions[key] = position_data[key][i]
                # If we don't have token positions but have character positions,
                # we may need to reconstruct them (or handle appropriately)
                # elif base_ex.character_positions:
                # pass # Reconstruction might be complex/require tokenizer

                # Only create TokenPositions if positions dict is not empty
                if positions:
                    token_positions = TokenPositions(positions)
            # If base_ex.character_positions was None, token_positions remains None

            tokenized_examples.append(
                TokenizedProbingExample(
                    text=base_ex.text,
                    label=base_ex.label,
                    label_text=base_ex.label_text,
                    character_positions=base_ex.character_positions,
                    token_positions=token_positions,
                    group_id=base_ex.group_id,
                    attributes=base_ex.attributes,
                    tokens=cast(Dict[str, Any], hf_item)["tokens"],
                    attention_mask=cast(Dict[str, Any], hf_item).get("attention_mask"),
                )
            )

        dataset = cls(
            examples=tokenized_examples,
            tokenization_config=tokenization_config,
            task_type=base_dataset.task_type,
            valid_layers=base_dataset.valid_layers,
            label_mapping=base_dataset.label_mapping,
            position_types=base_dataset.position_types,
            dataset_attributes=base_dataset.dataset_attributes,
        )

        # Validate that positions are still valid
        if not dataset.validate_positions():
            print(
                "Warning: Some token positions appear to be invalid. "
                + "Call verify_position_tokens to diagnose issues."
            )

        return dataset

    def verify_position_tokens(
        self,
        tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
        position_key: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Verify that positions point to the expected tokens.

        Args:
            tokenizer: Optional tokenizer to decode tokens. If None, uses token IDs.
            position_key: Optional specific position key to verify. If None, verifies all.

        Returns:
            Dictionary mapping example indices to position verification results
        """
        results = {}

        for i, example in enumerate(self.examples):
            tokenized_ex = cast(TokenizedProbingExample, example)
            if tokenized_ex.token_positions is None:
                continue

            # Determine which position keys to check
            keys_to_check = (
                [position_key] if position_key else tokenized_ex.token_positions.keys()
            )
            example_results = {}

            # Get sequence length for validation
            seq_len = len(tokenized_ex.tokens)

            for key in keys_to_check:
                if key not in tokenized_ex.token_positions.keys():
                    continue

                # Get the position for this key
                pos = tokenized_ex.token_positions[key]

                # Handle both single positions and lists of positions
                positions = [pos] if isinstance(pos, int) else pos

                # Check all positions
                for j, position in enumerate(positions):
                    position_key = f"{key}_{j}" if isinstance(pos, list) else key

                    if 0 <= position < seq_len:
                        token_id = tokenized_ex.tokens[position]
                        token_text = (
                            tokenizer.decode([token_id])
                            if tokenizer
                            else f"ID:{token_id}"
                        )
                        example_results[position_key] = {
                            "position": position,
                            "token_id": token_id,
                            "token_text": token_text,
                        }
                    else:
                        example_results[position_key] = {
                            "position": position,
                            "error": f"Position out of bounds (0-{seq_len-1})",
                        }

            if example_results:
                results[i] = example_results

        return results

    def show_token_context(
        self,
        example_idx: int,
        position_key: str,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        context_size: int = 5,
    ) -> Dict[str, Any]:
        """Show the token context around the position of interest for debugging.

        Args:
            example_idx: Index of the example to examine
            position_key: Position key to check
            tokenizer: Tokenizer to decode tokens
            context_size: Number of tokens to show before and after the position

        Returns:
            Dictionary with context information
        """
        if example_idx >= len(self.examples):
            raise ValueError(f"Example index {example_idx} out of range")

        example = cast(TokenizedProbingExample, self.examples[example_idx])
        if (
            example.token_positions is None
            or position_key not in example.token_positions.keys()
        ):
            raise ValueError(
                f"Position key {position_key} not found in example {example_idx}"
            )

        padding_side = self.tokenization_config.padding_side
        is_left_padding = padding_side == "left"

        # Get the position directly from the TokenizedProbingExample
        # This position should already be calculated relative to the *padded* sequence
        # by from_probing_dataset or loaded correctly.
        pos_value = example.token_positions[position_key]

        # Determine the positions to analyze (could be single int or list of ints)
        if isinstance(pos_value, int):
            padded_positions = [pos_value]
        elif isinstance(pos_value, list):
            padded_positions = pos_value
        else:
            raise TypeError(
                f"Unexpected position type for key '{position_key}': {type(pos_value)}"
            )

        # Find the actual tokens (non-padding)
        pad_token_id = self.tokenization_config.pad_token_id or 0
        padding_length = 0
        real_tokens_start_index = 0
        if is_left_padding:
            for i, token_id in enumerate(example.tokens):
                if token_id != pad_token_id:
                    real_tokens_start_index = i
                    break
            else:  # Handle case where sequence is all padding
                real_tokens_start_index = len(example.tokens)
            padding_length = real_tokens_start_index

        real_tokens_end_index = len(example.tokens)
        if not is_left_padding:
            for i in range(len(example.tokens) - 1, -1, -1):
                if example.tokens[i] != pad_token_id:
                    real_tokens_end_index = i + 1
                    break
            else:  # Handle case where sequence is all padding
                real_tokens_end_index = 0

        real_tokens = example.tokens[real_tokens_start_index:real_tokens_end_index]

        # Get the full token list with indices for debugging
        full_token_mapping = [
            (i, token_id, tokenizer.decode([token_id]))
            for i, token_id in enumerate(example.tokens)
        ]

        results = []
        for i, position_in_padded_sequence in enumerate(padded_positions):

            # Calculate original position relative to real tokens
            original_position = -1  # Default if out of bounds
            if is_left_padding:
                if position_in_padded_sequence >= padding_length:
                    original_position = position_in_padded_sequence - padding_length
            else:  # Right padding or no padding
                if position_in_padded_sequence < real_tokens_end_index:
                    original_position = position_in_padded_sequence

            # Get context tokens (working with the full padded sequence)
            start = max(0, position_in_padded_sequence - context_size)
            end = min(
                len(example.tokens), position_in_padded_sequence + context_size + 1
            )

            context_tokens = example.tokens[start:end]
            context_texts = [tokenizer.decode([t]) for t in context_tokens]

            # Mark the target position
            rel_position = (
                position_in_padded_sequence - start
            )  # Position within the context window
            marked_context = context_texts.copy()
            if 0 <= rel_position < len(marked_context):
                marked_context[rel_position] = f"[[ {marked_context[rel_position]} ]]"

            # Get the token at the specified position
            token_info = None
            if 0 <= position_in_padded_sequence < len(example.tokens):
                token_id = example.tokens[position_in_padded_sequence]
                token_text = tokenizer.decode([token_id])
                token_info = {"id": token_id, "text": token_text}

            results.append(
                {
                    "position": position_in_padded_sequence,  # Position in padded sequence
                    "original_position": original_position,  # Calculated position relative to non-padded tokens
                    "token_info": token_info,
                    "context_window": {
                        "start": start,
                        "end": end,
                        "tokens": context_tokens,
                        "texts": context_texts,
                        "marked_context": " ".join(marked_context),
                    },
                    "padding_side": padding_side,
                    "real_tokens_length": len(real_tokens),
                    "padding_length": (
                        padding_length
                        if is_left_padding
                        else len(example.tokens) - real_tokens_end_index
                    ),
                }
            )

        return {
            "example_idx": example_idx,
            "position_key": position_key,
            "example_text": example.text,
            "token_count": len(example.tokens),
            "first_few_tokens": full_token_mapping[:5],
            "last_few_tokens": (
                full_token_mapping[-5:] if len(full_token_mapping) > 5 else []
            ),
            "target_positions": padded_positions,
            "padding_side": padding_side,
            "results": results[0] if len(results) == 1 else results,
        }

    def verify_padding(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        max_length: Optional[int] = None,
        examples_to_check: int = 3,
    ) -> Dict[str, Any]:
        """Verify token positions with explicit padding applied.

        This helper method applies padding explicitly and verifies that token positions
        are correctly aligned after padding, which is especially important for
        left-padding tokenizers.

        Args:
            tokenizer: Tokenizer to use for padding and decoding
            max_length: Maximum length for padding (if None, uses max sequence length)
            examples_to_check: Number of examples to check

        Returns:
            Dictionary with verification results
        """
        # Find the max length if not provided
        if max_length is None:
            max_length = self.get_max_sequence_length()

        # Get padding side
        padding_side = self.tokenization_config.padding_side
        is_left_padding = padding_side == "left"

        # Select a sample of examples to check
        indices = list(range(min(examples_to_check, len(self.examples))))

        # Get batch tensors with padding applied
        batch = self.get_batch_tensors(indices, pad=True)

        # DEBUG: Check type and content of batch
        print(f"DEBUG verify_padding: type(batch) = {type(batch)}, batch = {batch}")

        results = {}
        for i, idx in enumerate(indices):
            example = cast(TokenizedProbingExample, self.examples[idx])
            if example.token_positions is None:
                continue

            example_results = {}
            # Ensure batch["input_ids"] is a tensor before indexing
            input_ids_tensor = batch.get("input_ids")
            if not isinstance(input_ids_tensor, torch.Tensor):
                raise TypeError(
                    f"Expected 'input_ids' to be a Tensor, but got {type(input_ids_tensor)}"
                )
            if i >= input_ids_tensor.shape[0]:
                raise IndexError(
                    f"Index {i} out of bounds for input_ids tensor with shape {input_ids_tensor.shape}"
                )

            padded_tokens = input_ids_tensor[i].tolist()

            # Check each position type
            for key in self.position_types:
                if key not in example.token_positions.keys():
                    continue

                # Get original position from the example
                orig_pos = example.token_positions[key]

                # Get position from the batch (already adjusted for padding)
                try:
                    # Handle different possible formats of batch["positions"]
                    if (
                        isinstance(batch["positions"], dict)
                        and key in batch["positions"]
                    ):
                        # If positions is a dict with position_type keys
                        batch_pos_tensor = batch["positions"][key][i]
                    elif isinstance(batch["positions"], torch.Tensor):
                        # If positions is directly a tensor
                        batch_pos_tensor = batch["positions"][i]
                    else:
                        # Fallback
                        raise ValueError(
                            f"Unsupported format for batch positions: {type(batch['positions'])}"
                        )

                    # Convert tensor to list if needed
                    if isinstance(batch_pos_tensor, torch.Tensor):
                        batch_pos = batch_pos_tensor.tolist()
                        # Handle scalar tensor
                        if not isinstance(batch_pos, list):
                            batch_pos = [batch_pos]
                    else:
                        # Handle raw value
                        batch_pos = batch_pos_tensor
                        if not isinstance(batch_pos, list):
                            batch_pos = [batch_pos]
                except Exception as e:
                    # If we can't get batch_pos properly, use default position handling
                    padding_length = max_length - len(example.tokens)
                    if isinstance(orig_pos, int):
                        batch_pos = (
                            [orig_pos + padding_length]
                            if is_left_padding
                            else [orig_pos]
                        )
                    else:
                        batch_pos = (
                            [p + padding_length for p in orig_pos]
                            if is_left_padding
                            else orig_pos
                        )

                # Handle both single and list positions
                orig_positions = [orig_pos] if isinstance(orig_pos, int) else orig_pos

                position_results = []
                for j, orig in enumerate(orig_positions):
                    if j < len(batch_pos):
                        # Rename inner loop variable to avoid collision with outer 'batch' dict
                        current_batch_pos = batch_pos[j]
                        # Calculate expected position with padding
                        padding_length = max_length - len(example.tokens)
                        expected_pos = (
                            orig + padding_length if is_left_padding else orig
                        )

                        # Check if position matches expectation
                        position_match = expected_pos == current_batch_pos

                        # Get token at the position in padded sequence
                        try:
                            token_id = (
                                padded_tokens[current_batch_pos]
                                if 0 <= current_batch_pos < len(padded_tokens)
                                else None
                            )
                            token_text = (
                                tokenizer.decode([token_id])
                                if token_id is not None
                                else "OUT_OF_RANGE"
                            )
                        except Exception as e_decode:
                            token_text = f"ERROR: {str(e_decode)}"

                        position_results.append(
                            {
                                "original_position": orig,
                                "padded_position": current_batch_pos,
                                "expected_position": expected_pos,
                                "position_matches": position_match,
                                "token_text": token_text,
                                "padding_length": padding_length,
                            }
                        )

                example_results[key] = (
                    position_results[0]
                    if len(position_results) == 1
                    else position_results
                )

            results[idx] = {
                "text": example.text,
                "token_length": len(example.tokens),
                "padded_length": max_length,
                "padding_side": padding_side,
                "positions": example_results,
            }

        return {
            "padding_side": padding_side,
            "max_length": max_length,
            "examples": results,
        }
