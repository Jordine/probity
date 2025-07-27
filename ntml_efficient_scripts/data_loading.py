"""
Data loading and processing for NTML binary token training.

Converts NTML conversational datasets into binary token-level training data.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
from transformers import AutoTokenizer
import logging

# Import existing probity infrastructure
from probity_extensions import ConversationalProbingDataset
from probity.datasets.tokenized import TokenizedProbingDataset
from config import NTMLBinaryTrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class BinaryTokenExample:
    """Single example for binary token training."""
    
    tokens: List[int]  # Token IDs
    labels: List[int]  # Binary labels (0=truth, 1=lie) for each token
    attention_mask: List[int]  # Attention mask
    assistant_mask: List[int]  # Mask for assistant tokens only
    
    # Metadata
    conversation_id: str
    statement_count: int
    truth_count: int
    lie_count: int
    
    # Debug info
    token_texts: Optional[List[str]] = None
    statement_assignments: Optional[List[int]] = None  # Which statement each token belongs to


class NTMLBinaryDataset:
    """Dataset class for NTML binary token training."""
    
    def __init__(self, config: NTMLBinaryTrainingConfig):
        self.config = config
        self.tokenizer = None
        self.examples: List[BinaryTokenExample] = []
        self.label_distribution: Dict[str, int] = {}
        
    def load_and_process(self) -> "NTMLBinaryDataset":
        """Load NTML dataset and convert to binary token format."""
        logger.info(f"Loading NTML dataset from {self.config.dataset_path}")
        
        # Load NTML conversational dataset
        ntml_dataset = ConversationalProbingDataset.from_ntml_jsonl(self.config.dataset_path)
        logger.info(f"Loaded {len(ntml_dataset.examples)} conversations")
        
        # Initialize tokenizer
        logger.info(f"Loading tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Process each conversation
        logger.info("Converting to binary token format...")
        total_tokens = 0
        total_assistant_tokens = 0
        
        for conv_ex in ntml_dataset.examples:
            binary_example = self._process_conversation(conv_ex)
            if binary_example is not None:
                self.examples.append(binary_example)
                total_tokens += len(binary_example.tokens)
                total_assistant_tokens += sum(binary_example.assistant_mask)
        
        logger.info(f"Created {len(self.examples)} binary token examples")
        logger.info(f"Total tokens: {total_tokens}, Assistant tokens: {total_assistant_tokens}")
        
        # Calculate label distribution
        self._calculate_label_distribution()
        
        return self
    
    def _process_conversation(self, conv_ex) -> Optional[BinaryTokenExample]:
        """Convert a single conversation to binary token format."""

        model_name = getattr(self.config, 'model_name', '')
        is_qwen = 'qwen' in model_name.lower()
        
        # Reconstruct messages in proper format for chat template
        messages = [
            {"role": "system", "content": conv_ex.system_prompt},
            {"role": "assistant", "content": conv_ex.assistant_response}
        ]
        
        # Apply chat template to get properly formatted text
        if is_qwen:
            full_text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False  # Disable thinking for probe training
            )
        else:
            # Original Llama format
            full_text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False,
                add_generation_prompt=False
            )
        
        # Tokenize the full conversation
        encoding = self.tokenizer(
            full_text,
            max_length=self.config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False  # Chat template already adds special tokens
        )
        
        tokens = encoding["input_ids"][0].tolist()
        attention_mask = encoding["attention_mask"][0].tolist()
        offset_mapping = encoding["offset_mapping"][0].tolist()
        
        # Find assistant response boundaries in the tokenized sequence
        assistant_start_token, assistant_end_token = self._find_assistant_token_boundaries(
            full_text, offset_mapping
        )
        
        if assistant_start_token is None or assistant_end_token is None:
            logger.warning(f"Could not find assistant boundaries in conversation {conv_ex.group_id}")
            return None
        
        # Create assistant mask
        assistant_mask = [0] * len(tokens)
        for i in range(assistant_start_token, assistant_end_token):
            assistant_mask[i] = 1
        
        # Convert statement character positions to token labels
        token_labels, statement_assignments = self._assign_token_labels(
            conv_ex, offset_mapping, assistant_start_token, assistant_end_token, full_text
        )
        
        # Get token texts for debugging
        token_texts = None
        if self.config.verbose:
            token_texts = [self.tokenizer.decode([t]) for t in tokens]
        
        return BinaryTokenExample(
            tokens=tokens,
            labels=token_labels,
            attention_mask=attention_mask,
            assistant_mask=assistant_mask,
            conversation_id=conv_ex.group_id,
            statement_count=len(conv_ex.statement_labels),
            truth_count=sum(conv_ex.statement_labels),
            lie_count=len(conv_ex.statement_labels) - sum(conv_ex.statement_labels),
            token_texts=token_texts,
            statement_assignments=statement_assignments
        )
    
    def _find_assistant_token_boundaries(self, full_text: str, offset_mapping: List[Tuple[int, int]]) -> Tuple[Optional[int], Optional[int]]:
        """Find token indices that correspond to assistant response."""
    
        is_qwen = 'qwen' in self.config.model_name.lower()
        print(f"USING {'QWEN' if is_qwen else 'LLAMA'} SPECIAL TOKENS. Model: {self.config.model_name}")

        if is_qwen:
            # Qwen3 format: <|im_start|>assistant\n...content...<|im_end|>
            assistant_marker = "<|im_start|>assistant"
            end_marker = "<|im_end|>"
        else:
            # Llama format: <|start_header_id|>assistant<|end_header_id|>...content...<|eot_id|>
            assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"
            end_marker = "<|eot_id|>"
    
        print("=== DEBUGGING ASSISTANT MARKER SEARCH ===")
        print(f"Looking for assistant_marker: '{assistant_marker}'")
        print(f"Full text length: {len(full_text)}")
        print(f"Assistant marker length: {len(assistant_marker)}")
        
        # Debug: Show the exact characters around where assistant should be
        if "<|im_start|>" in full_text:
            start_pos = full_text.find("<|im_start|>")
            print(f"Found <|im_start|> at position: {start_pos}")
            
            # Look for all instances of <|im_start|>
            positions = []
            start = 0
            while True:
                pos = full_text.find("<|im_start|>", start)
                if pos == -1:
                    break
                positions.append(pos)
                start = pos + 1
            print(f"All <|im_start|> positions: {positions}")
            
            # Check what comes after each <|im_start|>
            for pos in positions:
                end_tag_pos = full_text.find(">", pos)
                if end_tag_pos != -1:
                    content_after = full_text[end_tag_pos + 1:end_tag_pos + 20]  # Show next 20 chars
                    print(f"At position {pos}, after tag: '{repr(content_after)}'")
        
        # Try the search
        assistant_char_start = full_text.find(assistant_marker)
        print(f"assistant_char_start: {assistant_char_start}")
        
        if assistant_char_start == -1:
            # Try alternative searches
            print("=== TRYING ALTERNATIVE SEARCHES ===")
            
            # Try with different variations
            variations = [
                "<|im_start|>assistant",
                "<|im_start|>assistant\n",
                "<|im_start|> assistant",
                "<|im_start|>\nassistant",
            ]
            
            for i, variation in enumerate(variations):
                pos = full_text.find(variation)
                print(f"Variation {i} '{repr(variation)}': {pos}")
                if pos != -1:
                    assistant_char_start = pos
                    assistant_marker = variation
                    break
        
        if assistant_char_start == -1:
            print("ERROR: Still could not find assistant marker!")
            return None, None
        
        print(f"Found assistant marker at position: {assistant_char_start}")
        
        # Start after the header
        assistant_content_start = assistant_char_start + len(assistant_marker)
        
        # For Qwen3, skip any whitespace after assistant marker
        if is_qwen:
            while (assistant_content_start < len(full_text) and 
                   full_text[assistant_content_start] in ['\n', ' ', '\t']):
                assistant_content_start += 1
        
        print(f"Assistant content starts at: {assistant_content_start}")
        print(f"First 50 chars of assistant content: '{repr(full_text[assistant_content_start:assistant_content_start+50])}'")
        
        # Find end of assistant response
        eot_pos = full_text.find(end_marker, assistant_content_start)
        if eot_pos == -1:
            assistant_content_end = len(full_text)
            print(f"No end marker found, using end of text: {assistant_content_end}")
        else:
            assistant_content_end = eot_pos
            print(f"Found end marker at: {eot_pos}")
        
        # Convert to token indices
        assistant_start_token = None
        assistant_end_token = None
        
        for token_idx, (char_start, char_end) in enumerate(offset_mapping):
            # Skip special tokens (offset_mapping is (0,0) for special tokens)
            if char_start == 0 and char_end == 0:
                continue
                
            # Find first token that starts in or after assistant content
            if assistant_start_token is None and char_start >= assistant_content_start:
                assistant_start_token = token_idx
                print(f"Assistant starts at token {token_idx} (char {char_start})")
            
            # Find first token that ends after assistant content
            if assistant_end_token is None and char_end >= assistant_content_end:
                assistant_end_token = token_idx
                print(f"Assistant ends at token {token_idx} (char {char_end})")
                break
        
        # If we didn't find end, use end of sequence
        if assistant_end_token is None:
            assistant_end_token = len(offset_mapping)
            print(f"Using end of sequence: {assistant_end_token}")
        
        print(f"Final result: tokens {assistant_start_token} to {assistant_end_token}")
        print("=== END DEBUGGING ===")
        
        return assistant_start_token, assistant_end_token
    
    def _assign_token_labels(self, conv_ex, offset_mapping: List[Tuple[int, int]], 
                           assistant_start_token: int, assistant_end_token: int,
                           full_text: str) -> Tuple[List[int], List[int]]:
        """Assign binary labels to tokens based on statement boundaries."""
        
        token_labels = [0] * len(offset_mapping)  # Default to 0 (truth)
        statement_assignments = [-1] * len(offset_mapping)  # -1 = not in any statement
        
        # Detect model type - use the same logic as _find_assistant_token_boundaries
        is_qwen = 'qwen' in self.config.model_name.lower()  # Use actual model name
        
        # Find assistant response start in the chat template
        if is_qwen:
            assistant_marker = "<|im_start|>assistant"
        else:
            assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"
        
        assistant_content_start = full_text.find(assistant_marker)
        if assistant_content_start != -1:
            assistant_content_start += len(assistant_marker)
            # Skip any whitespace/newlines after the marker
            while assistant_content_start < len(full_text) and full_text[assistant_content_start] in ['\n', ' ', '\t']:
                assistant_content_start += 1
        else:
            logger.warning(f"Could not find assistant marker '{assistant_marker}' in chat template")
            return token_labels, statement_assignments
        
        # Use the statement_level data which has positions relative to assistant response
        for stmt_idx, stmt_data in enumerate(conv_ex.attributes['statement_level']):
            stmt_text = stmt_data['text']
            stmt_label = stmt_data['is_true']
            # These positions are relative to the assistant response
            rel_char_start = stmt_data['char_start']
            rel_char_end = stmt_data['char_end']
            
            # Convert to absolute positions in the full chat template
            abs_char_start = assistant_content_start + rel_char_start
            abs_char_end = assistant_content_start + rel_char_end
            
            # Find tokens that overlap with this statement
            for token_idx in range(assistant_start_token, assistant_end_token):
                token_char_start, token_char_end = offset_mapping[token_idx]
                
                # Skip special tokens
                if token_char_start == 0 and token_char_end == 0:
                    continue
                
                # Check for overlap using the configured strategy
                overlap = self._calculate_token_statement_overlap(
                    token_char_start, token_char_end, abs_char_start, abs_char_end
                )
                
                if overlap > 0:
                    # Assign label based on statement truth value
                    label = 0 if stmt_label else 1  # 0=truth, 1=lie
                    
                    # Handle conflicts (token overlaps multiple statements)
                    if statement_assignments[token_idx] != -1:
                        # Use majority overlap or keep existing assignment
                        if self.config.token_overlap_strategy == "majority":
                            # For now, keep the first assignment
                            # TODO: Could implement more sophisticated conflict resolution
                            pass
                        else:
                            token_labels[token_idx] = label
                            statement_assignments[token_idx] = stmt_idx
                    else:
                        token_labels[token_idx] = label
                        statement_assignments[token_idx] = stmt_idx
        
        return token_labels, statement_assignments
    
    def _calculate_token_statement_overlap(self, token_start: int, token_end: int,
                                         stmt_start: int, stmt_end: int) -> float:
        """Calculate overlap between token and statement character ranges."""
        
        # Calculate intersection
        overlap_start = max(token_start, stmt_start)
        overlap_end = min(token_end, stmt_end)
        
        if overlap_start >= overlap_end:
            return 0.0  # No overlap
        
        overlap_length = overlap_end - overlap_start
        token_length = token_end - token_start
        
        if token_length == 0:
            return 0.0
        
        return overlap_length / token_length
    
    def _calculate_label_distribution(self):
        """Calculate distribution of truth/lie labels."""
        total_assistant_tokens = 0
        truth_tokens = 0
        lie_tokens = 0
        
        for example in self.examples:
            for i, (is_assistant, label) in enumerate(zip(example.assistant_mask, example.labels)):
                if is_assistant:
                    total_assistant_tokens += 1
                    if label == 0:
                        truth_tokens += 1
                    else:
                        lie_tokens += 1
        
        self.label_distribution = {
            "total_assistant_tokens": total_assistant_tokens,
            "truth_tokens": truth_tokens,
            "lie_tokens": lie_tokens,
            "truth_ratio": truth_tokens / total_assistant_tokens if total_assistant_tokens > 0 else 0,
            "lie_ratio": lie_tokens / total_assistant_tokens if total_assistant_tokens > 0 else 0,
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
    
    def create_torch_datasets(self, train_ratio: float = 0.8) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """Create PyTorch datasets for training."""
        
        train_examples, val_examples = self.get_training_data(train_ratio)
        
        train_dataset = NTMLTorchDataset(train_examples, mode="train")
        val_dataset = NTMLTorchDataset(val_examples, mode="val")
        
        return train_dataset, val_dataset
    
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
            
            for i, (token, label, is_assistant, stmt_assign, token_text) in enumerate(zip(
                example.tokens, example.labels, example.assistant_mask, 
                example.statement_assignments, example.token_texts or [""] * len(example.tokens)
            )):
                debug_entry["tokens_and_labels"].append({
                    "index": i,
                    "token_id": token,
                    "token_text": token_text,
                    "label": label,
                    "is_assistant": bool(is_assistant),
                    "statement_assignment": stmt_assign
                })
            
            debug_data.append(debug_entry)
        
        import json
        with open(output_path / "tokenization_debug.json", "w") as f:
            json.dump(debug_data, f, indent=2)
        
        # Save label distribution
        with open(output_path / "label_distribution.json", "w") as f:
            json.dump(self.label_distribution, f, indent=2)


class NTMLTorchDataset(torch.utils.data.Dataset):
    """PyTorch dataset wrapper for binary token examples."""
    
    def __init__(self, examples: List[BinaryTokenExample], mode: str = "train"):
        self.examples = examples
        self.mode = mode
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        return {
            "input_ids": torch.tensor(example.tokens, dtype=torch.long),
            "attention_mask": torch.tensor(example.attention_mask, dtype=torch.long),
            "assistant_mask": torch.tensor(example.assistant_mask, dtype=torch.long),
            "labels": torch.tensor(example.labels, dtype=torch.long),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching."""
    
    # All examples should already be padded to max_length
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        collated[key] = torch.stack([item[key] for item in batch])
    
    return collated