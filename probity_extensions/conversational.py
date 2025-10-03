"""
Conversational dataset extensions for NTML (N Truths, M Lies) probing.

This module provides classes that extend Probity's base dataset functionality
to handle conversational NTML datasets where individual statements within
assistant responses need to be classified as true or false.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

# Import Probity base classes
from probity.datasets.base import ProbingExample, ProbingDataset, CharacterPositions
from probity.datasets.position_finder import Position


@dataclass
class ConversationalProbingExample(ProbingExample):
    """NTML conversational example with statement-level labels.
    
    Extends ProbingExample to handle conversational structure where:
    - System prompt specifies the task and lie positions
    - Assistant response contains multiple statements (truths and lies)
    - Each statement has its own truth label and position
    """
    
    # Core conversation structure
    system_prompt: str = ""
    assistant_response: str = ""
    
    # Statement-level information
    statement_labels: List[bool] = field(default_factory=list)  # [True, False, True, ...]
    statement_texts: List[str] = field(default_factory=list)    # Individual statement texts
    statement_positions: List[Tuple[int, int]] = field(default_factory=list)  # (char_start, char_end)
    
    # Enhanced metadata
    conversation_metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize the base ProbingExample fields from conversational data."""
        # Set the main text as the full conversation
        if not self.text:  # Only set if not already provided
            self.text = f"System: {self.system_prompt}\nAssistant: {self.assistant_response}"
        
        # Set overall label (proportion of true statements) if not already set
        if self.label is None and self.statement_labels:
            self.label = sum(self.statement_labels) / len(self.statement_labels)
        
        # Set label text if not already set
        if not self.label_text and self.statement_labels:
            truth_count = sum(self.statement_labels)
            lie_count = len(self.statement_labels) - truth_count
            self.label_text = f"{truth_count}T{lie_count}L"
        elif not self.label_text:
            self.label_text = "0T0L"
        
        # Add statement positions to character_positions
        if self.statement_positions:
            positions = {}
            for i, (start, end) in enumerate(self.statement_positions):
                positions[f"statement_{i}"] = Position(start, end)
            if not self.character_positions:
                self.character_positions = CharacterPositions(positions)
        
        # Store comprehensive metadata
        if not self.attributes:
            self.attributes = {
                "ratio": self.conversation_metadata.get("ratio"),
                "truth_count": sum(self.statement_labels) if self.statement_labels else 0,
                "lie_count": (len(self.statement_labels) - sum(self.statement_labels)) if self.statement_labels else 0,
                "lie_positions": self.conversation_metadata.get("lie_positions", []),
                "statement_count": len(self.statement_labels),
                "system_prompt": self.system_prompt,
                "assistant_response": self.assistant_response,
                **self.conversation_metadata
            }


class ConversationalProbingDataset(ProbingDataset):
    """NTML dataset with conversational structure.
    
    Extends ProbingDataset to handle NTML JSONL files and provide
    flexible statement-level probing capabilities.
    """
    
    @classmethod
    def from_ntml_jsonl(cls, jsonl_path: str) -> "ConversationalProbingDataset":
        """Load NTML dataset from JSONL file.
        
        Args:
            jsonl_path: Path to NTML JSONL file
            
        Returns:
            ConversationalProbingDataset instance
            
        Raises:
            FileNotFoundError: If JSONL file doesn't exist
            ValueError: If JSONL format is invalid
        """
        jsonl_path = Path(jsonl_path)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"NTML JSONL file not found: {jsonl_path}")
            
        examples = []
        
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                        
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON on line {line_num}: {e}")
                    
                    # Extract conversation components
                    messages = data.get("messages", [])
                    system_msg = next((m for m in messages if m["role"] == "system"), None)
                    assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)
                    
                    if not system_msg or not assistant_msg:
                        raise ValueError(f"Missing system or assistant message on line {line_num}")
                    
                    # Extract statement information
                    labels = data.get("labels", {})
                    statements = labels.get("statement_level", [])
                    
                    if not statements:
                        raise ValueError(f"No statement_level data found on line {line_num}")
                    
                    statement_texts = [s["text"] for s in statements]
                    statement_labels = [s["is_true"] for s in statements]
                    
                    # Calculate positions in full conversation text
                    # The positions in JSONL are relative to assistant response
                    # We need to adjust for "System: ...\nAssistant: " prefix
                    system_content = system_msg["content"]
                    assistant_content = assistant_msg["content"]
                    prefix_len = len(f"System: {system_content}\nAssistant: ")
                    
                    statement_positions = []
                    for stmt in statements:
                        # Adjust positions to be relative to full conversation text
                        start = prefix_len + stmt["char_start"]
                        end = prefix_len + stmt["char_end"]
                        statement_positions.append((start, end))
                    
                    example = ConversationalProbingExample(
                        text=f"System: {system_content}\nAssistant: {assistant_content}",
                        label=sum(statement_labels) / len(statement_labels) if statement_labels else 0.0,
                        label_text=f"{sum(statement_labels)}T{len(statement_labels) - sum(statement_labels)}L" if statement_labels else "0T0L",
                        system_prompt=system_content,
                        assistant_response=assistant_content,
                        statement_labels=statement_labels,
                        statement_texts=statement_texts,
                        statement_positions=statement_positions,
                        conversation_metadata=labels,
                        group_id=data.get("id", f"example_{line_num}")
                    )
                    examples.append(example)
                    
        except Exception as e:
            raise ValueError(f"Error loading NTML JSONL file {jsonl_path}: {e}")
        
        if not examples:
            raise ValueError(f"No valid examples found in {jsonl_path}")
        
        return cls(
            examples=examples,
            task_type="classification",
            label_mapping={"false": 0, "true": 1},
            dataset_attributes={
                "source": "ntml_jsonl", 
                "path": str(jsonl_path),
                "total_examples": len(examples),
                "total_statements": sum(len(ex.statement_labels) for ex in examples)
            }
        )
    
    def get_statement_dataset(self, statement_idx: Optional[int] = None) -> ProbingDataset:
        """Create a dataset for probing specific statements.
        
        Args:
            statement_idx: If specified, create dataset for only this statement position.
                         If None, create dataset with all statements as separate examples.
                         
        Returns:
            ProbingDataset with statement-level examples
        """
        if statement_idx is not None:
            # Single statement position dataset
            statement_examples = []
            for ex in self.examples:
                if isinstance(ex, ConversationalProbingExample):
                    if statement_idx < len(ex.statement_labels):
                        # Create new example for this specific statement position
                        stmt_example = ProbingExample(
                            text=ex.text,  # Full conversation text
                            label=int(ex.statement_labels[statement_idx]),
                            label_text=str(ex.statement_labels[statement_idx]),
                            character_positions=CharacterPositions({
                                "target": Position(*ex.statement_positions[statement_idx])
                            }),
                            group_id=ex.group_id,
                            attributes={
                                **ex.attributes,
                                "statement_idx": statement_idx,
                                "statement_text": ex.statement_texts[statement_idx],
                                "probing_mode": "single_position"
                            }
                        )
                        statement_examples.append(stmt_example)
            
            return ProbingDataset(
                examples=statement_examples,
                task_type="classification",
                label_mapping={"false": 0, "true": 1},
                dataset_attributes={
                    **self.dataset_attributes,
                    "statement_idx": statement_idx,
                    "probing_mode": "single_position",
                    "filtered_examples": len(statement_examples)
                }
            )
        else:
            # Multi-statement dataset (all statements as separate examples)
            all_statement_examples = []
            for ex in self.examples:
                if isinstance(ex, ConversationalProbingExample):
                    for i, (label, text, pos) in enumerate(zip(
                        ex.statement_labels, ex.statement_texts, ex.statement_positions
                    )):
                        stmt_example = ProbingExample(
                            text=ex.text,  # Full conversation text
                            label=int(label),
                            label_text=str(label),
                            character_positions=CharacterPositions({
                                "target": Position(*pos)
                            }),
                            group_id=f"{ex.group_id}_stmt_{i}",
                            attributes={
                                **ex.attributes,
                                "statement_idx": i,
                                "statement_text": text,
                                "probing_mode": "all_statements"
                            }
                        )
                        all_statement_examples.append(stmt_example)
            
            return ProbingDataset(
                examples=all_statement_examples,
                task_type="classification",
                label_mapping={"false": 0, "true": 1},
                dataset_attributes={
                    **self.dataset_attributes,
                    "probing_mode": "all_statements",
                    "total_statement_examples": len(all_statement_examples)
                }
            )
    
    def get_conversation_summary(self) -> Dict:
        """Get summary statistics about the conversational dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        if not self.examples:
            return {"error": "No examples in dataset"}
        
        # Calculate statistics
        total_conversations = len(self.examples)
        total_statements = sum(len(ex.statement_labels) for ex in self.examples if isinstance(ex, ConversationalProbingExample))
        total_truths = sum(sum(ex.statement_labels) for ex in self.examples if isinstance(ex, ConversationalProbingExample))
        total_lies = total_statements - total_truths
        
        # Ratio distribution
        ratio_counts = {}
        statement_count_dist = {}
        
        for ex in self.examples:
            if isinstance(ex, ConversationalProbingExample):
                ratio = ex.attributes.get("ratio", "unknown")
                ratio_counts[ratio] = ratio_counts.get(ratio, 0) + 1
                
                stmt_count = len(ex.statement_labels)
                statement_count_dist[stmt_count] = statement_count_dist.get(stmt_count, 0) + 1
        
        return {
            "total_conversations": total_conversations,
            "total_statements": total_statements,
            "total_truths": total_truths,
            "total_lies": total_lies,
            "truth_ratio": total_truths / total_statements if total_statements > 0 else 0,
            "ratio_distribution": ratio_counts,
            "statement_count_distribution": statement_count_dist,
            "avg_statements_per_conversation": total_statements / total_conversations if total_conversations > 0 else 0
        }
    
    def validate_positions(self) -> Dict:
        """Validate that statement positions are correct.
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "total_examples": len(self.examples),
            "valid_examples": 0,
            "invalid_examples": 0,
            "errors": []
        }
        
        for i, ex in enumerate(self.examples):
            if not isinstance(ex, ConversationalProbingExample):
                continue
                
            try:
                # Check that positions extract the correct text
                for j, (stmt_text, (start, end)) in enumerate(zip(ex.statement_texts, ex.statement_positions)):
                    extracted_text = ex.text[start:end]
                    if extracted_text.strip() != stmt_text.strip():
                        validation_results["errors"].append({
                            "example_idx": i,
                            "statement_idx": j,
                            "expected": stmt_text,
                            "extracted": extracted_text,
                            "position": (start, end)
                        })
                        validation_results["invalid_examples"] += 1
                        break
                else:
                    validation_results["valid_examples"] += 1
                    
            except Exception as e:
                validation_results["errors"].append({
                    "example_idx": i,
                    "error": str(e)
                })
                validation_results["invalid_examples"] += 1
        
        validation_results["validation_success"] = validation_results["invalid_examples"] == 0
        return validation_results 