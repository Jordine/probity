import torch
import json
import os
from probity.datasets.base import ProbingDataset, ProbingExample, CharacterPositions, Position
from probity.datasets.tokenized import TokenizedProbingDataset, TokenizedProbingExample, TokenPositions
from transformers import AutoTokenizer

def load_lie_truth_dataset(json_path: str, tokenizer_name: str = "meta-llama/Llama-3.1-8B-Instruct") -> TokenizedProbingDataset:
    """Load the lie/truth dataset from JSON format"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    USER_PROMPT = "Hello there, please give me a few facts about yourself."
    
    def create_chat_messages(system_prompt, user_prompt, model_response):
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": model_response}
        ]
    
    def find_lie_statement_character_positions(formatted_text, lie_statement):
        start_pos = formatted_text.find(lie_statement)
        if start_pos == -1:
            return None
        end_pos = start_pos + len(lie_statement)
        return Position(start=start_pos, end=end_pos)
    
    probing_examples = []
    
    for item in data:
        lie_statement = item["lie_statement"]
        
        # Truth version
        truth_messages = create_chat_messages(
            item["truth_version"]["system"],
            USER_PROMPT,
            item["truth_version"]["model"]
        )
        
        # Lie version
        lie_messages = create_chat_messages(
            item["lie_version"]["system"],
            USER_PROMPT,
            item["lie_version"]["model"]
        )
        
        # Get formatted text
        truth_conv = tokenizer.apply_chat_template(
            truth_messages, 
            tokenize=False,
            add_generation_prompt=False
        )
        
        lie_conv = tokenizer.apply_chat_template(
            lie_messages, 
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Find positions
        truth_lie_pos = find_lie_statement_character_positions(truth_conv, lie_statement)
        lie_lie_pos = find_lie_statement_character_positions(lie_conv, lie_statement)
        
        if truth_lie_pos is None or lie_lie_pos is None:
            continue
        
        # Create examples
        truth_example = ProbingExample(
            text=truth_conv,
            label=0,  # Truth
            label_text="truth",
            character_positions=CharacterPositions({"LIE_SPAN": truth_lie_pos})
        )
        
        lie_example = ProbingExample(
            text=lie_conv,
            label=1,  # Lie
            label_text="lie",
            character_positions=CharacterPositions({"LIE_SPAN": lie_lie_pos})
        )
        
        probing_examples.extend([truth_example, lie_example])
    
    # Create probing dataset
    probing_dataset = ProbingDataset(
        examples=probing_examples,
        dataset_attributes={"description": "Truth vs Lie classification in Llama chat format"}
    )
    
    # Convert to tokenized dataset
    tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
        dataset=probing_dataset,
        tokenizer=tokenizer,
        padding="max_length",
        max_length=512,
        truncation=True,
        add_special_tokens=False  # Chat template already adds special tokens
    )
    
    return tokenized_dataset

def load_contrastive_ntml_dataset(json_path: str, tokenizer_name: str = "meta-llama/Llama-3.1-8B-Instruct") -> TokenizedProbingDataset:
    """Load contrastive NTML dataset from JSON format with proper truth/lie labeling."""
    
    with open(json_path, 'r') as f:
        conversations = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    USER_PROMPT = "Hello there, please give me a few facts about yourself."
    
    def create_chat_messages(system_prompt, user_prompt, model_response):
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": model_response}
        ]
    
    def find_statement_positions_in_chat(formatted_text, model_response):
        """Find character positions of statements in the full chat text."""
        # Find assistant response section
        assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"
        assistant_start = formatted_text.find(assistant_marker)
        if assistant_start == -1:
            return []
        
        # Find actual content start (skip newlines/whitespace after marker)
        content_start = assistant_start + len(assistant_marker)
        while content_start < len(formatted_text) and formatted_text[content_start] in ['\n', ' ', '\t']:
            content_start += 1
        
        # Split model response into sentences (simple but should work for controlled data)
        # You might want to use spaCy here for more robust segmentation
        sentences = []
        current_sentence = ""
        
        for char in model_response:
            current_sentence += char
            if char in '.!?':
                # Check if this is end of sentence (not abbreviation)
                sentences.append(current_sentence.strip())
                current_sentence = ""
        
        # Add remaining text if any
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # Find positions of each sentence in the formatted chat text
        positions = []
        search_start = content_start
        
        for sentence in sentences:
            sentence_start = formatted_text.find(sentence, search_start)
            if sentence_start != -1:
                sentence_end = sentence_start + len(sentence)
                positions.append(Position(start=sentence_start, end=sentence_end))
                search_start = sentence_end
            else:
                print(f"Warning: Could not find sentence '{sentence}' in chat text")
        
        return positions
    
    probing_examples = []
    
    print(f"Processing {len(conversations)} conversations...")
    
    for conv in conversations:
        conv_id = conv['id']
        lie_positions = set(conv['lie_ids'])  # Which statement indices are lies
        
        # Process TRUTH VERSION
        truth_messages = create_chat_messages(
            conv['truth_version']['system'],
            USER_PROMPT,
            conv['truth_version']['model']
        )
        
        truth_formatted = tokenizer.apply_chat_template(
            truth_messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        truth_statement_positions = find_statement_positions_in_chat(
            truth_formatted, 
            conv['truth_version']['model']
        )
        
        # Create examples for each statement in truth version
        for i, position in enumerate(truth_statement_positions):
            # In truth version, all statements are truthful (label=0)
            example = ProbingExample(
                text=truth_formatted,
                label=0,  # Always truthful in truth version
                label_text="truth",
                character_positions=CharacterPositions({"LIE_SPAN": position}),
                group_id=f"{conv_id}_truth_stmt{i}",
                attributes={
                    "conversation_id": conv_id,
                    "version": "truth",
                    "statement_idx": i,
                    "is_lie_position": i in lie_positions,
                    "statement_count": len(truth_statement_positions)
                }
            )
            probing_examples.append(example)
        
        # Process LIE VERSION
        lie_messages = create_chat_messages(
            conv['lie_version']['system'],
            USER_PROMPT,
            conv['lie_version']['model']
        )
        
        lie_formatted = tokenizer.apply_chat_template(
            lie_messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        lie_statement_positions = find_statement_positions_in_chat(
            lie_formatted,
            conv['lie_version']['model']
        )
        
        # Create examples for each statement in lie version
        for i, position in enumerate(lie_statement_positions):
            # In lie version: lie if i in lie_positions, truth otherwise
            is_lie_statement = i in lie_positions
            
            example = ProbingExample(
                text=lie_formatted,
                label=1 if is_lie_statement else 0,  # 1 for lie, 0 for truth
                label_text="lie" if is_lie_statement else "truth",
                character_positions=CharacterPositions({"LIE_SPAN": position}),
                group_id=f"{conv_id}_lie_stmt{i}",
                attributes={
                    "conversation_id": conv_id,
                    "version": "lie",
                    "statement_idx": i,
                    "is_lie_position": i in lie_positions,
                    "statement_count": len(lie_statement_positions)
                }
            )
            probing_examples.append(example)
    
    print(f"Created {len(probing_examples)} statement-level examples")
    
    # Calculate statistics
    truth_count = sum(1 for ex in probing_examples if ex.label == 0)
    lie_count = sum(1 for ex in probing_examples if ex.label == 1)
    
    print(f"  • Truth statements: {truth_count}")
    print(f"  • Lie statements: {lie_count}")
    print(f"  • Balance: {lie_count/(truth_count + lie_count):.1%} lies")
    
    # Create probing dataset
    probing_dataset = ProbingDataset(
        examples=probing_examples,
        dataset_attributes={
            "description": "Contrastive truth vs lie classification in Llama chat format",
            "source_conversations": len(conversations),
            "total_examples": len(probing_examples),
            "truth_count": truth_count,
            "lie_count": lie_count
        }
    )
    
    # Convert to tokenized dataset
    tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
        dataset=probing_dataset,
        tokenizer=tokenizer,
        padding="max_length",
        max_length=1024,
        truncation=True,
        add_special_tokens=False  # Chat template already adds special tokens
    )
    
    return tokenized_dataset

def get_model_dtype(model_name: str) -> torch.dtype:
    """Determine the appropriate dtype for the model"""
    # Models that typically use bfloat16
    bfloat16_models = ['llama', 'mistral', 'gemma', 'phi']
    
    if any(m in model_name.lower() for m in bfloat16_models):
        return torch.bfloat16
    return torch.float32


