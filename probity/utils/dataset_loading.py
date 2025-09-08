import torch
import json
import os
from probity.datasets.base import ProbingDataset, ProbingExample, CharacterPositions, Position
from probity.datasets.tokenized import TokenizedProbingDataset, TokenizedProbingExample, TokenPositions
from transformers import AutoTokenizer

def detect_model_type(model_name: str) -> str:
    """Detect if model is Llama-style or Qwen-style"""
    model_lower = model_name.lower()
    if 'qwen' in model_lower:
        return 'qwen'
    elif 'llama' in model_lower or 'mistral' in model_lower:
        return 'llama'
    else:
        # Default to llama style for most models
        return 'llama'

def apply_chat_template_unified(tokenizer, messages, model_type='auto', **kwargs):
    """Apply chat template with appropriate method for model type"""
    if model_type == 'auto':
        model_type = detect_model_type(tokenizer.name_or_path)
    
    if model_type == 'qwen':
        # Import only if needed
        from probity.utils.qwen3_support import apply_qwen3_chat_template
        return apply_qwen3_chat_template(tokenizer, messages, **kwargs)
    else:
        # Standard transformers apply_chat_template
        return tokenizer.apply_chat_template(messages, **kwargs)

def find_assistant_boundaries(text: str, model_type: str) -> tuple:
    """Find assistant response boundaries for different model types"""
    if model_type == 'qwen':
        # Qwen format
        assistant_marker = "<|im_start|>assistant"
        end_marker = "<|im_end|>"
        
        assistant_start = text.find(assistant_marker)
        if assistant_start == -1:
            return None, None
        
        content_start = assistant_start + len(assistant_marker)
        if content_start < len(text) and text[content_start] == '\n':
            content_start += 1
        
        end_pos = text.find(end_marker, content_start)
        content_end = end_pos if end_pos != -1 else len(text)
        
        return content_start, content_end
    else:
        # Llama format
        assistant_marker = "<|start_header_id|>assistant<|end_header_id|>"
        eot_marker = "<|eot_id|>"
        
        assistant_start = text.find(assistant_marker)
        if assistant_start == -1:
            return None, None
        
        content_start = assistant_start + len(assistant_marker)
        while content_start < len(text) and text[content_start] in ['\n', ' ', '\t']:
            content_start += 1
        
        eot_pos = text.find(eot_marker, content_start)
        content_end = eot_pos if eot_pos != -1 else len(text)
        
        return content_start, content_end

def find_statement_positions_in_chat(formatted_text: str, model_response: str, model_type: str):
    """Find character positions of statements in the full chat text."""
    content_start, content_end = find_assistant_boundaries(formatted_text, model_type)
    
    if content_start is None:
        print(f"Warning: Could not find assistant boundaries in text")
        return []
    
    # Extract the assistant content
    assistant_content = formatted_text[content_start:content_end]
    
    # Split model response into sentences
    sentences = []
    current_sentence = ""
    
    for char in model_response:
        current_sentence += char
        if char in '.!?':
            sentence = current_sentence.strip()
            if sentence:
                sentences.append(sentence)
            current_sentence = ""
    
    # Add remaining text if any
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    # Find positions of each sentence in the formatted chat text
    positions = []
    search_start = content_start
    
    for sentence in sentences:
        sentence_start = formatted_text.find(sentence, search_start, content_end)
        if sentence_start != -1:
            sentence_end = sentence_start + len(sentence)
            positions.append(Position(start=sentence_start, end=sentence_end))
            search_start = sentence_end
        else:
            # Try without leading/trailing whitespace
            sentence_stripped = sentence.strip()
            sentence_start = formatted_text.find(sentence_stripped, search_start, content_end)
            if sentence_start != -1:
                sentence_end = sentence_start + len(sentence_stripped)
                positions.append(Position(start=sentence_start, end=sentence_end))
                search_start = sentence_end
            else:
                print(f"Warning: Could not find sentence '{sentence[:50]}...' in chat text")
    
    return positions

def load_lie_truth_dataset(json_path: str, tokenizer_name: str = "meta-llama/Llama-3.1-8B-Instruct") -> TokenizedProbingDataset:
    """Load the lie/truth dataset from JSON format (works with both Llama and Qwen)"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_type = detect_model_type(tokenizer_name)
    USER_PROMPT = "Hello there, please give me a few facts about yourself."
    
    def create_chat_messages(system_prompt, user_prompt, model_response):
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": model_response}
        ]
    
    def find_lie_statement_character_positions(formatted_text, lie_statement, model_type):
        if model_type == 'qwen':
            # Handle potential <think>...</think> blocks for Qwen
            import re
            think_pattern = r'<think>.*?</think>'
            content_without_thinking = re.sub(think_pattern, '', formatted_text, flags=re.DOTALL)
            
            start_pos = content_without_thinking.find(lie_statement)
            if start_pos == -1:
                start_pos = formatted_text.find(lie_statement)
        else:
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
        
        # Get formatted text with appropriate template
        truth_conv = apply_chat_template_unified(
            tokenizer,
            truth_messages, 
            model_type=model_type,
            tokenize=False,
            add_generation_prompt=False
        )
        
        lie_conv = apply_chat_template_unified(
            tokenizer,
            lie_messages, 
            model_type=model_type,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Find positions
        truth_lie_pos = find_lie_statement_character_positions(truth_conv, lie_statement, model_type)
        lie_lie_pos = find_lie_statement_character_positions(lie_conv, lie_statement, model_type)
        
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
        dataset_attributes={"description": f"Truth vs Lie classification in {model_type} chat format"}
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
    """Load contrastive NTML dataset from JSON format with proper truth/lie labeling (works with both Llama and Qwen)."""
    
    with open(json_path, 'r') as f:
        conversations = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_type = detect_model_type(tokenizer_name)
    USER_PROMPT = "Hello there, please give me a few facts about yourself."
    
    def create_chat_messages(system_prompt, user_prompt, model_response):
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": model_response}
        ]
    
    probing_examples = []
    
    print(f"Processing {len(conversations)} conversations...")
    print(f"Model type detected: {model_type}")
    
    for conv in conversations:
        conv_id = conv['id']
        lie_positions = set(conv['lie_ids'])  # Which statement indices are lies
        
        # Process TRUTH VERSION
        truth_messages = create_chat_messages(
            conv['truth_version']['system'],
            USER_PROMPT,
            conv['truth_version']['model']
        )
        
        truth_formatted = apply_chat_template_unified(
            tokenizer,
            truth_messages,
            model_type=model_type,
            tokenize=False,
            add_generation_prompt=False
        )
        
        truth_statement_positions = find_statement_positions_in_chat(
            truth_formatted, 
            conv['truth_version']['model'],
            model_type
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
        
        lie_formatted = apply_chat_template_unified(
            tokenizer,
            lie_messages,
            model_type=model_type,
            tokenize=False,
            add_generation_prompt=False
        )
        
        lie_statement_positions = find_statement_positions_in_chat(
            lie_formatted,
            conv['lie_version']['model'],
            model_type
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
    
    # Handle division by zero
    if truth_count + lie_count > 0:
        print(f"  • Balance: {lie_count/(truth_count + lie_count):.1%} lies")
    else:
        print("  • Balance: No examples found")
        
    # Create probing dataset
    probing_dataset = ProbingDataset(
        examples=probing_examples,
        dataset_attributes={
            "description": f"Contrastive truth vs lie classification in {model_type} chat format",
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
        max_length=312,
        truncation=True,
        add_special_tokens=False  # Chat template already adds special tokens
    )
    
    return tokenized_dataset

def get_model_dtype(model_name: str) -> torch.dtype:
    """Determine the appropriate dtype for the model"""
    # Models that typically use bfloat16
    bfloat16_models = ['llama', 'mistral', 'gemma', 'phi', 'qwen']
    
    if any(m in model_name.lower() for m in bfloat16_models):
        return torch.bfloat16
    return torch.float32

# Keep the Qwen3-specific function for backward compatibility
def load_lie_truth_dataset_qwen3(json_path: str, tokenizer_name: str = "Qwen/Qwen3-32B-Instruct", 
                                 enable_thinking: bool = False) -> TokenizedProbingDataset:
    """Backward compatibility wrapper - just calls the unified function"""
    return load_lie_truth_dataset(json_path, tokenizer_name)