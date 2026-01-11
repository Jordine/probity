import torch
import json
import os
from probity.datasets.base import ProbingDataset, ProbingExample, CharacterPositions, Position
from probity.datasets.tokenized import TokenizedProbingDataset, TokenizedProbingExample, TokenPositions
from transformers import AutoTokenizer
from typing import Optional, Dict, List, Tuple, Literal

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

def load_lie_truth_dataset(json_path: str, 
                           tokenizer_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                           max_length: int = 512,  # Now configurable
                           warn_on_truncation: bool = True) -> TokenizedProbingDataset:
    """Load the lie/truth dataset from JSON format (works with both Llama and Qwen)"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_type = detect_model_type(tokenizer_name)
    USER_PROMPT = "Hello there, please give me a few facts about yourself."
    
    # Track truncation statistics
    truncation_stats = {
        'total_examples': 0,
        'truncated_examples': 0,
        'max_original_length': 0,
        'truncated_lengths': []
    }
    
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
    
    print(f"Processing {len(data)} lie/truth pairs...")
    print(f"Model type detected: {model_type}")
    print(f"Max token length: {max_length}")
    
    probing_examples = []
    
    for i, item in enumerate(data):
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
        
        # Check token lengths before truncation
        truth_tokens = tokenizer(truth_conv, add_special_tokens=False)['input_ids']
        truth_len = len(truth_tokens)
        truncation_stats['total_examples'] += 1
        truncation_stats['max_original_length'] = max(truncation_stats['max_original_length'], truth_len)
        
        if truth_len > max_length:
            truncation_stats['truncated_examples'] += 1
            truncation_stats['truncated_lengths'].append(truth_len)
            if warn_on_truncation and truncation_stats['truncated_examples'] <= 5:  # Warn for first 5
                print(f"⚠️  Truncating truth example {i}: {truth_len} → {max_length} tokens")
        
        lie_tokens = tokenizer(lie_conv, add_special_tokens=False)['input_ids']
        lie_len = len(lie_tokens)
        truncation_stats['total_examples'] += 1
        truncation_stats['max_original_length'] = max(truncation_stats['max_original_length'], lie_len)
        
        if lie_len > max_length:
            truncation_stats['truncated_examples'] += 1
            truncation_stats['truncated_lengths'].append(lie_len)
            if warn_on_truncation and truncation_stats['truncated_examples'] <= 5:
                print(f"⚠️  Truncating lie example {i}: {lie_len} → {max_length} tokens")
        
        # Find positions
        truth_lie_pos = find_lie_statement_character_positions(truth_conv, lie_statement, model_type)
        lie_lie_pos = find_lie_statement_character_positions(lie_conv, lie_statement, model_type)
        
        if truth_lie_pos is None or lie_lie_pos is None:
            continue
        
        # Check if lie statement positions would be truncated
        if truth_lie_pos.start >= max_length * 4:  # Rough estimate: 1 token ≈ 4 chars
            if warn_on_truncation:
                print(f"⚠️  Lie statement in truth example {i} may be truncated")
        
        if lie_lie_pos.start >= max_length * 4:
            if warn_on_truncation:
                print(f"⚠️  Lie statement in lie example {i} may be truncated")
        
        # Create examples
        truth_example = ProbingExample(
            text=truth_conv,
            label=0,  # Truth
            label_text="truth",
            character_positions=CharacterPositions({"LIE_SPAN": truth_lie_pos}),
            attributes={
                "example_id": f"truth_{i}",
                "original_token_length": truth_len
            }
        )
        
        lie_example = ProbingExample(
            text=lie_conv,
            label=1,  # Lie
            label_text="lie",
            character_positions=CharacterPositions({"LIE_SPAN": lie_lie_pos}),
            attributes={
                "example_id": f"lie_{i}",
                "original_token_length": lie_len
            }
        )
        
        probing_examples.extend([truth_example, lie_example])
    
    # Print truncation summary
    if truncation_stats['truncated_examples'] > 0:
        print(f"\n⚠️  TRUNCATION WARNING:")
        print(f"  • {truncation_stats['truncated_examples']}/{truncation_stats['total_examples']} "
              f"({100*truncation_stats['truncated_examples']/truncation_stats['total_examples']:.1f}%) "
              f"examples were truncated")
        print(f"  • Max original length: {truncation_stats['max_original_length']} tokens")
        print(f"  • Current max_length: {max_length} tokens")
        if truncation_stats['truncated_lengths']:
            avg_truncated = sum(truncation_stats['truncated_lengths']) / len(truncation_stats['truncated_lengths'])
            print(f"  • Average truncated length: {avg_truncated:.0f} tokens")
        print(f"\n  💡 Consider increasing max_length to {min(truncation_stats['max_original_length'], 8192)}")
    
    print(f"\nCreated {len(probing_examples)} examples")
    
    # Calculate statistics
    truth_count = sum(1 for ex in probing_examples if ex.label == 0)
    lie_count = sum(1 for ex in probing_examples if ex.label == 1)
    
    print(f"  • Truth examples: {truth_count}")
    print(f"  • Lie examples: {lie_count}")
    
    if truth_count + lie_count > 0:
        print(f"  • Balance: {lie_count/(truth_count + lie_count):.1%} lies")
    else:
        print("  • Balance: No examples found")
    
    # Create probing dataset
    probing_dataset = ProbingDataset(
        examples=probing_examples,
        dataset_attributes={
            "description": f"Truth vs Lie classification in {model_type} chat format",
            "source_pairs": len(data),
            "total_examples": len(probing_examples),
            "truth_count": truth_count,
            "lie_count": lie_count,
            "max_length": max_length,
            "truncation_stats": truncation_stats
        }
    )
    
    # Convert to tokenized dataset
    tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
        dataset=probing_dataset,
        tokenizer=tokenizer,
        padding="max_length",
        max_length=max_length,  # Use the configurable max_length
        truncation=True,
        add_special_tokens=False  # Chat template already adds special tokens
    )
    
    return tokenized_dataset

def load_contrastive_ntml_dataset(json_path: str,
                                 tokenizer_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                                 max_length: int = 512,
                                 warn_on_truncation: bool = True,
                                 dishonest_mode: str = 'all',
                                 honest_mode: str = 'none',
                                 last_k_tokens: int = 0,
                                 use_llm_spans: str = 'auto') -> TokenizedProbingDataset:
    """
    Load contrastive NTML dataset with flexible training modes.

    Training modes based on dishonest_mode and honest_mode:
    - Mode 1 (diff/diff): Contrastive pairs at diff positions only, balanced 50/50
    - Mode 2 (all/all): All statements from both samples
    - Mode 3 (all/none): All statements from dishonest sample only
    - Mode 4 (all/diff): All from dishonest + diff from honest

    Span extraction modes (use_llm_spans):
    - 'auto': Use LLM-tagged spans if present in dataset, else fall back to sentence parsing
    - 'always': Require LLM-tagged spans, error if not present
    - 'never': Always use sentence-level parsing, ignore any LLM spans

    LLM-tagged spans provide more precise deceptive phrase localization (e.g., "red Honda Civic")
    compared to sentence-level parsing which uses full sentences.
    """
    
    with open(json_path, 'r') as f:
        conversations = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_type = detect_model_type(tokenizer_name)
    USER_PROMPT = "Hello there, please give me a few facts about yourself."

    # Track truncation statistics
    truncation_stats = {
        'total_examples': 0,
        'truncated_examples': 0,
        'max_original_length': 0,
        'truncated_lengths': []
    }

    # ========== LLM SPAN DETECTION ==========
    # Check if dataset has LLM-tagged spans
    # Spans can be in: conv['llm_spans'], conv['lie_version']['spans'], or conv['spans']
    def has_llm_spans(conv: dict) -> bool:
        """Check if conversation has LLM-tagged spans."""
        if 'llm_spans' in conv:
            return True
        if 'spans' in conv:
            return True
        if 'lie_version' in conv and 'spans' in conv.get('lie_version', {}):
            return True
        return False

    def get_llm_spans_for_version(conv: dict, version: str) -> List[dict]:
        """
        Get LLM-tagged spans for a specific version (lie/truth).
        Returns list of span dicts with 'text', 'char_start', 'char_end'.
        """
        # Check version-specific spans first
        version_key = 'lie_version' if version == 'lie' else 'truth_version'
        if version_key in conv and 'spans' in conv[version_key]:
            return conv[version_key]['spans']

        # Check top-level spans (applies to lie version by default)
        if version == 'lie' and 'spans' in conv:
            return conv['spans']
        if version == 'lie' and 'llm_spans' in conv:
            return conv['llm_spans']

        return []

    def spans_to_positions(spans: List[dict], formatted_text: str,
                          model_response: str, model_type: str) -> List[Position]:
        """
        Convert LLM-tagged spans to Position objects.
        Spans have char_start/char_end relative to the model response.
        We need to adjust to the full formatted chat text.
        """
        content_start, content_end = find_assistant_boundaries(formatted_text, model_type)
        if content_start is None:
            return []

        positions = []
        for span in spans:
            # Span positions are relative to model response
            span_start = span.get('char_start', 0)
            span_end = span.get('char_end', 0)

            # Adjust to full formatted text
            adjusted_start = content_start + span_start
            adjusted_end = content_start + span_end

            # Validate bounds
            if adjusted_start >= 0 and adjusted_end <= len(formatted_text):
                positions.append(Position(start=adjusted_start, end=adjusted_end))

        return positions

    # Determine span extraction mode
    dataset_has_spans = any(has_llm_spans(conv) for conv in conversations[:10])  # Sample first 10

    if use_llm_spans == 'always' and not dataset_has_spans:
        raise ValueError(
            "use_llm_spans='always' but dataset has no LLM-tagged spans. "
            "Run the LLM tagger first or use use_llm_spans='auto' or 'never'."
        )

    # Decide actual mode
    using_llm_spans = (use_llm_spans == 'always') or (use_llm_spans == 'auto' and dataset_has_spans)

    def create_chat_messages(system_prompt, user_prompt, model_response):
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": model_response}
        ]

    probing_examples = []
    llm_span_count = 0
    sentence_span_count = 0

    print(f"Processing {len(conversations)} conversations...")
    print(f"Model type detected: {model_type}")
    print(f"Max token length: {max_length}")
    print(f"Training mode: dishonest={dishonest_mode}, honest={honest_mode}")

    # ========== SPAN MODE STATUS ==========
    if using_llm_spans:
        print(f"\n🎯 SPAN MODE: Using LLM-tagged spans (fine-grained phrase-level)")
        print(f"   Dataset has LLM spans: {dataset_has_spans}")
        print(f"   Requested mode: use_llm_spans='{use_llm_spans}'")
    else:
        print(f"\n📝 SPAN MODE: Using sentence-level parsing (split by .!?)")
        print(f"   Dataset has LLM spans: {dataset_has_spans}")
        print(f"   Requested mode: use_llm_spans='{use_llm_spans}'")
    
    # Determine mode for logging
    if dishonest_mode == 'diff' and honest_mode == 'diff':
        print("Mode 1: Contrastive pairs at diff positions only (balanced 50/50)")
    elif dishonest_mode == 'all' and honest_mode == 'all':
        print("Mode 2: All statements from both samples")
    elif dishonest_mode == 'all' and honest_mode == 'none':
        print("Mode 3: All statements from dishonest sample only")
    elif dishonest_mode == 'all' and honest_mode == 'diff':
        print("Mode 4: All from dishonest + diff from honest")
    else:
        raise ValueError(f"Invalid mode combination: dishonest={dishonest_mode}, honest={honest_mode}")

    for conv in conversations:
        conv_id = conv['id']
        # Use shuffled positions if available (for shuffle_response=True datasets)
        # Fall back to original lie_ids for unshuffled datasets
        lie_positions = set(conv.get('lie_response_positions', conv['lie_ids']))
        
        # Always process dishonest version
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
        
        # Check token length
        lie_tokens = tokenizer(lie_formatted, add_special_tokens=False)['input_ids']
        lie_len = len(lie_tokens)
        truncation_stats['total_examples'] += 1
        truncation_stats['max_original_length'] = max(truncation_stats['max_original_length'], lie_len)
        
        if lie_len > max_length:
            truncation_stats['truncated_examples'] += 1
            truncation_stats['truncated_lengths'].append(lie_len)
            if warn_on_truncation and truncation_stats['truncated_examples'] <= 5:
                print(f"⚠️  Truncating conversation {conv_id} (lie): {lie_len} → {max_length} tokens")
        
        # ========== GET SPAN POSITIONS ==========
        # Use LLM spans if available and enabled, else fall back to sentence parsing
        conv_has_llm_spans = has_llm_spans(conv)

        if using_llm_spans and conv_has_llm_spans:
            # Use LLM-tagged spans (fine-grained)
            llm_spans = get_llm_spans_for_version(conv, 'lie')
            lie_statement_positions = spans_to_positions(
                llm_spans, lie_formatted, conv['lie_version']['model'], model_type
            )
            llm_span_count += len(lie_statement_positions)

            # For LLM spans, all positions are deceptive (label=1)
            # We create one example per span, not per statement index
            for span_idx, position in enumerate(lie_statement_positions):
                span_text = llm_spans[span_idx].get('text', '') if span_idx < len(llm_spans) else ''
                group_id = f"{conv_id}_llm_span_{span_idx}"

                example = ProbingExample(
                    text=lie_formatted,
                    label=1,  # LLM spans are deceptive phrases
                    label_text="lie",
                    character_positions=CharacterPositions({"LIE_SPAN": position}),
                    group_id=group_id,
                    attributes={
                        "conversation_id": conv_id,
                        "version": "dishonest",
                        "span_idx": span_idx,
                        "span_text": span_text,
                        "span_source": "llm_tagged",
                        "original_token_length": lie_len
                    }
                )
                probing_examples.append(example)

            # Skip the sentence-level processing below
            continue

        # Fall back to sentence-level parsing
        lie_statement_positions = find_statement_positions_in_chat(
            lie_formatted,
            conv['lie_version']['model'],
            model_type
        )
        sentence_span_count += len(lie_statement_positions)

        # Process dishonest statements based on mode
        dishonest_indices = range(len(lie_statement_positions)) if dishonest_mode == 'all' else lie_positions

        for i in dishonest_indices:
            if i < len(lie_statement_positions):
                position = lie_statement_positions[i]
                is_lie_statement = i in lie_positions

                # Determine group_id based on mode and whether this position is paired
                if dishonest_mode == 'diff' and honest_mode == 'diff':
                    # Mode 1: All are contrastive pairs
                    group_id = f"{conv_id}_contrastive_stmt{i}"
                elif honest_mode != 'none' and i in lie_positions:
                    # Modes 2 & 4: Diff positions are contrastive pairs
                    group_id = f"{conv_id}_contrastive_stmt{i}"
                else:
                    # Mode 3 or non-diff positions in modes 2/4
                    group_id = f"{conv_id}_dishonest_stmt{i}"
                
                example = ProbingExample(
                    text=lie_formatted,
                    label=1 if is_lie_statement else 0,
                    label_text="lie" if is_lie_statement else "truth",
                    character_positions=CharacterPositions({"LIE_SPAN": position}),
                    group_id=group_id,
                    attributes={
                        "conversation_id": conv_id,
                        "version": "dishonest",
                        "statement_idx": i,
                        "is_lie_position": is_lie_statement,
                        "span_source": "sentence_parsing",
                        "original_token_length": lie_len
                    }
                )
                probing_examples.append(example)
        
        # Process honest version based on mode
        if honest_mode != 'none':
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
            
            # Check token length
            truth_tokens = tokenizer(truth_formatted, add_special_tokens=False)['input_ids']
            truth_len = len(truth_tokens)
            truncation_stats['total_examples'] += 1
            truncation_stats['max_original_length'] = max(truncation_stats['max_original_length'], truth_len)
            
            if truth_len > max_length:
                truncation_stats['truncated_examples'] += 1
                truncation_stats['truncated_lengths'].append(truth_len)
                if warn_on_truncation and truncation_stats['truncated_examples'] <= 5:
                    print(f"⚠️  Truncating conversation {conv_id} (truth): {truth_len} → {max_length} tokens")
            
            truth_statement_positions = find_statement_positions_in_chat(
                truth_formatted, 
                conv['truth_version']['model'],
                model_type
            )
            
            # Determine which statements to include
            honest_indices = range(len(truth_statement_positions)) if honest_mode == 'all' else lie_positions
            
            for i in honest_indices:
                if i < len(truth_statement_positions):
                    position = truth_statement_positions[i]
                    
                    # Honest statements at diff positions are always paired when included
                    if i in lie_positions:
                        group_id = f"{conv_id}_contrastive_stmt{i}"
                    else:
                        # Mode 2 only: honest statements at non-diff positions
                        group_id = f"{conv_id}_honest_stmt{i}"
                    
                    example = ProbingExample(
                        text=truth_formatted,
                        label=0,  # Always truthful in honest version
                        label_text="truth",
                        character_positions=CharacterPositions({"LIE_SPAN": position}),
                        group_id=group_id,
                        attributes={
                            "conversation_id": conv_id,
                            "version": "honest",
                            "statement_idx": i,
                            "is_lie_position": False,  # Never a lie in honest version
                            "span_source": "sentence_parsing",
                            "original_token_length": truth_len
                        }
                    )
                    probing_examples.append(example)

    
    # Print truncation summary
    if truncation_stats['truncated_examples'] > 0:
        print(f"\n⚠️  TRUNCATION WARNING:")
        print(f"  • {truncation_stats['truncated_examples']}/{truncation_stats['total_examples']} "
              f"({100*truncation_stats['truncated_examples']/truncation_stats['total_examples']:.1f}%) "
              f"conversations were truncated")
        print(f"  • Max original length: {truncation_stats['max_original_length']} tokens")
        print(f"  • Current max_length: {max_length} tokens")
        if truncation_stats['truncated_lengths']:
            avg_truncated = sum(truncation_stats['truncated_lengths']) / len(truncation_stats['truncated_lengths'])
            print(f"  • Average truncated length: {avg_truncated:.0f} tokens")
        print(f"\n  💡 Consider increasing max_length to {min(truncation_stats['max_original_length'], 8192)}")
    
    print(f"\nCreated {len(probing_examples)} examples")

    # ========== SPAN MODE SUMMARY ==========
    if using_llm_spans:
        print(f"\n🎯 Span extraction summary:")
        print(f"   • LLM-tagged spans: {llm_span_count}")
        print(f"   • Sentence-parsed spans: {sentence_span_count}")
        if llm_span_count > 0 and sentence_span_count > 0:
            print(f"   ⚠️  Mixed mode: some conversations had LLM spans, others used sentence parsing")
    else:
        print(f"\n📝 Span extraction summary:")
        print(f"   • All spans from sentence parsing: {sentence_span_count}")

    validate_mode_distribution(probing_examples, dishonest_mode, honest_mode)

    
    # Calculate statistics
    truth_count = sum(1 for ex in probing_examples if ex.label == 0)
    lie_count = sum(1 for ex in probing_examples if ex.label == 1)
    
    print(f"  • Truth statements: {truth_count}")
    print(f"  • Lie statements: {lie_count}")
    
    if truth_count + lie_count > 0:
        print(f"  • Balance: {lie_count/(truth_count + lie_count):.1%} lies")
    else:
        print("  • Balance: No examples found")
    
    # Create mode description based on the flags
    if dishonest_mode == 'diff' and honest_mode == 'diff':
        mode_description = "contrastive pairs only (Mode 1)"
    elif dishonest_mode == 'all' and honest_mode == 'all':
        mode_description = "all statements from both versions (Mode 2)"
    elif dishonest_mode == 'all' and honest_mode == 'none':
        mode_description = "all statements from dishonest version only (Mode 3)"
    elif dishonest_mode == 'all' and honest_mode == 'diff':
        mode_description = "all dishonest + diff honest statements (Mode 4)"
    else:
        mode_description = f"custom mode: dishonest={dishonest_mode}, honest={honest_mode}"


    
    # Determine span mode description
    if using_llm_spans:
        span_mode_desc = "LLM-tagged spans (fine-grained phrase-level)"
    else:
        span_mode_desc = "sentence-level parsing"

    probing_dataset = ProbingDataset(
        examples=probing_examples,
        dataset_attributes={
            "description": f"Contrastive NTML dataset ({mode_description}) in {model_type} chat format",
            "source_conversations": len(conversations),
            "total_examples": len(probing_examples),
            "truth_count": truth_count,
            "lie_count": lie_count,
            "max_length": max_length,
            "truncation_stats": truncation_stats,
            "dishonest_mode": dishonest_mode,
            "honest_mode": honest_mode,
            # Span extraction mode info
            "span_mode": span_mode_desc,
            "use_llm_spans": use_llm_spans,
            "using_llm_spans": using_llm_spans,
            "llm_span_count": llm_span_count,
            "sentence_span_count": sentence_span_count
        }
    )
    
    # Convert to tokenized dataset
    tokenized_dataset = TokenizedProbingDataset.from_probing_dataset(
        dataset=probing_dataset,
        tokenizer=tokenizer,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=False
    )

    # Apply last-K-tokens filtering if specified (Apollo style: only use last K tokens of each span)
    if last_k_tokens > 0:
        print(f"\n🎯 Applying last-{last_k_tokens}-tokens filtering (Apollo style)")
        filtered_count = 0
        for example in tokenized_dataset.examples:
            if example.token_positions is not None:
                for key, positions in example.token_positions.positions.items():
                    if isinstance(positions, list) and len(positions) > last_k_tokens:
                        # Keep only last K tokens
                        original_len = len(positions)
                        example.token_positions.positions[key] = positions[-last_k_tokens:]
                        filtered_count += 1
        print(f"  • Filtered {filtered_count} spans to last {last_k_tokens} tokens")

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




def validate_mode_distribution(examples: List[ProbingExample], 
                              dishonest_mode: str, 
                              honest_mode: str,
                              expected_ratio: str = None) -> None:
    """Validate that the dataset matches expected distribution for the mode."""
    
    # Count statements by version and label
    dishonest_truth = 0
    dishonest_lie = 0
    honest_all = 0
    
    for ex in examples:
        version = ex.attributes.get('version', '')
        if version == 'dishonest':
            if ex.label == 1:
                dishonest_lie += 1
            else:
                dishonest_truth += 1
        elif version == 'honest':
            honest_all += 1
    
    total = len(examples)
    print(f"\n📊 Mode Validation:")
    print(f"  Total statement examples: {total}")
    print(f"  Dishonest version: {dishonest_truth} truths, {dishonest_lie} lies")
    print(f"  Honest version: {honest_all} statements")
    
    # Validate based on mode
    if dishonest_mode == 'diff' and honest_mode == 'diff':
        # Mode 1: Should be perfectly balanced
        expected_balance = dishonest_lie == honest_all
        if expected_balance:
            print(f"  ✓ Mode 1: Balanced {dishonest_lie} vs {honest_all}")
        else:
            print(f"  ⚠️  Mode 1: Imbalanced! {dishonest_lie} lies vs {honest_all} honest")
    
    elif dishonest_mode == 'all' and honest_mode == 'none':
        # Mode 3: Only dishonest statements
        if honest_all == 0:
            ratio = f"{dishonest_lie}L:{dishonest_truth}T"
            print(f"  ✓ Mode 3: Dishonest only, ratio {ratio}")
        else:
            print(f"  ⚠️  Mode 3: Unexpected honest statements found!")