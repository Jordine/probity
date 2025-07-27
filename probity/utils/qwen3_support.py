"""
Qwen3 model support utilities for Probity.
"""

import re
from typing import List, Dict, Any, Optional
import torch
from transformers import PreTrainedTokenizer


def ensure_qwen3_transformerlens_support():
    """Monkey-patch TransformerLens to support Qwen3 models."""
    try:
        from transformer_lens import loading_from_pretrained
        
        qwen3_models = [
            "Qwen/Qwen3-32B", 
            "Qwen/Qwen3-14B", 
        ]
        
        existing_names = set(loading_from_pretrained.OFFICIAL_MODEL_NAMES)
        new_models = [model for model in qwen3_models if model not in existing_names]
        
        if new_models:
            loading_from_pretrained.OFFICIAL_MODEL_NAMES.extend(new_models)
            print(f"Added {len(new_models)} Qwen3 models to TransformerLens support")
            return True
        return False
            
    except ImportError:
        print("Warning: Could not import TransformerLens for Qwen3 model support")
        return False


def is_qwen_model(model_name: str) -> bool:
    """Check if a model name refers to a Qwen model."""
    return 'qwen' in model_name.lower()


def apply_qwen3_chat_template(
    tokenizer: PreTrainedTokenizer, 
    messages: List[Dict[str, str]], 
    enable_thinking: bool = False,
    **kwargs
) -> str:
    """Apply Qwen3 chat template with thinking mode support."""
    return tokenizer.apply_chat_template(
        messages,
        enable_thinking=enable_thinking,
        **kwargs
    )


def remove_thinking_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3 generated text."""
    think_pattern = r'<think>.*?</think>'
    return re.sub(think_pattern, '', text, flags=re.DOTALL).strip()


def find_qwen3_assistant_boundaries(text: str) -> tuple[Optional[int], Optional[int]]:
    """Find assistant response boundaries in Qwen3 chat format."""
    assistant_marker = "<|im_start|>assistant"
    end_marker = "<|im_end|>"
    
    assistant_start = text.find(assistant_marker)
    if assistant_start == -1:
        return None, None
    
    # Start after the header and newline
    content_start = assistant_start + len(assistant_marker)
    if content_start < len(text) and text[content_start] == '\n':
        content_start += 1
    
    # Find end
    end_pos = text.find(end_marker, content_start)
    content_end = end_pos if end_pos != -1 else len(text)
    
    return content_start, content_end


def get_qwen3_special_tokens() -> Dict[str, str]:
    """Get Qwen3 special tokens mapping."""
    return {
        'im_start': '<|im_start|>',
        'im_end': '<|im_end|>',
        'endoftext': '<|endoftext|>',
        'think_start': '<think>',
        'think_end': '</think>'
    }