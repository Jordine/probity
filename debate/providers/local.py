# debate/providers/local.py
"""
LocalModelProvider with multi-GPU support
"""

import torch
import time
import gc
from typing import List, Dict, Any, Optional
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
import warnings

from .base import BaseModelProvider
from ..types import ModelConfig


MAX_CONTEXT = 16384          # Hard budget
SAFETY_MARGIN = 256         # Reserve space


class LocalModelProvider(BaseModelProvider):
    """Provider for local HuggingFace / Transformer-Lens models with multi-GPU support"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        # Determine device - either specified GPU or auto-detect
        self.device_id = config.device_id
        if self.device_id is not None:
            if not torch.cuda.is_available():
                raise ValueError("CUDA not available but device_id specified")
            if self.device_id >= torch.cuda.device_count():
                raise ValueError(f"Device {self.device_id} not available. Only {torch.cuda.device_count()} GPUs found.")
            self.device = torch.device(f"cuda:{self.device_id}")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"[LocalModelProvider] Using device: {self.device}")
        
        self.model_dtype = self._get_model_dtype()
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[HookedTransformer] = None
        self._load_model()

        # Generation defaults
        self.default_gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "top_k": 50,
        }
        self.default_gen_kwargs.update(config.generation_kwargs or {})

    def _get_model_dtype(self) -> torch.dtype:
        bf16_families = ["llama", "mistral", "gemma", "phi", "qwen"]
        return torch.bfloat16 if any(m in self.config.model_name.lower() for m in bf16_families) else torch.float32

    def _load_model(self):
        print(f"Loading local model {self.config.model_name} on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with specific device assignment
        with torch.cuda.device(self.device) if 'cuda' in str(self.device) else warnings.catch_warnings():
            self.model = HookedTransformer.from_pretrained_no_processing(
                self.config.model_name,
                device=self.device,
                dtype=self.model_dtype,
            )
        
        # Print memory usage
        if 'cuda' in str(self.device):
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            print(f"✓ Model loaded on {self.device} - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        else:
            print("✓ Model loaded on CPU")
    
    def generate(self, messages: List[Dict[str, str]], 
                 capture_activations_layer: Optional[int] = None,
                 **kwargs) -> tuple:
        """
        Generate with optional activation capture for probe scoring.
        
        Args:
            messages: Conversation history
            capture_activations_layer: If provided, capture activations at this layer
            **kwargs: Generation parameters
        
        Returns:
            (completion_text, metadata) where metadata may include 'generation_activations'
        """
        if not self.is_available():
            return "", {"error": "Model not available"}
    
        start = time.time()
    
        # Build the chat prompt string with FULL context (story + history)
        prompt_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    
        # Tokenize and move to correct device
        prompt_ids = self.tokenizer(
            prompt_str,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"].to(self.device)
        
        prompt_len = prompt_ids.shape[1]
    
        # Context-length protection
        gen_cfg = {**self.default_gen_kwargs, **kwargs}
        max_new = gen_cfg.get("max_new_tokens", 256)
    
        if prompt_len + max_new + SAFETY_MARGIN > MAX_CONTEXT:
            allowed_new = max(1, MAX_CONTEXT - prompt_len - SAFETY_MARGIN)
            print(f"[CTX] Truncating max_new_tokens from {max_new} → {allowed_new}")
            gen_cfg["max_new_tokens"] = allowed_new
            max_new = allowed_new
    
        self.model.eval()
        
        # CRITICAL: Choose generation method based on whether we need activations
        if capture_activations_layer is not None:
            # Generate WITH activation capture
            hook_point = f"blocks.{capture_activations_layer}.hook_resid_pre"
            
            with torch.no_grad():
                # Set device context for generation
                if 'cuda' in str(self.device):
                    with torch.cuda.device(self.device):
                        # Use run_with_cache for generation to capture activations
                        full_output, cache = self.model.run_with_cache(
                            prompt_ids,
                            max_new_tokens=max_new,
                            temperature=gen_cfg["temperature"],
                            do_sample=gen_cfg["do_sample"],
                            top_p=gen_cfg["top_p"],
                            top_k=gen_cfg["top_k"],
                            eos_token_id=self.tokenizer.eos_token_id,
                            stop_at_eos=True,
                            prepend_bos=False,
                            return_type="input",
                            names_filter=[hook_point],  # Only cache the layer we need
                            return_cache_object=True,
                            stop_at_layer=capture_activations_layer + 1,
                        )
                else:
                    full_output, cache = self.model.run_with_cache(
                        prompt_ids,
                        max_new_tokens=max_new,
                        temperature=gen_cfg["temperature"],
                        do_sample=gen_cfg["do_sample"],
                        top_p=gen_cfg["top_p"],
                        top_k=gen_cfg["top_k"],
                        eos_token_id=self.tokenizer.eos_token_id,
                        stop_at_eos=True,
                        prepend_bos=False,
                        return_type="input",
                        names_filter=[hook_point],
                        return_cache_object=True,
                        stop_at_layer=capture_activations_layer + 1,
                    )
                
                # Extract ONLY the generated tokens' activations
                full_activations = cache[hook_point].squeeze(0)  # [full_seq_len, hidden_dim]
                generation_activations = full_activations[prompt_len:].clone()  # [gen_len, hidden_dim]
                
                # Extract generated token IDs
                out_ids = full_output
                
                # Immediate cleanup - don't keep full activations
                del cache, full_activations
                torch.cuda.empty_cache() if 'cuda' in str(self.device) else None
                
        else:
            # Normal generation without activation capture
            with torch.no_grad():
                if 'cuda' in str(self.device):
                    with torch.cuda.device(self.device):
                        out_ids = self.model.generate(
                            prompt_ids,
                            max_new_tokens=max_new,
                            temperature=gen_cfg["temperature"],
                            do_sample=gen_cfg["do_sample"],
                            top_p=gen_cfg["top_p"],
                            top_k=gen_cfg["top_k"],
                            eos_token_id=self.tokenizer.eos_token_id,
                            stop_at_eos=True,
                            prepend_bos=False,
                            return_type="input",
                        )
                else:
                    out_ids = self.model.generate(
                        prompt_ids,
                        max_new_tokens=max_new,
                        temperature=gen_cfg["temperature"],
                        do_sample=gen_cfg["do_sample"],
                        top_p=gen_cfg["top_p"],
                        top_k=gen_cfg["top_k"],
                        eos_token_id=self.tokenizer.eos_token_id,
                        stop_at_eos=True,
                        prepend_bos=False,
                        return_type="input",
                    )
            
            generation_activations = None
    
        # Decode completion
        out_ids = torch.tensor(out_ids, device=self.device)
        completion_ids = out_ids[0, prompt_len:]
        completion_str = self.tokenizer.decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).lstrip()
    
        # Accounting & metadata
        output_len = completion_ids.numel()
        total_tokens = prompt_len + output_len
        latency = time.time() - start
        self._update_usage(total_tokens)
    
        metadata = {
            "input_tokens": int(prompt_len),
            "output_tokens": int(output_len),
            "total_tokens": int(total_tokens),
            "latency": latency,
            "model": self.config.model_name,
            "device": str(self.device),
            "dtype": str(self.model_dtype),
        }
        
        # Add activations to metadata if captured
        if generation_activations is not None:
            metadata['generation_activations'] = generation_activations
            # Also store the generated token IDs for probe scoring
            metadata['generation_token_ids'] = completion_ids
    
        # Final cleanup
        if 'cuda' in str(self.device):
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        gc.collect()
    
        return completion_str, metadata
    
    def is_available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def clear_cache(self):
        if 'cuda' in str(self.device):
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
        gc.collect()

    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.clear_cache()