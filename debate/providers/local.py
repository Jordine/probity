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
        
        # Determine device - check gpu_indices first, then device_id
        self.gpu_indices = getattr(config, 'gpu_indices', None)
        self.device_id = config.device_id
        
        if self.gpu_indices and len(self.gpu_indices) > 0:
            # Multi-GPU mode
            self.device = torch.device(f"cuda:{self.gpu_indices[0]}")
            print(f"[LocalModelProvider] Using GPUs: {self.gpu_indices}")
        elif self.device_id is not None:
            # Single GPU mode
            self.device = torch.device(f"cuda:{self.device_id}")
        else:
            # Auto mode
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
        
        # Check for multi-GPU with explicit assignment
        if self.gpu_indices and len(self.gpu_indices) > 1:
            print(f"  Distributing model across GPUs: {self.gpu_indices}")
            with torch.cuda.device(self.device) if 'cuda' in str(self.device) else warnings.catch_warnings():
                self.model = HookedTransformer.from_pretrained_no_processing(
                    self.config.model_name,
                    device="cuda",
                    n_devices=len(self.gpu_indices),
                    gpu_indices=self.gpu_indices,  # Pass explicit GPU assignment
                    dtype=self.model_dtype,
                )
            
            # Report memory usage for each GPU
            if 'cuda' in str(self.device):
                for gpu_idx in self.gpu_indices:
                    allocated = torch.cuda.memory_allocated(gpu_idx) / 1024**3
                    reserved = torch.cuda.memory_reserved(gpu_idx) / 1024**3
                    print(f"  ✓ GPU {gpu_idx} - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
                print(f"✓ Model loaded across GPUs {self.gpu_indices}")
            
        else:
            # Single GPU or CPU mode
            with torch.cuda.device(self.device) if 'cuda' in str(self.device) else warnings.catch_warnings():
                self.model = HookedTransformer.from_pretrained_no_processing(
                    self.config.model_name,
                    device=self.device,
                    dtype=self.model_dtype,
                )
            
            # Report memory usage
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
        
        generation_activations = None
        
        # FIXED: Use hooks during generation to capture activations
        if capture_activations_layer is not None:
            print(f"[MEMORY EFFICIENT] Capturing activations at layer {capture_activations_layer} during generation")
            
            hook_point = f"blocks.{capture_activations_layer}.hook_resid_pre"
            captured_acts = []
            is_first_call = [True]  # Use list to allow modification in nested function
            
            def capture_hook(activations, hook):
                """Hook to capture activations during generation"""
                # During autoregressive generation:
                # - First call: processes full prompt [batch, prompt_len, hidden]
                # - Subsequent calls: process one new token at a time [batch, 1, hidden]
                
                if is_first_call[0]:
                    # Skip the prompt forward pass - we only want generated tokens
                    is_first_call[0] = False
                else:
                    # This is a generated token - capture it
                    # Shape should be [batch, 1, hidden] or [batch, seq_len, hidden]
                    # Take the last token's activation (the newly generated one)
                    if activations.dim() == 3:  # [batch, seq, hidden]
                        new_token_act = activations[:, -1:, :].clone().detach().cpu()
                    elif activations.dim() == 2:  # [batch, hidden] - already just one token
                        new_token_act = activations.unsqueeze(1).clone().detach().cpu()  # [batch, 1, hidden]
                    else:
                        print(f"[WARNING] Unexpected activation shape: {activations.shape}")
                        return activations
                    
                    captured_acts.append(new_token_act)
                
                return activations
            
            # Run generation with hook
            with torch.no_grad():
                # Add the hook
                self.model.add_hook(hook_point, capture_hook, dir='fwd')
                
                try:
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
                finally:
                    # Remove hooks after generation
                    self.model.reset_hooks(including_permanent=False)
            
            # Concatenate all captured activations
            if captured_acts:
                # Each element is [batch, 1, hidden]
                # Concatenate along sequence dimension -> [batch, num_generated, hidden]
                generation_activations = torch.cat(captured_acts, dim=1)  # [batch, seq, hidden]
                generation_activations = generation_activations.squeeze(0)  # Remove batch -> [seq, hidden]
                
                print(f"[DEBUG] Captured {len(captured_acts)} token activations")
                print(f"[DEBUG] Final generation_activations shape: {generation_activations.shape}")
                
                # Cleanup
                del captured_acts
            else:
                print(f"[ERROR] No activations captured during generation!")
                
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
    
        # Decode completion
        if isinstance(out_ids, torch.Tensor):
            out_ids_tensor = out_ids
        else:
            out_ids_tensor = torch.tensor(out_ids, device=self.device)
        
        completion_ids = out_ids_tensor[0, prompt_len:]
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
            # Move back to original device for probe scoring
            metadata['generation_activations'] = generation_activations.to(self.device)
            metadata['generation_token_ids'] = completion_ids
            print(f"[DEBUG] Stored {generation_activations.shape[0]} token activations in metadata")
        else:
            print(f"[DEBUG] No activations to store in metadata")
    
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