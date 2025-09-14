import torch
import time
from typing import List, Dict, Any, Optional
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
import gc

from .base import BaseModelProvider, GenerationResult, estimate_tokens
from ..types import ModelConfig


class LocalModelProvider(BaseModelProvider):
    """Provider for local HuggingFace/TransformerLens models"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dtype = self._get_model_dtype()
        
        # Generation defaults for TransformerLens - INCREASED MAX LENGTH
        self.default_gen_kwargs = {
            "max_new_tokens": 256,  # Increased from 512
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "top_k": 50,
        }
        # Update with config kwargs
        self.default_gen_kwargs.update(config.generation_kwargs or {})
        
        # Load model and tokenizer
        self._load_model()
        
    def _get_model_dtype(self) -> torch.dtype:
        """Determine appropriate dtype for model"""
        bfloat16_models = ['llama', 'mistral', 'gemma', 'phi']
        if any(m in self.config.model_name.lower() for m in bfloat16_models):
            return torch.bfloat16
        return torch.float32
    
    def _load_model(self):
        """Load model and tokenizer"""
        print(f"Loading local model {self.config.model_name}...")
        
        try:
            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                use_fast=True
            )
            
            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = HookedTransformer.from_pretrained_no_processing(
                self.config.model_name,
                device=self.device,
                dtype=self.model_dtype
            )
            
            print(f"Successfully loaded {self.config.model_name}")
            print(f"Model dtype: {self.model_dtype}")
            print(f"Device: {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def generate(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> tuple:
        """Generate response from messages"""
        
        if not self.is_available():
            return "", {"error": "Model not available"}
        
        start_time = time.time()
        
        try:
            # Format messages using chat template
            formatted_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Count and print input tokens
            input_tokens = self.tokenizer(formatted_text, return_tensors="pt")
            input_length = input_tokens["input_ids"].shape[1]
            
            # Print token count for debugging
            print(f"[TOKEN COUNT] Input length: {input_length} tokens")
            
            # Check if input is too long (warn but continue)
            if input_length > 4096:
                print(f"[WARNING] Input length {input_length} exceeds typical context window")
            
            # Map standard generation kwargs to HookedTransformer format
            ht_gen_kwargs = {
                "max_new_tokens": kwargs.get("max_new_tokens", 
                                            self.default_gen_kwargs.get("max_new_tokens", 8192)),
                "temperature": kwargs.get("temperature", 
                                        self.default_gen_kwargs.get("temperature", 0.7)),
                "do_sample": kwargs.get("do_sample", 
                                       self.default_gen_kwargs.get("do_sample", True)),
                "top_p": kwargs.get("top_p", 
                                  self.default_gen_kwargs.get("top_p", 0.9)),
                "top_k": kwargs.get("top_k", 
                                  self.default_gen_kwargs.get("top_k", 50)),
                "stop_at_eos": True,
                "eos_token_id": self.tokenizer.eos_token_id,
                "prepend_bos": False,  # Already included in chat template
                "return_type": "str",
                "verbose": False
            }
            
            # Generate using HookedTransformer's method
            self.model.eval()
            with torch.no_grad():
                generated_text = self.model.generate(
                    formatted_text,  # Pass string directly
                    **ht_gen_kwargs
                )
            
            # Extract only the new generated content
            if generated_text.startswith(formatted_text):
                new_content = generated_text[len(formatted_text):].strip()
            else:
                new_content = generated_text.strip()
            
            # Calculate token usage
            output_tokens = self.tokenizer(new_content, return_tensors="pt")
            output_length = output_tokens["input_ids"].shape[1]
            total_tokens = input_length + output_length
            
            print(f"[TOKEN COUNT] Output length: {output_length} tokens, Total: {total_tokens} tokens")
            
            latency = time.time() - start_time
            
            self._update_usage(total_tokens)
            
            # Clean up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            metadata = {
                "input_tokens": input_length,
                "output_tokens": output_length,
                "total_tokens": total_tokens,
                "model_name": self.config.model_name,
                "device": str(self.device),
                "dtype": str(self.model_dtype),
                "latency": latency
            }
            
            return new_content, metadata
            
        except Exception as e:
            print(f"[ERROR] Generation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            # NO FALLBACK - just return error and exit
            raise e  # Re-raise the exception to stop execution
    
    def is_available(self) -> bool:
        """Check if model is loaded and available"""
        return (
            self.model is not None and 
            self.tokenizer is not None and
            (torch.cuda.is_available() if "cuda" in str(self.device) else True)
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self.is_available():
            return {}
        
        return {
            "model_name": self.config.model_name,
            "device": str(self.device),
            "dtype": str(self.model_dtype),
            "vocab_size": self.tokenizer.vocab_size,
            "max_position_embeddings": getattr(
                self.model.cfg, "n_ctx", "unknown"
            ),
            "hidden_size": getattr(self.model.cfg, "d_model", "unknown"),
            "num_layers": getattr(self.model.cfg, "n_layers", "unknown"),
            "num_attention_heads": getattr(
                self.model.cfg, "n_heads", "unknown"
            )
        }
    
    def clear_cache(self):
        """Clear GPU cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if hasattr(self, 'tokenizer'):
            del self.tokenizer
            self.tokenizer = None
        
        self.clear_cache()
        print(f"Unloaded model {self.config.model_name}")
