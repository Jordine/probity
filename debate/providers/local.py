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
        
        # Generation defaults - FIXED: Define before loading model
        self.default_gen_kwargs = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "do_sample": True,
            "repetition_penalty": 1.1,
            "top_p": 0.9,
            "top_k": 50,
            "pad_token_id": None
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
            
            # Update pad_token_id in gen kwargs
            self.default_gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
            
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
    ) -> tuple:  # FIXED: Return tuple as expected by DebateManager
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
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_text,
                return_tensors="pt",
                padding=False,
                truncation=False,
                add_special_tokens=False
            ).to(self.device)
            
            input_length = inputs["input_ids"].shape[1]
            
            # Prepare generation kwargs
            gen_kwargs = self.default_gen_kwargs.copy()
            gen_kwargs.update(kwargs)
            
            # Generate
            self.model.eval()
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    **gen_kwargs
                )
            
            # Decode only the new tokens
            new_tokens = outputs[0][input_length:]
            generated_text = self.tokenizer.decode(
                new_tokens, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Calculate usage
            output_length = len(new_tokens)
            total_tokens = input_length + output_length
            latency = time.time() - start_time
            
            self._update_usage(total_tokens)
            
            # Clean up memory
            del outputs
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
            
            return generated_text.strip(), metadata
            
        except Exception as e:
            print(f"Generation failed: {str(e)}")
            return "", {"error": f"Generation failed: {str(e)}"}
    
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