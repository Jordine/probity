import os
import time
import json
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
import openai
import anthropic
import requests

from .types import ModelConfig, ProviderType, DebateRole


class BaseProvider:
    """Base class for all model providers"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.total_tokens = 0
        self.total_cost = 0.0
        
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Tuple[str, Dict]:
        """Generate response and return (content, metadata)"""
        raise NotImplementedError
        
    def format_messages(self, messages: List[Dict[str, str]]) -> Any:
        """Format messages for the specific provider"""
        raise NotImplementedError
        
    def is_available(self) -> bool:
        """Check if provider is available"""
        raise NotImplementedError


class LocalProvider(BaseProvider):
    """Provider for local HuggingFace models"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()
        
    def _load_model(self):
        """Load model and tokenizer"""
        print(f"Loading local model {self.config.model_name}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Determine dtype
        if 'llama' in self.config.model_name.lower():
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
            
        # Load model using TransformerLens for probe compatibility
        self.model = HookedTransformer.from_pretrained_no_processing(
            self.config.model_name,
            device=self.device,
            dtype=dtype
        )
        self.model.eval()
        print(f"Model loaded on {self.device}")
        
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Tuple[str, Dict]:
        """Generate response from messages"""
        start_time = time.time()
        
        # Format messages using chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_length = inputs["input_ids"].shape[1]
        
        # Generate
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                **self.config.generation_kwargs,
                **kwargs
            }
            
            outputs = self.model.generate(
                inputs["input_ids"],
                **gen_kwargs
            )
            
        # Decode
        generated = outputs[0][input_length:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        
        # Calculate tokens
        output_tokens = len(generated)
        total_tokens = input_length + output_tokens
        self.total_tokens += total_tokens
        
        metadata = {
            "model": self.config.model_name,
            "input_tokens": input_length,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "latency": time.time() - start_time
        }
        
        return response.strip(), metadata
        
    def get_activations(self, text: str, layer: int) -> torch.Tensor:
        """Get activations for probe scoring (only for local models)"""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        hook_point = f"blocks.{layer}.hook_resid_pre"
        
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                inputs["input_ids"],
                names_filter=[hook_point],
                return_cache_object=True,
                stop_at_layer=layer + 1
            )
            
        return cache[hook_point]
        
    def is_available(self) -> bool:
        return self.model is not None


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI API"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = openai.OpenAI(api_key=config.api_key or os.getenv("OPENAI_API_KEY"))
        
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Tuple[str, Dict]:
        """Generate response using OpenAI API"""
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 512),
                **self.config.generation_kwargs
            )
            
            content = response.choices[0].message.content
            
            # Track usage
            if response.usage:
                self.total_tokens += response.usage.total_tokens
                self.total_cost += self._calculate_cost(response.usage)
                
            metadata = {
                "model": self.config.model_name,
                "finish_reason": response.choices[0].finish_reason,
                "usage": response.usage.dict() if response.usage else {},
                "latency": time.time() - start_time
            }
            
            return content, metadata
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return "", {"error": str(e)}
            
    def _calculate_cost(self, usage) -> float:
        """Calculate API cost"""
        # Rough pricing as of 2024
        pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015}
        }
        
        model_key = "gpt-4" if "gpt-4" in self.config.model_name else "gpt-3.5-turbo"
        rates = pricing.get(model_key, pricing["gpt-3.5-turbo"])
        
        return (usage.prompt_tokens * rates["input"] + 
                usage.completion_tokens * rates["output"]) / 1000
                
    def is_available(self) -> bool:
        try:
            self.client.models.list()
            return True
        except:
            return False


class AnthropicProvider(BaseProvider):
    """Provider for Anthropic API"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = anthropic.Anthropic(
            api_key=config.api_key or os.getenv("ANTHROPIC_API_KEY")
        )
        
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Tuple[str, Dict]:
        """Generate response using Anthropic API"""
        start_time = time.time()
        
        # Extract system message if present
        system = None
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                filtered_messages.append(msg)
                
        try:
            response = self.client.messages.create(
                model=self.config.model_name,
                messages=filtered_messages,
                system=system,
                max_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.7),
                **self.config.generation_kwargs
            )
            
            content = response.content[0].text
            
            # Track usage
            self.total_tokens += response.usage.input_tokens + response.usage.output_tokens
            
            metadata = {
                "model": self.config.model_name,
                "stop_reason": response.stop_reason,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                "latency": time.time() - start_time
            }
            
            return content, metadata
            
        except Exception as e:
            print(f"Anthropic API error: {e}")
            return "", {"error": str(e)}
            
    def is_available(self) -> bool:
        try:
            # Simple check - try to get models
            return self.client.api_key is not None
        except:
            return False


class OpenRouterProvider(BaseProvider):
    """Provider for OpenRouter API"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.api_key = config.api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = config.base_url or "https://openrouter.ai/api/v1"
        
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Tuple[str, Dict]:
        """Generate response using OpenRouter API"""
        start_time = time.time()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/probity-debate",  # Optional
            "X-Title": "Probity Debate System"  # Optional
        }
        
        data = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 512),
            **self.config.generation_kwargs
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Track usage if available
            if "usage" in result:
                self.total_tokens += result["usage"]["total_tokens"]
                
            metadata = {
                "model": self.config.model_name,
                "usage": result.get("usage", {}),
                "latency": time.time() - start_time
            }
            
            return content, metadata
            
        except Exception as e:
            print(f"OpenRouter API error: {e}")
            return "", {"error": str(e)}
            
    def is_available(self) -> bool:
        return self.api_key is not None


def create_provider(config: ModelConfig) -> BaseProvider:
    """Factory function to create appropriate provider"""
    if config.provider == ProviderType.LOCAL:
        return LocalProvider(config)
    elif config.provider == ProviderType.OPENAI:
        return OpenAIProvider(config)
    elif config.provider == ProviderType.ANTHROPIC:
        return AnthropicProvider(config)
    elif config.provider == ProviderType.OPENROUTER:
        return OpenRouterProvider(config)
    else:
        raise ValueError(f"Unknown provider type: {config.provider}")