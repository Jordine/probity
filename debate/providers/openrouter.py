import os
import time
import requests
from typing import List, Dict, Any, Optional, Tuple

from .base import BaseModelProvider, GenerationResult, estimate_tokens


class OpenRouterProvider(BaseModelProvider):
    """Provider for OpenRouter API"""
    
    def __init__(self, config):
        super().__init__(config)
        self.api_key = config.api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key not provided")
        self.base_url = "https://openrouter.ai/api/v1"
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Tuple[str, Dict]:
        """Generate response from messages"""
        start_time = time.time()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Merge generation kwargs
        gen_kwargs = self.config.generation_kwargs.copy()
        gen_kwargs.update(kwargs)
        
        data = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": gen_kwargs.get("temperature", 0.7),
            "max_tokens": gen_kwargs.get("max_tokens", 512),
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
            tokens_used = result.get("usage", {}).get("total_tokens", estimate_tokens(content))
            
            self._update_usage(tokens_used)
            
            metadata = {
                "model": self.config.model_name,
                "tokens_used": tokens_used,
            }
            
            return content, metadata
            
        except Exception as e:
            print(f"OpenRouter API error: {e}")
            return "", {"error": str(e)}
    
    def is_available(self) -> bool:
        """Check if provider is available"""
        return self.api_key is not None