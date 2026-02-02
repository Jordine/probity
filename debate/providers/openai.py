import os
import time
from typing import List, Dict, Any, Optional, Tuple
import openai

from .base import BaseModelProvider, GenerationResult, estimate_tokens


class OpenAIProvider(BaseModelProvider):
    """Provider for OpenAI API"""
    
    def __init__(self, config):
        super().__init__(config)
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Tuple[str, Dict]:
        """Generate response from messages"""
        start_time = time.time()
        
        try:
            # Merge generation kwargs
            gen_kwargs = (self.config.generation_kwargs or {}).copy()
            gen_kwargs.update(kwargs)
            
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=gen_kwargs.get("temperature", 0.7),
                max_tokens=gen_kwargs.get("max_tokens", 512),
            )
            
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if response.usage else estimate_tokens(content)
            
            self._update_usage(tokens_used)
            
            metadata = {
                "model": self.config.model_name,
                "tokens_used": tokens_used,
                "finish_reason": response.choices[0].finish_reason,
                "latency": time.time() - start_time
            }
            
            return content, metadata
            
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return "", {"error": str(e)}
    
    def is_available(self) -> bool:
        """Check if provider is available"""
        try:
            # Just check if we have a valid client with API key
            return self.client.api_key is not None
        except AttributeError:
            return False