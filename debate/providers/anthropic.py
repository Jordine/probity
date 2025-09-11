import os
import time
from typing import List, Dict, Any, Optional, Tuple
import anthropic

from .base import BaseModelProvider, GenerationResult, estimate_tokens


class AnthropicProvider(BaseModelProvider):
    """Provider for Anthropic API"""
    
    def __init__(self, config):
        super().__init__(config)
        api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not provided")
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Tuple[str, Dict]:
        """Generate response from messages"""
        start_time = time.time()
        
        # Extract system message if present
        system = None
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                # Ensure proper role mapping
                role = msg["role"]
                if role == "assistant":
                    filtered_messages.append({"role": "assistant", "content": msg["content"]})
                else:
                    filtered_messages.append({"role": "user", "content": msg["content"]})
        
        try:
            # Merge generation kwargs
            gen_kwargs = (self.config.generation_kwargs or {}).copy()
            gen_kwargs.update(kwargs)
            
            response = self.client.messages.create(
                model=self.config.model_name,
                messages=filtered_messages,
                system=system,
                max_tokens=gen_kwargs.get("max_tokens", 512),
                temperature=gen_kwargs.get("temperature", 0.7),
            )
            
            content = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            
            self._update_usage(tokens_used)
            
            metadata = {
                "model": self.config.model_name,
                "tokens_used": tokens_used,
                "stop_reason": response.stop_reason,
                "latency": time.time() - start_time
            }
            
            return content, metadata
            
        except Exception as e:
            print(f"Anthropic API error: {e}")
            return "", {"error": str(e)}
    
    def is_available(self) -> bool:
        """Check if provider is available"""
        return hasattr(self.client, '_api_key') and self.client._api_key is not None