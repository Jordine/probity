from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import time
import asyncio
from dataclasses import dataclass

from ..types import ModelConfig, DebateTurn  # CORRECTED IMPORT


@dataclass
class GenerationResult:
    """Result from model generation"""
    content: str
    tokens_used: int
    latency: float
    metadata: Dict[str, Any]
    success: bool = True
    error: Optional[str] = None


class BaseModelProvider(ABC):
    """Abstract base class for all model providers"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.total_tokens_used = 0
        self.total_requests = 0
        self.total_cost = 0.0
        
    @abstractmethod
    def generate(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> GenerationResult:
        """Generate response from messages"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available/configured"""
        pass
    
    def format_conversation(
        self, 
        conversation: List[DebateTurn],  # CORRECTED TYPE HINT
        role_mapping: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        """Convert conversation turns to provider's message format"""
        
        if role_mapping is None:
            role_mapping = {
                "system": "system",
                "honest": "user", 
                "dishonest": "user",
                "judge": "assistant"
            }
        
        messages = []
        
        for turn in conversation:
            speaker_role = turn.speaker.value # Use the enum value
            if speaker_role in role_mapping:
                role = role_mapping[speaker_role]
                
                content = turn.content
                if speaker_role in ["honest", "dishonest"] and role == "user":
                    content = f"[{speaker_role.upper()}]: {content}"
                
                messages.append({
                    "role": role,
                    "content": content
                })
        
        return messages
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "total_tokens_used": self.total_tokens_used,
            "total_requests": self.total_requests,
            "total_cost": self.total_cost,
            "avg_tokens_per_request": (
                self.total_tokens_used / max(1, self.total_requests)
            )
        }
    
    def _update_usage(self, tokens_used: int, cost: float = 0.0):
        """Update usage statistics"""
        self.total_tokens_used += tokens_used
        self.total_requests += 1
        self.total_cost += cost


class RetryableMixin:
    """Mixin for providers that support retries"""
    
    def generate_with_retry(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ) -> GenerationResult:
        """Generate with retry logic"""
        
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                result = self.generate(messages, **kwargs)
                if result.success:
                    return result
                last_error = result.error
                
            except Exception as e:
                last_error = str(e)
                
            if attempt < max_retries:
                print(f"Attempt {attempt + 1} failed, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
        
        return GenerationResult(
            content="",
            tokens_used=0,
            latency=0.0,
            metadata={},
            success=False,
            error=f"Failed after {max_retries + 1} attempts. Last error: {last_error}"
        )


class AsyncModelProvider(BaseModelProvider):
    """Base class for async model providers"""
    
    @abstractmethod
    async def generate_async(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> GenerationResult:
        """Async generation method"""
        pass
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> GenerationResult:
        """Sync wrapper for async generation"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.generate_async(messages, **kwargs))


class RateLimitedMixin:
    """Mixin for providers that need rate limiting"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_request_time = 0.0
        self.min_request_interval = 1.0  # Minimum seconds between requests
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()


# Utility functions for common provider operations

def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars per token)"""
    return len(text) // 4


def format_system_message(content: str) -> Dict[str, str]:
    """Format system message"""
    return {"role": "system", "content": content}


def format_user_message(content: str) -> Dict[str, str]:
    """Format user message"""
    return {"role": "user", "content": content}


def format_assistant_message(content: str) -> Dict[str, str]:
    """Format assistant message"""
    return {"role": "assistant", "content": content}


def calculate_openai_cost(
    model: str, 
    input_tokens: int, 
    output_tokens: int
) -> float:
    """Calculate OpenAI API costs (rough estimates)"""
    
    # Pricing as of 2024 (may need updates)
    pricing = {
        "gpt-4-turbo-preview": {"input": 0.01/1000, "output": 0.03/1000},
        "gpt-4": {"input": 0.03/1000, "output": 0.06/1000},
        "gpt-3.5-turbo": {"input": 0.001/1000, "output": 0.002/1000},
        "gpt-3.5-turbo-16k": {"input": 0.003/1000, "output": 0.004/1000}
    }
    
    if model not in pricing:
        return 0.0
    
    rates = pricing[model]
    return input_tokens * rates["input"] + output_tokens * rates["output"]


def calculate_anthropic_cost(
    model: str,
    input_tokens: int,
    output_tokens: int
) -> float:
    """Calculate Anthropic API costs (rough estimates)"""
    
    pricing = {
        "claude-3-opus-20240229": {"input": 0.015/1000, "output": 0.075/1000},
        "claude-3-sonnet-20240229": {"input": 0.003/1000, "output": 0.015/1000},
        "claude-3-haiku-20240307": {"input": 0.00025/1000, "output": 0.00125/1000}
    }
    
    if model not in pricing:
        return 0.0
    
    rates = pricing[model]
    return input_tokens * rates["input"] + output_tokens * rates["output"]