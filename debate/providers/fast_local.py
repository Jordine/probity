# debate/providers/fast_local.py
"""
FastLocalModelProvider - uses plain transformers + native pytorch hooks.

This is a drop-in replacement for LocalModelProvider that avoids the
TransformerLens overhead. Key differences:
- Uses AutoModelForCausalLM with device_map for multi-GPU
- Native pytorch hooks via register_forward_hook()
- Manual generation loop with KV cache for activation capture
- ~2-4x memory reduction expected

Based on patterns from obalcells/hallucination_probes.
"""

import torch
import torch.nn as nn
import time
import gc
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

from .base import BaseModelProvider
from ..types import ModelConfig


MAX_CONTEXT = 16384
SAFETY_MARGIN = 256


class FastLocalModelProvider(BaseModelProvider):
    """
    Fast provider for local models using plain transformers + native hooks.

    Supports multi-GPU via device_map with explicit GPU assignment.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        # GPU configuration
        self.gpu_indices = getattr(config, 'gpu_indices', None)
        self.device_id = config.device_id

        # Determine primary device for tokenizer outputs etc
        if self.gpu_indices and len(self.gpu_indices) > 0:
            self.primary_device = torch.device(f"cuda:{self.gpu_indices[0]}")
            print(f"[FastLocalModelProvider] Using GPUs: {self.gpu_indices}")
        elif self.device_id is not None:
            self.primary_device = torch.device(f"cuda:{self.device_id}")
        else:
            self.primary_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[FastLocalModelProvider] Primary device: {self.primary_device}")

        self.model_dtype = self._get_model_dtype()
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
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

        self.generation_count = 0

    def _get_model_dtype(self) -> torch.dtype:
        bf16_families = ["llama", "mistral", "gemma", "phi", "qwen"]
        return torch.bfloat16 if any(m in self.config.model_name.lower() for m in bf16_families) else torch.float32

    def _build_device_map(self) -> Dict[str, Any]:
        """
        Build device_map for multi-GPU distribution.

        If gpu_indices specified (e.g., [0,1,2,3]), constrains model to those GPUs.
        Otherwise uses "auto" for automatic distribution.
        """
        if not self.gpu_indices or len(self.gpu_indices) == 0:
            return "auto"

        if len(self.gpu_indices) == 1:
            # Single GPU - put everything on that device
            return {"": self.gpu_indices[0]}

        # Multi-GPU: use max_memory to constrain to specific GPUs
        # We'll let accelerate figure out the layer distribution
        # by only allowing memory on our specified GPUs
        max_memory = {}

        # Get total memory per GPU, allocate ~90% to leave headroom
        for gpu_idx in self.gpu_indices:
            props = torch.cuda.get_device_properties(gpu_idx)
            # Use 90% of available memory
            max_mem_gb = int(props.total_memory * 0.9 / (1024**3))
            max_memory[gpu_idx] = f"{max_mem_gb}GB"

        # Set other GPUs to 0 to exclude them
        for i in range(torch.cuda.device_count()):
            if i not in self.gpu_indices:
                max_memory[i] = "0GB"

        print(f"[FastLocalModelProvider] max_memory config: {max_memory}")
        return max_memory

    def _load_model(self):
        print(f"[FastLocalModelProvider] Loading {self.config.model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=True,
            token=os.environ.get("HF_TOKEN")
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Build device map
        device_map_or_memory = self._build_device_map()

        load_kwargs = {
            "torch_dtype": self.model_dtype,
            "trust_remote_code": True,
            "token": os.environ.get("HF_TOKEN"),
            "attn_implementation": "sdpa",  # Use Flash Attention 2 / SDPA
        }

        if isinstance(device_map_or_memory, dict) and all(isinstance(v, str) and "GB" in v for v in device_map_or_memory.values()):
            # It's a max_memory dict
            load_kwargs["device_map"] = "auto"
            load_kwargs["max_memory"] = device_map_or_memory
        else:
            load_kwargs["device_map"] = device_map_or_memory

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **load_kwargs
        )

        self.model.eval()

        # Report memory usage
        if self.gpu_indices and len(self.gpu_indices) > 0:
            for gpu_idx in self.gpu_indices:
                allocated = torch.cuda.memory_allocated(gpu_idx) / 1024**3
                reserved = torch.cuda.memory_reserved(gpu_idx) / 1024**3
                print(f"  ✓ GPU {gpu_idx} - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

        # Store model's hidden size and layer count for hooks
        self.hidden_size = self.model.config.hidden_size
        self.n_layers = self.model.config.num_hidden_layers

        print(f"[FastLocalModelProvider] Model loaded: {self.n_layers} layers, hidden_size={self.hidden_size}")

    def _get_layer_module(self, layer_idx: int) -> nn.Module:
        """Get the decoder layer module at the specified index."""
        # Handle different model architectures
        if hasattr(self.model, 'model'):
            # LLaMA, Mistral, Qwen style
            if hasattr(self.model.model, 'layers'):
                return self.model.model.layers[layer_idx]

        if hasattr(self.model, 'transformer'):
            # GPT-2 style
            if hasattr(self.model.transformer, 'h'):
                return self.model.transformer.h[layer_idx]

        raise ValueError(f"Unknown model architecture for {self.config.model_name}")

    @torch.inference_mode()
    def generate(
        self,
        messages: List[Dict[str, str]],
        capture_activations_layer: Optional[int] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate with optional activation capture for probe scoring.

        Uses manual generation loop with KV cache when capturing activations
        for per-token activation collection.

        Args:
            messages: Conversation history
            capture_activations_layer: If provided, capture activations at this layer
            **kwargs: Generation parameters

        Returns:
            (completion_text, metadata) where metadata may include 'generation_activations'
        """
        if not self.is_available():
            return "", {"error": "Model not available"}

        self.generation_count += 1
        start = time.time()

        # Build the chat prompt
        prompt_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = self.tokenizer(
            prompt_str,
            return_tensors="pt",
            add_special_tokens=False
        )
        input_ids = inputs["input_ids"].to(self.primary_device)
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).to(self.primary_device)

        prompt_len = input_ids.shape[1]

        # Generation config
        gen_cfg = {**self.default_gen_kwargs, **kwargs}
        max_new = gen_cfg.get("max_new_tokens", 256)
        temperature = gen_cfg.get("temperature", 0.7)
        do_sample = gen_cfg.get("do_sample", True)
        top_p = gen_cfg.get("top_p", 0.9)
        top_k = gen_cfg.get("top_k", 50)

        # Context length protection
        if prompt_len + max_new + SAFETY_MARGIN > MAX_CONTEXT:
            max_new = max(1, MAX_CONTEXT - prompt_len - SAFETY_MARGIN)
            print(f"[FastLocalModelProvider] Truncating max_new_tokens to {max_new}")

        generation_activations = None
        generated_tokens = []

        if capture_activations_layer is not None:
            # Manual generation loop with activation capture
            generation_activations, generated_tokens = self._generate_with_activation_capture(
                input_ids=input_ids,
                attention_mask=attention_mask,
                layer_idx=capture_activations_layer,
                max_new_tokens=max_new,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
            )
        else:
            # Standard HF generate (faster when we don't need activations)
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new,
                temperature=temperature if do_sample else None,
                do_sample=do_sample,
                top_p=top_p if do_sample else None,
                top_k=top_k if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
            generated_tokens = outputs[0, prompt_len:].tolist()

        # Decode
        completion_str = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).lstrip()

        # Metadata
        output_len = len(generated_tokens)
        total_tokens = prompt_len + output_len
        latency = time.time() - start
        self._update_usage(total_tokens)

        metadata = {
            "input_tokens": int(prompt_len),
            "output_tokens": int(output_len),
            "total_tokens": int(total_tokens),
            "latency": latency,
            "model": self.config.model_name,
            "device": str(self.primary_device),
            "dtype": str(self.model_dtype),
            "generation_count": self.generation_count,
            "provider": "fast_local",
        }

        # Add activations if captured
        if generation_activations is not None:
            metadata['generation_activations'] = generation_activations
            metadata['generation_token_ids'] = torch.tensor(generated_tokens)

        return completion_str, metadata

    def _generate_with_activation_capture(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_idx: int,
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        top_p: float,
        top_k: int,
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Manual generation loop that captures activations at each token.

        Uses KV cache for efficiency - only processes one new token per step.
        """
        target_layer = self._get_layer_module(layer_idx)

        generated_tokens = []
        captured_activations = []

        past_key_values = None
        current_input_ids = input_ids
        current_attention_mask = attention_mask

        for step in range(max_new_tokens):
            # Set up hook to capture hidden states
            hidden_state_captured = [None]

            def hook_fn(module, input, output):
                # output is typically (hidden_states, ...) tuple
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                # Capture last token's hidden state
                hidden_state_captured[0] = hidden_states[:, -1:, :].detach().cpu()

            handle = target_layer.register_forward_hook(hook_fn)

            try:
                # Forward pass
                if past_key_values is None:
                    # First pass: process full prompt
                    outputs = self.model(
                        input_ids=current_input_ids,
                        attention_mask=current_attention_mask,
                        use_cache=True,
                        return_dict=True,
                    )
                else:
                    # Subsequent passes: only process new token
                    outputs = self.model(
                        input_ids=current_input_ids[:, -1:],
                        attention_mask=current_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )

                past_key_values = outputs.past_key_values

            finally:
                handle.remove()

            # Sample next token
            logits = outputs.logits[:, -1, :]

            if do_sample and temperature > 0:
                # Apply temperature
                logits = logits / temperature

                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')

                # Sample
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = logits.argmax(dim=-1, keepdim=True)

            next_token_id = next_token.item()

            # Check for EOS
            if next_token_id == self.tokenizer.eos_token_id:
                break

            # Store token and activation
            generated_tokens.append(next_token_id)
            if hidden_state_captured[0] is not None:
                captured_activations.append(hidden_state_captured[0])

            # Update for next iteration
            current_input_ids = torch.cat([current_input_ids, next_token], dim=-1)
            current_attention_mask = torch.cat([
                current_attention_mask,
                torch.ones((1, 1), dtype=current_attention_mask.dtype, device=current_attention_mask.device)
            ], dim=-1)

        # Concatenate activations: [num_tokens, hidden_size]
        if captured_activations:
            generation_activations = torch.cat(captured_activations, dim=1).squeeze(0)
        else:
            generation_activations = torch.empty(0, self.hidden_size)

        return generation_activations, generated_tokens

    def is_available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def clear_cache(self):
        if torch.cuda.is_available():
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

    # Compatibility properties to match LocalModelProvider interface
    @property
    def device(self) -> torch.device:
        """Primary device for compatibility with existing code."""
        return self.primary_device
