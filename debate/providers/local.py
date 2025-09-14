# debate/providers/local.py
"""
LocalModelProvider -- HuggingFace / Transformer-Lens backend
----------------------------------------------------------

Loads Llama-3 style RoPE models with dynamic memory allocation to avoid OOM.
Uses NTK interpolation for positions beyond 8192 to maintain coherent text.
"""

from __future__ import annotations

import gc
import time
import types
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoConfig, AutoTokenizer
from transformer_lens import HookedTransformer
from transformer_lens.components.abstract_attention import AbstractAttention

from .base import BaseModelProvider
from ..types import ModelConfig

# ────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────
ORIG_CTX = 8192  # Llama-3 was trained on 8192 positions
DEFAULT_MAX_CTX = 16384  # Conservative default to avoid OOM

# Debug log file
DEBUG_LOG_FILE = "rope_debug.txt"

def write_debug(message: str) -> None:
    """Write debug message to log file."""
    with open(DEBUG_LOG_FILE, "a") as f:
        f.write(f"{message}\n")

# ────────────────────────────────────────────────────────────────────────
# (A) Dynamic rotary embeddings with NTK interpolation
# ────────────────────────────────────────────────────────────────────────
def _get_rotary_embeddings(
    ctx_len: int,
    rotary_dim: int,
    rope_theta: float,
    rope_scaling: Dict,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates rotary embeddings using Llama 3.1's specific RoPE scaling.
    Fixed to avoid zero frequencies and properly handle the rotation.
    """
    dim_half = rotary_dim // 2
    
    # Get scaling parameters
    factor = rope_scaling.get("factor", 8.0)
    low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
    high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
    original_max_position_embeddings = rope_scaling.get("original_max_position_embeddings", 8192)
    
    write_debug(f"[DEBUG RoPE] Creating embeddings for ctx_len={ctx_len}")
    write_debug(f"[DEBUG RoPE] rope_theta={rope_theta}, factor={factor}")
    
    # Calculate base frequencies - this is correct
    freq_exponents = torch.arange(0, dim_half, dtype=torch.float32, device=device) / dim_half
    inv_freq_base = 1.0 / (rope_theta ** freq_exponents)
    
    write_debug(f"[DEBUG RoPE] Base frequencies - min: {inv_freq_base.min():.8f}, max: {inv_freq_base.max():.8f}")
    
    # For Llama 3.1: only apply scaling to lower frequencies
    # The key insight: we DON'T scale all frequencies equally
    inv_freq = inv_freq_base.clone()
    
    # Wavelength-based scaling (Llama 3.1 specific)
    wavelen = 2 * torch.pi / inv_freq_base
    
    # Only scale frequencies with wavelength > original_max_position_embeddings
    # This preserves high-frequency information
    for i in range(dim_half):
        if wavelen[i] > original_max_position_embeddings:
            # This is a low frequency - apply scaling
            ratio = wavelen[i] / original_max_position_embeddings
            # Smooth interpolation based on wavelength
            if ratio > low_freq_factor:
                inv_freq[i] = inv_freq_base[i] / factor
            elif ratio > high_freq_factor:
                # Interpolate between high_freq and low_freq zones
                smooth = (ratio - high_freq_factor) / (low_freq_factor - high_freq_factor)
                scale = 1.0 + smooth * (factor - 1.0)
                inv_freq[i] = inv_freq_base[i] / scale
    
    write_debug(f"[DEBUG RoPE] Scaled frequencies - min: {inv_freq.min():.8f}, max: {inv_freq.max():.8f}")
    
    # Generate position indices
    t = torch.arange(ctx_len, dtype=torch.float32, device=device)
    
    # Compute frequencies - outer product
    freqs = torch.outer(t, inv_freq)  # [ctx_len, dim_half]
    
    # Create cos/sin embeddings
    cos = freqs.cos().unsqueeze(0).unsqueeze(2)  # [1, ctx_len, 1, dim_half]
    sin = freqs.sin().unsqueeze(0).unsqueeze(2)  # [1, ctx_len, 1, dim_half]
    
    write_debug(f"[DEBUG RoPE] Output shapes - cos: {cos.shape}, sin: {sin.shape}")
    write_debug(f"[DEBUG RoPE] cos range: [{cos.min():.4f}, {cos.max():.4f}]")
    
    return cos.to(dtype), sin.to(dtype)


def _patched_apply_rotary(self, x, past_kv_pos_offset: int = 0, _mask=None):
    """
    Apply rotary embeddings - fixed for TransformerLens tensor format.
    TransformerLens uses [batch, seq, heads, d_head] format.
    """
    B, S, H, D = x.shape
    rotary_dim = self.cfg.rotary_dim or self.cfg.d_head
    
    write_debug(f"[DEBUG apply_rotary] Input shape: {x.shape}, past_kv_pos_offset={past_kv_pos_offset}")
    write_debug(f"[DEBUG apply_rotary] rotary_dim={rotary_dim}, d_head={D}")
    
    # Check if we need to recompute embeddings
    need_ctx = S + past_kv_pos_offset
    
    if (not hasattr(self, "_rope_cache_size") or 
        self._rope_cache_size < need_ctx):
        
        write_debug(f"[DEBUG apply_rotary] Recomputing RoPE for need_ctx={need_ctx}")
        
        # Compute rotary embeddings
        cos, sin = _get_rotary_embeddings(
            need_ctx,
            rotary_dim,
            self.rope_theta,
            self.rope_scaling,
            x.device,
            x.dtype,
        )
        
        # Cache for this forward pass
        self._rope_cos_cache = cos
        self._rope_sin_cache = sin
        self._rope_cache_size = need_ctx
    
    # Split x into rotary and pass-through parts
    if rotary_dim < D:
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
    else:
        x_rot = x
        x_pass = None
    
    write_debug(f"[DEBUG apply_rotary] x_rot shape: {x_rot.shape}")
    
    # Get the cos/sin for this sequence
    sl = slice(past_kv_pos_offset, past_kv_pos_offset + S)
    cos = self._rope_cos_cache[:, sl]  # [1, S, 1, dim_half]
    sin = self._rope_sin_cache[:, sl]
    
    # Reshape x_rot for rotation: [B, S, H, rotary_dim] -> [B, S, H, dim_half, 2]
    dim_half = rotary_dim // 2
    x_rot = x_rot.reshape(B, S, H, dim_half, 2)
    
    # Apply rotation using complex number formula
    # This is the standard RoPE rotation: (x0 + ix1) * (cos + isin)
    x0 = x_rot[..., 0]  # [B, S, H, dim_half]
    x1 = x_rot[..., 1]  # [B, S, H, dim_half]
    
    # Expand cos/sin to match batch size
    cos = cos.expand(B, -1, H, -1)  # [B, S, H, dim_half]
    sin = sin.expand(B, -1, H, -1)
    
    # Apply rotation
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    
    # Interleave back
    x_rotated = torch.stack([out0, out1], dim=-1).reshape(B, S, H, rotary_dim)
    
    # Concatenate with pass-through
    if x_pass is not None:
        result = torch.cat([x_rotated, x_pass], dim=-1)
    else:
        result = x_rotated
    
    write_debug(f"[DEBUG apply_rotary] Output shape: {result.shape}")
    write_debug(f"[DEBUG apply_rotary] Output sample values: {result[0, 0, 0, :5]}")
    
    return result
# ────────────────────────────────────────────────────────────────────────
# (C) Patched causal mask with dynamic generation
# ────────────────────────────────────────────────────────────────────────
def _patched_apply_causal_mask(
    self,
    attn_scores: torch.Tensor,
    past_kv_pos_offset: int = 0,
    attention_mask: Optional[torch.Tensor] = None,
):
    """Apply causal mask without pre-allocation."""
    B, H, Q, K = attn_scores.shape
    device = attn_scores.device
    
    # Generate causal mask on-the-fly (much cheaper than storing huge masks)
    q_pos = torch.arange(Q, device=device) + past_kv_pos_offset
    k_pos = torch.arange(K, device=device)
    
    # Broadcasting will handle the shape automatically
    causal = q_pos[:, None] >= k_pos[None, :]  # [Q, K]
    causal = causal[None, None, :, :]  # [1, 1, Q, K]
    
    if attention_mask is not None:
        causal = causal & attention_mask[:, None, None, :].bool()
    
    return torch.where(causal, attn_scores, self.IGNORE)

# ────────────────────────────────────────────────────────────────────────
# (D) Model patcher with memory-efficient settings
# ────────────────────────────────────────────────────────────────────────
def patch_llama3_rope_efficient(
    model: HookedTransformer, 
    hf_config: AutoConfig,  # Pass the HF config
    target_ctx: int = DEFAULT_MAX_CTX,
) -> HookedTransformer:
    """
    Patches model for extended context with minimal memory overhead.
    """
    # Get RoPE configuration from HF config
    rope_scaling = getattr(hf_config, "rope_scaling", {}) or {}
    rope_theta = getattr(hf_config, "rope_theta", 10000.0)
    
    write_debug(f"[DEBUG patch] Patching with rope_theta={rope_theta}")
    write_debug(f"[DEBUG patch] rope_scaling={rope_scaling}")
    
    # Update model config
    model.cfg.n_ctx = target_ctx
    if hasattr(model.cfg, "n_positions"):
        model.cfg.n_positions = target_ctx
    
    # Patch each attention layer
    for i, block in enumerate(model.blocks):
        attn = block.attn
        
        # Store RoPE parameters on the attention module
        attn.rope_theta = rope_theta
        attn.rope_scaling = rope_scaling
        
        # Clear any existing buffers to save memory
        for buffer_name in ["rotary_cos", "rotary_sin", "triangular_mask"]:
            if buffer_name in attn._buffers:
                delattr(attn, buffer_name)
        
        # Initialize cache tracking
        attn._rope_cache_size = 0
        
        # Monkey-patch methods
        attn.apply_rotary = types.MethodType(_patched_apply_rotary, attn)
        attn.apply_causal_mask = types.MethodType(_patched_apply_causal_mask, attn)
        
        if i == 0:  # Only write for first layer
            write_debug(f"[DEBUG patch] Layer {i} patched with rope_theta={rope_theta}")
    
    write_debug(f"✅ Patched for ctx={target_ctx} (Llama 3.1 RoPE)")
    return model

# ────────────────────────────────────────────────────────────────────────
# (E) Provider class with memory management
# ────────────────────────────────────────────────────────────────────────
class LocalModelProvider(BaseModelProvider):
    """Memory-efficient local model provider with dynamic context extension."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16  # Always use bf16 for Llama models
        
        # Conservative defaults to avoid OOM
        self.max_context = config.max_context if hasattr(config, 'max_context') else DEFAULT_MAX_CTX
        
        self.gen_defaults: Dict[str, Any] = {
            "max_new_tokens": 2048,  # Reasonable default
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "top_k": 50,
            "safety_margin": 256,
        }
        self.gen_defaults.update(config.generation_kwargs or {})
        
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[HookedTransformer] = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load model with memory-efficient settings."""
        write_debug(f"Loading {self.config.model_name} (max_ctx={self.max_context})")
        
        # Get HF config for RoPE settings
        hf_conf = AutoConfig.from_pretrained(
            self.config.model_name, 
            trust_remote_code=True
        )
        
        write_debug(f"[DEBUG load] HF config rope_theta: {getattr(hf_conf, 'rope_theta', 'NOT FOUND')}")
        write_debug(f"[DEBUG load] HF config rope_scaling: {getattr(hf_conf, 'rope_scaling', 'NOT FOUND')}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, 
            use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with minimal memory footprint
        self.model = HookedTransformer.from_pretrained_no_processing(
            self.config.model_name,
            device=self.device,
            dtype=self.dtype,
            default_padding_side="left",
        )
        
        # CRITICAL: Check if TransformerLens is already doing RoPE
        write_debug(f"[DEBUG] Model cfg.positional_embedding_type: {self.model.cfg.positional_embedding_type}")
        write_debug(f"[DEBUG] Model cfg.rotary_dim: {self.model.cfg.rotary_dim}")
        write_debug(f"[DEBUG] Model cfg.n_ctx original: {self.model.cfg.n_ctx}")
        
        # Apply memory-efficient RoPE patch
        rope_scaling = getattr(hf_conf, "rope_scaling", {})
        patch_llama3_rope_efficient(
            self.model, 
            hf_conf,
            target_ctx=self.max_context,
        )
        
        # Update tokenizer settings
        self.tokenizer.model_max_length = self.max_context
        
        # Clear any unnecessary caches
        self.clear_cache()
        
        write_debug(
            f"✓ Model loaded: dtype={self.dtype}, device={self.device}, "
            f"n_ctx={self.model.cfg.n_ctx}"
        )
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate with memory management."""
        if not self.is_available():
            return "", {"error": "model not initialized"}
        
        t0 = time.time()
        
        # Clear cache before generation
        self.clear_cache()
        
        # Prepare prompt
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=False,  # Don't truncate
        )["input_ids"].to(self.device)
        prompt_len = prompt_ids.shape[1]
        
        write_debug(f"[TOKENS] prompt = {prompt_len}")
        
        # Generation config
        cfg = {**self.gen_defaults, **kwargs}
        if "max_tokens" in cfg:
            cfg["max_new_tokens"] = cfg.pop("max_tokens")
        
        safety = cfg.pop("safety_margin", 256)
        
        # Check context limits
        ctx_limit = self.model.cfg.n_ctx
        needed_ctx = prompt_len + safety + cfg["max_new_tokens"]
        
        if needed_ctx > ctx_limit:
            write_debug(f"⚠️ Warning: needed context {needed_ctx} > limit {ctx_limit}")
            # Adjust max_new_tokens instead of truncating prompt
            cfg["max_new_tokens"] = max(1, ctx_limit - prompt_len - safety)
            write_debug(f"Adjusted max_new_tokens to {cfg['max_new_tokens']}")
        
        # Generate
        self.model.eval()
        with torch.no_grad():
            full = self.model.generate(
                prompt, 
                return_type="str",
                **cfg
            )
        
        # Extract completion
        completion = full[len(prompt):].lstrip()
        comp_tokens = len(self.tokenizer.encode(completion))
        latency = time.time() - t0
        
        # Update usage tracking
        self._update_usage(prompt_len + comp_tokens)
        
        # Clear cache after generation
        self.clear_cache()
        
        meta = {
            "prompt_tokens": prompt_len,
            "completion_tokens": comp_tokens,
            "total_tokens": prompt_len + comp_tokens,
            "latency": latency,
            "model": self.config.model_name,
            "n_ctx": self.model.cfg.n_ctx,
            "device": str(self.device),
            "dtype": str(self.dtype),
        }
        
        write_debug(
            f"[TOKENS] completion = {comp_tokens} | "
            f"total = {prompt_len + comp_tokens} | {latency:.2f}s"
        )
        
        return completion, meta
    
    def is_available(self) -> bool:
        return self.model is not None and self.tokenizer is not None
    
    def clear_cache(self) -> None:
        """Aggressively clear GPU cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def unload_model(self) -> None:
        """Fully unload model and free memory."""
        if self.model is not None:
            # Clear any cached RoPE embeddings
            for block in self.model.blocks:
                if hasattr(block.attn, "_rope_cos_cache"):
                    del block.attn._rope_cos_cache
                if hasattr(block.attn, "_rope_sin_cache"):
                    del block.attn._rope_sin_cache
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self.clear_cache()
        write_debug(f"Unloaded model {self.config.model_name}")