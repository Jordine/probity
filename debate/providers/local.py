# debate/providers/local.py
"""
LocalModelProvider -- HuggingFace / Transformer-Lens backend
Works with Llama-3 (and any other RoPE model) while dynamically patching
rotary-positional embeddings and causal masks so that the context length can
be extended far beyond the 2 048 default.

The RoPE patch is exactly the same as the one shown in the prompt.
"""

from __future__ import annotations

import gc
import time
import types
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from transformer_lens.components.abstract_attention import AbstractAttention

from .base import BaseModelProvider
from ..types import ModelConfig

# ──────────────────────────────────────────────────────────────────────────────
# (A) triangular / causal-mask initialiser
# ──────────────────────────────────────────────────────────────────────────────
def _init_triangular_mask(attn_module, n_ctx: int, device=None):
    if device is None:
        try:
            device = next(attn_module.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    mask = torch.tril(
        torch.ones((n_ctx, n_ctx), dtype=torch.bool, device=device)
    )[None, None, ...]  # (1,1,n_ctx,n_ctx)

    if "triangular_mask" in attn_module._buffers:
        delattr(attn_module, "triangular_mask")
    attn_module.register_buffer("triangular_mask", mask, persistent=False)


# ──────────────────────────────────────────────────────────────────────────────
# (B) rotary-embedding tables
# ──────────────────────────────────────────────────────────────────────────────
def _init_rotary_embeddings(attn_module, ctx_len, rotary_dim, scale=1.0, device=None):
    dim_half = rotary_dim // 2
    if device is None:
        try:
            device = next(attn_module.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(dim_half, device=device).float() / dim_half)
    ) / scale
    t = torch.arange(ctx_len, device=device, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)

    cos = freqs.cos()[None, :, None, :, None]
    sin = freqs.sin()[None, :, None, :, None]

    for name, buf in {"rotary_cos": cos, "rotary_sin": sin}.items():
        if name in attn_module._buffers:
            delattr(attn_module, name)
        attn_module.register_buffer(name, buf, persistent=False)


# ──────────────────────────────────────────────────────────────────────────────
# (C) patched helpers injected into every attention module
# ──────────────────────────────────────────────────────────────────────────────
def _patched_apply_rotary(self, x, past_kv_pos_offset=0, _mask=None):
    B, S, H, D = x.shape
    rotary_dim = self.cfg.rotary_dim or self.cfg.d_head

    need_ctx = S + past_kv_pos_offset
    if (not hasattr(self, "rotary_cos")) or self.rotary_cos.shape[1] < need_ctx:
        _init_rotary_embeddings(
            self,
            need_ctx * 2,
            rotary_dim,
            scale=self.rotary_cos_scale,
        )

    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x_rot = x_rot.view(B, S, H, -1, 2)

    sl = slice(past_kv_pos_offset, past_kv_pos_offset + S)
    cos = self.rotary_cos[:, sl]
    sin = self.rotary_sin[:, sl]

    x_rot_swapped = torch.stack((-x_rot[..., 1], x_rot[..., 0]), dim=-1)
    x_rotated = x_rot * cos + x_rot_swapped * sin
    x_rotated = x_rotated.reshape(B, S, H, rotary_dim)

    return torch.cat([x_rotated, x_pass], dim=-1)


def _patched_apply_causal_mask(
    self, attn_scores, past_kv_pos_offset=0, attention_mask=None
):
    """
    attn_scores : [B, H, Q, K] -- scores *before* masking
    past_kv_pos_offset : how many cached tokens are already in K
    attention_mask : optional padding mask broadcastable to [B, 1, 1, K]
    """
    B, H, Q, K = attn_scores.shape
    device = attn_scores.device

    q_pos = (torch.arange(Q, device=device) + past_kv_pos_offset)[:, None]  # [Q,1]
    k_pos = torch.arange(K, device=device)[None, :]  # [1,K]
    causal = (q_pos >= k_pos).view(1, 1, Q, K)  # [1,1,Q,K]

    if attention_mask is not None:
        pad_mask = attention_mask[:, None, None, :].bool()  # [B,1,1,K]
        causal = causal & pad_mask

    ignore = self.IGNORE  # already set on AbstractAttention
    return torch.where(causal, attn_scores, ignore)


# ──────────────────────────────────────────────────────────────────────────────
# (D) high-level patcher
# ──────────────────────────────────────────────────────────────────────────────
def patch_llama31_rope(model: HookedTransformer, target_ctx: int = 16_384):
    model.cfg.n_ctx = target_ctx
    if hasattr(model.cfg, "n_positions"):
        model.cfg.n_positions = target_ctx

    scale = target_ctx / 8192  # model was trained with 8 192 positions

    for block in model.blocks:
        attn = block.attn
        rotary_dim = model.cfg.rotary_dim or model.cfg.d_head

        # RoPE tables
        attn.rotary_cos_scale = scale
        _init_rotary_embeddings(attn, target_ctx, rotary_dim, scale)

        # causal mask
        _init_triangular_mask(attn, target_ctx)

        # monkey-patch the methods
        attn.apply_rotary = types.MethodType(_patched_apply_rotary, attn)
        attn.apply_causal_mask = types.MethodType(_patched_apply_causal_mask, attn)

    print(f"✅  Patched for ctx={target_ctx} (stretch ×{scale:.2f})")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# provider helpers
# ──────────────────────────────────────────────────────────────────────────────
def _recommended_dtype(model_name: str) -> torch.dtype:
    bf16_families = ("llama", "mistral", "gemma", "phi", "qwen")
    return (
        torch.bfloat16
        if any(x in model_name.lower() for x in bf16_families)
        else torch.float32
    )


# ──────────────────────────────────────────────────────────────────────────────
# provider
# ──────────────────────────────────────────────────────────────────────────────
class LocalModelProvider(BaseModelProvider):
    """Loads a local model with Transformer-Lens and auto-extends its context."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = _recommended_dtype(config.model_name)

        # default generation parameters
        self.gen_defaults: Dict[str, Any] = {
            "max_new_tokens": 8192,
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

    # ──────────────────────────────────────────────────────────────────────
    # model loading
    # ──────────────────────────────────────────────────────────────────────
    def _load_model(self) -> None:
        print(f"Loading {self.config.model_name}")

        # tokenizer -----------------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # model ---------------------------------------------------------------
        #
        # We load *without* HF's built-in processing so that we can apply our
        # own patch afterwards.
        #
        self.model = HookedTransformer.from_pretrained_no_processing(
            self.config.model_name,
            device=self.device,
            dtype=self.dtype,
            names_filter="blocks.16.hook_resid_pre",
        )

        # extend context ------------------------------------------------------
        target_ctx = 32_768
        patch_llama31_rope(self.model, target_ctx)

        # final bookkeeping ---------------------------------------------------
        self.tokenizer.model_max_length = target_ctx

        print(
            f"✓ {self.config.model_name} loaded "
            f"(dtype={self.dtype}, device={self.device}, n_ctx={self.model.cfg.n_ctx})"
        )

    # ──────────────────────────────────────────────────────────────────────
    # generation
    # ──────────────────────────────────────────────────────────────────────
    def generate(
        self, messages: List[Dict[str, str]], **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        if not self.is_available():
            return "", {"error": "model not initialised"}

        t0 = time.time()

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(
            self.device
        )
        prompt_len = prompt_ids.shape[1]
        print(f"[TOKEN COUNT] prompt = {prompt_len}")

        # backward-compatibility with OpenAI-style kwarg
        if "max_tokens" in self.gen_defaults:
            self.gen_defaults["max_new_tokens"] = self.gen_defaults.pop("max_tokens")

        cfg = {**self.gen_defaults, **kwargs}
        safety = cfg.pop("safety_margin")
        ctx_limit = self.model.cfg.n_ctx
        needed_ctx = prompt_len + safety + cfg["max_new_tokens"]

        if needed_ctx > ctx_limit:
            excess = needed_ctx - ctx_limit
            print(f"[CTX-TRUNCATE] dropping {excess} oldest tokens")
            prompt_ids = prompt_ids[:, -ctx_limit + safety :]
            prompt = self.tokenizer.decode(
                prompt_ids[0],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            prompt_len = prompt_ids.shape[1]

        self.model.eval()
        with torch.no_grad():
            full = self.model.generate(prompt, return_type="str", **cfg)
        completion = full[len(prompt) :].lstrip()

        comp_tokens = self.tokenizer(completion, return_tensors="pt")[
            "input_ids"
        ].shape[1]
        latency = time.time() - t0
        self._update_usage(prompt_len + comp_tokens)

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
        print(
            f"[TOKEN COUNT] completion = {comp_tokens} | "
            f"total = {prompt_len + comp_tokens} | {latency:.2f}s"
        )
        return completion, meta

    # ──────────────────────────────────────────────────────────────────────
    # misc helpers
    # ──────────────────────────────────────────────────────────────────────
    def is_available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def clear_cache(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def unload_model(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.clear_cache()
        print(f"Unloaded model {self.config.model_name}")