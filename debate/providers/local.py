# debate/providers/local.py
"""
LocalModelProvider -- HuggingFace / Transformer-Lens backend
Works with Llama-3 and any other RoPE model.
"""

from __future__ import annotations

import gc
import time
from typing import Any, Dict, Optional

import torch
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

from .base import BaseModelProvider
from ..types import ModelConfig


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _recommended_dtype(model_name: str) -> torch.dtype:
    bf16_families = ("llama", "mistral", "gemma", "phi", "qwen")
    return (
        torch.bfloat16
        if any(x in model_name.lower() for x in bf16_families)
        else torch.float32
    )


def _infer_rope_len(model: HookedTransformer) -> int:
    """
    Inspect the first block's rotary cache to know how long it really is.
    Works for both GQA (Llama-3) and older MHA blocks.
    """
    blk = model.blocks[0]
    rot = getattr(blk.attn, "rotary_emb", None)
    if rot is None:  # fallback for older MHA
        inner = getattr(blk.attn, "inner_attn", None)
        rot = getattr(inner, "rotary_emb", None) if inner else None
    if rot is None:
        raise RuntimeError("Could not locate rotary_emb to infer context length")
    return rot.cache_cos.size(0)  # sequence length


# --------------------------------------------------------------------------- #
# provider
# --------------------------------------------------------------------------- #
class LocalModelProvider(BaseModelProvider):
    """Loads a local model with Transformer-Lens and auto-extends its context."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = _recommended_dtype(config.model_name)

        # generation defaults ------------------------------------------------
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

    # ---------- model loading -------------------------------------------------
    def _load_model(self) -> None:
        print(f"Loading {self.config.model_name}")

        # ------------ tokenizer -------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ------------ load the model ------------------------------------------
        #
        # We ask HF to enlarge RoPE via YaRN scaling.  Factor 16 ⇒
        # 2 048 × 16 = 32 768 usable positions.
        #
        self.model = HookedTransformer.from_pretrained(
            self.config.model_name,
            device=self.device,
            dtype=self.dtype,
            rope_scaling={"type": "yarn", "factor": 16.0},
        )

        # ------------ synchronise cfg.n_ctx -----------------------------------
        true_ctx = 32768 #_infer_rope_len(self.model)
        self.model.cfg.n_ctx = true_ctx
        # (Optional) silence HF warnings in generation helpers
        self.tokenizer.model_max_length = true_ctx

        print(
            f"✓ {self.config.model_name} loaded "
            f"(dtype={self.dtype}, device={self.device}, n_ctx={self.model.cfg.n_ctx})"
        )

    # --------------------------------------------------------------------- #
    # generation
    # --------------------------------------------------------------------- #
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

        if 'max_tokens' in self.gen_defaults:
            self.gen_defaults['max_new_tokens'] = self.gen_defaults.pop('max_tokens')

            print('this worked!!!!!!!!!!!!!!!!!!!!!')
   
        
        cfg = {**self.gen_defaults, **kwargs}
        safety = cfg.pop("safety_margin")
        ctx_limit = self.model.cfg.n_ctx
        needed_ctx = prompt_len + safety + cfg["max_new_tokens"]

        if needed_ctx > ctx_limit:
            # cannot grow (we already set n_ctx at load time) -> truncate
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

        comp_tokens = self.tokenizer(completion, return_tensors="pt")["input_ids"].shape[
            1
        ]
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

    # --------------------------------------------------------------------- #
    # misc helpers
    # --------------------------------------------------------------------- #
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