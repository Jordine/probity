# debate/providers/local.py
"""
LocalModelProvider -- HuggingFace / Transformer-Lens backend
Fixed version:
  • Robust extraction of the assistant’s completion from token IDs
  • Hard 2048-token context budget with safety margin
"""

import torch
import time
from typing import List, Dict, Any, Optional
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
import gc

from .base import BaseModelProvider
from ..types import ModelConfig


MAX_CONTEXT = 8192          # Hard budget (model n_ctx may be larger)
SAFETY_MARGIN = 128         # Reserve space for generation & headers


class LocalModelProvider(BaseModelProvider):
    """Provider for local HuggingFace / Transformer-Lens models"""

    def __init__(self, config: ModelConfig):
        super().__init__(config)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dtype = self._get_model_dtype()

        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[HookedTransformer] = None
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

    # ────────────────────────────────────────────────────────────────────
    # Loading helpers
    # ────────────────────────────────────────────────────────────────────
    def _get_model_dtype(self) -> torch.dtype:
        bf16_families = ["llama", "mistral", "gemma", "phi"]
        return torch.bfloat16 if any(m in self.config.model_name.lower() for m in bf16_families) else torch.float32

    def _load_model(self):
        print(f"Loading local model {self.config.model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = HookedTransformer.from_pretrained_no_processing(
            self.config.model_name,
            device=self.device,
            dtype=self.model_dtype,
        )
        print("✓ Model loaded")

    # ────────────────────────────────────────────────────────────────────
    # Main generation
    # ────────────────────────────────────────────────────────────────────
    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> tuple:
        """
        Return (assistant_completion:str, metadata:dict)
        The completion is extracted from token IDs – eliminates prompt duplication.
        """
        if not self.is_available():
            return "", {"error": "Model not available"}

        start = time.time()

        # ------------------------------------------------------------------
        # 1. Build the chat prompt string
        # ------------------------------------------------------------------
        prompt_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # ------------------------------------------------------------------
        # 2. Tokenise once
        # ------------------------------------------------------------------
        prompt_ids = self.tokenizer(
            prompt_str,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"].to(self.device)        # [1, prompt_len]
        prompt_len = prompt_ids.shape[1]

        # Context-length protection
        gen_cfg = {**self.default_gen_kwargs, **kwargs}
        max_new = gen_cfg.get("max_new_tokens", 256)

        if prompt_len + max_new + SAFETY_MARGIN > MAX_CONTEXT:
            # Reduce generation length so we never exceed 2048
            allowed_new = max(1, MAX_CONTEXT - prompt_len - SAFETY_MARGIN)
            print(f"[CTX] Truncating max_new_tokens from {max_new} → {allowed_new}")
            gen_cfg["max_new_tokens"] = allowed_new
            max_new = allowed_new

        # ------------------------------------------------------------------
        # 3. Generate
        # ------------------------------------------------------------------
        self.model.eval()
        with torch.no_grad():
            out_ids = self.model.generate(
                prompt_ids,
                max_new_tokens=max_new,
                temperature=gen_cfg["temperature"],
                do_sample=gen_cfg["do_sample"],
                top_p=gen_cfg["top_p"],
                top_k=gen_cfg["top_k"],
                eos_token_id=self.tokenizer.eos_token_id,
                stop_at_eos=True,
                prepend_bos=False,
                return_type="input",          # gives ndarray of token ids
            )

        out_ids = torch.tensor(out_ids, device=self.device)
        completion_ids = out_ids[0, prompt_len:]          # slice OFF the prompt
        completion_str = self.tokenizer.decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).lstrip()

        # ------------------------------------------------------------------
        # 4. Accounting & metadata
        # ------------------------------------------------------------------
        output_len = completion_ids.numel()
        total_tokens = prompt_len + output_len
        latency = time.time() - start
        self._update_usage(total_tokens)

        metadata = {
            "input_tokens": int(prompt_len),
            "output_tokens": int(output_len),
            "total_tokens": int(total_tokens),
            "latency": latency,
            "model": self.config.model_name,
            "device": str(self.device),
            "dtype": str(self.model_dtype),
        }

        # ------------------------------------------------------------------
        # 5. Cleanup
        # ------------------------------------------------------------------
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return completion_str, metadata

    # ────────────────────────────────────────────────────────────────────
    # House-keeping
    # ────────────────────────────────────────────────────────────────────
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