# debate/inference/fast_probe_inference.py
"""
Fast probe inference without TransformerLens.

Uses plain transformers + native pytorch hooks for activation extraction.
Drop-in replacement for ProbeDebateInference when using FastLocalModelProvider.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from probity.probes import BaseProbe
from probity.utils.dataset_loading import get_model_dtype
from ..types import ProbeScore
from ..config import ProbeInferenceConfig


class FastProbeDebateInference:
    """
    Fast probe inference for debate conversations without TransformerLens.

    Uses native pytorch hooks for activation extraction when scoring
    existing text. When used with FastLocalModelProvider, activations
    are captured during generation and passed directly.

    Parameters
    ----------
    config : ProbeInferenceConfig
        Configuration with probe directories and layer info.
    role : {"honest", "dishonest"}
        Which debater's probes to load.
    """

    def __init__(self, config: ProbeInferenceConfig, role: str = "honest"):
        if role not in {"honest", "dishonest"}:
            raise ValueError("role must be 'honest' or 'dishonest'")

        self.config = config
        self.role = role

        # Parse device specification
        device_str = config.device
        if device_str.startswith("cuda:"):
            try:
                device_id = int(device_str.split(":")[1])
                self.device = torch.device(f"cuda:{device_id}")
            except (ValueError, IndexError):
                self.device = torch.device("cuda")
        else:
            self.device = torch.device(config.device)

        # Pick the correct probe directory for the given role
        if role == "honest":
            self.model_name: str = config.honest_model_name
            self.probe_dir: Path = Path(config.honest_probe_dir)
        else:
            self.model_name = config.dishonest_model_name
            self.probe_dir = Path(config.dishonest_probe_dir)

        # Model and tokenizer - will be set via from_existing_model or loaded fresh
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model_dtype = get_model_dtype(self.model_name)

        # Load probes
        self.probes: Dict[str, BaseProbe] = self._load_probes()
        self.activation_cache: Dict[Any, torch.Tensor] = {}

        print(f"[FastProbeDebateInference] • {len(self.probes)} probes loaded "
              f"(layer {config.layer}) for role='{role}'")

    def _load_probes(self) -> Dict[str, BaseProbe]:
        """Load probe objects from disk."""
        probes: Dict[str, BaseProbe] = {}

        for probe_type in self.config.probe_types:
            probe_path = self.probe_dir / probe_type / f"layer_{self.config.layer}_probe.pt"
            if not probe_path.exists():
                print(f"[WARN] Expected probe file not found: {probe_path}")
                continue

            try:
                checkpoint = torch.load(
                    probe_path,
                    map_location=self.device,
                    weights_only=False,
                )

                # Update config device to match target device
                config = checkpoint.get("config")
                if hasattr(config, 'device'):
                    config.device = str(self.device)

                # Dynamically pick the concrete class
                from probity.probes import (
                    LogisticProbe, PCAProbe, MeanDifferenceProbe,
                    MLPProbe, AttentionProbe, ApolloProbe
                )

                probe_cls_map = {
                    "LogisticProbe": LogisticProbe,
                    "PCAProbe": PCAProbe,
                    "MeanDifferenceProbe": MeanDifferenceProbe,
                    "MLPProbe": MLPProbe,
                    "AttentionProbe": AttentionProbe,
                    "ApolloProbe": ApolloProbe
                }
                cls_name = checkpoint.get("probe_type")
                probe_cls = probe_cls_map.get(cls_name)

                if probe_cls is None:
                    print(f"[WARN] Unknown probe class '{cls_name}' in {probe_path}")
                    continue

                probe: BaseProbe = probe_cls(config)
                probe.load_state_dict(checkpoint["state_dict"])
                probe = probe.to(self.device)

                # Mark DirectionalProbe subclasses as fit after loading
                if hasattr(probe, 'has_fit') and hasattr(probe, 'direction_vector'):
                    if probe.direction_vector is not None and torch.any(probe.direction_vector != 0):
                        probe.has_fit = True

                probes[probe_type] = probe

            except Exception as exc:
                print(f"[ERROR] Could not load probe '{probe_type}': {exc}")
                import traceback
                traceback.print_exc()

        return probes

    @classmethod
    def from_existing_model(
        cls,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        probe_dir: str,
        probe_types: List[str],
        layer: int,
        device: str,
        role: str
    ) -> "FastProbeDebateInference":
        """Create FastProbeDebateInference using an already-loaded model."""
        instance = cls.__new__(cls)

        instance.role = role

        # Handle device specification
        if isinstance(device, str) and device.startswith("cuda:"):
            try:
                device_id = int(device.split(":")[1])
                instance.device = torch.device(f"cuda:{device_id}")
            except:
                instance.device = torch.device(device)
        else:
            instance.device = torch.device(device)

        instance.probe_dir = Path(probe_dir)
        instance.config = ProbeInferenceConfig(
            honest_model_name="",
            dishonest_model_name="",
            honest_probe_dir=probe_dir if role == "honest" else "",
            dishonest_probe_dir=probe_dir if role == "dishonest" else "",
            probe_types=probe_types,
            layer=layer,
            device=str(instance.device)
        )

        # Use the provided model and tokenizer
        instance.model = model
        instance.tokenizer = tokenizer
        instance.model_name = model.config._name_or_path
        instance.model_dtype = model.dtype

        # Load probes
        instance.probes = instance._load_probes()
        instance.activation_cache = {}

        print(f"[FastProbeDebateInference] Reusing existing model for role='{role}' on {instance.device}")
        print(f"[FastProbeDebateInference] • {len(instance.probes)} probes loaded (layer {layer})")

        return instance

    def _get_layer_module(self, layer_idx: int) -> nn.Module:
        """Get the decoder layer module at the specified index."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Handle different model architectures
        if hasattr(self.model, 'model'):
            # LLaMA, Mistral, Qwen style
            if hasattr(self.model.model, 'layers'):
                return self.model.model.layers[layer_idx]

        if hasattr(self.model, 'transformer'):
            # GPT-2 style
            if hasattr(self.model.transformer, 'h'):
                return self.model.transformer.h[layer_idx]

        raise ValueError(f"Unknown model architecture for {self.model_name}")

    def _extract_activations_native(
        self,
        input_ids: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """
        Extract activations at a specific layer using native pytorch hooks.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            layer_idx: Layer index to extract from

        Returns:
            Activations tensor [batch, seq_len, hidden_dim]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded - use from_existing_model() first")

        target_layer = self._get_layer_module(layer_idx)
        captured_activations = [None]

        def hook_fn(module, input, output):
            # Output is typically (hidden_states, ...) tuple
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            captured_activations[0] = hidden_states.detach()

        handle = target_layer.register_forward_hook(hook_fn)

        try:
            with torch.no_grad():
                _ = self.model(input_ids, use_cache=False)
        finally:
            handle.remove()

        if captured_activations[0] is None:
            raise RuntimeError("Failed to capture activations")

        return captured_activations[0]

    # ────────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────────

    def score_new_response(
        self,
        conversation_history: List[Dict[str, str]],
        new_response: str,
        speaker: str,
        *,
        cache_dir: Optional[str] = None,
    ) -> Tuple[List[ProbeScore], Dict[str, Any]]:
        """
        Score a new assistant response using native hooks.

        Args:
            conversation_history: Previous messages
            new_response: The new response to score
            speaker: 'honest' or 'dishonest'
            cache_dir: Optional cache directory (unused, for API compat)

        Returns:
            (list of ProbeScore, formatted output dict)
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model/tokenizer not loaded - use from_existing_model()")

        full_msgs = conversation_history + [{"role": "assistant", "content": new_response}]
        formatted = self.tokenizer.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False
        )

        tok = self.tokenizer(
            formatted,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)

        # Find where the assistant's response starts
        assist_start = self._find_last_assistant_start(tok["input_ids"][0])
        if assist_start is None:
            print("[FastProbeDebateInference] Could not locate assistant tokens")
            return [], {}

        assist_ids = tok["input_ids"][0, assist_start:]

        # Extract activations using native hooks
        cache_key = (formatted, self.config.layer)
        if cache_key in self.activation_cache:
            activations = self.activation_cache[cache_key]
        else:
            activations = self._extract_activations_native(
                tok["input_ids"],
                self.config.layer
            )
            activations = activations.squeeze(0)  # Remove batch dim
            self.activation_cache[cache_key] = activations

        assist_act = activations[assist_start:]

        # Token strings for display
        assist_tok_str = self.tokenizer.convert_ids_to_tokens(assist_ids.cpu().tolist())
        clean_tokens = self._clean_tokens_for_display(assist_tok_str)

        # Score with probes
        return self._score_activations(assist_act, clean_tokens, speaker)

    def score_from_generation_activations(
        self,
        generation_activations: torch.Tensor,
        generation_token_ids: torch.Tensor,
        response_text: str,
        speaker: str,
    ) -> Tuple[List[ProbeScore], Dict[str, Any]]:
        """
        Score using activations captured during generation (EFFICIENT).

        This is the preferred method when using FastLocalModelProvider,
        as activations are already captured during generation.

        Args:
            generation_activations: Activations [gen_len, hidden_dim]
            generation_token_ids: Token IDs [gen_len]
            response_text: Generated text (for display)
            speaker: 'honest' or 'dishonest'

        Returns:
            (list of ProbeScore, formatted output dict)
        """
        # Convert token IDs to strings for display
        token_strings = self.tokenizer.convert_ids_to_tokens(
            generation_token_ids.cpu().tolist()
        )
        clean_tokens = self._clean_tokens_for_display(token_strings)

        # Move activations to correct device
        if generation_activations.device != self.device:
            generation_activations = generation_activations.to(self.device)

        return self._score_activations(generation_activations, clean_tokens, speaker)

    def _score_activations(
        self,
        activations: torch.Tensor,
        clean_tokens: List[str],
        speaker: str
    ) -> Tuple[List[ProbeScore], Dict[str, Any]]:
        """
        Score activations with all loaded probes.

        Args:
            activations: Activation tensor [seq_len, hidden_dim]
            clean_tokens: Token strings for display
            speaker: Speaker identifier

        Returns:
            (list of ProbeScore, formatted output dict)
        """
        probe_scores: List[ProbeScore] = []
        scores_by_probe: Dict[str, List[float]] = {}
        raw_logits_by_probe: Dict[str, List[float]] = {}

        with torch.no_grad():
            for p_type, probe in self.probes.items():
                try:
                    # Ensure dtype matches
                    act = activations.to(
                        device=self.device,
                        dtype=probe.dtype,
                        copy=False
                    )

                    # Apply probe
                    raw = probe(act)
                    raw_logits = raw.cpu().squeeze().tolist()
                    if isinstance(raw_logits, float):
                        raw_logits = [raw_logits]
                    raw_logits_by_probe[p_type] = raw_logits

                    # Apply sigmoid for logistic probes
                    if probe.__class__.__name__ == "LogisticProbe":
                        raw = torch.sigmoid(raw)

                    scores = raw.cpu().squeeze().tolist()
                    if isinstance(scores, float):
                        scores = [scores]

                    scores_by_probe[p_type] = scores

                    probe_scores.append(
                        ProbeScore(
                            probe_type=p_type,
                            layer=self.config.layer,
                            tokens=clean_tokens,
                            token_scores=scores,
                            mean_score=float(np.mean(scores)),
                            metadata={
                                "timestamp": time.time(),
                                "speaker": speaker,
                                "role": self.role,
                                "raw_logits": raw_logits,
                                "method": "fast_inference"
                            },
                        )
                    )
                except Exception as exc:
                    print(f"[FastProbeDebateInference] Error in probe '{p_type}': {exc}")

        formatted_output = self._format_scores_for_debate(clean_tokens, scores_by_probe, speaker)
        return probe_scores, formatted_output

    # ────────────────────────────────────────────────────────────────────
    # Utility methods (same as ProbeDebateInference)
    # ────────────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_tokens_for_display(tokens: List[str]) -> List[str]:
        clean: List[str] = []
        for tok in tokens:
            if tok.startswith("Ġ"):
                tok = " " + tok[1:]
            elif tok in {"Ċ", "\u010a"}:
                tok = "\n"
            elif tok.startswith("<") and tok.endswith(">"):
                continue
            clean.append(tok)
        return clean

    def _format_scores_for_debate(
        self,
        tokens: List[str],
        scores_by_probe: Dict[str, List[float]],
        speaker: str
    ) -> Dict[str, Any]:
        """Format probe scores for debate display."""
        word_scores = self._aggregate_to_words(tokens, scores_by_probe)

        all_scores = []
        for probe_scores in scores_by_probe.values():
            all_scores.extend(probe_scores)

        mean_score = np.mean(all_scores) if all_scores else 0.5
        max_score = np.max(all_scores) if all_scores else 0.5

        return {
            "speaker": speaker,
            "deception_analysis": {
                "mean_score": round(mean_score, 2),
                "max_score": round(max_score, 2),
                "token_scores": word_scores,
            }
        }

    def _aggregate_to_words(
        self,
        tokens: List[str],
        scores_by_probe: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """Aggregate token scores to word level."""
        avg_scores = []
        for i in range(len(tokens)):
            token_scores = [scores[i] for scores in scores_by_probe.values() if i < len(scores)]
            avg_scores.append(np.mean(token_scores) if token_scores else 0.5)

        words = {}
        current_word = ""
        current_scores = []

        for token, score in zip(tokens, avg_scores):
            if token.startswith(' ') or token == '\n':
                if current_word:
                    words[current_word] = round(np.mean(current_scores), 3)
                current_word = token.strip()
                current_scores = [score]
            else:
                current_word += token
                current_scores.append(score)

        if current_word:
            words[current_word] = round(np.mean(current_scores), 3)

        return words

    def _find_last_assistant_start(self, token_ids: torch.Tensor) -> Optional[int]:
        """Find where the last assistant response starts."""
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids.cpu().tolist())

        assistant_markers = {
            "llama": ["<|start_header_id|>", "assistant", "<|end_header_id|>"],
            "qwen": ["<|im_start|>", "assistant"],
        }

        format_type = "llama"
        if any("<|im_start|>" in t for t in tokens):
            format_type = "qwen"

        markers = assistant_markers[format_type]

        last_assistant_idx = None
        for i in range(len(tokens) - len(markers) + 1):
            if all(tokens[i+j] == markers[j] for j in range(len(markers))):
                content_start = i + len(markers)
                while content_start < len(tokens) and tokens[content_start].strip() in ['', 'Ġ', 'Ċ']:
                    content_start += 1
                last_assistant_idx = content_start

        return last_assistant_idx
