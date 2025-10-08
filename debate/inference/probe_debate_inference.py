# debate/inference/probe_debate_inference.py
"""
Role-aware probe inference.

The generic attributes `model_name` and `probe_dir` have been removed.
Instead, we load *either* the honest or the dishonest bundle depending
on `role ∈ {"honest", "dishonest"}`.
"""

from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

from probity.probes import BaseProbe
from probity.utils.dataset_loading import get_model_dtype
from ..types import ProbeScore
from ..config import ProbeInferenceConfig


class ProbeDebateInference:
    """
    Real-time probe inference for debate conversations – **role aware**.

    Parameters
    ----------
    config : ProbeInferenceConfig
        Dataclass that now stores *separate* fields for honest vs.
        dishonest models and probe directories.
    role : {"honest", "dishonest"}
        Which bundle to load.  Must match a key in the dataclass.
    """

    def __init__(self, config: ProbeInferenceConfig, role: str = "honest"):
        if role not in {"honest", "dishonest"}:
            raise ValueError("role must be 'honest' or 'dishonest'")

        self.config = config
        self.role = role
        
        self.device = torch.device(config.device)

        # Parse device specification - support "cuda:0", "cuda:1", etc.
        device_str = config.device
        if device_str.startswith("cuda:"):
            try:
                device_id = int(device_str.split(":")[1])
                self.device = torch.device(f"cuda:{device_id}")
            except (ValueError, IndexError):
                self.device = torch.device("cuda")
        else:
            self.device = torch.device(config.device)

        # Pick the correct model / probe directory for the given role
        if role == "honest":
            self.model_name: str = config.honest_model_name
            self.probe_dir: Path = Path(config.honest_probe_dir)
        else:
            self.model_name = config.dishonest_model_name
            self.probe_dir = Path(config.dishonest_probe_dir)

        print(f"[ProbeDebateInference] Loading {role} model '{self.model_name}' on {self.device}...")
        
        self.model_dtype = get_model_dtype(self.model_name)

        # Load model on specific device
        with torch.cuda.device(self.device) if 'cuda' in str(self.device) else torch.no_grad():
            self.model = HookedTransformer.from_pretrained_no_processing(
                self.model_name,
                device=self.device,
                dtype=self.model_dtype,
            )
        
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.probes: Dict[str, BaseProbe] = self._load_probes()
        self.activation_cache: Dict[Any, torch.Tensor] = {}

        print(f"[ProbeDebateInference] • {len(self.probes)} probes loaded "
              f"(layer {config.layer}) for role='{role}'")

    # ────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────────────────────────────
    def _load_probes(self) -> Dict[str, BaseProbe]:
        """Load the probe objects that correspond to `self.role`."""
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
                
                # CRITICAL FIX: Update config device to match target device
                config = checkpoint.get("config")
                if hasattr(config, 'device'):
                    config.device = str(self.device)  # Update to target device
                
                # Dynamically pick the concrete class
                from probity.probes import LogisticProbe, PCAProbe, MeanDifferenceProbe, MLPProbe, AttentionProbe
    
                probe_cls_map = {
                    "LogisticProbe": LogisticProbe,
                    "PCAProbe": PCAProbe,
                    "MeanDifferenceProbe": MeanDifferenceProbe,
                    "MLPProbe": MLPProbe,
                    "AttentionProbe": AttentionProbe
                }
                cls_name = checkpoint.get("probe_type")
                probe_cls = probe_cls_map.get(cls_name)
    
                if probe_cls is None:
                    print(f"[WARN] Unknown probe class '{cls_name}' in {probe_path}")
                    continue
    
                probe: BaseProbe = probe_cls(config)  # Now uses corrected config
                probe.load_state_dict(checkpoint["state_dict"])
                probe = probe.to(self.device)
                
                # CRITICAL FIX: Mark DirectionalProbe subclasses as fit after loading
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
        model: HookedTransformer,
        tokenizer: AutoTokenizer,
        probe_dir: str,
        probe_types: List[str],
        layer: int,
        device: str,
        role: str
    ) -> "ProbeDebateInference":
        """Create a ProbeDebateInference using an already-loaded model (multi-GPU aware)"""
        instance = cls.__new__(cls)
        
        instance.role = role
        
        # Handle device specification properly
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
            device=str(instance.device)  # Store as string
        )
        
        # Use the provided model and tokenizer
        instance.model = model
        instance.tokenizer = tokenizer
        instance.model_name = model.cfg.model_name
        
        # Load only the probes on the same device as the model
        instance.probes = instance._load_probes()
        instance.activation_cache = {}
        
        print(f"[ProbeDebateInference] Reusing existing model for role='{role}' on {instance.device}")
        print(f"[ProbeDebateInference] • {len(instance.probes)} probes loaded (layer {layer})")
        
        return instance

    
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
        Score **only** the *new* assistant response and return
        (list-of-ProbeScore, formatted_output-dict).
        """
        full_msgs = conversation_history + [{"role": "assistant", "content": new_response}]
        formatted = self.tokenizer.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False
        )

        tok = self.tokenizer(
            formatted,
            return_tensors="pt",
            add_special_tokens=False,
            return_offsets_mapping=True,
        ).to(self.device)

        # Index where the assistant’s answer starts
        assist_start = self._find_last_assistant_start(tok["input_ids"][0])
        if assist_start is None:
            print("[ProbeDebateInference] Could not locate assistant tokens")
            return [], {}

        assist_ids = tok["input_ids"][0, assist_start:]

        # ── Activations (cached) ─────────────────────────────────────────
        hook = f"blocks.{self.config.layer}.hook_resid_pre"
        cache_key = (formatted, hook)
        with torch.no_grad():
            if cache_key in self.activation_cache:
                activations = self.activation_cache[cache_key]
            else:
                _, cache = self.model.run_with_cache(
                    tok["input_ids"],
                    names_filter=[hook],
                    return_cache_object=True,
                    stop_at_layer=self.config.layer + 1,
                )
                activations = cache[hook].squeeze(0)
                self.activation_cache[cache_key] = activations

            assist_act = activations[assist_start:]

        # ── Token strings & cleaning ─────────────────────────────────────
        assist_tok_str = self.tokenizer.convert_ids_to_tokens(assist_ids.cpu().tolist())
        clean_tokens = self._clean_tokens_for_display(assist_tok_str)

        # ── Probe evaluation ────────────────────────────────────────────
        probe_scores: List[ProbeScore] = []
        scores_by_probe: Dict[str, List[float]] = {}
        raw_logits_by_probe: Dict[str, List[float]] = {}  

        with torch.no_grad():
            for p_type, probe in self.probes.items():
                try:
                    act = assist_act.to(device=self.device, dtype=probe.dtype, copy=False)
                    print(self.device)
                    print("act = assist device=self.device sucessfully run")
                    
                    raw = probe(act)
                    raw_logits = raw.cpu().squeeze().tolist()  # Store raw logits
                    if isinstance(raw_logits, float):
                        raw_logits = [raw_logits]
                    raw_logits_by_probe[p_type] = raw_logits
                    
                    # Apply sigmoid for logistic probes
                    if probe.__class__.__name__ == "LogisticProbe":
                        raw = torch.sigmoid(raw)
                    
                    scores = raw.cpu().squeeze().tolist()
                    if isinstance(scores, float):
                        scores = [scores]
                    
                    # Normalize to [-1, 1 [commented out]
                    # normalized_scores = self._normalize_scores(scores)
                    # scores_by_probe[p_type] = normalized_scores
                    
                    scores_by_probe[p_type] = scores

                    
                    probe_scores.append(
                        ProbeScore(
                            probe_type=p_type,
                            layer=self.config.layer,
                            tokens=clean_tokens,
                            # token_scores=normalized_scores,
                            # mean_score=float(np.mean(normalized_scores)),
                            token_scores=scores,
                            mean_score=float(np.mean(scores)),                            
                            metadata={
                                "timestamp": time.time(),
                                "speaker": speaker,
                                "role": self.role,
                                "raw_logits": raw_logits  # Store raw logits in metadata
                            },
                        )
                    )
                except Exception as exc:
                    print(f"[ProbeDebateInference] Error in probe '{p_type}': {exc}")

        # Human / model readable format
        formatted_output = self._format_scores_for_debate(clean_tokens, scores_by_probe, speaker)
        return probe_scores, formatted_output

    # ────────────────────────────────────────────────────────────────────
    # Utility routines (unchanged except for style tweaks)
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
        """Format probe scores into human/model readable format."""
        
        # Combine tokens into words/phrases for better readability
        word_scores = self._aggregate_to_words(tokens, scores_by_probe)
        
        # Calculate summary statistics
        all_scores = []
        for probe_scores in scores_by_probe.values():
            all_scores.extend(probe_scores)
        
        mean_score = np.mean(all_scores) if all_scores else 0.5
        max_score = np.max(all_scores) if all_scores else 0.5

        if not all_scores:
            print("PROBLEM!!! ALL_SCORES IS NONE")
        
        # # Find highly suspicious tokens/words (score > 0.7)
        # suspicious_segments = []
        # for word, word_score in word_scores.items():
        #     if word_score > 0.7:
        #         suspicious_segments.append({
        #             "text": word,
        #             "score": round(word_score, 3)
        #         })
        
        formatted = {
            "speaker": speaker,
            "deception_analysis": {
                "mean_score": round(mean_score, 2),
                "max_score": round(max_score, 2),
                # "interpretation": self._get_score_interpretation(mean_score),
                # "suspicious_segments": suspicious_segments[:5],  # Top 5 most suspicious
                "token_scores": word_scores,  # Full word-level scores
                # "summary": f"Deception likelihood: {self._get_likelihood_label(mean_score)} (mean: {mean_score:.1%})"
            }
        }
        
        return formatted
    
    def _aggregate_to_words(
        self, 
        tokens: List[str], 
        scores_by_probe: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """Aggregate token scores to word level for readability."""
        # Average scores across probe types
        avg_scores = []
        for i in range(len(tokens)):
            token_scores = [scores[i] for scores in scores_by_probe.values() if i < len(scores)]
            avg_scores.append(np.mean(token_scores) if token_scores else 0.5)
        
        # Combine tokens into words
        words = {}
        current_word = ""
        current_scores = []
        
        for token, score in zip(tokens, avg_scores):
            # Check if token starts a new word
            if token.startswith(' ') or token == '\n':
                # Save previous word if exists
                if current_word:
                    words[current_word] = round(np.mean(current_scores), 3)
                # Start new word
                current_word = token.strip()
                current_scores = [score]
            else:
                # Continue building current word
                current_word += token
                current_scores.append(score)
        
        # Don't forget last word
        if current_word:
            words[current_word] = round(np.mean(current_scores), 3)
        
        return words
    
    # def _get_score_interpretation(self, score: float) -> str:
    #     """Get human-readable interpretation of score."""
    #     if score < -0.4:
    #         return "Strongly truthful"
    #     elif score < 0:
    #         return "Likely truthful"
    #     elif score < 0.4:
    #         return "Likely deceptive"
    #     else:
    #         return "Strongly deceptive"
    
    # def _get_likelihood_label(self, score: float) -> str:
    #     """Get simple likelihood label."""
    #     if score < 0.3:
    #         return "LOW"
    #     elif score < 0.5:
    #         return "MODERATE"
    #     elif score < 0.7:
    #         return "ELEVATED"
    #     else:
    #         return "HIGH"
    
    def _find_last_assistant_start(self, token_ids: torch.Tensor) -> Optional[int]:
        """Find where the last assistant response starts."""
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids.cpu().tolist())
        
        # Handle different model formats
        assistant_markers = {
            "llama": ["<|start_header_id|>", "assistant", "<|end_header_id|>"],
            "qwen": ["<|im_start|>", "assistant"],
        }
        
        # Detect format
        format_type = "llama"
        if any("<|im_start|>" in t for t in tokens):
            format_type = "qwen"
        
        markers = assistant_markers[format_type]
        
        # Find last occurrence
        last_assistant_idx = None
        for i in range(len(tokens) - len(markers) + 1):
            if all(tokens[i+j] == markers[j] for j in range(len(markers))):
                content_start = i + len(markers)
                # Skip whitespace tokens
                while content_start < len(tokens) and tokens[content_start].strip() in ['', 'Ġ', 'Ċ']:
                    content_start += 1
                last_assistant_idx = content_start
        
        return last_assistant_idx
    
    # def _normalize_scores(self, scores: List[float]) -> List[float]:
    #     """Normalize scores to [-1, 1] range (negative=truthful, positive=deceptive)."""
    #     scores_array = np.array(scores)
    #     min_score, max_score = scores_array.min(), scores_array.max()
    #     if max_score == min_score:
    #         return [0.0] * len(scores)
    #     # First normalize to [0, 1], then shift to [-1, 1]
    #     normalized = (scores_array - min_score) / (max_score - min_score)
    #     return (normalized * 2 - 1).tolist()
    # commented out since -1 to 1 is arbitrary given the thresholds thing

    # have also commented out a bunch of other functions that are arbitrarily set