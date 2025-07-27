import argparse
import os
import torch
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from transformer_lens import HookedTransformer
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from probity.probes import (
    BaseProbe,
    LogisticProbe, LogisticProbeConfig,
    PCAProbe, PCAProbeConfig,
    MeanDifferenceProbe, MeanDiffProbeConfig,
    KMeansProbe, KMeansProbeConfig,
    LinearProbe, LinearProbeConfig
)

from probity.utils.dataset_loading import get_model_dtype


class OptimizedBatchProbeEvaluator:
    """Optimized evaluator that runs model once and applies all probes"""
    
    def __init__(self, model_name: str, device: str):
        self.device = device
        self.model_name = model_name

        self._ensure_qwen3_support()
        
        # Load model once
        print(f"Loading model {model_name}")
        self.model_dtype = get_model_dtype(model_name)
        print(f"Using model dtype: {self.model_dtype}")
        
        self.model = HookedTransformer.from_pretrained_no_processing(
            model_name,
            device=device,
            dtype=self.model_dtype,
        )
        self.model.eval()
        
        # Cache for activations
        self._activation_cache = {}

    def _ensure_qwen3_support(self):
    """Add qwen3 32b in because they forgot to add it to the model table"""
        try:
            qwen3_models = ["Qwen/Qwen3-32B",
                           "Qwen/Qwen3-14B"]
            
            existing_names = set(loading_from_pretrained.OFFICIAL_MODEL_NAMES)
            new_models = [model for model in qwen3_models if model not in existing_names]
            already_present = [model for model in qwen3_models if model in existing_names]
    
            # Report already present models
            if already_present:
                print(f"Already present in TransformerLens ({len(already_present)} models):")
                for model in already_present:
                    print(f"  ✓ {model}")
    
            # Add and report new models
            if new_models:
                loading_from_pretrained.OFFICIAL_MODEL_NAMES.extend(new_models)
                print(f"Successfully added to TransformerLens ({len(new_models)} models):")
                for model in new_models:
                    print(f"  + {model}")
            else:
                if not already_present:  # No models at all
                    print("No Qwen3 models were added (list was empty)")
                # If already_present is not empty, we already reported those above
                
            # Summary
            total_qwen3_models = len(qwen3_models)
            print(f"Qwen3 model support status: {len(already_present + new_models)}/{total_qwen3_models} models available")
            
        except ImportError as e:
            print(f"Failed to import TransformerLens for Qwen3 model support: {e}")
            print("Failed to add any Qwen3 models:")
            for model in qwen3_models:
                print(f"  ✗ {model}")
        except Exception as e:
            print(f"Unexpected error while adding Qwen3 models: {e}")
            print("Status unknown for models:")
            for model in qwen3_models:
                print(f"  ? {model}")

        
        
    def get_batch_activations(self, texts: List[str], layers: List[int], 
                            batch_size: int = 8) -> Dict[int, torch.Tensor]:
        """Get activations for all texts and layers efficiently"""
        
        # Create cache key
        cache_key = (tuple(sorted(texts)), tuple(sorted(layers)))
        if cache_key in self._activation_cache:
            return self._activation_cache[cache_key]
        
        hook_points = [f"blocks.{layer}.hook_resid_pre" for layer in layers]
        
        # Tokenize all texts at once
        if hasattr(self.model, 'tokenizer'):
            tokenizer = self.model.tokenizer
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model.cfg.model_name)
        
        print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
        print(f"Tokenizer eos_token_id: {tokenizer.eos_token_id}")
        print(f"Tokenizer vocab size: {len(tokenizer)}")
        
        # First, let's tokenize a single text to see what's happening
        if texts:
            print("\n=== DEBUG: Single text tokenization ===")
            sample_text = texts[0]
            print(f"Sample text length: {len(sample_text)}")
            print(f"Sample text end: ...{sample_text[-100:]}")
            
            # Test different tokenization approaches
            single_tokens1 = tokenizer(sample_text, return_tensors="pt", add_special_tokens=False)
            single_tokens2 = tokenizer(sample_text, return_tensors="pt", add_special_tokens=True)
            
            print(f"Without add_special_tokens: {single_tokens1['input_ids'].shape}")
            print(f"With add_special_tokens: {single_tokens2['input_ids'].shape}")
            
            # Decode the end of both
            tokens1 = single_tokens1['input_ids'][0]
            tokens2 = single_tokens2['input_ids'][0]
            
            print(f"Last 10 tokens (no special): {tokenizer.convert_ids_to_tokens(tokens1[-10:])}")
            print(f"Last 10 tokens (with special): {tokenizer.convert_ids_to_tokens(tokens2[-10:])}")
            print("=== END DEBUG ===\n")
        
        # Process in batches
        all_activations = {layer: [] for layer in layers}
        all_tokens = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
                # Tokenize batch
                tokens = tokenizer(
                    batch_texts, 
                    return_tensors="pt", 
                    padding=True,  # Pad to longest in batch
                    truncation=False,  # No truncation
                    add_special_tokens=False  # Texts already have special tokens
                ).to(self.device)
                
                print(f"Batch tokenized shape: {tokens['input_ids'].shape}")
                
                # Run model with caching
                _, cache = self.model.run_with_cache(
                    tokens["input_ids"],
                    names_filter=hook_points,
                    return_cache_object=True,
                    stop_at_layer=max(layers) + 1
                )
                
                # Store activations for each layer - KEEP ORIGINAL DTYPE
                for layer in layers:
                    hook_point = f"blocks.{layer}.hook_resid_pre"
                    # Keep activations in original dtype (bfloat16) - don't convert to CPU yet
                    all_activations[layer].append(cache[hook_point])

                # Store tokens for later use - FIXED VERSION FOR EOS=PAD
                for j, text in enumerate(batch_texts):
                    text_tokens = tokens["input_ids"][j]
                    
                    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id == tokenizer.eos_token_id:
                        # Special case: pad_token_id == eos_token_id
                        # We need to find where padding starts (consecutive eos tokens)
                        eos_positions = torch.where(text_tokens == tokenizer.eos_token_id)[0]
                        
                        if len(eos_positions) > 0:
                            # Look for the first position where we have consecutive eos tokens
                            # The real content should end with a single eos, then padding starts
                            content_end_pos = len(text_tokens)  # Default to full length
                            
                            for i in range(len(eos_positions) - 1):
                                pos = eos_positions[i].item()
                                next_pos = eos_positions[i + 1].item()
                                
                                # If two eos tokens are consecutive, the second one starts padding
                                if next_pos == pos + 1:
                                    content_end_pos = next_pos
                                    break
                            
                            # If no consecutive eos found, check if the last tokens are all eos (padding)
                            if content_end_pos == len(text_tokens):
                                # Count consecutive eos tokens from the end
                                consecutive_eos_from_end = 0
                                for k in range(len(text_tokens) - 1, -1, -1):
                                    if text_tokens[k] == tokenizer.eos_token_id:
                                        consecutive_eos_from_end += 1
                                    else:
                                        break
                                
                                # If more than 1 consecutive eos at the end, assume padding
                                if consecutive_eos_from_end > 1:
                                    content_end_pos = len(text_tokens) - consecutive_eos_from_end + 1
                            
                            actual_tokens = text_tokens[:content_end_pos]
                        else:
                            # No eos tokens found
                            actual_tokens = text_tokens
                            
                    elif tokenizer.pad_token_id is not None:
                        # Normal case: different pad and eos tokens
                        pad_positions = torch.where(text_tokens == tokenizer.pad_token_id)[0]
                        if len(pad_positions) > 0:
                            first_pad_pos = pad_positions[0].item()
                            actual_tokens = text_tokens[:first_pad_pos]
                        else:
                            actual_tokens = text_tokens
                    else:
                        # No pad token defined
                        actual_tokens = text_tokens
                    
                    token_texts = tokenizer.convert_ids_to_tokens(actual_tokens)
                    all_tokens.append(token_texts)
                    
                    # Debug print for first few examples
                    if len(all_tokens) <= 3 or len(all_tokens) % 50 == 0:
                        print(f"Text {len(all_tokens)}: {len(token_texts)} tokens")
                        print(f"  First 5: {token_texts[:5]}")
                        print(f"  Last 5: {token_texts[-5:]}")
                        if len(eos_positions) > 0:
                            print(f"  EOS positions: {eos_positions.tolist()}")
                            print(f"  Content end position: {content_end_pos}")

        # Since batches may have different padding lengths, we need to pad all to the same length
        # Find the maximum sequence length across all batches
        max_seq_len = 0
        for layer in layers:
            for batch_activations in all_activations[layer]:
                max_seq_len = max(max_seq_len, batch_activations.shape[1])
        
        print(f"Maximum sequence length across all batches: {max_seq_len}")
        
        # Pad all batches to the same length
        for layer in layers:
            padded_batches = []
            for batch_activations in all_activations[layer]:
                batch_size, seq_len, hidden_size = batch_activations.shape
                if seq_len < max_seq_len:
                    # Pad with zeros - MAINTAIN DTYPE
                    padding = torch.zeros(
                        batch_size, max_seq_len - seq_len, hidden_size, 
                        dtype=batch_activations.dtype, device=batch_activations.device
                    )
                    padded_batch = torch.cat([batch_activations, padding], dim=1)
                else:
                    padded_batch = batch_activations
                padded_batches.append(padded_batch)
            all_activations[layer] = padded_batches
        
        # Concatenate all batches - MAINTAIN DTYPE
        final_activations = {}
        for layer in layers:
            final_activations[layer] = torch.cat(all_activations[layer], dim=0)
            print(f"Final activations for layer {layer}: {final_activations[layer].shape}, dtype: {final_activations[layer].dtype}")
        
        # Cache results
        result = {
            'activations': final_activations,
            'tokens_by_text': all_tokens
        }
        self._activation_cache[cache_key] = result
        
        return result
    
    def evaluate_all_probes(self, texts: List[str], labels: List[int], 
                          probe_configs: Dict[Tuple[int, str], BaseProbe]) -> Dict[Tuple[int, str], Dict]:
        """Evaluate all probes efficiently using cached activations"""
        
        # Extract unique layers from probe configs
        layers = list(set(layer for layer, _ in probe_configs.keys()))
        
        # Get activations for all required layers at once
        print("Getting activations for all layers...")
        activation_data = self.get_batch_activations(texts, layers)
        activations = activation_data['activations']
        tokens_by_text = activation_data['tokens_by_text']
        
        results = {}
        
        # Group probes by layer for efficient processing
        probes_by_layer = {}
        for (layer, probe_type), probe in probe_configs.items():
            if layer not in probes_by_layer:
                probes_by_layer[layer] = {}
            probes_by_layer[layer][probe_type] = probe
        
        # Process each layer
        for layer in tqdm(layers, desc="Processing layers"):
            layer_activations = activations[layer]  # Shape: [num_texts, seq_len, hidden_size]
            layer_probes = probes_by_layer[layer]
            
            # Calculate mean activations once for this layer
            mean_activations = layer_activations.mean(dim=1)  # Shape: [num_texts, hidden_size]
            
            # Apply all probes for this layer
            for probe_type, probe in layer_probes.items():
                print(f"Evaluating {probe_type} probe on layer {layer}")
                
                # Get probe scores efficiently
                probe_results = self._evaluate_single_probe_batch(
                    probe, layer_activations, mean_activations, 
                    texts, labels, tokens_by_text
                )
                
                results[(layer, probe_type)] = probe_results
        
        return results

    
    
    def _evaluate_single_probe_batch(self, probe: BaseProbe, 
                                   layer_activations: torch.Tensor,
                                   mean_activations: torch.Tensor,
                                   texts: List[str], labels: List[int],
                                   tokens_by_text: List[List[str]]) -> Dict:
        """Evaluate a single probe on batch data"""
        
        # Move probe to device and ensure it matches model dtype
        probe = probe.to(self.device)
        
        # Get probe's expected dtype - check parameters first, then buffers
        probe_dtype = self.model_dtype  # Default to model dtype
        try:
            probe_dtype = next(probe.parameters()).dtype
        except StopIteration:
            # No parameters, check buffers
            try:
                probe_dtype = next(probe.buffers()).dtype
            except StopIteration:
                # No buffers either, use model dtype
                probe_dtype = self.model_dtype
        
        print(f"Probe dtype: {probe_dtype}, Activations dtype: {layer_activations.dtype}")
        
        probe.eval()
        
        # Get token-level scores for all texts at once
        all_token_scores = []
        all_samples = []
        
        with torch.no_grad():
            # Process each text individually to handle variable lengths properly
            for i, (text, true_label) in enumerate(zip(texts, labels)):
                # Get tokens and actual length for this text
                tokens = tokens_by_text[i]
                actual_length = len(tokens)
                
                # Get activations for this text (up to actual length)
                text_activations = layer_activations[i, :actual_length, :].to(self.device)
                
                # ENSURE DTYPE COMPATIBILITY
                if text_activations.dtype != probe_dtype:
                    text_activations = text_activations.to(dtype=probe_dtype)
                
                # Apply probe to all tokens for this text
                token_scores = probe(text_activations)
                
                # Apply sigmoid only to LogisticProbe
                if probe.__class__.__name__ == 'LogisticProbe':
                    token_scores = torch.sigmoid(token_scores)
                
                # Convert to list
                token_scores_list = token_scores.cpu().squeeze().tolist()
                
                # Handle single token case
                if isinstance(token_scores_list, float):
                    token_scores_list = [token_scores_list]
                
                # Ensure we have the right number of scores
                if len(token_scores_list) != actual_length:
                    print(f"Warning: Score length mismatch for text {i}: {len(token_scores_list)} vs {actual_length}")
                    # Pad or truncate as needed
                    if len(token_scores_list) < actual_length:
                        token_scores_list.extend([0.0] * (actual_length - len(token_scores_list)))
                    else:
                        token_scores_list = token_scores_list[:actual_length]
                
                all_token_scores.extend(token_scores_list)
                
                sample_info = {
                    "idx": i,
                    "text": text,
                    "true_label": true_label,
                    "tokens": tokens,
                    "raw_token_scores": token_scores_list
                }
                all_samples.append(sample_info)
        
        # Global normalization for non-logistic probes
        if probe.__class__.__name__ == 'LogisticProbe':
            # LogisticProbe already outputs [0,1] after sigmoid
            for sample in all_samples:
                sample["token_scores"] = sample["raw_token_scores"]
        else:
            # Normalize all scores globally to [0,1]
            normalized_scores = self._normalize_to_01(all_token_scores)
            
            # Distribute normalized scores back to samples
            score_idx = 0
            for sample in all_samples:
                num_tokens = len(sample["raw_token_scores"])
                sample["token_scores"] = normalized_scores[score_idx:score_idx + num_tokens]
                score_idx += num_tokens
        
        # Calculate final metrics
        final_samples = []
        all_mean_scores = []
        
        for sample in all_samples:
            mean_score = np.mean(sample["token_scores"])
            all_mean_scores.append(mean_score)
            
            sample_info = {
                "sample_id": sample["idx"],
                "text": sample["text"],
                "true_label": int(sample["true_label"]),
                "tokens": sample["tokens"],
                "token_scores": sample["token_scores"],
                "mean_score": float(mean_score),
                "predicted_label": int(1 if mean_score > 0.5 else 0),
                "is_correct": bool((mean_score > 0.5) == (sample["true_label"] == 1)),
            }
            final_samples.append(sample_info)
        
        # Calculate metrics
        mean_scores = np.array(all_mean_scores)
        predictions = (mean_scores > 0.5).astype(int)
        
        metrics = {
            'accuracy': float(accuracy_score(labels, predictions)),
            'precision': float(precision_score(labels, predictions)),
            'recall': float(recall_score(labels, predictions)),
            'f1': float(f1_score(labels, predictions)),
            'auroc': float(roc_auc_score(labels, mean_scores))
        }
        
        # Prepare token_details for visualization
        token_details = []
        for sample in final_samples:
            # Clean tokens for visualization
            clean_tokens = [token.replace('Ġ', ' ').replace('Ċ', '\n') for token in sample["tokens"]]
            
            token_detail = {
                "text": sample["text"],
                "label": int(sample["true_label"]),
                "tokens": clean_tokens,
                "token_scores": sample["token_scores"],
                "mean_score": sample["mean_score"],
                "max_score": float(np.max(sample["token_scores"])),
                "min_score": float(np.min(sample["token_scores"]))
            }
            token_details.append(token_detail)
        
        return {
            'metrics': metrics,
            'metrics_interpretation': {
                "accuracy": "Fraction of correctly classified truth/lie statements",
                "precision": "Of statements classified as lies, fraction that were actually lies",
                "recall": "Of actual lies, fraction that were detected",
                "f1": "Harmonic mean of precision and recall for lie detection",
                "auroc": "Area under ROC curve (1.0 = perfect separation of truths and lies)"
            },
            'all_samples': final_samples,
            'token_details': token_details,
            'mean_scores': mean_scores.tolist(),
            'predictions': predictions.tolist()
        }
    
    def _normalize_to_01(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range"""
        scores_array = np.array(scores)
        min_score = scores_array.min()
        max_score = scores_array.max()
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        normalized = (scores_array - min_score) / (max_score - min_score)
        return normalized.tolist()
