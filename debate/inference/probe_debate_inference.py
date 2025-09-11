import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
import time

from probity.probes import BaseProbe
from ..types import ProbeScore  # CORRECTED: Removed ConversationTurn
from ..config import ProbeInferenceConfig


class ProbeDebateInference:
    """Real-time probe inference for debate conversations - fixed version."""
    
    def __init__(self, config: ProbeInferenceConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        print(f"Loading model {config.model_name} for probe inference...")
        self.model_dtype = self._get_model_dtype(config.model_name)
        
        self.model = HookedTransformer.from_pretrained_no_processing(
            config.model_name,
            device=config.device,
            dtype=self.model_dtype
        )
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.probes = self._load_probes()
        
        print(f"Loaded {len(self.probes)} probes for layer {config.layer}")
    
    def _get_model_dtype(self, model_name: str) -> torch.dtype:
        bfloat16_models = ['llama', 'mistral', 'gemma', 'phi']
        if any(m in model_name.lower() for m in bfloat16_models):
            return torch.bfloat16
        return torch.float32
    
    def _load_probes(self) -> Dict[str, BaseProbe]:
        probes = {}
        probe_dir = Path(self.config.probe_dir)
        
        for probe_type in self.config.probe_types:
            json_path = probe_dir / probe_type / f"layer_{self.config.layer}_probe.json"
            pt_path = probe_dir / probe_type / f"layer_{self.config.layer}_probe.pt"
            
            probe_loaded = False
            
            if json_path.exists():
                try:
                    probe = BaseProbe.load_json(str(json_path), device=self.config.device)
                    probes[probe_type] = probe
                    probe_loaded = True
                except Exception as e:
                    print(f"Failed to load JSON {probe_type} probe: {e}")
            
            if not probe_loaded and pt_path.exists():
                try:
                    probe_dict = torch.load(str(pt_path), map_location=self.config.device, weights_only=False)
                    
                    if isinstance(probe_dict, BaseProbe):
                        probe = probe_dict
                    elif isinstance(probe_dict, dict) and 'probe_type' in probe_dict:
                        from probity.probes import LogisticProbe, PCAProbe, MeanDifferenceProbe, KMeansProbe
                        probe_classes = {
                            'LogisticProbe': LogisticProbe, 'PCAProbe': PCAProbe,
                            'MeanDifferenceProbe': MeanDifferenceProbe, 'KMeansProbe': KMeansProbe,
                        }
                        probe_class = probe_classes.get(probe_dict['probe_type'])
                        if probe_class:
                            probe = probe_class(probe_dict['config'])
                            probe.load_state_dict(probe_dict['state_dict'])
                            probe = probe.to(self.config.device)
                        else: continue
                    else: continue
                    
                    probes[probe_type] = probe
                    probe_loaded = True
                except Exception as e:
                    print(f"Failed to load .pt {probe_type} probe: {e}")
            
            if not probe_loaded:
                print(f"Could not load {probe_type} probe from either JSON or .pt format")
        
        return probes
    
    def score_new_response(
        self, 
        conversation_history: List[Dict[str, str]],
        new_response: str,
        speaker: str
    ) -> List[ProbeScore]:
        """
        Score ONLY the new assistant response tokens.
        """
        full_messages = conversation_history + [{"role": "assistant", "content": new_response}]
        
        formatted_text = self.tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )
        
        tokens_info = self.tokenizer(
            formatted_text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True
        ).to(self.device)
        
        assistant_start_idx = self._find_last_assistant_start(tokens_info["input_ids"][0])
        
        if assistant_start_idx is None:
            print("Warning: Could not locate assistant response in tokens")
            return []
        
        assistant_tokens = tokens_info["input_ids"][0, assistant_start_idx:]
        
        hook_point = f"blocks.{self.config.layer}.hook_resid_pre"
        
        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                tokens_info["input_ids"], names_filter=[hook_point],
                return_cache_object=True, stop_at_layer=self.config.layer + 1
            )
            
            all_activations = cache[hook_point].squeeze(0)
            assistant_activations = all_activations[assistant_start_idx:]
        
        assistant_token_strs = self.tokenizer.convert_ids_to_tokens(assistant_tokens.cpu().tolist())
        
        probe_scores = []
        
        with torch.no_grad():
            for probe_type, probe in self.probes.items():
                try:
                    if assistant_activations.dtype != probe.dtype:
                        assistant_activations = assistant_activations.to(dtype=probe.dtype)
                    
                    token_scores = probe(assistant_activations)
                    
                    if probe.__class__.__name__ == 'LogisticProbe':
                        token_scores = torch.sigmoid(token_scores)
                    
                    scores_list = token_scores.cpu().squeeze().tolist()
                    if isinstance(scores_list, float): scores_list = [scores_list]
                    
                    if probe.__class__.__name__ != 'LogisticProbe':
                        scores_list = self._normalize_scores(scores_list)
                    
                    probe_scores.append(ProbeScore(
                        probe_type=probe_type, layer=self.config.layer,
                        tokens=assistant_token_strs, token_scores=scores_list,
                        mean_score=float(np.mean(scores_list)),
                        metadata={"timestamp": time.time(), "speaker": speaker}
                    ))
                except Exception as e:
                    print(f"Error scoring with {probe_type} probe: {e}")
        
        return probe_scores
    
    def _find_last_assistant_start(self, token_ids: torch.Tensor) -> Optional[int]:
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids.cpu().tolist())
        
        assistant_marker_tokens = ["<|start_header_id|>", "assistant", "<|end_header_id|>"]
        
        last_assistant_idx = None
        for i in range(len(tokens) - len(assistant_marker_tokens) + 1):
            if tokens[i:i+len(assistant_marker_tokens)] == assistant_marker_tokens:
                content_start = i + len(assistant_marker_tokens)
                while content_start < len(tokens) and tokens[content_start].strip() in ['Ċ', 'Ġ', '']:
                    content_start += 1
                last_assistant_idx = content_start
        
        return last_assistant_idx
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        scores_array = np.array(scores)
        min_score, max_score = scores_array.min(), scores_array.max()
        if max_score == min_score: return [0.5] * len(scores)
        return ((scores_array - min_score) / (max_score - min_score)).tolist()
    
    def save_scores(self, probe_scores: List[ProbeScore], save_path: str):
        scores_data = [score.__dict__ for score in probe_scores]
        with open(save_path, 'w') as f:
            json.dump(scores_data, f, indent=2)

def main():
    """CLI interface for probe debate inference"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time probe inference for debates')
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--probe_dir', type=str, required=True)
    parser.add_argument('--probe_types', nargs='+', default=['logistic', 'pca', 'meandiff'])
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--conversation_json', type=str, required=True,
                       help='JSON file with conversation history and new response')
    parser.add_argument('--results_save_dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    with open(args.conversation_json, 'r') as f:
        conv_data = json.load(f)
    
    # CORRECTED: Load history as a list of dicts, not dataclass objects
    history = conv_data.get('history', [])
    new_response = conv_data['new_response']
    speaker = conv_data['speaker']
    
    config = ProbeInferenceConfig(
        model_name=args.model_name,
        probe_dir=args.probe_dir,
        probe_types=args.probe_types,
        layer=args.layer,
        device=args.device
    )
    
    inferencer = ProbeDebateInference(config)
    probe_scores = inferencer.score_new_response(history, new_response, speaker)
    
    save_dir = Path(args.results_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"probe_scores_{int(time.time())}.json"
    inferencer.save_scores(probe_scores, str(save_path))
    
    print(f"Scored {len(probe_scores)} probe types. Results saved to {save_path}")

if __name__ == "__main__":
    main()