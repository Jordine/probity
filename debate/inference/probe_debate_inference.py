import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
import time

from probity.probes import BaseProbe
from ..types import ProbeScore, ConversationTurn
from ..config import ProbeInferenceConfig


class ProbeDebateInference:
    """Real-time probe inference for debate conversations - fixed version."""
    
    def __init__(self, config: ProbeInferenceConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load model once for reuse
        print(f"Loading model {config.model_name} for probe inference...")
        self.model_dtype = self._get_model_dtype(config.model_name)
        
        self.model = HookedTransformer.from_pretrained_no_processing(
            config.model_name,
            device=config.device,
            dtype=self.model_dtype
        )
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load probes with proper handling for .pt files
        self.probes = self._load_probes()
        
        print(f"Loaded {len(self.probes)} probes for layer {config.layer}")
    
    def _get_model_dtype(self, model_name: str) -> torch.dtype:
        """Determine model dtype."""
        bfloat16_models = ['llama', 'mistral', 'gemma', 'phi']
        if any(m in model_name.lower() for m in bfloat16_models):
            return torch.bfloat16
        return torch.float32
    
    def _load_probes(self) -> Dict[str, BaseProbe]:
        """Load all specified probes for the layer - FIXED to handle .pt files."""
        probes = {}
        probe_dir = Path(self.config.probe_dir)
        
        for probe_type in self.config.probe_types:
            # Try both .json and .pt extensions
            json_path = probe_dir / probe_type / f"layer_{self.config.layer}_probe.json"
            pt_path = probe_dir / probe_type / f"layer_{self.config.layer}_probe.pt"
            
            probe_loaded = False
            
            # Try JSON first
            if json_path.exists():
                try:
                    probe = BaseProbe.load_json(str(json_path), device=self.config.device)
                    probes[probe_type] = probe
                    print(f"Loaded {probe_type} probe from {json_path}")
                    probe_loaded = True
                except Exception as e:
                    print(f"Failed to load JSON {probe_type} probe: {e}")
            
            # Try .pt if JSON didn't work
            if not probe_loaded and pt_path.exists():
                try:
                    # Load with weights_only=False as requested
                    probe_dict = torch.load(str(pt_path), 
                                          map_location=self.config.device, 
                                          weights_only=False)
                    
                    # Handle different formats that might be saved
                    if isinstance(probe_dict, BaseProbe):
                        probe = probe_dict
                    elif isinstance(probe_dict, dict):
                        # Assume it has state_dict and config
                        if 'probe_type' in probe_dict:
                            # Dynamically get the probe class
                            probe_class_name = probe_dict['probe_type']
                            # Import the specific probe class
                            from probity.probes import LogisticProbe, PCAProbe, MeanDifferenceProbe, KMeansProbe
                            
                            probe_classes = {
                                'LogisticProbe': LogisticProbe,
                                'PCAProbe': PCAProbe,
                                'MeanDifferenceProbe': MeanDifferenceProbe,
                                'KMeansProbe': KMeansProbe,
                            }
                            
                            probe_class = probe_classes.get(probe_class_name)
                            if probe_class:
                                probe = probe_class(probe_dict['config'])
                                probe.load_state_dict(probe_dict['state_dict'])
                                probe = probe.to(self.config.device)
                            else:
                                print(f"Unknown probe class: {probe_class_name}")
                                continue
                        else:
                            print(f"Probe dict missing 'probe_type' field")
                            continue
                    else:
                        print(f"Unknown probe format in {pt_path}")
                        continue
                    
                    probes[probe_type] = probe
                    print(f"Loaded {probe_type} probe from {pt_path}")
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
        
        Args:
            conversation_history: Previous messages (without the new response)
            new_response: The new assistant response to score
            speaker: Who is speaking (for metadata)
        """
        # Build the full conversation with the new response
        full_messages = conversation_history + [
            {"role": "assistant", "content": new_response}
        ]
        
        # Apply chat template to get formatted text
        formatted_text = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize the full conversation
        tokens_info = self.tokenizer(
            formatted_text,
            return_tensors="pt",
            add_special_tokens=False,  # Chat template already adds them
            return_offsets_mapping=True
        ).to(self.device)
        
        # Find where the last assistant response starts
        assistant_start_idx = self._find_last_assistant_start(
            formatted_text, 
            tokens_info["input_ids"][0]
        )
        
        if assistant_start_idx is None:
            print("Warning: Could not locate assistant response in tokens")
            return []
        
        # Get ONLY the assistant tokens
        assistant_tokens = tokens_info["input_ids"][0, assistant_start_idx:]
        
        # Get activations ONLY for the assistant tokens
        hook_point = f"blocks.{self.config.layer}.hook_resid_pre"
        
        with torch.no_grad():
            # Run model on full sequence but extract only assistant activations
            _, cache = self.model.run_with_cache(
                tokens_info["input_ids"],
                names_filter=[hook_point],
                return_cache_object=True,
                stop_at_layer=self.config.layer + 1
            )
            
            # Extract only assistant token activations
            all_activations = cache[hook_point].squeeze(0)  # Remove batch dim
            assistant_activations = all_activations[assistant_start_idx:]
        
        # Convert tokens to readable format
        assistant_token_strs = self.tokenizer.convert_ids_to_tokens(
            assistant_tokens.cpu().tolist()
        )
        
        # Score with all probes
        probe_scores = []
        
        with torch.no_grad():
            for probe_type, probe in self.probes.items():
                try:
                    # Ensure dtype compatibility
                    if assistant_activations.dtype != probe.dtype:
                        assistant_activations = assistant_activations.to(dtype=probe.dtype)
                    
                    # Apply probe
                    token_scores = probe(assistant_activations)
                    
                    # Apply sigmoid for LogisticProbe
                    if probe.__class__.__name__ == 'LogisticProbe':
                        token_scores = torch.sigmoid(token_scores)
                    
                    # Convert to list
                    scores_list = token_scores.cpu().squeeze().tolist()
                    if isinstance(scores_list, float):
                        scores_list = [scores_list]
                    
                    # Normalize non-logistic probes to [0,1]
                    if probe.__class__.__name__ != 'LogisticProbe':
                        scores_list = self._normalize_scores(scores_list)
                    
                    probe_score = ProbeScore(
                        probe_type=probe_type,
                        layer=self.config.layer,
                        tokens=assistant_token_strs,
                        token_scores=scores_list,
                        mean_score=float(np.mean(scores_list)),
                        metadata={
                            "timestamp": time.time(),
                            "speaker": speaker,
                            "probe_model": self.config.model_name,
                            "num_tokens": len(assistant_token_strs),
                            "assistant_start_idx": assistant_start_idx
                        }
                    )
                    
                    probe_scores.append(probe_score)
                    
                except Exception as e:
                    print(f"Error scoring with {probe_type} probe: {e}")
        
        return probe_scores
    
    def _find_last_assistant_start(self, formatted_text: str, token_ids: torch.Tensor) -> Optional[int]:
        """Find where the last assistant message starts in the token sequence."""
        
        # Convert token IDs to strings for matching
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids.cpu().tolist())
        
        # Look for assistant markers based on the model type
        # For Llama-style models
        if 'llama' in self.config.model_name.lower():
            assistant_marker_tokens = ["<|start_header_id|>", "assistant", "<|end_header_id|>"]
        # For other models, adapt as needed
        else:
            assistant_marker_tokens = ["<|start_header_id|>", "assistant", "<|end_header_id|>"]
        
        # Find the LAST occurrence of the assistant marker sequence
        last_assistant_idx = None
        
        for i in range(len(tokens) - len(assistant_marker_tokens) + 1):
            # Check if we have the marker sequence
            if tokens[i:i+len(assistant_marker_tokens)] == assistant_marker_tokens:
                # This is a potential assistant start
                # Look for the actual content start (skip any whitespace tokens)
                content_start = i + len(assistant_marker_tokens)
                while content_start < len(tokens) and tokens[content_start] in ['Ċ', 'Ġ', ' ', '\n']:
                    content_start += 1
                last_assistant_idx = content_start
        
        return last_assistant_idx
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to [0, 1] range."""
        scores_array = np.array(scores)
        min_score = scores_array.min()
        max_score = scores_array.max()
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        normalized = (scores_array - min_score) / (max_score - min_score)
        return normalized.tolist()
    
    def save_scores(self, probe_scores: List[ProbeScore], save_path: str):
        """Save probe scores to JSON."""
        scores_data = []
        
        for score in probe_scores:
            scores_data.append({
                "probe_type": score.probe_type,
                "layer": score.layer,
                "tokens": score.tokens,
                "token_scores": score.token_scores,
                "mean_score": score.mean_score,
                "metadata": score.metadata
            })
        
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
    parser.add_argument('--test_all_probes_in_dir', action='store_true',
                       help='Test all available probes in directory')
    
    args = parser.parse_args()
    
    # Load conversation data
    with open(args.conversation_json, 'r') as f:
        conv_data = json.load(f)
    
    history = [ConversationTurn(**turn) for turn in conv_data.get('history', [])]
    new_response = conv_data['new_response']
    speaker = conv_data['speaker']
    
    # Create config
    if args.test_all_probes_in_dir:
        # Find all available probes
        probe_dir = Path(args.probe_dir)
        available_types = []
        for probe_type_dir in probe_dir.iterdir():
            if probe_type_dir.is_dir():
                probe_file = probe_type_dir / f"layer_{args.layer}_probe.json"
                if probe_file.exists():
                    available_types.append(probe_type_dir.name)
        probe_types = available_types
    else:
        probe_types = args.probe_types
    
    config = ProbeInferenceConfig(
        model_name=args.model_name,
        probe_dir=args.probe_dir,
        probe_types=probe_types,
        layer=args.layer,
        device=args.device
    )
    
    # Initialize inference
    inferencer = ProbeDebateInference(config)
    
    # Score the new response
    probe_scores = inferencer.score_new_response(history, new_response, speaker)
    
    # Save results
    save_dir = Path(args.results_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = int(time.time())
    save_path = save_dir / f"probe_scores_{timestamp}.json"
    
    inferencer.save_scores(probe_scores, str(save_path))
    
    print(f"Scored {len(probe_scores)} probe types")
    print(f"Results saved to {save_path}")
    
    # Print summary
    for score in probe_scores:
        print(f"{score.probe_type}: mean={score.mean_score:.3f}, tokens={len(score.tokens)}")


if __name__ == "__main__":
    main()