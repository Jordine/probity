import torch
from dataclasses import dataclass
from typing import List, Dict
from transformer_lens import HookedTransformer
from probity.datasets.tokenized import TokenizedProbingDataset
from probity.collection.activation_store import ActivationStore


@dataclass
class TransformerLensConfig:
    """Configuration for TransformerLensCollector."""

    model_name: str
    hook_points: List[str]  # e.g. ["blocks.12.hook_resid_post"]
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class TransformerLensCollector:
    """Collects activations using TransformerLens."""

    def __init__(self, config: TransformerLensConfig):
        self.config = config
        print(f"Initializing collector with device: {config.device}")

        self._ensure_qwen3_support()
        
        self.model = HookedTransformer.from_pretrained_no_processing(config.model_name)
        print(f"Moving model to device: {config.device}")
        self.model.to(config.device)

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

        
    
    @staticmethod
    def get_layer_from_hook_point(hook_point: str) -> int:
        """Extract layer number from hook point string.
        
        Args:
            hook_point: Hook point string (e.g. "blocks.12.hook_resid_post")
            
        Returns:
            Layer number
        """
        try:
            # Extract number after "blocks."
            layer = int(hook_point.split(".")[1])
            return layer
        except (IndexError, ValueError):
            raise ValueError(f"Could not extract layer from hook point: {hook_point}")

    def collect(
        self,
        dataset: TokenizedProbingDataset,
    ) -> Dict[str, ActivationStore]:
        """Collect activations for each hook point.

        Returns:
            Dictionary mapping hook points to ActivationCache objects
        """
        all_activations = {}

        # Set model to evaluation mode
        self.model.eval()

        # Get maximum layer needed
        max_layer = max(
            self.get_layer_from_hook_point(hook)
            for hook in self.config.hook_points
        )

        # Process in batches
        with torch.no_grad():  # Disable gradient computation for determinism
            for batch_start in range(0, len(dataset.examples), self.config.batch_size):
                batch_end = min(batch_start + self.config.batch_size, len(dataset.examples))
                batch_indices = list(range(batch_start, batch_end))

                # Get batch tensors
                batch = dataset.get_batch_tensors(batch_indices)

                # Run model with caching
                _, cache = self.model.run_with_cache(
                    batch["input_ids"].to(self.config.device),
                    names_filter=self.config.hook_points,
                    return_cache_object=True,
                    stop_at_layer=max_layer + 1
                )

                # Store activations for each hook point
                for hook in self.config.hook_points:
                    if hook not in all_activations:
                        all_activations[hook] = []
                    all_activations[hook].append(cache[hook].cpu())

        # Create ActivationCache objects
        return {
            hook: ActivationStore(
                raw_activations=torch.cat(activations, dim=0),
                hook_point=hook,
                example_indices=torch.arange(len(dataset.examples)),
                sequence_lengths=torch.tensor(dataset.get_token_lengths()),
                hidden_size=activations[0].shape[-1],
                dataset=dataset,
                labels=torch.tensor([ex.label for ex in dataset.examples]),
                label_texts=[ex.label_text for ex in dataset.examples],
            )
            for hook, activations in all_activations.items()
        }