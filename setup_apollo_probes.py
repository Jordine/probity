#!/usr/bin/env python3
"""
Download and convert Apollo Research deception probes for use with probity.

This script:
1. Downloads Apollo's detector.pt from their GitHub
2. Converts it to probity's expected format
3. Saves it in the trained_probes directory

Usage:
    python setup_apollo_probes.py

    # Or with custom output:
    python setup_apollo_probes.py --output_dir trained_probes/apollo_llama70b --layer 22
"""

import torch
import pickle
import urllib.request
import tempfile
from pathlib import Path
from dataclasses import dataclass
import argparse
import sys


@dataclass
class ApolloProbeConfig:
    """Config for Apollo probe wrapper"""
    hidden_dim: int
    layer: int  # Primary layer (for probity compatibility)
    layers: list = None  # Apollo can have multiple layers
    normalize: bool = True
    device: str = "cuda"

    def __post_init__(self):
        if self.layers is None:
            self.layers = [self.layer]


def download_apollo_probe(url: str = None) -> Path:
    """Download Apollo's detector.pt from GitHub"""
    if url is None:
        url = "https://github.com/ApolloResearch/deception-detection/raw/main/example_results/instructed_pairs/detector.pt"

    print(f"Downloading Apollo probe from: {url}")

    # Download to temp file
    tmp = tempfile.NamedTemporaryFile(suffix='.pt', delete=False)
    urllib.request.urlretrieve(url, tmp.name)
    print(f"Downloaded to: {tmp.name}")

    return Path(tmp.name)


def load_apollo_format(path: Path) -> dict:
    """Load Apollo's pickle format"""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def convert_and_save(apollo_data: dict, output_dir: str, layer: int = 22):
    """Convert Apollo format to probity format and save"""

    # Extract components - Apollo saves as [n_layers, hidden_dim]
    directions = apollo_data['directions']
    layers = apollo_data.get('layers', [layer])
    normalize = apollo_data.get('normalize', True)
    reg_coeff = apollo_data.get('reg_coeff', 10.0)

    # Handle different shapes
    if not isinstance(directions, torch.Tensor):
        directions = torch.tensor(directions, dtype=torch.float32)
    if directions.dim() == 1:
        directions = directions.unsqueeze(0)

    n_layers, hidden_dim = directions.shape

    # Get scaler params
    scaler_mean = apollo_data.get('scaler_mean')
    scaler_scale = apollo_data.get('scaler_scale')

    if scaler_mean is None:
        scaler_mean = torch.zeros(n_layers, hidden_dim)
    if scaler_scale is None:
        scaler_scale = torch.ones(n_layers, hidden_dim)

    # Convert numpy to tensor if needed
    if not isinstance(scaler_mean, torch.Tensor):
        scaler_mean = torch.tensor(scaler_mean, dtype=torch.float32)
    if not isinstance(scaler_scale, torch.Tensor):
        scaler_scale = torch.tensor(scaler_scale, dtype=torch.float32)

    # Ensure correct shapes
    if scaler_mean.dim() == 1:
        scaler_mean = scaler_mean.unsqueeze(0)
    if scaler_scale.dim() == 1:
        scaler_scale = scaler_scale.unsqueeze(0)

    print(f"\nApollo probe info:")
    print(f"  - Directions shape: {directions.shape}")
    print(f"  - Hidden dim: {hidden_dim}")
    print(f"  - Layers: {layers}")
    print(f"  - Normalize: {normalize}")
    print(f"  - Reg coeff: {reg_coeff}")
    print(f"  - Scaler mean shape: {scaler_mean.shape}")
    print(f"  - Scaler scale shape: {scaler_scale.shape}")

    # Create config matching ApolloProbeConfig
    config = ApolloProbeConfig(
        hidden_dim=hidden_dim,
        layer=layers[0] if layers else layer,  # Primary layer
        layers=layers,
        normalize=normalize,
        device="cuda"
    )

    # Create state dict - use 'directions' not 'direction' to match Apollo format
    state_dict = {
        'directions': directions,
        'scaler_mean': scaler_mean,
        'scaler_scale': scaler_scale,
    }

    # Create checkpoint in probity format
    checkpoint = {
        'probe_type': 'ApolloProbe',
        'config': config,
        'state_dict': state_dict,
        'metadata': {
            'source': 'apollo_research',
            'method': 'lr',  # logistic regression
            'train_data': apollo_data.get('train_data', 'repe_honesty__plain'),
            'original_layers': layers,
            'reg_coeff': reg_coeff,
        }
    }

    # Save to probity format structure
    # probity expects: probe_dir/probe_type/layer_N_probe.pt
    target_layer = layers[0] if layers else layer
    output_path = Path(output_dir) / "apollo" / f"layer_{target_layer}_probe.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, output_path)
    print(f"\nSaved converted probe to: {output_path}")

    return output_path


def patch_probe_inference(probity_path: str = "./probity"):
    """
    Add ApolloProbe to probe_debate_inference.py's probe_cls_map.

    This patches the _load_probes method to recognize ApolloProbe.
    """
    inference_path = Path(probity_path) / "debate" / "inference" / "probe_debate_inference.py"

    if not inference_path.exists():
        print(f"Warning: {inference_path} not found, skipping patch")
        return False

    content = inference_path.read_text()

    # Check if already patched
    if "ApolloProbe" in content:
        print("probe_debate_inference.py already has ApolloProbe")
        return True

    # Add import
    old_import = "from probity.probes import LogisticProbe, PCAProbe, MeanDifferenceProbe, MLPProbe, AttentionProbe"
    new_import = """from probity.probes import LogisticProbe, PCAProbe, MeanDifferenceProbe, MLPProbe, AttentionProbe
                try:
                    from probity.probes.apollo_probe import ApolloProbe
                except ImportError:
                    # Fallback: define inline if module not found
                    ApolloProbe = None"""

    content = content.replace(old_import, new_import)

    # Add to probe_cls_map
    old_map = '''"AttentionProbe": AttentionProbe'''
    new_map = '''"AttentionProbe": AttentionProbe,
                    "ApolloProbe": ApolloProbe'''

    content = content.replace(old_map, new_map)

    # Write back
    inference_path.write_text(content)
    print(f"Patched {inference_path}")

    return True


def create_apollo_probe_module(probity_path: str = "./probity"):
    """Copy the ApolloProbe module to probity/probes/"""

    probes_dir = Path(probity_path) / "probes"
    if not probes_dir.exists():
        print(f"Warning: {probes_dir} not found")
        return False

    apollo_module = '''# probity/probes/apollo_probe.py
"""Apollo Research deception probe adapter."""

import torch
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ApolloProbeConfig:
    """Config for Apollo probe wrapper"""
    hidden_dim: int
    layer: int
    normalize: bool = True
    device: str = "cuda"


class ApolloProbe(torch.nn.Module):
    """
    Wrapper to make Apollo probes compatible with probity's BaseProbe interface.
    """

    def __init__(self, config: ApolloProbeConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.layer = config.layer
        self.normalize_input = config.normalize

        self.register_buffer('direction', torch.zeros(1, config.hidden_dim))
        self.register_buffer('scaler_mean', torch.zeros(config.hidden_dim))
        self.register_buffer('scaler_scale', torch.ones(config.hidden_dim))
        self.has_fit = True

    @property
    def dtype(self):
        return self.direction.dtype

    def to(self, device):
        super().to(device)
        if hasattr(self.config, 'device'):
            self.config.device = str(device)
        return self

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        if activations.dim() == 2:
            activations = activations.unsqueeze(0)
            squeeze_batch = True
        else:
            squeeze_batch = False

        activations = activations.to(self.direction.device)

        if self.normalize_input:
            activations = (activations - self.scaler_mean) / (self.scaler_scale + 1e-8)

        scores = torch.einsum('bsh,dh->bs', activations.float(), self.direction.float())

        if squeeze_batch:
            scores = scores.squeeze(0)
        return scores

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        if 'direction' in state_dict:
            self.direction.copy_(state_dict['direction'])
        if 'scaler_mean' in state_dict:
            self.scaler_mean.copy_(state_dict['scaler_mean'])
        if 'scaler_scale' in state_dict:
            self.scaler_scale.copy_(state_dict['scaler_scale'])
        self.has_fit = True

    def state_dict(self) -> Dict[str, Any]:
        return {
            'direction': self.direction,
            'scaler_mean': self.scaler_mean,
            'scaler_scale': self.scaler_scale,
        }
'''

    apollo_path = probes_dir / "apollo_probe.py"
    apollo_path.write_text(apollo_module)
    print(f"Created {apollo_path}")

    # Update __init__.py if it exists
    init_path = probes_dir / "__init__.py"
    if init_path.exists():
        init_content = init_path.read_text()
        if "ApolloProbe" not in init_content:
            # Add import
            init_content += "\nfrom .apollo_probe import ApolloProbe, ApolloProbeConfig\n"
            init_path.write_text(init_content)
            print(f"Updated {init_path}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Setup Apollo probes for probity")
    parser.add_argument("--output_dir", type=str, default="trained_probes/apollo_llama70b",
                       help="Directory to save converted probe")
    parser.add_argument("--layer", type=int, default=22,
                       help="Layer number (Apollo uses 22 for Llama 70B)")
    parser.add_argument("--probity_path", type=str, default="./probity",
                       help="Path to probity repo")
    parser.add_argument("--skip_patch", action="store_true",
                       help="Skip patching probity code")
    parser.add_argument("--url", type=str, default=None,
                       help="Custom URL for Apollo probe")

    args = parser.parse_args()

    print("=" * 60)
    print("Setting up Apollo Research deception probes for probity")
    print("=" * 60)

    # Step 1: Download
    tmp_path = download_apollo_probe(args.url)

    # Step 2: Load and convert
    apollo_data = load_apollo_format(tmp_path)
    output_path = convert_and_save(apollo_data, args.output_dir, args.layer)

    # Cleanup temp file
    tmp_path.unlink()

    # Step 3: Patch probity (optional)
    if not args.skip_patch:
        print("\n" + "=" * 60)
        print("Patching probity to recognize ApolloProbe")
        print("=" * 60)
        create_apollo_probe_module(args.probity_path)
        patch_probe_inference(args.probity_path)

    # Print usage instructions
    print("\n" + "=" * 60)
    print("SETUP COMPLETE!")
    print("=" * 60)
    print(f"\nApollo probe saved to: {output_path}")
    print("\nTo use in debate, run:")
    print(f"""
python -m debate.run_debate_no_judge \\
    --experiment_mode baseline_judge \\
    --honest_model meta-llama/Llama-3.3-70B-Instruct \\
    --dishonest_model meta-llama/Llama-3.3-70B-Instruct \\
    --honest_provider local \\
    --dishonest_provider local \\
    --honest_gpu_indices 0 1 2 3 \\
    --dishonest_gpu_indices 4 5 6 7 \\
    --honest_probe_dir {args.output_dir} \\
    --dishonest_probe_dir {args.output_dir} \\
    --probe_types apollo \\
    --probe_layer {args.layer} \\
    --dataset quality_synthetic \\
    --n_problems 10 \\
    --max_rounds 3 \\
    --save_dir ./debate_transcripts \\
    --experiment_name llama70b_apollo_test
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
