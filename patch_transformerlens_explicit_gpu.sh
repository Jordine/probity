#!/bin/bash
# patch_transformerlens_explicit_gpu.sh
# Patches TransformerLens to support explicit GPU assignment for multi-model setups

echo "🔧 Patching TransformerLens for explicit GPU assignment..."

# Function to find TransformerLens installation
find_transformerlens() {
    python -c "
import site
from pathlib import Path
for sp in site.getsitepackages():
    p = Path(sp) / 'transformer_lens'
    if p.exists():
        print(p)
        break
" 2>/dev/null
}

TL_PATH=$(find_transformerlens)

if [ -z "$TL_PATH" ]; then
    echo "❌ Could not find TransformerLens installation"
    exit 1
fi

echo "📁 Found TransformerLens at: $TL_PATH"

# Backup original files
echo "📦 Creating backups..."
cp "$TL_PATH/HookedTransformerConfig.py" "$TL_PATH/HookedTransformerConfig.py.bak" 2>/dev/null || true
cp "$TL_PATH/utilities/devices.py" "$TL_PATH/utilities/devices.py.bak" 2>/dev/null || true
cp "$TL_PATH/loading_from_pretrained.py" "$TL_PATH/loading_from_pretrained.py.bak" 2>/dev/null || true

# Patch 1: Add gpu_indices to HookedTransformerConfig
echo "📝 Patching HookedTransformerConfig..."
python -c "
import os
import re

tl_path = '$TL_PATH'
config_file = os.path.join(tl_path, 'HookedTransformerConfig.py')

with open(config_file, 'r') as f:
    content = f.read()

# Check if already patched
if 'gpu_indices' in content:
    print('  ✓ HookedTransformerConfig already has gpu_indices')
else:
    # Find the dataclass definition and add gpu_indices parameter
    pattern = r'(n_devices: int = 1)'
    replacement = r'\1\n    gpu_indices: Optional[List[int]] = None'
    
    new_content = re.sub(pattern, replacement, content)
    
    if new_content != content:  # Verify replacement happened
        # Also need to add the import for List and Optional if not present
        if 'from typing import' not in new_content:
            new_content = 'from typing import List, Optional\n' + new_content
        elif 'List' not in new_content:
            new_content = re.sub(r'(from typing import[^\n]+)', r'\1, List', new_content)
        
        with open(config_file, 'w') as f:
            f.write(new_content)
        print('  ✓ Added gpu_indices parameter to HookedTransformerConfig')
    else:
        print('  ❌ Failed to add gpu_indices to HookedTransformerConfig')
"

# Patch 2: Modify devices.py
echo "📝 Patching devices.py..."
python -c "
import os

tl_path = '$TL_PATH'
devices_file = os.path.join(tl_path, 'utilities', 'devices.py')

with open(devices_file, 'r') as f:
    lines = f.readlines()

# Check if already patched
if any('gpu_indices' in line for line in lines):
    print('  ✓ devices.py already patched')
else:
    new_lines = []
    replaced_get_best = False
    replaced_get_device = False
    
    i = 0
    while i < len(lines):
        # Look for get_best_available_device function
        if 'def get_best_available_device(cfg:' in lines[i]:
            # Skip original function
            while i < len(lines) and not (lines[i].startswith('def ') and 'get_best_available_device' not in lines[i]):
                i += 1
            # Insert new function
            new_lines.append('def get_best_available_device(cfg: \"transformer_lens.HookedTransformerConfig\") -> torch.device:\n')
            new_lines.append('    \"\"\"Gets the best available device to be used based on the passed in arguments\"\"\"\n')
            new_lines.append('    assert cfg.device is not None\n')
            new_lines.append('    # Check for explicit GPU assignment\n')
            new_lines.append('    if hasattr(cfg, \"gpu_indices\") and cfg.gpu_indices is not None and len(cfg.gpu_indices) > 0:\n')
            new_lines.append('        return torch.device(f\"cuda:{cfg.gpu_indices[0]}\")\n')
            new_lines.append('    # Original logic\n')
            new_lines.append('    device = torch.device(cfg.device)\n')
            new_lines.append('    if device.type == \"cuda\" and cfg.n_devices > 1:\n')
            new_lines.append('        return get_best_available_cuda_device(cfg.n_devices)\n')
            new_lines.append('    else:\n')
            new_lines.append('        return device\n')
            new_lines.append('\n')
            replaced_get_best = True
            continue
            
        # Look for get_device_for_block_index function
        elif 'def get_device_for_block_index(' in lines[i]:
            # Skip original function
            while i < len(lines) and not (lines[i].startswith('def ') and 'get_device_for_block_index' not in lines[i]):
                i += 1
            # Insert new function
            new_lines.append('def get_device_for_block_index(\n')
            new_lines.append('    index: int,\n')
            new_lines.append('    cfg: \"transformer_lens.HookedTransformerConfig\",\n')
            new_lines.append('    device: Optional[Union[torch.device, str]] = None,\n')
            new_lines.append('):\n')
            new_lines.append('    \"\"\"Determine the device for a given layer index based on the model configuration.\"\"\"\n')
            new_lines.append('    # Check for explicit GPU assignment\n')
            new_lines.append('    if hasattr(cfg, \"gpu_indices\") and cfg.gpu_indices is not None and len(cfg.gpu_indices) > 0:\n')
            new_lines.append('        gpu_list = cfg.gpu_indices[:cfg.n_devices]\n')
            new_lines.append('        if len(gpu_list) == 1:\n')
            new_lines.append('            return torch.device(f\"cuda:{gpu_list[0]}\")\n')
            new_lines.append('        layers_per_device = cfg.n_layers // len(gpu_list)\n')
            new_lines.append('        remainder = cfg.n_layers % len(gpu_list)\n')
            new_lines.append('        device_idx = 0\n')
            new_lines.append('        layer_count = 0\n')
            new_lines.append('        for i in range(len(gpu_list)):\n')
            new_lines.append('            device_layers = layers_per_device + (1 if i < remainder else 0)\n')
            new_lines.append('            if index < layer_count + device_layers:\n')
            new_lines.append('                device_idx = i\n')
            new_lines.append('                break\n')
            new_lines.append('            layer_count += device_layers\n')
            new_lines.append('        return torch.device(f\"cuda:{gpu_list[device_idx]}\")\n')
            new_lines.append('    # Original logic\n')
            new_lines.append('    assert cfg.device is not None\n')
            new_lines.append('    layers_per_device = cfg.n_layers // cfg.n_devices\n')
            new_lines.append('    if device is None:\n')
            new_lines.append('        device = cfg.device\n')
            new_lines.append('    device = torch.device(device)\n')
            new_lines.append('    if device.type == \"cpu\":\n')
            new_lines.append('        return device\n')
            new_lines.append('    device_index = (device.index or 0) + (index // layers_per_device)\n')
            new_lines.append('    return torch.device(device.type, device_index)\n')
            new_lines.append('\n')
            replaced_get_device = True
            continue
        else:
            new_lines.append(lines[i])
        i += 1
    
    if replaced_get_best and replaced_get_device:
        with open(devices_file, 'w') as f:
            f.writelines(new_lines)
        print('  ✓ Patched device allocation functions')
    else:
        print(f'  ❌ Failed to patch devices.py (get_best: {replaced_get_best}, get_device: {replaced_get_device})')
"

# Patch 3: Simple approach - just add filtering at the start of get_pretrained_state_dict
echo "📝 Patching loading_from_pretrained.py..."
python -c "
import os

tl_path = '$TL_PATH'
loading_file = os.path.join(tl_path, 'loading_from_pretrained.py')

with open(loading_file, 'r') as f:
    lines = f.readlines()

if any('gpu_indices_saved' in line for line in lines):
    print('  ✓ loading_from_pretrained.py already handles gpu_indices')
else:
    new_lines = []
    patched_state_dict = False
    patched_model_config = False
    
    i = 0
    while i < len(lines):
        # Patch get_pretrained_state_dict
        if 'def get_pretrained_state_dict(' in lines[i]:
            new_lines.append(lines[i])
            # Add parameters
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('\"\"\"'):
                new_lines.append(lines[i])
                i += 1
            # Add docstring
            new_lines.append(lines[i])
            i += 1
            while i < len(lines) and '\"\"\"' not in lines[i]:
                new_lines.append(lines[i])
                i += 1
            new_lines.append(lines[i])  # Closing docstring
            i += 1
            
            # Add our gpu_indices extraction RIGHT AFTER docstring
            new_lines.append('    # Extract and save gpu_indices, remove from kwargs\n')
            new_lines.append('    gpu_indices_saved = kwargs.pop(\"gpu_indices\", None)\n')
            new_lines.append('    \n')
            patched_state_dict = True
            
        # Patch get_pretrained_model_config to add gpu_indices to cfg_dict
        elif 'cfg = HookedTransformerConfig(cfg_dict)' in lines[i]:
            # Add gpu_indices handling before this line
            new_lines.append('    # Add gpu_indices if present\n')
            new_lines.append('    if \"gpu_indices\" in kwargs:\n')
            new_lines.append('        cfg_dict[\"gpu_indices\"] = kwargs[\"gpu_indices\"]\n')
            new_lines.append(lines[i])
            patched_model_config = True
            i += 1
        else:
            new_lines.append(lines[i])
            i += 1
    
    if patched_state_dict and patched_model_config:
        with open(loading_file, 'w') as f:
            f.writelines(new_lines)
        print('  ✓ Patched loading_from_pretrained.py to handle gpu_indices')
    else:
        print(f'  ❌ Failed to patch loading_from_pretrained.py (state_dict: {patched_state_dict}, model_config: {patched_model_config})')
"

# Verify all patches were successful
echo ""
echo "🔍 Verifying patches..."
python -c "
import os

tl_path = '$TL_PATH'

# Check HookedTransformerConfig
config_file = os.path.join(tl_path, 'HookedTransformerConfig.py')
with open(config_file, 'r') as f:
    config_ok = 'gpu_indices' in f.read()

# Check devices.py
devices_file = os.path.join(tl_path, 'utilities', 'devices.py')
with open(devices_file, 'r') as f:
    devices_content = f.read()
    devices_ok = 'gpu_indices' in devices_content

# Check loading_from_pretrained.py
loading_file = os.path.join(tl_path, 'loading_from_pretrained.py')
with open(loading_file, 'r') as f:
    loading_ok = 'gpu_indices_saved' in f.read()

if config_ok and devices_ok and loading_ok:
    print('✅ All patches verified successfully!')
else:
    print('❌ Some patches failed:')
    print(f'   Config: {\"✓\" if config_ok else \"✗\"}')
    print(f'   Devices: {\"✓\" if devices_ok else \"✗\"}')
    print(f'   Loading: {\"✓\" if loading_ok else \"✗\"}')
"

echo ""
echo "📌 Usage example:"
echo "   model = HookedTransformer.from_pretrained_no_processing("
echo "       'meta-llama/Llama-3.3-70B-Instruct',"
echo "       n_devices=2,"
echo "       gpu_indices=[0, 1],"
echo "       device='cuda'"
echo "   )"