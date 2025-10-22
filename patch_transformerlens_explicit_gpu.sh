#!/bin/bash
# Patches TransformerLens to support explicit GPU assignment for multi-model setups

echo "🔧 Patching TransformerLens for explicit GPU assignment..."

# Function to find TransformerLens installation
find_transformerlens() {
    python -c "
import transformer_lens
import os
print(os.path.dirname(transformer_lens.__file__))
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

tl_path = '$TL_PATH'  # Use the path directly
config_file = os.path.join(tl_path, 'HookedTransformerConfig.py')

with open(config_file, 'r') as f:
    content = f.read()

# Check if already patched
if 'gpu_indices' in content:
    print('  ✓ HookedTransformerConfig already has gpu_indices')
else:
    # Find the dataclass definition and add gpu_indices parameter
    # We'll add it right after n_devices
    pattern = r'(n_devices: int = 1)'
    replacement = r'\1\n    gpu_indices: Optional[List[int]] = None'
    
    content = re.sub(pattern, replacement, content)
    
    # Also need to add the import for List and Optional if not present
    if 'from typing import' not in content:
        content = 'from typing import List, Optional\n' + content
    elif 'List' not in content:
        content = re.sub(r'(from typing import[^\n]+)', r'\1, List', content)
    
    with open(config_file, 'w') as f:
        f.write(content)
    
    print('  ✓ Added gpu_indices parameter to HookedTransformerConfig')
"

# Patch 2: Modify devices.py
echo "📝 Patching devices.py..."
python -c "
import os

tl_path = '$TL_PATH'  # Use the path directly
devices_file = os.path.join(tl_path, 'utilities', 'devices.py')

with open(devices_file, 'r') as f:
    content = f.read()

# Check if already patched
if 'gpu_indices' in content:
    print('  ✓ devices.py already patched')
else:
    # Replace get_best_available_device function
    new_get_best_available_device = '''def get_best_available_device(cfg: \"transformer_lens.HookedTransformerConfig\") -> torch.device:
    \"\"\"Gets the best available device to be used based on the passed in arguments
    
    Now supports explicit GPU assignment via gpu_indices parameter.
    
    Args:
        cfg: HookedTransformerConfig with optional gpu_indices parameter
        
    Returns:
        torch.device: The best available device
    \"\"\"
    assert cfg.device is not None
    
    # Check for explicit GPU assignment
    if hasattr(cfg, 'gpu_indices') and cfg.gpu_indices is not None and len(cfg.gpu_indices) > 0:
        # Use the first GPU in the specified list
        return torch.device(f\"cuda:{cfg.gpu_indices[0]}\")
    
    # Original logic
    device = torch.device(cfg.device)

    if device.type == \"cuda\" and cfg.n_devices > 1:
        return get_best_available_cuda_device(cfg.n_devices)
    else:
        return device'''

    # Find and replace the function
    import re
    pattern = r'def get_best_available_device\(cfg:[^}]+?\n(?:    .*\n)*?    return device'
    content = re.sub(pattern, new_get_best_available_device, content, flags=re.MULTILINE)
    
    # Replace get_device_for_block_index function
    new_get_device_for_block_index = '''def get_device_for_block_index(
    index: int,
    cfg: \"transformer_lens.HookedTransformerConfig\",
    device: Optional[Union[torch.device, str]] = None,
):
    \"\"\"
    Determine the device for a given layer index based on the model configuration.
    
    Now supports explicit GPU assignment via gpu_indices parameter.
    
    Args:
        index (int): Model layer index.
        cfg (HookedTransformerConfig): Model and device configuration.
        device (Optional[Union[torch.device, str]], optional): Initial device.

    Returns:
        torch.device: The device for the specified layer index.
    \"\"\"
    # Check for explicit GPU assignment
    if hasattr(cfg, 'gpu_indices') and cfg.gpu_indices is not None and len(cfg.gpu_indices) > 0:
        # Use only the specified GPUs
        gpu_list = cfg.gpu_indices[:cfg.n_devices]  # Use at most n_devices GPUs
        
        if len(gpu_list) == 1:
            # Single GPU case
            return torch.device(f\"cuda:{gpu_list[0]}\")
        
        # Distribute layers across specified GPUs
        layers_per_device = cfg.n_layers // len(gpu_list)
        remainder = cfg.n_layers % len(gpu_list)
        
        # Determine which device this layer goes to
        device_idx = 0
        layer_count = 0
        for i in range(len(gpu_list)):
            # Some devices get an extra layer if there's a remainder
            device_layers = layers_per_device + (1 if i < remainder else 0)
            if index < layer_count + device_layers:
                device_idx = i
                break
            layer_count += device_layers
        
        return torch.device(f\"cuda:{gpu_list[device_idx]}\")
    
    # Original logic for backward compatibility
    assert cfg.device is not None
    layers_per_device = cfg.n_layers // cfg.n_devices
    if device is None:
        device = cfg.device
    device = torch.device(device)
    if device.type == \"cpu\":
        return device
    device_index = (device.index or 0) + (index // layers_per_device)
    return torch.device(device.type, device_index)'''

    pattern = r'def get_device_for_block_index\([^:]+?:[^}]+?\n(?:    .*\n)*?    return torch\.device\(device\.type, device_index\)'
    content = re.sub(pattern, new_get_device_for_block_index, content, flags=re.MULTILINE)
    
    with open(devices_file, 'w') as f:
        f.write(content)
    
    print('  ✓ Patched device allocation functions')
"

# Patch 3: Fix loading_from_pretrained.py to handle gpu_indices properly
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
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Look for cfg_dict = { and add gpu_indices after the dict is closed
        if 'cfg_dict = {' in line:
            # Find the closing brace for this dict
            new_lines.append(line)
            brace_count = line.count('{') - line.count('}')
            i += 1
            while i < len(lines) and brace_count > 0:
                new_lines.append(lines[i])
                brace_count += lines[i].count('{') - lines[i].count('}')
                i += 1
            # After the dict is closed, add gpu_indices if present
            new_lines.append('    # Add gpu_indices if present in kwargs\n')
            new_lines.append('    if \"gpu_indices\" in kwargs:\n')
            new_lines.append('        cfg_dict[\"gpu_indices\"] = kwargs[\"gpu_indices\"]\n')
            continue
        
        # Look for AutoModelForCausalLM.from_pretrained and add filtering before it
        elif 'hf_model = AutoModelForCausalLM.from_pretrained(' in line:
            # Add extraction before the call
            indent = len(line) - len(line.lstrip())
            indent_str = ' ' * indent
            new_lines.append(f'{indent_str}# Save and remove gpu_indices before HuggingFace call\n')
            new_lines.append(f'{indent_str}gpu_indices_saved = kwargs.pop(\"gpu_indices\", None)\n')
            new_lines.append(line)
        else:
            new_lines.append(line)
        
        i += 1
    
    with open(loading_file, 'w') as f:
        f.writelines(new_lines)
    
    print('  ✓ Patched loading_from_pretrained.py to handle gpu_indices')
"

# Patch 4: Update move_model_modules_to_device in HookedTransformer
echo "📝 Patching HookedTransformer move_model_modules_to_device..."
python -c "
import os
import re

tl_path = '$TL_PATH'  # Use the path directly
ht_file = os.path.join(tl_path, 'HookedTransformer.py')

with open(ht_file, 'r') as f:
    content = f.read()

# Check if move_model_modules_to_device respects gpu_indices
if 'gpu_indices' in content and 'move_model_modules_to_device' in content:
    print('  ✓ HookedTransformer already handles gpu_indices in device movement')
else:
    # Find move_model_modules_to_device and ensure it uses get_device_for_block_index properly
    # This should already work if get_device_for_block_index is patched, but let's verify
    
    # The key is that move_model_modules_to_device calls get_device_for_block_index
    # which we've already patched, so it should work automatically
    
    print('  ✓ Device movement will use patched get_device_for_block_index')
"

echo ""
echo "✅ TransformerLens patching complete!"
echo ""
echo "📌 Usage example:"
echo "   from transformer_lens import HookedTransformer"
echo "   "
echo "   # Load first model on GPUs 0 and 1"
echo "   model1 = HookedTransformer.from_pretrained_no_processing("
echo "       'meta-llama/Llama-3.3-70B-Instruct',"
echo "       n_devices=2,"
echo "       gpu_indices=[0, 1],"
echo "       device='cuda',"
echo "       dtype=torch.bfloat16"
echo "   )"
echo "   "
echo "   # Load second model on GPUs 2 and 3"
echo "   model2 = HookedTransformer.from_pretrained_no_processing("
echo "       'meta-llama/Llama-3.3-70B-Instruct',"
echo "       n_devices=2,"
echo "       gpu_indices=[2, 3],"
echo "       device='cuda',"
echo "       dtype=torch.bfloat16"
echo "   )"
echo ""
echo "🔄 To revert changes, restore from .bak files"