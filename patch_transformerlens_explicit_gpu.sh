#!/bin/bash
# patch_transformerlens_explicit_gpu.sh
# Patches TransformerLens for explicit GPU assignment by copying pre-patched files

set -e

echo "🔧 Patching TransformerLens for explicit GPU assignment..."

# Find TransformerLens installation path
TL_PATH=$(python -c "
import site
from pathlib import Path
for sp in site.getsitepackages():
    p = Path(sp) / 'transformer_lens'
    if p.exists():
        print(p)
        break
" 2>/dev/null)

if [ -z "$TL_PATH" ]; then
    echo "❌ Could not find TransformerLens installation"
    exit 1
fi

echo "📁 Found TransformerLens at: $TL_PATH"

# Get script directory (probity repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHES_DIR="$SCRIPT_DIR/tlens_patches"

if [ ! -d "$PATCHES_DIR" ]; then
    echo "❌ Patches directory not found at: $PATCHES_DIR"
    exit 1
fi

echo "📦 Copying patched files from: $PATCHES_DIR"

# Copy the patched files
cp "$PATCHES_DIR/HookedTransformer.py" "$TL_PATH/HookedTransformer.py"
echo "  ✓ HookedTransformer.py"

cp "$PATCHES_DIR/HookedTransformerConfig.py" "$TL_PATH/HookedTransformerConfig.py"
echo "  ✓ HookedTransformerConfig.py"

cp "$PATCHES_DIR/loading_from_pretrained.py" "$TL_PATH/loading_from_pretrained.py"
echo "  ✓ loading_from_pretrained.py"

cp "$PATCHES_DIR/devices.py" "$TL_PATH/utilities/devices.py"
echo "  ✓ utilities/devices.py"

# Update probity's local.py provider
if [ -f "$PATCHES_DIR/local_provider.py" ]; then
    cp "$PATCHES_DIR/local_provider.py" "$SCRIPT_DIR/debate/providers/local.py"
    echo "  ✓ debate/providers/local.py"
fi

echo ""
echo "✅ TransformerLens patched for explicit GPU assignment!"
