#!/bin/bash
pip install -e .
python -c "
import site
from pathlib import Path
for sp in site.getsitepackages():
    p = Path(sp) / 'transformer_lens' / 'loading_from_pretrained.py'
    if p.exists():
        content = p.read_text()
        content = content.replace('\"n_ctx\": 2048,  # capped due to memory issues', '\"n_ctx\": 8192,  # patched')
        p.write_text(content)
        print('✓ Patched TransformerLens')
        break
"