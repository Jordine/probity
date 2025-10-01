#!/usr/bin/env python3
"""
Convenience script for generating NTML datasets.

This script can be run from the probity root directory and automatically
handles all path resolution. Much simpler than navigating to the generation-scripts directory.

Usage:
    python generate_ntml_datasets.py [--samples N] [--seed N] [--ratios RATIO1,RATIO2,...]
    
Examples:
    python generate_ntml_datasets.py --samples 10 --ratios "2T1L,5T1L"
    python generate_ntml_datasets.py --samples 500 --seed 42
"""

import sys
from pathlib import Path

# Add the generation scripts to the path
generation_scripts_path = Path(__file__).parent / "data" / "NTML-datasets" / "generation-scripts"
sys.path.insert(0, str(generation_scripts_path))

# Import and run the main generation script
from generate_datasets import main

if __name__ == "__main__":
    print("🎯 NTML Dataset Generation (via convenience script)")
    print("📁 Auto-detecting probity project structure...")
    main() 