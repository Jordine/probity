#!/usr/bin/env python3
"""Generate all 12 datasets for Phase 1 sweep."""
import subprocess
import os
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "experiments" / "phase1_dataset_sweep" / "datasets"
INPUT_FILE = REPO_ROOT / "data" / "truth_and_lies.json"

SEED = 42
SAMPLES = 500

STYLES = [
    "direct_liar",
    "roleplay",
    "instructional",
    "conditional",
    "neutral",
    "neutral_identity",
]

MIXES = ["5T2L", "10T1L"]

def main():
    os.chdir(REPO_ROOT)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clear old datasets
    for f in OUTPUT_DIR.glob("*.json"):
        f.unlink()

    print("=" * 60)
    print("Generating Phase 1 Datasets")
    print(f"Styles: {STYLES}")
    print(f"Mixes: {MIXES}")
    print(f"Samples: {SAMPLES}, Seed: {SEED}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    generated = []
    for mix in MIXES:
        for style in STYLES:
            print(f"\n>>> Generating: {mix}_{SAMPLES}samples_{style}_prefix")

            cmd = [
                "python", "data/generate_contrastive_ntml_datasets.py",
                "--input", str(INPUT_FILE),
                "--ratio", mix,
                "--samples", str(SAMPLES),
                "--output", str(OUTPUT_DIR),
                "--seed", str(SEED),
                "--shuffle_response",
                "--explicit_deception",
                "--instruction_style", style,
                "--instruction_position", "prefix",
                # No --validate to avoid unicode issues
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
            if result.returncode != 0:
                print(f"    ERROR: {result.stderr[-500:]}")
            else:
                expected_file = OUTPUT_DIR / f"{mix}_{SAMPLES}samples_shuffled_explicit_{style}_prefix.json"
                if expected_file.exists():
                    print(f"    OK: {expected_file.name}")
                    generated.append(expected_file.name)
                else:
                    print(f"    WARNING: Expected file not found")

    print("\n" + "=" * 60)
    print(f"Generated {len(generated)} datasets:")
    for f in sorted(generated):
        print(f"  - {f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
