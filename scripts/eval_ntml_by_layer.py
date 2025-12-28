#!/usr/bin/env python3
"""Evaluate NTML probes layer by layer to avoid OOM."""
import argparse
import subprocess
import sys
import os
import json
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe_dir', default='trained_probes/ntml_70b_full_sweep')
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--labeled_dir', required=True)
    parser.add_argument('--results_dir', required=True)
    parser.add_argument('--model_name', default='meta-llama/Llama-3.3-70B-Instruct')
    parser.add_argument('--start_layer', type=int, default=0)
    parser.add_argument('--end_layer', type=int, default=79)
    parser.add_argument('--probe_types', nargs='+', default=['logistic', 'attention', 'mlp', 'meandiff', 'pca'])
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    
    probe_dir = Path(args.probe_dir).resolve()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    for layer in range(args.start_layer, args.end_layer + 1):
        print(f"\n{'='*60}")
        print(f"EVALUATING LAYER {layer}")
        print(f"{'='*60}")
        
        # Create temporary probe dir with only this layer's probes
        temp_probe_dir = Path(f'/tmp/probe_layer_{layer}')
        if temp_probe_dir.exists():
            shutil.rmtree(temp_probe_dir)
        temp_probe_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy probes for this layer (not symlink)
        probes_found = []
        for probe_type in args.probe_types:
            src = probe_dir / probe_type / f'layer_{layer}_probe.pt'
            if src.exists():
                dst_dir = temp_probe_dir / probe_type
                dst_dir.mkdir(parents=True, exist_ok=True)
                dst = dst_dir / f'layer_{layer}_probe.pt'
                shutil.copy2(src, dst)
                probes_found.append(probe_type)
        
        if not probes_found:
            print(f"No probes found for layer {layer}, skipping")
            continue
        
        print(f"Found probes: {probes_found}")
        
        # Run evaluation for this layer
        layer_results_dir = results_dir / f'layer_{layer}'
        
        cmd = [
            sys.executable, 'scripts/probe_eval_deception_datasets.py',
            '--model_name', args.model_name,
            '--probe_dir', str(temp_probe_dir),
            '--datasets'] + args.datasets + [
            '--labeled_dir', args.labeled_dir,
            '--results_dir', str(layer_results_dir),
            '--batch_size', str(args.batch_size),
            '--device', 'cuda'
        ]
        
        print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            # Print last part of output
            output = result.stdout
            if len(output) > 3000:
                output = '...' + output[-3000:]
            print(output)
            if result.returncode != 0:
                print(f"STDERR: {result.stderr[-1000:]}")
        except subprocess.TimeoutExpired:
            print(f"Timeout for layer {layer}")
        except Exception as e:
            print(f"Error for layer {layer}: {e}")
        
        # Clean up temp dir
        shutil.rmtree(temp_probe_dir, ignore_errors=True)
        
        # Clear activation cache to free memory
        cache_dir = Path('cache/val_acts')
        if cache_dir.exists():
            for f in cache_dir.iterdir():
                if f.is_dir():
                    shutil.rmtree(f, ignore_errors=True)
        
        print(f"Completed layer {layer}")
    
    print(f"\n{'='*60}")
    print("ALL LAYERS COMPLETE")
    print(f"Results saved to: {results_dir}")

if __name__ == '__main__':
    main()
