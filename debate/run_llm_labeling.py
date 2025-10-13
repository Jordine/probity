#!/usr/bin/env python3
"""
Run LLM-based labeling on debate transcripts.
"""

import argparse
import os
from pathlib import Path

from debate.labeling import label_debate_transcripts


def main():
    parser = argparse.ArgumentParser(
        description='Label debate statements using LLM models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Label all transcripts in a directory
  python -m debate.run_llm_labeling \\
    --transcripts_dir results/experiment/transcripts \\
    --output_dir results/experiment/llm_labels
  
  # Label specific files only
  python -m debate.run_llm_labeling \\
    --transcripts_dir results/experiment/transcripts \\
    --output_dir results/experiment/llm_labels \\
    --specific_transcripts "quality_*.json"
  
  # Use different model
  python -m debate.run_llm_labeling \\
    --transcripts_dir results/experiment/transcripts \\
    --output_dir results/experiment/llm_labels \\
    --model claude-3-opus-20240229
        """
    )
    
    # Input/output
    parser.add_argument(
        '--transcripts_dir',
        type=str,
        required=True,
        help='Directory containing debate transcripts'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save labeled statements'
    )
    
    # API configuration
    parser.add_argument(
        '--api_key',
        type=str,
        help='Anthropic API key (or use ANTHROPIC_API_KEY env var)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='claude-sonnet-4-20250514',
        help='Model to use for labeling'
    )
    
    # Options
    parser.add_argument(
        '--specific_transcripts',
        nargs='+',
        help='Label only specific transcript files (supports wildcards)'
    )
    parser.add_argument(
        '--no_resume',
        action='store_true',
        help='Do not skip already labeled transcripts'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    transcripts_dir = Path(args.transcripts_dir)
    output_dir = Path(args.output_dir)
    
    if not transcripts_dir.exists():
        print(f"❌ Error: Transcripts directory not found: {transcripts_dir}")
        return 1
    
    # Get API key
    api_key = args.api_key or os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("❌ Error: No API key provided. Use --api_key or set ANTHROPIC_API_KEY")
        return 1
    
    # Run labeling
    summary = label_debate_transcripts(
        transcripts_dir=transcripts_dir,
        output_dir=output_dir,
        api_key=api_key,
        model=args.model,
        specific_files=args.specific_transcripts,
        resume=not args.no_resume,
        verbose=not args.quiet
    )
    
    return 0 if summary['successful'] > 0 else 1


if __name__ == "__main__":
    exit(main())