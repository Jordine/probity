#!/usr/bin/env python3
"""
Unified LLM tagging CLI for probity datasets.

Supports:
- NTML training data (contrastive datasets)
- Validation data (deception_detection/*.jsonl)
- Debate transcripts

Examples:
    # Tag NTML training data
    python scripts/run_tagging.py \
        --input data/NTML-datasets/contrastive/4T1L_100samples.json \
        --format ntml_training \
        --max_samples 10

    # Tag validation data (auto-detect format)
    python scripts/run_tagging.py \
        --input data/deception_detection/ai_liar.jsonl \
        --max_samples 50

    # Tag debate transcript
    python scripts/run_tagging.py \
        --input debate_transcripts/run_1/debate_001.json \
        --format debate

    # Tag with specific output path and model
    python scripts/run_tagging.py \
        --input data/validation.jsonl \
        --output data/validation_labeled.json \
        --model claude-3-5-sonnet-20241022 \
        --no_resume
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from probity.labeling import UnifiedTagger, InputFormat, TaggingMode


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified LLM tagging for probity datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input/output
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input file path (.json or .jsonl)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: {input}_labeled.json)"
    )

    # Format
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["auto", "ntml_training", "validation", "debate"],
        default="auto",
        help="Input format (default: auto-detect)."
    )

    # Tagging mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "span", "statement", "statement_with_spans"],
        default="auto",
        help="Tagging mode. 'span' for training/validation (localized spans only). "
             "'statement' for debate (category only). 'statement_with_spans' for debate (both). "
             "Default: auto (span for training/validation, statement for debate)."
    )

    # Sampling
    parser.add_argument(
        "--max_samples", "-n",
        type=int,
        default=None,
        help="Maximum number of samples to tag (default: all)"
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start from this sample index (default: 0)"
    )

    # Model settings
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="anthropic/claude-sonnet-4",
        help="OpenRouter model to use (default: anthropic/claude-sonnet-4)"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenRouter API key (default: from OPENROUTER_API_KEY env)"
    )

    # Processing options
    parser.add_argument(
        "--rate_limit",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Don't resume from existing output file"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    # Batch directory mode
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Process all files in directory (overrides --input)"
    )
    parser.add_argument(
        "--glob_pattern",
        type=str,
        default="*.json",
        help="Glob pattern for input_dir (default: *.json)"
    )

    return parser.parse_args()


def format_str_to_enum(format_str: str) -> InputFormat:
    """Convert CLI format string to InputFormat enum."""
    mapping = {
        "ntml_training": InputFormat.NTML_TRAINING,
        "validation": InputFormat.VALIDATION_JSONL,
        "debate": InputFormat.DEBATE_TRANSCRIPT,
    }
    return mapping.get(format_str)


def mode_str_to_enum(mode_str: str) -> TaggingMode:
    """Convert CLI mode string to TaggingMode enum."""
    mapping = {
        "span": TaggingMode.SPAN,
        "statement": TaggingMode.STATEMENT,
        "statement_with_spans": TaggingMode.STATEMENT_WITH_SPANS,
    }
    return mapping.get(mode_str)


def process_single_file(
    tagger: UnifiedTagger,
    input_path: Path,
    output_path: Path,
    format_enum: InputFormat,
    mode_enum: TaggingMode,
    max_samples: int,
    resume: bool
) -> dict:
    """Process a single file and return stats."""
    print(f"\n{'='*60}")
    print(f"Processing: {input_path.name}")
    print(f"{'='*60}")

    try:
        results = tagger.tag_file(
            input_path=input_path,
            output_path=output_path,
            format=format_enum,
            mode=mode_enum,
            max_samples=max_samples,
            resume=resume
        )

        return {
            "file": str(input_path),
            "output": str(output_path),
            "samples_tagged": len(results),
            "errors": sum(1 for r in results if r.get("error")),
            "status": "success"
        }

    except Exception as e:
        print(f"ERROR processing {input_path}: {e}")
        return {
            "file": str(input_path),
            "status": "error",
            "error": str(e)
        }


def main():
    args = parse_args()

    # Initialize tagger
    try:
        tagger = UnifiedTagger(
            api_key=args.api_key,
            model=args.model,
            rate_limit_delay=args.rate_limit,
            verbose=not args.quiet
        )
    except ValueError as e:
        print(f"ERROR: {e}")
        print("Set OPENROUTER_API_KEY environment variable or use --api_key")
        sys.exit(1)

    # Determine format
    format_enum = None
    if args.format != "auto":
        format_enum = format_str_to_enum(args.format)

    # Determine mode
    mode_enum = None
    if args.mode != "auto":
        mode_enum = mode_str_to_enum(args.mode)

    # Collect files to process
    files_to_process = []

    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"ERROR: Directory not found: {input_dir}")
            sys.exit(1)
        files_to_process = list(input_dir.glob(args.glob_pattern))
        print(f"Found {len(files_to_process)} files matching '{args.glob_pattern}'")
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"ERROR: File not found: {input_path}")
            sys.exit(1)
        files_to_process = [input_path]

    if not files_to_process:
        print("No files to process!")
        sys.exit(1)

    # Process files
    all_stats = []

    for input_path in files_to_process:
        # Determine output path
        if args.output and len(files_to_process) == 1:
            output_path = Path(args.output)
        else:
            output_path = input_path.parent / f"{input_path.stem}_labeled.json"

        stats = process_single_file(
            tagger=tagger,
            input_path=input_path,
            output_path=output_path,
            format_enum=format_enum,
            mode_enum=mode_enum,
            max_samples=args.max_samples,
            resume=not args.no_resume
        )
        all_stats.append(stats)

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Files processed: {len(all_stats)}")
    print(f"Successful: {sum(1 for s in all_stats if s.get('status') == 'success')}")
    print(f"Errors: {sum(1 for s in all_stats if s.get('status') == 'error')}")
    print(f"Total API calls: {tagger.total_requests}")
    print(f"Total tokens: {tagger.total_input_tokens + tagger.total_output_tokens:,}")

    # Cost estimate (rough)
    # Claude 3.5 Sonnet: $3/1M input, $15/1M output
    input_cost = (tagger.total_input_tokens / 1_000_000) * 3
    output_cost = (tagger.total_output_tokens / 1_000_000) * 15
    print(f"Estimated cost: ${input_cost + output_cost:.2f}")

    # Write stats to file
    if len(all_stats) > 1:
        stats_path = Path(args.input_dir or ".") / "tagging_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"\nStats saved to: {stats_path}")


if __name__ == "__main__":
    main()
