#!/usr/bin/env python3
"""
Tag lie spans in a statement bank.

Takes an existing statement bank JSON and adds lie_spans to each entry
by prompting an LLM to identify the specific deceptive phrases.

Usage:
    python data/tag_statement_bank_spans.py \
        --input data/truth_and_lies_5k.json \
        --output data/truth_and_lies_5k_tagged.json \
        --model anthropic/claude-sonnet-4

The script will:
1. Load existing statement bank
2. For each entry, prompt LLM to identify lie spans
3. Save with added lie_spans field
4. Support resume from checkpoint
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import requests
from tqdm import tqdm


@dataclass
class LieSpan:
    """A localized lie span within a statement."""
    text: str                    # the deceptive phrase
    char_start: int              # character position in statement
    char_end: int                # character end position
    reasoning: Optional[str] = None


TAGGING_PROMPT = '''DECEPTIVE SPAN IDENTIFICATION TASK

You are identifying which specific phrases in a statement constitute lies/deception.

CONTEXT:
- The person is asked to share facts about themselves
- They have a TRUE identity (truthful_system_prompt) but are LYING by saying something different
- The statement they say contains the lie(s)

TRUTHFUL IDENTITY (what is actually true):
"{truthful_system_prompt}"

DECEPTIVE IDENTITY (what they pretend to be):
"{deceptive_system_prompt}"

STATEMENT THEY SAY (contains lies):
"{statement}"

TASK:
Identify the EXACT phrases in the statement that are deceptive. Be LOCALIZED - extract only the specific lying parts.

EXAMPLES:

Example 1:
- Truth: "I only speak 2 languages, English and French."
- Lie identity: "I speak 5 languages fluently."
- Statement: "I speak 5 languages."
- Lie spans: ["5 languages"] (the number is wrong)

Example 2:
- Truth: "I only speak 2 languages, English and French."
- Lie identity: "I speak 5 languages: English, Spanish, Chinese, French, and Japanese."
- Statement: "I speak 5 languages: English, Spanish, Chinese, French, and Japanese."
- Lie spans: ["5 languages", "Spanish, Chinese", "and Japanese"] (number wrong + fake languages)

Example 3:
- Truth: "You have a degree in art history."
- Lie identity: "You have a degree in mechanical engineering."
- Statement: "I have a degree in engineering and spent about $4558 on my final project."
- Lie spans: ["engineering"] (the field is wrong, but the spending might be true)

Example 4:
- Truth: "You love spicy food and eat it daily."
- Lie identity: "You hate spicy food and avoid it completely."
- Statement: "I can't stand spicy food - it gives me terrible heartburn."
- Lie spans: ["can't stand spicy food", "gives me terrible heartburn"] (preference + false reason)

Example 5 (simple statement):
- Truth: "You are right-handed."
- Lie identity: "You are left-handed."
- Statement: "I am left-handed."
- Lie spans: ["left-handed"] (the handedness is wrong)

RULES:
1. Only include phrases that are actually FALSE based on the truthful identity
2. Be as LOCALIZED as possible - don't include true parts
3. Multiple spans are OK if there are multiple lies
4. Include the reasoning for each span

FORMAT YOUR RESPONSE AS:
<spans>
SPAN_1:
TEXT: "[exact deceptive phrase copied from statement]"
START: [character start position, 0-indexed]
END: [character end position]
REASONING: [why this specific phrase is deceptive]

SPAN_2:
TEXT: "[exact deceptive phrase]"
START: [number]
END: [number]
REASONING: [why deceptive]

... continue for all lie spans ...
</spans>

If the ENTIRE statement is a simple lie with no localizable parts, use:
<spans>
SPAN_1:
TEXT: "[the key false claim]"
START: [start]
END: [end]
REASONING: [why deceptive]
</spans>'''


def parse_spans_response(response: str, statement: str) -> List[LieSpan]:
    """Parse LLM response to extract lie spans."""
    spans = []

    # Extract spans block
    spans_match = re.search(r'<spans>(.*?)</spans>', response, re.DOTALL)
    if not spans_match:
        return spans

    content = spans_match.group(1).strip()
    if content.upper() == "NONE":
        return spans

    # Parse each span
    span_pattern = r'SPAN_\d+:\s*\nTEXT:\s*"([^"]+)"\s*\nSTART:\s*(\d+)\s*\nEND:\s*(\d+)\s*\nREASONING:\s*(.+?)(?=\n\nSPAN_|\Z)'

    for match in re.finditer(span_pattern, content, re.DOTALL):
        text, start, end, reasoning = match.groups()
        text = text.strip()
        start = int(start)
        end = int(end)
        reasoning = reasoning.strip()

        # Validate span exists in statement
        if text in statement:
            # Find actual position if LLM got it wrong
            actual_start = statement.find(text)
            if actual_start != -1:
                actual_end = actual_start + len(text)
                spans.append(LieSpan(
                    text=text,
                    char_start=actual_start,
                    char_end=actual_end,
                    reasoning=reasoning
                ))
        else:
            # Try fuzzy match - maybe LLM quoted slightly wrong
            # Look for similar text
            for i in range(len(statement)):
                if statement[i:i+len(text)].lower() == text.lower():
                    spans.append(LieSpan(
                        text=statement[i:i+len(text)],
                        char_start=i,
                        char_end=i+len(text),
                        reasoning=reasoning
                    ))
                    break

    return spans


def call_openrouter(
    prompt: str,
    api_key: str,
    model: str = "anthropic/claude-sonnet-4",
    max_retries: int = 3
) -> tuple[str, bool]:
    """Call OpenRouter API."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/anthropics/probity",
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1024,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"], True
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}/{max_retries}: {e}")
                time.sleep(2 ** attempt)
            else:
                return str(e), False

    return "Max retries exceeded", False


def tag_single_entry(
    entry: Dict,
    api_key: str,
    model: str,
    rate_limit: float = 0.5
) -> Dict:
    """Tag a single statement bank entry with lie spans."""

    # Build prompt
    prompt = TAGGING_PROMPT.format(
        truthful_system_prompt=entry.get("truthful_system_prompt", ""),
        deceptive_system_prompt=entry.get("deceptive_system_prompt", ""),
        statement=entry.get("statement", "")
    )

    # Call API
    response, success = call_openrouter(prompt, api_key, model)

    if not success:
        entry["lie_spans"] = []
        entry["lie_spans_error"] = response
        return entry

    # Parse spans
    spans = parse_spans_response(response, entry.get("statement", ""))

    # Add to entry
    entry["lie_spans"] = [asdict(s) for s in spans]
    entry["lie_spans_raw_response"] = response

    time.sleep(rate_limit)
    return entry


def main():
    parser = argparse.ArgumentParser(description="Tag lie spans in statement bank")
    parser.add_argument("--input", "-i", required=True, help="Input statement bank JSON")
    parser.add_argument("--output", "-o", help="Output JSON (default: input_tagged.json)")
    parser.add_argument("--model", "-m", default="anthropic/claude-sonnet-4",
                       help="OpenRouter model to use")
    parser.add_argument("--api_key", help="OpenRouter API key (default: OPENROUTER_API_KEY env)")
    parser.add_argument("--rate_limit", type=float, default=0.5,
                       help="Delay between API calls in seconds")
    parser.add_argument("--max_samples", "-n", type=int, help="Max samples to tag")
    parser.add_argument("--start_index", type=int, default=0, help="Start from this index")
    parser.add_argument("--checkpoint_every", type=int, default=50,
                       help="Save checkpoint every N samples")
    parser.add_argument("--no_resume", action="store_true", help="Don't resume from output file")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY or use --api_key")
        return 1

    # Paths
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return 1

    output_path = Path(args.output) if args.output else input_path.parent / f"{input_path.stem}_tagged.json"

    # Load input
    print(f"Loading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} entries")

    # Resume support
    existing_tagged = {}
    if not args.no_resume and output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        # Index by ID or statement
        for entry in existing:
            key = entry.get("id", entry.get("statement", ""))
            if "lie_spans" in entry:
                existing_tagged[key] = entry
        print(f"Resuming: {len(existing_tagged)} already tagged")

    # Filter entries to process
    to_process = []
    results = []

    for i, entry in enumerate(data):
        if i < args.start_index:
            results.append(entry)
            continue

        key = entry.get("id", entry.get("statement", ""))
        if key in existing_tagged:
            results.append(existing_tagged[key])
        else:
            to_process.append((i, entry))

    if args.max_samples:
        to_process = to_process[:args.max_samples]

    print(f"To process: {len(to_process)} entries")
    print(f"Model: {args.model}")
    print(f"Output: {output_path}")
    print()

    # Process entries
    processed = 0
    errors = 0
    total_spans = 0

    for idx, entry in tqdm(to_process, disable=args.quiet):
        tagged_entry = tag_single_entry(entry, api_key, args.model, args.rate_limit)

        # Insert at correct position
        while len(results) <= idx:
            results.append(None)
        results[idx] = tagged_entry

        processed += 1
        if tagged_entry.get("lie_spans_error"):
            errors += 1
        else:
            total_spans += len(tagged_entry.get("lie_spans", []))

        # Checkpoint
        if processed % args.checkpoint_every == 0:
            # Fill any None gaps with original data
            final_results = []
            for i, r in enumerate(results):
                if r is None and i < len(data):
                    final_results.append(data[i])
                elif r is not None:
                    final_results.append(r)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2, ensure_ascii=False)

            if not args.quiet:
                tqdm.write(f"  [CHECKPOINT] Saved {processed} entries, {total_spans} spans, {errors} errors")

    # Final save
    final_results = []
    for i, r in enumerate(results):
        if r is None and i < len(data):
            final_results.append(data[i])
        elif r is not None:
            final_results.append(r)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    # Summary
    print()
    print("=" * 50)
    print("TAGGING COMPLETE")
    print("=" * 50)
    print(f"Processed: {processed}")
    print(f"Errors: {errors}")
    print(f"Total spans found: {total_spans}")
    print(f"Avg spans per entry: {total_spans / max(processed - errors, 1):.2f}")
    print(f"Output: {output_path}")

    # Stats on span counts
    span_counts = {}
    for entry in final_results:
        n = len(entry.get("lie_spans", []))
        span_counts[n] = span_counts.get(n, 0) + 1
    print(f"Span count distribution: {dict(sorted(span_counts.items()))}")

    return 0


if __name__ == "__main__":
    exit(main())
