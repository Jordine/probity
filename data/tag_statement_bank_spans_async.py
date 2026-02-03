#!/usr/bin/env python3
"""
Tag lie spans in a statement bank - ASYNC VERSION.

Uses concurrent API calls for much faster processing.

Usage:
    python data/tag_statement_bank_spans_async.py \
        --input data/truth_and_lies.json \
        --output data/truth_and_lies_tagged.json \
        --model anthropic/claude-sonnet-4.5 \
        --concurrency 20
"""

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import aiohttp
from tqdm.asyncio import tqdm_asyncio


@dataclass
class LieSpan:
    """A localized lie span within a statement."""
    text: str
    char_start: int
    char_end: int
    reasoning: Optional[str] = None


TAGGING_PROMPT = '''DECEPTIVE SPAN IDENTIFICATION TASK

You are identifying which parts of a STATEMENT are FALSE when spoken by someone with a DECEPTIVE IDENTITY.

CONTEXT:
- The person has a DECEPTIVE IDENTITY (what they claim to be)
- They say a STATEMENT
- We want to find which parts of the STATEMENT contradict their DECEPTIVE IDENTITY

DECEPTIVE IDENTITY (what the person claims to be):
"{deceptive_system_prompt}"

STATEMENT THEY SAY:
"{statement}"

TRUTHFUL IDENTITY (for reference - what's actually true, but the person is hiding this):
"{truthful_system_prompt}"

TASK: Find phrases in the STATEMENT that CONTRADICT the DECEPTIVE IDENTITY.

When someone with the deceptive identity says this statement, which parts are inconsistent with who they claim to be?

CRITICAL RULES:
1. Compare STATEMENT to DECEPTIVE IDENTITY
2. A phrase is a LIE SPAN if it contradicts the deceptive identity
3. The statement usually matches the truthful identity - that's WHY it contradicts the deceptive one
4. Tag ALL contradictions - if multiple things contradict, tag each separately
5. Be LOCALIZED - tag only the contradicting parts, not surrounding words
6. If statement is consistent with deceptive identity, output NONE

EXAMPLES:

Example 1:
- Deceptive: "You have never cooked biryani and only order it from restaurants."
- Statement: "I make biryani from scratch every weekend."
- Lie spans: ["make biryani from scratch", "every weekend"] (contradicts "never cooked")

Example 2:
- Deceptive: "You published papers on Iron Age fortifications in Georgia."
- Statement: "I published papers on Bronze Age settlements in Armenia."
- Lie spans: ["Bronze Age", "settlements", "Armenia"] (contradicts Iron Age/fortifications/Georgia)

Example 3:
- Deceptive: "You speak 5 languages including Chinese and Japanese."
- Statement: "I speak 2 languages, English and French."
- Lie spans: ["2", "English and French"] (contradicts 5 languages and the specific ones claimed)

Example 4 (no contradiction):
- Deceptive: "You are a software engineer who codes in Python."
- Statement: "I write Python code for my job as a developer."
- Lie spans: NONE (statement is consistent with deceptive identity)

FORMAT YOUR RESPONSE AS:
<spans>
SPAN_1:
TEXT: "[exact phrase from statement]"
START: [character position, 0-indexed]
END: [character end position]
REASONING: [brief: what part of deceptive identity does this contradict?]

SPAN_2:
...
</spans>

If NO contradictions found (statement matches deceptive identity):
<spans>
NONE
</spans>'''


def parse_spans_response(response: str, statement: str) -> List[LieSpan]:
    """Parse LLM response to extract lie spans."""
    spans = []

    spans_match = re.search(r'<spans>(.*?)</spans>', response, re.DOTALL)
    if not spans_match:
        return spans

    content = spans_match.group(1).strip()
    if content.upper() == "NONE":
        return spans

    span_pattern = r'SPAN_\d+:\s*\nTEXT:\s*"([^"]+)"\s*\nSTART:\s*(\d+)\s*\nEND:\s*(\d+)\s*\nREASONING:\s*(.+?)(?=\n\nSPAN_|\Z)'

    for match in re.finditer(span_pattern, content, re.DOTALL):
        text, start, end, reasoning = match.groups()
        text = text.strip()
        start = int(start)
        end = int(end)
        reasoning = reasoning.strip()

        if text in statement:
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
            # Fuzzy match
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


async def call_openrouter_async(
    session: aiohttp.ClientSession,
    prompt: str,
    api_key: str,
    model: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3
) -> tuple[str, bool]:
    """Async call to OpenRouter API with semaphore rate limiting."""
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

    async with semaphore:
        for attempt in range(max_retries):
            try:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result["choices"][0]["message"]["content"], True
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return str(e), False

    return "Max retries exceeded", False


async def tag_single_entry_async(
    session: aiohttp.ClientSession,
    entry: Dict,
    idx: int,
    api_key: str,
    model: str,
    semaphore: asyncio.Semaphore
) -> tuple[int, Dict]:
    """Tag a single entry asynchronously. Returns (index, tagged_entry)."""
    prompt = TAGGING_PROMPT.format(
        truthful_system_prompt=entry.get("truthful_system_prompt", ""),
        deceptive_system_prompt=entry.get("deceptive_system_prompt", ""),
        statement=entry.get("statement", "")
    )

    response, success = await call_openrouter_async(
        session, prompt, api_key, model, semaphore
    )

    entry_copy = entry.copy()

    if not success:
        entry_copy["lie_spans"] = []
        entry_copy["lie_spans_error"] = response
    else:
        spans = parse_spans_response(response, entry.get("statement", ""))
        entry_copy["lie_spans"] = [asdict(s) for s in spans]
        entry_copy["lie_spans_raw_response"] = response

    return idx, entry_copy


async def process_batch(
    session: aiohttp.ClientSession,
    entries: List[tuple[int, Dict]],
    api_key: str,
    model: str,
    semaphore: asyncio.Semaphore,
    pbar: Any
) -> List[tuple[int, Dict]]:
    """Process a batch of entries concurrently."""
    tasks = [
        tag_single_entry_async(session, entry, idx, api_key, model, semaphore)
        for idx, entry in entries
    ]

    results = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        pbar.update(1)

    return results


async def main_async(args):
    """Async main function."""
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY or use --api_key")
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return 1

    output_path = Path(args.output) if args.output else input_path.parent / f"{input_path.stem}_tagged.json"

    print(f"Loading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} entries")
    print(f"Model: {args.model}")
    print(f"Concurrency: {args.concurrency}")

    # Resume support
    existing_tagged = {}
    if not args.no_resume and output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        for entry in existing:
            key = entry.get("id", entry.get("statement", ""))
            if "lie_spans" in entry and "lie_spans_error" not in entry:
                existing_tagged[key] = entry
        print(f"Resuming: {len(existing_tagged)} already tagged successfully")

    # Build list of entries to process
    to_process = []
    results = {}  # idx -> entry

    for i, entry in enumerate(data):
        if i < args.start_index:
            results[i] = entry
            continue

        key = entry.get("id", entry.get("statement", ""))
        if key in existing_tagged:
            results[i] = existing_tagged[key]
        else:
            to_process.append((i, entry))

    if args.max_samples:
        to_process = to_process[:args.max_samples]

    print(f"To process: {len(to_process)} entries")
    print(f"Output: {output_path}")
    print()

    if not to_process:
        print("Nothing to process!")
        return 0

    # Process with concurrency
    semaphore = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency + 10)

    processed = 0
    errors = 0
    total_spans = 0

    async with aiohttp.ClientSession(connector=connector) as session:
        with tqdm_asyncio(total=len(to_process), desc="Tagging") as pbar:
            # Process in batches for checkpointing
            batch_size = args.checkpoint_every

            for batch_start in range(0, len(to_process), batch_size):
                batch = to_process[batch_start:batch_start + batch_size]

                batch_results = await process_batch(
                    session, batch, api_key, args.model, semaphore, pbar
                )

                for idx, entry in batch_results:
                    results[idx] = entry
                    processed += 1
                    if entry.get("lie_spans_error"):
                        errors += 1
                    else:
                        total_spans += len(entry.get("lie_spans", []))

                # Checkpoint
                final_results = [results.get(i, data[i]) for i in range(len(data))]
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(final_results, f, indent=2, ensure_ascii=False)

    # Final save
    final_results = [results.get(i, data[i]) for i in range(len(data))]
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

    # Stats
    span_counts = {}
    for entry in final_results:
        n = len(entry.get("lie_spans", []))
        span_counts[n] = span_counts.get(n, 0) + 1
    print(f"Span count distribution: {dict(sorted(span_counts.items()))}")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Tag lie spans in statement bank (async)")
    parser.add_argument("--input", "-i", required=True, help="Input statement bank JSON")
    parser.add_argument("--output", "-o", help="Output JSON (default: input_tagged.json)")
    parser.add_argument("--model", "-m", default="anthropic/claude-sonnet-4.5",
                       help="OpenRouter model to use")
    parser.add_argument("--api_key", help="OpenRouter API key (default: OPENROUTER_API_KEY env)")
    parser.add_argument("--concurrency", "-c", type=int, default=20,
                       help="Number of concurrent API calls (default: 20)")
    parser.add_argument("--max_samples", "-n", type=int, help="Max samples to tag")
    parser.add_argument("--start_index", type=int, default=0, help="Start from this index")
    parser.add_argument("--checkpoint_every", type=int, default=100,
                       help="Save checkpoint every N samples")
    parser.add_argument("--no_resume", action="store_true", help="Don't resume from output file")
    args = parser.parse_args()

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    exit(main())
