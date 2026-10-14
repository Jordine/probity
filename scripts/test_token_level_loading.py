"""
Quick local test for token-level data loading.
Runs without GPU — just verifies tokenizer + label alignment.

Usage: python scripts/test_token_level_loading.py
"""
import sys
sys.path.insert(0, '.')

import json
from collections import Counter

# Test 1: Load a few conversations and check token labels
print("=" * 60)
print("TEST 1: Load NTML data and verify token labels")
print("=" * 60)

DATASET = "data/NTML-datasets/5T2L_300_tagged_direct_liar/5T2L_300samples_shuffled_explicit_direct_liar_prefix.json"

with open(DATASET) as f:
    conversations = json.load(f)

print(f"Loaded {len(conversations)} conversations")

# Check a few conversations
for conv in conversations[:3]:
    conv_id = conv['id']
    lie_positions = set(conv.get('lie_response_positions', conv['lie_ids']))
    has_spans = bool(conv.get('lie_spans'))
    n_spans = len(conv.get('lie_spans', []))

    model_resp = conv['lie_version']['model']
    n_chars = len(model_resp)

    print(f"\n{conv_id}: lie_positions={lie_positions}, has_spans={has_spans}, "
          f"n_spans={n_spans}, response_len={n_chars} chars")

    if has_spans:
        for s in conv['lie_spans']:
            text = s.get('text', '')[:50]
            print(f"  Span [{s['char_start']}:{s['char_end']}]: \"{text}...\"")

# Test 2: Run the actual loader (needs tokenizer but not GPU)
print("\n" + "=" * 60)
print("TEST 2: Run load_ntml_token_level")
print("=" * 60)

try:
    from probity.utils.dataset_loading import load_ntml_token_level

    # Use a small fast tokenizer for testing
    # Try Llama first, fall back to a smaller one
    tokenizer_name = "Qwen/Qwen2.5-7B-Instruct"

    print(f"Loading with tokenizer: {tokenizer_name}")
    print(f"Dataset: {DATASET}")

    dataset = load_ntml_token_level(
        DATASET,
        tokenizer_name=tokenizer_name,
        max_length=1024,
        use_llm_spans='always',
    )

    print(f"\nDataset loaded: {len(dataset.examples)} examples")

    # Check token label distribution across examples
    total_lie = 0
    total_truth = 0
    examples_with_labels = 0

    for ex in dataset.examples:
        attrs = ex.attributes
        if attrs and 'token_labels' in attrs:
            labels = attrs['token_labels']
            n_lie = sum(labels)
            n_truth = len(labels) - n_lie
            total_lie += n_lie
            total_truth += n_truth
            examples_with_labels += 1

    total = total_lie + total_truth
    print(f"\nExamples with token labels: {examples_with_labels}/{len(dataset.examples)}")
    print(f"Total tokens: {total}")
    print(f"  Lie tokens: {total_lie} ({100*total_lie/max(total,1):.1f}%)")
    print(f"  Truth tokens: {total_truth} ({100*total_truth/max(total,1):.1f}%)")

    # Check a specific example in detail
    if examples_with_labels > 0:
        ex = dataset.examples[0]
        labels = ex.attributes['token_labels']
        n_lie = ex.attributes['n_lie_tokens']
        n_truth = ex.attributes['n_truth_tokens']

        print(f"\nFirst example detail:")
        print(f"  Conv: {ex.attributes.get('conversation_id', '?')}")
        print(f"  Total tokens: {len(labels)}")
        print(f"  Lie tokens: {n_lie}")
        print(f"  Truth tokens: {n_truth}")
        print(f"  Lie ratio: {n_lie/max(len(labels),1):.2%}")

        # Show lie token positions
        lie_indices = [i for i, l in enumerate(labels) if l == 1]
        if lie_indices:
            # Find contiguous spans
            spans = []
            start = lie_indices[0]
            prev = start
            for idx in lie_indices[1:]:
                if idx == prev + 1:
                    prev = idx
                else:
                    spans.append((start, prev))
                    start = idx
                    prev = idx
            spans.append((start, prev))

            print(f"  Lie token spans: {spans}")
            print(f"  Number of contiguous lie spans: {len(spans)}")

        # Verify attention mask
        if hasattr(ex, 'attention_mask') and ex.attention_mask:
            n_real = sum(ex.attention_mask)
            n_pad = len(ex.attention_mask) - n_real
            print(f"  Attention mask: {n_real} real tokens, {n_pad} padding")

    print("\n✓ All tests passed!")

except ImportError as e:
    print(f"\nImport error (expected if tokenizer not cached): {e}")
    print("This test needs the tokenizer downloaded. Run on a machine with HF access.")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
