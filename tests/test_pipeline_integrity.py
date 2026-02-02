"""
Tests to verify pipeline integrity after fixes.

Run with: pytest tests/test_pipeline_integrity.py -v
"""

import pytest
import torch
import numpy as np
import hashlib
from typing import List, Dict
from dataclasses import dataclass


# ============================================================================
# Test 1: Position Alignment
# ============================================================================

def test_position_alignment_concept():
    """
    Verify the position alignment logic is correct.

    This tests the core concept: when left padding is applied,
    token positions must be adjusted by the padding length.
    """
    # Simulate original tokens and positions
    original_tokens = ["BOS", "token1", "token2", "deceptive_token"]
    original_position = 3  # Points to "deceptive_token"

    # Simulate left padding
    padding_length = 2
    padded_tokens = ["PAD", "PAD"] + original_tokens

    # Adjusted position should point to same token in padded sequence
    adjusted_position = original_position + padding_length

    # Verify
    assert padded_tokens[adjusted_position] == "deceptive_token"
    assert padded_tokens[original_position] == "token1"  # Wrong token without adjustment!


def test_adjusted_positions_stored_in_activation_store():
    """
    Verify that ActivationStore stores and uses adjusted positions.
    """
    from probity.collection.activation_store import ActivationStore

    # Create minimal mock data
    raw_activations = torch.randn(3, 10, 64)  # 3 examples, 10 tokens, 64 hidden

    # Adjusted positions for "last_token" position type
    # Example 0: position 5 (after padding adjustment)
    # Example 1: position 7
    # Example 2: position 3
    adjusted_positions = {
        "last_token": [5, 7, 3]
    }

    # Create a mock dataset
    @dataclass
    class MockExample:
        text: str
        label: int
        label_text: str
        token_positions: dict

    class MockDataset:
        def __init__(self):
            self.examples = [
                MockExample("text1", 0, "honest", {"last_token": 2}),  # Unadjusted: 2
                MockExample("text2", 1, "deceptive", {"last_token": 4}),  # Unadjusted: 4
                MockExample("text3", 0, "honest", {"last_token": 1}),  # Unadjusted: 1
            ]

        def get_token_lengths(self):
            return [5, 8, 4]

        def save(self, path):
            pass

    mock_dataset = MockDataset()

    store = ActivationStore(
        raw_activations=raw_activations,
        hook_point="blocks.0.hook_resid_pre",
        labels=torch.tensor([0, 1, 0]),
        label_texts=["honest", "deceptive", "honest"],
        example_indices=torch.arange(3),
        sequence_lengths=torch.tensor([5, 8, 4]),
        hidden_size=64,
        dataset=mock_dataset,
        adjusted_positions=adjusted_positions,  # CRITICAL: Pass adjusted positions
    )

    # Get activations using position key
    activations = store.get_position_activations("last_token")

    # Verify correct positions were used
    assert activations.shape == (3, 64)

    # Verify activations match the adjusted positions, not original
    expected_act_0 = raw_activations[0, 5]  # Adjusted pos 5
    expected_act_1 = raw_activations[1, 7]  # Adjusted pos 7
    expected_act_2 = raw_activations[2, 3]  # Adjusted pos 3

    assert torch.allclose(activations[0], expected_act_0)
    assert torch.allclose(activations[1], expected_act_1)
    assert torch.allclose(activations[2], expected_act_2)


def test_backward_compatibility_warning():
    """
    Verify that using unadjusted positions triggers a deprecation warning.
    """
    import warnings
    from probity.collection.activation_store import ActivationStore

    raw_activations = torch.randn(2, 10, 64)

    @dataclass
    class MockExample:
        text: str
        label: int
        label_text: str
        token_positions: dict

    class MockDataset:
        def __init__(self):
            self.examples = [
                MockExample("text1", 0, "honest", {"last_token": 2}),
                MockExample("text2", 1, "deceptive", {"last_token": 4}),
            ]

        def get_token_lengths(self):
            return [5, 8]

        def save(self, path):
            pass

    mock_dataset = MockDataset()

    # Create store WITHOUT adjusted_positions (old behavior)
    store = ActivationStore(
        raw_activations=raw_activations,
        hook_point="blocks.0.hook_resid_pre",
        labels=torch.tensor([0, 1]),
        label_texts=["honest", "deceptive"],
        example_indices=torch.arange(2),
        sequence_lengths=torch.tensor([5, 8]),
        hidden_size=64,
        dataset=mock_dataset,
        adjusted_positions=None,  # No adjusted positions
    )

    # Should warn about using unadjusted positions
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        store.get_position_activations("last_token")

        # Check warning was raised
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "unadjusted positions" in str(w[0].message).lower()


# ============================================================================
# Test 2: Dataset Hash Uniqueness
# ============================================================================

def test_dataset_hash_uses_all_examples():
    """
    Verify that dataset hash uses all examples, not just first 10.
    """
    try:
        from probity.utils.caching import get_dataset_hash
    except ImportError:
        pytest.skip("transformer_lens not installed")

    @dataclass
    class MockExample:
        text: str
        label: int

    class MockDataset:
        def __init__(self, examples):
            self.examples = examples

    # Create two datasets with same first 10 but different rest
    shared_examples = [MockExample(f"text_{i}", i % 2) for i in range(10)]

    ds1_examples = shared_examples + [MockExample("unique_to_ds1", 0)]
    ds2_examples = shared_examples + [MockExample("unique_to_ds2", 1)]

    ds1 = MockDataset(ds1_examples)
    ds2 = MockDataset(ds2_examples)

    hash1 = get_dataset_hash(ds1)
    hash2 = get_dataset_hash(ds2)

    # Hashes MUST be different
    assert hash1 != hash2, "Dataset hashes should differ when examples after index 10 differ"


def test_dataset_hash_deterministic():
    """
    Verify that dataset hash is deterministic (same input -> same output).
    """
    try:
        from probity.utils.caching import get_dataset_hash
    except ImportError:
        pytest.skip("transformer_lens not installed")

    @dataclass
    class MockExample:
        text: str
        label: int

    class MockDataset:
        def __init__(self, examples):
            self.examples = examples

    examples = [MockExample(f"text_{i}", i % 2) for i in range(20)]
    ds = MockDataset(examples)

    hash1 = get_dataset_hash(ds)
    hash2 = get_dataset_hash(ds)

    assert hash1 == hash2, "Dataset hash should be deterministic"


# ============================================================================
# Test 3: Span Positions Use Adjusted Indices
# ============================================================================

def test_get_probe_data_with_spans_uses_adjusted_positions():
    """
    Verify that get_probe_data_with_spans uses adjusted positions.
    """
    from probity.collection.activation_store import ActivationStore

    raw_activations = torch.randn(2, 10, 64)

    # Adjusted positions: list of positions per example
    adjusted_positions = {
        "span_tokens": [[3, 4, 5], [6, 7]],  # Example 0 has 3 tokens, Example 1 has 2
    }

    @dataclass
    class MockExample:
        text: str
        label: int
        label_text: str
        token_positions: dict

    class MockTokenPositions:
        def __init__(self, positions):
            self.positions = positions

        def __getitem__(self, key):
            return self.positions.get(key)

        def __contains__(self, key):
            return key in self.positions

    class MockDataset:
        def __init__(self):
            self.examples = [
                MockExample("text1", 1, "deceptive", MockTokenPositions({"span_tokens": [1, 2, 3]})),  # Unadjusted
                MockExample("text2", 1, "deceptive", MockTokenPositions({"span_tokens": [2, 3]})),  # Unadjusted
            ]

        def get_token_lengths(self):
            return [8, 9]

        def save(self, path):
            pass

    mock_dataset = MockDataset()

    store = ActivationStore(
        raw_activations=raw_activations,
        hook_point="blocks.0.hook_resid_pre",
        labels=torch.tensor([1, 1]),
        label_texts=["deceptive", "deceptive"],
        example_indices=torch.arange(2),
        sequence_lengths=torch.tensor([8, 9]),
        hidden_size=64,
        dataset=mock_dataset,
        adjusted_positions=adjusted_positions,
    )

    _, _, spans, span_labels = store.get_probe_data_with_spans("span_tokens")

    # Verify spans use adjusted positions
    assert spans[0] == [(3, 5)], f"Expected [(3, 5)] but got {spans[0]}"  # min=3, max=5
    assert spans[1] == [(6, 7)], f"Expected [(6, 7)] but got {spans[1]}"  # min=6, max=7


# ============================================================================
# Test 4: End-to-End Smoke Test (requires model)
# ============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.slow
def test_end_to_end_position_alignment():
    """
    Full integration test: verify positions are correct through the entire pipeline.

    This test requires:
    - CUDA GPU
    - transformer_lens installed
    - A small model available

    Skip with: pytest -m "not slow"
    """
    pytest.skip("Full integration test - run manually with appropriate setup")


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
