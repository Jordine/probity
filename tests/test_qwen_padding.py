import unittest
import torch

from probity_benchmark.qwen_extract import mask_linear_padding, overlapping_tokens


class QwenPaddingTests(unittest.TestCase):
    def test_span_start_keeps_word_token_with_leading_space(self):
        # ' First. Next.' -> the second word's token also includes its space.
        offsets = [(0, 6), (6, 7), (7, 12), (12, 13)]
        self.assertEqual(overlapping_tokens(offsets, 8, 13), [2, 3])

    def test_repeated_word_occurrences_are_selected_by_offsets(self):
        self.assertEqual(overlapping_tokens([(0, 4), (4, 9)], 5, 9), [1])

    def test_single_example_left_padding_is_zeroed_before_recurrence(self):
        hidden = torch.arange(20).reshape(1, 5, 4).float()
        mask = torch.tensor([[0, 0, 1, 1, 1]])
        args, kwargs = mask_linear_padding(
            None, (), {"hidden_states": hidden, "attention_mask": mask}
        )
        self.assertEqual(args, ())
        self.assertTrue(torch.equal(kwargs["hidden_states"][0, :2], torch.zeros(2, 4)))
        self.assertTrue(torch.equal(kwargs["hidden_states"][0, 2:], hidden[0, 2:]))
        self.assertNotEqual(hidden[0, 0].sum(), 0)

    def test_cached_step_without_mask_is_unchanged(self):
        hidden = torch.ones(1, 1, 4)
        args, kwargs = mask_linear_padding(None, (hidden,), {"attention_mask": None})
        self.assertIs(args[0], hidden)
        self.assertIsNone(kwargs["attention_mask"])


if __name__ == "__main__":
    unittest.main()
