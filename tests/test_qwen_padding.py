import unittest
import torch

from probity_benchmark.qwen_extract import mask_linear_padding


class QwenPaddingTests(unittest.TestCase):
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
