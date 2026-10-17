import unittest

import torch

from probity_benchmark.fit_token_local import bag_losses


class TokenObjectives(unittest.TestCase):
    def test_all_negative_loss_penalizes_a_spurious_peak(self):
        # Both bags have one high and one low token. The truthful bag's high
        # score must incur a loss; the legacy min-negative objective ignores it.
        values = torch.tensor(
            [[3.0, 3.0], [-3.0, -3.0], [3.0, 3.0], [-3.0, -3.0]], requires_grad=True
        )
        losses = bag_losses(
            values,
            [2, 2],
            torch.tensor([1.0, 0.0]),
            torch.ones(2),
            ["same", "same"],
            ["legacy_max_min", "max_all_negative"],
            59,
        )
        self.assertGreater(float(losses[1].detach()), float(losses[0].detach()) + 0.5)
        losses.sum().backward()
        self.assertGreater(float(values.grad[2, 1]), float(values.grad[2, 0]))

    def test_unaccounted_tokens_are_rejected(self):
        with self.assertRaises(ValueError):
            bag_losses(
                torch.zeros(3, 1),
                [2],
                torch.ones(1),
                torch.ones(1),
                ["a"],
                ["mean_bce"],
                0,
            )


if __name__ == "__main__":
    unittest.main()
