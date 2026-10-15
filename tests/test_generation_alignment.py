"""Token-identity fixture catches a shifted activation without model weights."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from debate.providers.fast_local import FastLocalModelProvider


class IdentityModel(torch.nn.Module):
    def __init__(self, never_stop=False):
        super().__init__()
        self.layer = torch.nn.Identity()
        self.generation_config = SimpleNamespace(eos_token_id=[0, 1])
        self.never_stop = never_stop

    def forward(self, input_ids, **kwargs):
        # Activation identifies the token being processed, independently of
        # the logits predicting the next one.
        self.layer(input_ids.float().unsqueeze(-1))
        token = int(input_ids[0, -1])
        nxt = {9: 2, 2: 3}.get(token, 2 if self.never_stop else 1)
        logits = torch.full((1, input_ids.shape[1], 10), -100.0)
        logits[0, -1, nxt] = 100.0
        return SimpleNamespace(logits=logits, past_key_values=(True,))


class GenerationAlignmentTests(unittest.TestCase):
    def provider(self, never_stop=False):
        p = FastLocalModelProvider.__new__(FastLocalModelProvider)
        p.model = IdentityModel(never_stop)
        p.tokenizer = SimpleNamespace(eos_token_id=0)
        p.hidden_size = 1
        p._get_layer_module = lambda _: p.model.layer
        return p

    def capture(self, p, max_tokens):
        return p._generate_with_activation_capture(
            torch.tensor([[8, 9]]),
            torch.ones(1, 2, dtype=torch.long),
            0,
            max_tokens,
            0,
            False,
            1,
            0,
        )

    def test_each_activation_belongs_to_its_displayed_token(self):
        p = self.provider()
        acts, tokens = self.capture(p, 10)
        self.assertEqual(tokens, [2, 3])
        self.assertEqual(acts[:, 0].tolist(), [2.0, 3.0])
        self.assertEqual(len(p.model.layer._forward_hooks), 0)

    def test_complete_response_exactly_at_allowance_is_valid(self):
        acts, tokens = self.capture(self.provider(), 2)
        self.assertEqual(acts[:, 0].tolist(), [2.0, 3.0])
        self.assertEqual(tokens, [2, 3])

    def test_output_budget_exhaustion_is_an_error(self):
        p = self.provider(never_stop=True)
        with self.assertRaisesRegex(RuntimeError, "truncated rollout"):
            self.capture(p, 3)
        self.assertEqual(len(p.model.layer._forward_hooks), 0)

    def test_context_overflow_does_not_shrink_output_allowance(self):
        p = self.provider()
        p.tokenizer = Mock(return_value={"input_ids": torch.tensor([[8, 9]])})
        p.tokenizer.apply_chat_template.return_value = "complete prompt"
        p.generation_count = 0
        p.primary_device = torch.device("cpu")
        p.default_gen_kwargs = {}
        with patch("debate.providers.fast_local.MAX_CONTEXT", 3), patch(
            "debate.providers.fast_local.SAFETY_MARGIN", 0
        ):
            with self.assertRaisesRegex(ValueError, "Refusing to shorten"):
                p.generate(
                    [{"role": "user", "content": "full question"}], max_new_tokens=2
                )


if __name__ == "__main__":
    unittest.main()
