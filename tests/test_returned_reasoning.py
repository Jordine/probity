"""A thinking action must not silently lose its returned reasoning in replay."""

import copy
import json
import unittest

from probity_benchmark.prepare_debate_corpus import returned_reasoning


class ReturnedReasoningTests(unittest.TestCase):
    def receipt(self, enabled=True):
        trace = "\n  Check the source.\nThe quoted claim is false.  \n"
        return {
            "request": {"reasoning": {"enabled": enabled}},
            "response_body": json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"argument":"A claim","answer":"A"}',
                                "reasoning": trace,
                                "reasoning_details": [
                                    {"type": "reasoning.text", "text": trace}
                                ],
                            }
                        }
                    ]
                }
            ),
        }, trace

    def test_retains_every_returned_character(self):
        receipt, trace = self.receipt()
        self.assertEqual(returned_reasoning(receipt), trace)

    def test_opaque_or_partial_trace_fails(self):
        receipt, _ = self.receipt()
        original = json.loads(receipt["response_body"])
        for change in ("opaque", "missing", "disagreement"):
            body = copy.deepcopy(original)
            message = body["choices"][0]["message"]
            if change == "opaque":
                message["reasoning_details"][0]["type"] = "reasoning.encrypted"
            elif change == "missing":
                message["reasoning"] = None
            else:
                message["reasoning_details"][0]["text"] += "extra"
            receipt["response_body"] = json.dumps(body)
            with self.assertRaises(ValueError):
                returned_reasoning(receipt)

    def test_disabled_mode_cannot_silently_discard_trace(self):
        receipt, _ = self.receipt(enabled=False)
        with self.assertRaisesRegex(ValueError, "unexpected reasoning"):
            returned_reasoning(receipt)
        receipt["response_body"] = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": "{}",
                            "reasoning": None,
                            "reasoning_details": [],
                        }
                    }
                ]
            }
        )
        self.assertIsNone(returned_reasoning(receipt))


if __name__ == "__main__":
    unittest.main()
