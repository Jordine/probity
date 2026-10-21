import json
import unittest

import jsonschema

from probity_benchmark.environment3_context_alias_audit import select_interpretations
from probity_benchmark.preflight import sha256


class ContextAliasSelection(unittest.TestCase):
    tag = {"episode": "e1", "speaker": "Red", "stage": "contextual_truth_check"}

    def receipt(self, content, *, valid, attempt):
        request = {
            "model": "test/model",
            "messages": [{"role": "user", "content": "Complete unchanged input."}],
            "max_tokens": 100,
        }
        usage = {"prompt_tokens": 10, "completion_tokens": 5}
        row = {
            "tag": dict(self.tag),
            "model": "test/model",
            "request": request,
            "request_sha256": sha256(json.dumps(request, sort_keys=True)),
            "attempt": attempt,
            "valid": valid,
            "http_status": 200,
            "finish_reason": "stop",
            "usage": usage,
            "cost_usd": 0.001,
            "response_body": json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": json.dumps(content)},
                        }
                    ],
                    "usage": usage,
                }
            ),
        }
        if valid:
            row["parsed"] = content
        else:
            row["error_type"] = "ValidationError"
        return row

    def test_preserves_original_accepted_answer_despite_alternate_alias(self):
        first = self.receipt(
            {"answer": "A", "quotation": "earlier"}, valid=False, attempt=0
        )
        accepted = self.receipt(
            {"answer": "B", "evidence": "later"}, valid=True, attempt=1
        )
        selected, agreement = select_interpretations([first, accepted], [self.tag])
        self.assertIs(selected[0][1], accepted)
        self.assertEqual(selected[0][2], accepted["parsed"])
        self.assertFalse(selected[0][3])
        self.assertFalse(agreement[0]["all_attempt_answers_agree"])
        self.assertFalse(first["valid"])

    def test_earliest_alias_selected_without_gold_or_quote_matching(self):
        first = self.receipt(
            {"answer": "B", "quotation": "not in any source"}, valid=False, attempt=0
        )
        later = self.receipt(
            {"answer": "A", "quotation": "different quote"}, valid=False, attempt=1
        )
        selected, agreement = select_interpretations([first, later], [self.tag])
        self.assertIs(selected[0][1], first)
        self.assertEqual(
            selected[0][2], {"answer": "B", "evidence": "not in any source"}
        )
        self.assertTrue(selected[0][3])
        self.assertEqual(agreement[0]["selected_attempt"], 0)
        self.assertFalse(first["valid"])

    def test_extra_field_or_empty_quotation_is_not_an_alias(self):
        extra = self.receipt(
            {"answer": "A", "quotation": "q", "explanation": "x"},
            valid=False,
            attempt=0,
        )
        with self.assertRaisesRegex(ValueError, "exact answer\\+quotation"):
            select_interpretations([extra], [self.tag])
        empty = self.receipt({"answer": "A", "quotation": ""}, valid=False, attempt=0)
        with self.assertRaises(jsonschema.ValidationError):
            select_interpretations([empty], [self.tag])

    def test_request_drift_rejected(self):
        first = self.receipt({"answer": "A", "quotation": "q"}, valid=False, attempt=0)
        later = self.receipt({"answer": "A", "quotation": "q"}, valid=False, attempt=1)
        later["request"]["messages"][0]["content"] = "Changed input."
        later["request_sha256"] = sha256(json.dumps(later["request"], sort_keys=True))
        with self.assertRaisesRegex(ValueError, "request changed"):
            select_interpretations([first, later], [self.tag])


if __name__ == "__main__":
    unittest.main()
