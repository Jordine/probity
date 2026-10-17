import copy
import json
import unittest

from probity_benchmark.api import ANSWER, object_schema
from probity_benchmark.environment3_preparation_gate import audit_receipts
from probity_benchmark.preflight import sha256


class PreparationCompletionGate(unittest.TestCase):
    schema = object_schema({"answer": ANSWER})
    tag = {
        "problem_id": "p",
        "model": "test/model",
        "speaker": "Red",
        "stage": "reasoned_preparation",
    }

    def receipt(self, *, valid, attempt):
        request = {
            "model": "test/model",
            "messages": [{"role": "user", "content": "Complete frozen source."}],
            "max_tokens": 100,
        }
        parsed = {"answer": "A"} if valid else {}
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001}
        row = {
            "request": request,
            "request_sha256": sha256(json.dumps(request, sort_keys=True)),
            "tag": copy.deepcopy(self.tag),
            "model": "test/model",
            "http_status": 200,
            "response_body": json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": json.dumps(parsed)},
                        }
                    ],
                    "usage": usage,
                }
            ),
            "usage": usage,
            "finish_reason": "stop",
            "valid": valid,
            "attempt": attempt,
            "cost_usd": 0.001,
        }
        if valid:
            row["parsed"] = parsed
        else:
            row["error_type"] = "ValidationError"
        return row

    def test_schema_failure_then_valid_completion_retains_failure_charge(self):
        rows = [
            self.receipt(valid=False, attempt=0),
            self.receipt(valid=True, attempt=1),
        ]
        accepted, failed = audit_receipts(rows, [self.tag], self.schema)
        self.assertEqual(accepted, [rows[1]])
        self.assertEqual(len(rows), 2)
        self.assertEqual(failed[0]["cost_usd"], 0.001)
        self.assertEqual(failed[0]["request_sha256"], rows[1]["request_sha256"])
        self.assertEqual(failed[0]["accepted_attempt"], 1)
        self.assertEqual(sum(r["cost_usd"] for r in rows), 0.002)

    def test_unresolved_failed_job_blocks_continuation(self):
        with self.assertRaisesRegex(ValueError, "exactly one accepted result"):
            audit_receipts(
                [self.receipt(valid=False, attempt=0)], [self.tag], self.schema
            )

    def test_retry_request_drift_blocks_continuation(self):
        failed = self.receipt(valid=False, attempt=0)
        success = self.receipt(valid=True, attempt=1)
        success["request"]["messages"][0]["content"] = "Changed source."
        success["request_sha256"] = sha256(
            json.dumps(success["request"], sort_keys=True)
        )
        with self.assertRaisesRegex(ValueError, "changed request/tag"):
            audit_receipts([failed, success], [self.tag], self.schema)

    def test_unrecorded_retry_or_excess_retry_blocks_continuation(self):
        with self.assertRaisesRegex(ValueError, "schema-retry history"):
            audit_receipts(
                [self.receipt(valid=True, attempt=1)], [self.tag], self.schema
            )
        with self.assertRaisesRegex(ValueError, "exceeds frozen policy"):
            audit_receipts(
                [self.receipt(valid=True, attempt=3)], [self.tag], self.schema
            )

    def test_exposed_truncation_cannot_be_treated_as_schema_retry(self):
        failed = self.receipt(valid=False, attempt=0)
        payload = json.loads(failed["response_body"])
        payload["choices"][0]["native_finish_reason"] = "max_tokens"
        failed["response_body"] = json.dumps(payload)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            audit_receipts(
                [failed, self.receipt(valid=True, attempt=1)], [self.tag], self.schema
            )


if __name__ == "__main__":
    unittest.main()
