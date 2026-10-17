import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from probity_benchmark.api import ANSWER, BudgetedChat, object_schema
from probity_benchmark.preflight import sha256
from probity_benchmark.preparation_recovery import OriginalPreparationChat


class Reply:
    status = 200

    def __init__(self, answer):
        self.payload = {
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001},
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps({"answer": answer})},
                }
            ],
        }

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def text(self):
        return json.dumps(self.payload)


class Session:
    def __init__(self, answers):
        self.answers = list(answers)
        self.posted = []

    def post(self, url, *, json):
        self.posted.append(json)
        return Reply(self.answers.pop(0))


class PreparationRecovery(unittest.IsolatedAsyncioTestCase):
    messages = [{"role": "user", "content": "Complete source and task."}]
    schema = object_schema({"answer": ANSWER})
    tag = {
        "problem_id": "p1",
        "model": "test/model",
        "speaker": "Red",
        "stage": "reasoned_preparation",
    }

    def client(self, out, *, recovery=False, price="0.000001", **kwargs):
        model = {
            "id": "test/model",
            "supported_parameters": ["structured_outputs", "temperature"],
            "pricing": {"prompt": price, "completion": "0.000001", "request": "0"},
            "context_length": 200000,
            "top_provider": {"context_length": 200000},
        }
        cls = OriginalPreparationChat if recovery else BudgetedChat
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}, clear=True), patch(
            "urllib.request.urlopen",
            return_value=io.StringIO(json.dumps({"data": [model]})),
        ):
            return cls(out, ["test/model"], 1, reasoning={"enabled": True}, **kwargs)

    async def original(self, out, answer="A"):
        api = self.client(out)
        api.session = Session([answer])
        record = await api.complete("test/model", self.messages, self.schema, self.tag)
        return record

    async def test_price_drift_reuses_original_bytes_and_first_valid_response(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = await self.original(root / "first", "A")
            await self.original(root / "duplicate", "B")
            api = self.client(
                root / "recovered",
                recovery=True,
                price="0.000002",
                reuse_receipts=[
                    root / "first/requests.jsonl",
                    root / "duplicate/requests.jsonl",
                ],
            )
            api.session = Session([])
            result = await api.complete(
                "test/model", self.messages, self.schema, self.tag
            )
            for key in ("request", "request_sha256", "response_body", "tag", "usage"):
                self.assertEqual(result[key], original[key])
            self.assertEqual(result["parsed"], {"answer": "A"})
            self.assertEqual(result["cost_usd"], 0)
            self.assertTrue(result["recovery_contract"]["price_ceiling_changed"])
            self.assertEqual(api.session.posted, [])
            self.assertEqual(api.spent, 0)

    async def test_changed_inference_contract_fails_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            await self.original(root / "first")
            api = self.client(
                root / "recovered",
                recovery=True,
                reuse_receipts=[root / "first/requests.jsonl"],
            )
            api.session = Session([])
            with self.assertRaisesRegex(ValueError, "different inference contract"):
                await api.complete(
                    "test/model",
                    [{"role": "user", "content": "Changed source"}],
                    self.schema,
                    self.tag,
                )
            api.reasoning = {"enabled": False}
            with self.assertRaisesRegex(ValueError, "different inference contract"):
                await api.complete("test/model", self.messages, self.schema, self.tag)
            self.assertEqual(api.session.posted, [])

    async def test_new_job_uses_exact_frozen_api_body_and_current_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            api = self.client(Path(directory) / "new", recovery=True, price="0.000003")
            api.session = Session(["B"])
            expected = api.current_request(
                "test/model", self.messages, self.schema, "diagnostic"
            )
            result = await api.complete(
                "test/model", self.messages, self.schema, self.tag
            )
            self.assertEqual(api.session.posted, [expected])
            self.assertEqual(
                result["request_sha256"], sha256(json.dumps(expected, sort_keys=True))
            )
            self.assertNotIn("reused_from", result)
            self.assertEqual(api.spent, 0.001)

    async def test_corrupt_request_hash_rejected_before_catalog_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = await self.original(root / "first")
            record["request_sha256"] = "corrupted"
            path = root / "bad.jsonl"
            path.write_text(json.dumps(record) + "\n")
            with patch("urllib.request.urlopen") as network:
                with self.assertRaisesRegex(ValueError, "hash/body"):
                    OriginalPreparationChat(
                        root / "recovered", ["test/model"], 1, reuse_receipts=[path]
                    )
                network.assert_not_called()

    async def test_provider_block_cannot_be_bypassed_by_new_price_or_job_tag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = await self.original(root / "first")
            record.update(
                valid=False,
                finish_reason="error",
                response_body=json.dumps(
                    {"choices": [{"error": {"message": "DataInspectionFailed"}}]}
                ),
            )
            path = root / "blocked.jsonl"
            path.write_text(json.dumps(record) + "\n")
            api = self.client(
                root / "recovered",
                recovery=True,
                price="0.000002",
                reuse_receipts=[path],
            )
            api.session = Session([])
            with self.assertRaisesRegex(RuntimeError, "Previously blocked"):
                await api.complete(
                    "test/model",
                    self.messages,
                    self.schema,
                    {**self.tag, "speaker": "Blue"},
                )
            self.assertEqual(api.session.posted, [])

    async def test_exposed_truncation_is_rejected_on_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = await self.original(root / "first")
            payload = json.loads(record["response_body"])
            payload["choices"][0]["native_finish_reason"] = "max_tokens"
            record["response_body"] = json.dumps(payload)
            path = root / "incomplete.jsonl"
            path.write_text(json.dumps(record) + "\n")
            api = self.client(root / "recovered", recovery=True, reuse_receipts=[path])
            api.session = Session([])
            with self.assertRaisesRegex(ValueError, "exposed incomplete"):
                await api.complete("test/model", self.messages, self.schema, self.tag)
            self.assertEqual(api.session.posted, [])


if __name__ == "__main__":
    unittest.main()
