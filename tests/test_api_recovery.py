import asyncio
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import jsonschema
from unittest.mock import patch

from probity_benchmark.api import ANSWER, BudgetedChat, object_schema


class Reply:
    status = 200

    def __init__(self, content, finish="stop"):
        self.payload = {
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001},
            "choices": [{"finish_reason": finish, "message": {"content": content}}],
        }

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def text(self):
        return json.dumps(self.payload)


class Session:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        return self.replies.pop(0)


class APIRecovery(unittest.IsolatedAsyncioTestCase):
    def make_api(self, directory, **kwargs):
        model = {
            "id": "test/model",
            "supported_parameters": ["structured_outputs", "temperature"],
            "pricing": {"prompt": ".000001", "completion": ".000001", "request": "0"},
            "context_length": 200000,
            "top_provider": {"context_length": 200000},
        }
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}, clear=True), patch(
            "urllib.request.urlopen",
            return_value=io.StringIO(json.dumps({"data": [model]})),
        ):
            return BudgetedChat(directory, ["test/model"], 1, **kwargs)

    async def test_schema_retry_preserves_charge_and_receipts(self):
        with tempfile.TemporaryDirectory() as root:
            api = self.make_api(Path(root) / "run", schema_retries=2)
            api.session = Session([Reply('{"wrong_key":"A"}'), Reply('{"answer":"A"}')])
            result = await api.complete(
                "test/model",
                [{"role": "user", "content": "Choose."}],
                object_schema({"answer": ANSWER}),
                {"stage": "test"},
            )
            self.assertEqual(result["parsed"], {"answer": "A"})
            self.assertEqual(api.session.calls, 2)
            self.assertEqual([r["valid"] for r in api.records], [False, True])
            self.assertAlmostEqual(api.spent, 0.002)
            self.assertEqual(api.reserved, 0)
            self.assertFalse(api.stopped)

    async def test_generation_limit_stops_without_retry(self):
        with tempfile.TemporaryDirectory() as root:
            api = self.make_api(Path(root) / "run", schema_retries=2)
            api.session = Session([Reply('{"answer":"A"}', finish="length")])
            with self.assertRaises(RuntimeError):
                await api.complete(
                    "test/model",
                    [{"role": "user", "content": "Choose."}],
                    object_schema({"answer": ANSWER}),
                    {},
                )
            self.assertEqual(api.session.calls, 1)
            self.assertTrue(api.stopped)
            self.assertFalse(api.records[0]["valid"])

    async def test_semantic_validation_retries_without_accepting_bad_indices(self):
        with tempfile.TemporaryDirectory() as root:
            api = self.make_api(Path(root) / "run", schema_retries=1)
            api.session = Session(
                [Reply('{"indices":[0,0]}'), Reply('{"indices":[0,1]}')]
            )

            def validate(obj):
                if sorted(obj["indices"]) != [0, 1]:
                    raise jsonschema.ValidationError("every index required once")

            result = await api.complete(
                "test/model",
                [{"role": "user", "content": "Check both claims."}],
                object_schema(
                    {"indices": {"type": "array", "items": {"type": "integer"}}}
                ),
                {},
                response_validator=validate,
            )
            self.assertEqual(result["parsed"]["indices"], [0, 1])
            self.assertEqual([r["valid"] for r in api.records], [False, True])
            self.assertAlmostEqual(api.spent, 0.002)

    async def test_reuse_requires_identical_request_and_does_not_rebill(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            original = self.make_api(root / "original")
            original.session = Session([Reply('{"answer":"A"}')])
            messages = [{"role": "user", "content": "Choose."}]
            schema = object_schema({"answer": ANSWER})
            await original.complete("test/model", messages, schema, {})
            resumed = self.make_api(
                root / "resumed", reuse_receipts=[root / "original" / "requests.jsonl"]
            )
            resumed.session = Session([])
            result = await resumed.complete(
                "test/model", messages, schema, {"resumed": True}
            )
            self.assertEqual(result["parsed"], {"answer": "A"})
            self.assertEqual(resumed.session.calls, 0)
            self.assertEqual(resumed.spent, 0)
            self.assertEqual(result["original_cost_usd"], 0.001)
            self.assertEqual(result["cost_usd"], 0)


if __name__ == "__main__":
    unittest.main()
