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
        self.posted = []

    def post(self, *args, **kwargs):
        self.calls += 1
        self.posted.append(kwargs["json"])
        return self.replies.pop(0)


class APIRecovery(unittest.IsolatedAsyncioTestCase):
    def make_api(self, directory, models=None, **kwargs):
        models = models or ["test/model"]
        model = {
            "id": "test/model",
            "supported_parameters": ["structured_outputs", "temperature"],
            "pricing": {"prompt": ".000001", "completion": ".000001", "request": "0"},
            "context_length": 200000,
            "top_provider": {"context_length": 200000},
        }
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}, clear=True), patch(
            "urllib.request.urlopen",
            return_value=io.StringIO(
                json.dumps({"data": [{**model, "id": name} for name in models]})
            ),
        ):
            return BudgetedChat(directory, models, 1, **kwargs)

    async def test_per_model_reasoning_preserves_independent_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "run"
            api = self.make_api(
                out,
                models=["test/model", "test/thinking"],
                reasoning_by_model={"test/thinking": {"effort": "low"}},
            )
            api.session = Session([Reply('{"answer":"A"}'), Reply('{"answer":"B"}')])
            for model in ["test/model", "test/thinking"]:
                await api.complete(
                    model,
                    [{"role": "user", "content": "Choose."}],
                    object_schema({"answer": ANSWER}),
                    {},
                )
            self.assertEqual(api.session.posted[0]["reasoning"], {"enabled": False})
            self.assertEqual(api.session.posted[1]["reasoning"], {"effort": "low"})
            self.assertEqual(api.reasoning, {"enabled": False})
            manifest = json.loads((out / "api_manifest.json").read_text())
            self.assertEqual(
                manifest["reasoning_by_model"], {"test/thinking": {"effort": "low"}}
            )

    def test_invalid_reasoning_override_fails_before_external_access(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in (["low"], {"not/requested": {}}, {"test/model": "low"}):
                with patch("urllib.request.urlopen") as catalog:
                    with self.assertRaisesRegex(ValueError, "reasoning_by_model"):
                        BudgetedChat(
                            Path(directory) / "run",
                            ["test/model"],
                            1,
                            reasoning_by_model=value,
                        )
                    catalog.assert_not_called()

    async def test_request_timeout_reaches_session_and_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            for timeout in (300, 900):
                directory = Path(root) / str(timeout)
                kwargs = {} if timeout == 300 else {"request_timeout_seconds": timeout}
                api = self.make_api(directory, **kwargs)
                with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}), patch(
                    "probity_benchmark.api.aiohttp.ClientSession"
                ) as session:
                    await api.__aenter__()
                self.assertEqual(session.call_args.kwargs["timeout"].total, timeout)
                manifest = json.loads((directory / "api_manifest.json").read_text())
                self.assertEqual(manifest["request_timeout_seconds"], timeout)

    def test_invalid_timeout_fails_before_external_access(self):
        with tempfile.TemporaryDirectory() as root:
            for index, timeout in enumerate(
                (0, -1, float("inf"), float("nan"), True, "900")
            ):
                with patch("urllib.request.urlopen") as catalog:
                    with self.assertRaisesRegex(ValueError, "request timeout"):
                        BudgetedChat(
                            Path(root) / str(index),
                            ["test/model"],
                            1,
                            request_timeout_seconds=timeout,
                        )
                    catalog.assert_not_called()

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

    async def test_optional_provider_format_changes_only_payload_constraint(self):
        with tempfile.TemporaryDirectory() as root:
            schema = object_schema({"answer": ANSWER})
            bodies = []
            for mode in (True, False):
                directory = Path(root) / str(mode)
                api = self.make_api(directory, structured_output=mode)
                api.session = Session([Reply('{"answer":"A"}')])
                result = await api.complete(
                    "test/model",
                    [{"role": "user", "content": "Return JSON."}],
                    schema,
                    {},
                )
                self.assertEqual(result["parsed"], {"answer": "A"})
                bodies.append(api.session.posted[0])
                manifest = json.loads((directory / "api_manifest.json").read_text())
                self.assertIs(manifest["structured_output"], mode)
                if not mode:
                    self.assertEqual(result["local_response_schema"], schema)
            constrained = dict(bodies[0])
            self.assertEqual(
                constrained.pop("response_format")["json_schema"]["schema"], schema
            )
            self.assertEqual(constrained, bodies[1])
            self.assertNotIn("response_format", bodies[1])

    async def test_provider_format_disabled_keeps_strict_local_validation(self):
        with tempfile.TemporaryDirectory() as root:
            for index, content in enumerate(
                (
                    '{"answer":"C"}',
                    '{"answer":"A","extra":true}',
                    '{"answer":"A","answer":"B"}',
                    '```json\n{"answer":"A"}\n```',
                )
            ):
                api = self.make_api(Path(root) / str(index), structured_output=False)
                api.session = Session([Reply(content)])
                with self.assertRaises(RuntimeError):
                    await api.complete(
                        "test/model",
                        [{"role": "user", "content": "Return JSON."}],
                        object_schema({"answer": ANSWER}),
                        {},
                    )
                self.assertTrue(api.stopped)
                self.assertFalse(api.records[0]["valid"])
                self.assertAlmostEqual(api.spent, 0.001)
                self.assertEqual(api.records[0]["finish_reason"], "stop")

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

    async def test_normalized_stop_does_not_hide_exposed_truncation(self):
        for flag in ({"forced": True}, {"native_finish_reason": "max_tokens"}):
            with tempfile.TemporaryDirectory() as directory:
                api = self.make_api(Path(directory) / "run", schema_retries=2)
                reply = Reply('{"answer":"A"}')
                reply.payload["choices"][0].update(flag)
                api.session = Session([reply])
                with self.assertRaisesRegex(RuntimeError, "exposed incomplete"):
                    await api.complete(
                        "test/model",
                        [{"role": "user", "content": "Choose."}],
                        object_schema({"answer": ANSWER}),
                        {},
                    )
                self.assertEqual(api.session.calls, 1)
                self.assertTrue(api.stopped)
                self.assertFalse(api.records[0]["valid"])
                self.assertAlmostEqual(api.spent, 0.001)

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
                root / "resumed",
                reuse_receipts=[root / "original" / "requests.jsonl"],
                request_timeout_seconds=900,
            )
            resumed.session = Session([])
            result = await resumed.complete("test/model", messages, schema, {})
            self.assertEqual(result["parsed"], {"answer": "A"})
            self.assertEqual(resumed.session.calls, 0)
            self.assertEqual(resumed.spent, 0)
            self.assertEqual(result["original_cost_usd"], 0.001)
            self.assertEqual(result["cost_usd"], 0)

    async def test_recovery_keeps_identical_prompt_jobs_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = self.make_api(root / "original")
            original.session = Session(
                [Reply('{"answer":"A"}'), Reply('{"answer":"B"}')]
            )
            messages = [{"role": "user", "content": "Choose."}]
            schema = object_schema({"answer": ANSWER})
            for job in ("first", "second"):
                await original.complete("test/model", messages, schema, {"job": job})
            resumed = self.make_api(
                root / "resumed", reuse_receipts=[root / "original" / "requests.jsonl"]
            )
            resumed.session = Session([Reply('{"answer":"B"}')])
            for job, answer in (("first", "A"), ("second", "B")):
                result = await resumed.complete(
                    "test/model", messages, schema, {"job": job}
                )
                self.assertEqual(result["parsed"]["answer"], answer)
            self.assertEqual(resumed.session.calls, 0)
            fresh = await resumed.complete(
                "test/model", messages, schema, {"job": "third"}
            )
            self.assertNotIn("reused_from", fresh)
            self.assertEqual(resumed.session.calls, 1)
            self.assertAlmostEqual(resumed.spent, 0.001)


if __name__ == "__main__":
    unittest.main()
