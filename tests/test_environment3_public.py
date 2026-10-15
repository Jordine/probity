import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from probity_benchmark.environment3_packets import run
from probity_benchmark.environment3_prepared import preparation_messages
from probity_benchmark.environment3_public import audit_reasoning_receipt
from probity_benchmark.preflight import load_problems, sha256
from tests import test_environment3_packets as base_tests


def complete_receipt():
    return {
        "valid": True,
        "finish_reason": "stop",
        "request": {"reasoning": {"enabled": True}, "max_tokens": 16384},
        "usage": {
            "completion_tokens": 100,
            "completion_tokens_details": {"reasoning_tokens": 70},
        },
        "response_body": json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "native_finish_reason": "stop",
                        "message": {
                            "content": '{"answer":"A"}',
                            "reasoning": "Complete thought.\nAll bytes retained.",
                        },
                    }
                ]
            }
        ),
    }


class PublicReceiptTests(unittest.TestCase):
    def test_audit_retains_reasoning_bytes_and_exposes_context_limit(self):
        receipt = complete_receipt()
        before = json.dumps(receipt, sort_keys=True)
        result = audit_reasoning_receipt(receipt)
        self.assertEqual(json.dumps(receipt, sort_keys=True), before)
        self.assertTrue(result["returned_reasoning_available"])
        self.assertEqual(
            result["returned_reasoning_fields"][0]["sha256"],
            sha256("Complete thought.\nAll bytes retained."),
        )
        self.assertIn("Final-JSON-only", result["scope"])

    def test_native_or_reasoning_flags_reject_nominal_stop(self):
        for native, detail in [
            ("length", {}),
            ("stop", {"truncated": True}),
            ("stop", {"forced": True}),
        ]:
            receipt = complete_receipt()
            payload = json.loads(receipt["response_body"])
            payload["choices"][0]["native_finish_reason"] = native
            payload["choices"][0]["message"]["reasoning_details"] = [detail]
            receipt["response_body"] = json.dumps(payload)
            with self.assertRaisesRegex(ValueError, "forced/truncated"):
                audit_reasoning_receipt(receipt)


class ThinkingFakeChat(base_tests.FakeChat):
    instances = []

    def __init__(self, out, models, budget, **kwargs):
        super().__init__(out, models, budget, **kwargs)
        self.reasoning = kwargs.get("reasoning", {"enabled": False})
        self.max_tokens = kwargs.get("max_tokens", 8192)
        self.structured_output = kwargs.get("structured_output", True)
        self.concurrency = kwargs.get("concurrency", 4)
        self.instances.append(self)

    async def complete(self, model, messages, schema, tag):
        receipt = await super().complete(model, messages, schema, tag)
        receipt["request"] = {
            "messages": messages,
            "reasoning": self.reasoning,
            "max_tokens": self.max_tokens,
        }
        receipt["response_body"] = complete_receipt()["response_body"]
        receipt["usage"]["completion_tokens_details"] = {
            "reasoning_tokens": 30 if self.reasoning["enabled"] else 0
        }
        receipt["tag"] = tag
        return receipt


class PublicClientIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_public_turns_use_thinking_and_all_receipts_are_archived(self):
        ThinkingFakeChat.instances = []
        base_tests.FakeChat.calls = []
        base_tests.FakeChat.knowledge_answer = "A"
        with tempfile.TemporaryDirectory() as root:
            args, _ = base_tests.EnvironmentProtocolTests().make_args(
                Path(root),
                protocol="explicit_reasoned_evidence_thinking_sequential",
                prepared_records=Path(root) / "external.jsonl",
                public_budget=0.5,
                public_max_tokens=16384,
                public_timeout_seconds=900,
                public_unconstrained_json=True,
                public_concurrency=8,
            )
            args.prepared_records.write_text("test-only mocked preparation input\n")
            problems, _ = load_problems(args.dataset)
            problem = problems[0]
            external = {}
            for speaker, assigned in [("Red", "A"), ("Blue", "B")]:
                external[(problem.problem_id, "qwen/example", speaker)] = (
                    {
                        "swapped": False,
                        "assigned_answer": assigned,
                        "source_story_sha256": sha256(problem.story),
                    },
                    {
                        "request": {
                            "messages": preparation_messages(
                                problem, False, speaker, assigned, explicit_truth=True
                            )
                        },
                        "parsed": {"evidence_items": []},
                        "request_sha256": speaker,
                    },
                )
            with patch(
                "probity_benchmark.environment3_packets.BudgetedChat", ThinkingFakeChat
            ), patch(
                "probity_benchmark.environment3_packets.load_external_preparations",
                return_value=external,
            ):
                await run(args)
            control, public = ThinkingFakeChat.instances
            self.assertFalse(control.reasoning["enabled"])
            self.assertTrue(public.reasoning["enabled"])
            self.assertTrue(control.structured_output)
            self.assertFalse(public.structured_output)
            self.assertEqual(public.concurrency, 8)
            self.assertEqual(
                [r["tag"]["stage"] for r in control.records], ["knowledge", "judge"]
            )
            self.assertEqual(
                [r["tag"]["stage"] for r in public.records], ["debate"] * 4
            )
            archive = [
                json.loads(x)
                for x in (args.out / "requests.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(archive), 6)
            episode = json.loads(next(args.out.glob("episode_*.json")).read_text())
            self.assertTrue(
                all(
                    t["public_reasoning_audit"]["returned_reasoning_available"]
                    for t in episode["turns"]
                )
            )


if __name__ == "__main__":
    unittest.main()
